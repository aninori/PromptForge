"""backend/app/savings.py — counts work that never reached the paid model.

The old "tokens saved" figure compared against pasting the whole repo into a
prompt, which nobody would ever do. Copilot lives inside VS Code and reads files
itself, so avoided *pasting* is not a real saving.

What IS real is work that never became a Copilot call at all:

  answered_locally — a question the local model answered outright. Copilot was
                     never invoked, so its cost for that turn was zero.
  refined_locally  — a brief rewritten on this machine. Without it the bad
                     prompt goes to Copilot and gets corrected in a second
                     round-trip; each of those is billed.
  cache_hits       — a repeat question served from the cache. No model ran.

`tokens` on each event is the context that turn actually consumed — the assembled
prompt we measured, not a guess. The reasoning: answering the question required
reading that much code, and Copilot would have had to read comparable context to
answer it too. Code doesn't shrink because a different model reads it.

This is deliberately an UNDER-estimate. It counts input context only: not
Copilot's replies, and not the extra files Copilot reads while hunting.
Being wrong in the low direction is the safe direction.

Storage is global (beside the vector store, not inside the indexed repo), so the
totals survive switching projects. Events carry the project and chat they came
from; events recorded before that attribution existed live in `earlier`.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

_STORE_NAME = "savings.json"
_MAX_EVENTS = 5000  # oldest dropped past this; keeps the JSON small

# Kinds that represent an avoided Copilot call — these carry tokens.
_AVOIDED_KINDS = ("answered_locally", "refined_locally", "cache_hits")
# A forged brief is *meant* to go to Copilot, so it saves nothing. Recorded with
# tokens=0 purely so the query-type split can count it; excluded from every
# savings total. Counting its tokens would rebuild the inflated metric this
# module exists to replace.
_KINDS = _AVOIDED_KINDS + ("brief_built",)


@dataclass
class Event:
    kind: str      # one of _KINDS
    tokens: int    # context this turn consumed (measured, not estimated)
    project: str = ""   # repo basename at the time; "" when unknown
    session: str = ""   # chat id — counted distinct, never displayed
    date: str = ""      # YYYY-MM-DD
    mode: str = ""      # "answer" (reasoning) | "forge" (prompt optimization)


@dataclass
class Earlier:
    """Events recorded before per-event attribution existed. Kept separate and
    shown as its own bucket rather than being silently attributed to whichever
    project happened to be open when the format changed."""
    answered_locally: int = 0
    refined_locally: int = 0
    cache_hits: int = 0
    context_tokens: int = 0
    since: str = ""


_events: list[Event] = []
_earlier = Earlier()


def _store_path() -> Path:
    return Path(settings.chroma_dir) / _STORE_NAME


def _save() -> None:
    try:
        p = _store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({
                "events": [asdict(e) for e in _events],
                "earlier": asdict(_earlier),
            }),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Could not persist savings counters: %s", e)


def _load() -> None:
    global _events, _earlier
    p = _store_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "events" in data:
            _events = [Event(**e) for e in data.get("events", [])]
            _earlier = Earlier(**data.get("earlier", {}))
        else:
            # Pre-attribution format: the whole file was one aggregate. Migrate
            # it wholesale into `earlier` — those counts are real, we just can't
            # say which project or chat produced them.
            _earlier = Earlier(**data)
            _events = []
    except Exception as e:
        logger.warning("Could not read savings counters (%s); starting at zero", e)


_load()


def record(
    kind: str,
    context_tokens: int = 0,
    *,
    today: str = "",
    project: str = "",
    session: str = "",
    mode: str = "",
) -> None:
    """Log one turn. Never raises — a counter must not be able to fail the query
    that triggered it."""
    try:
        if kind not in _KINDS:
            return
        # A brief is sent onward to Copilot, so it contributes no saved tokens
        # however it was called.
        tokens = 0 if kind == "brief_built" else max(0, context_tokens)
        _events.append(Event(
            kind=kind,
            tokens=tokens,
            project=project or "",
            session=session or "",
            date=today or "",
            mode=mode or "",
        ))
        if len(_events) > _MAX_EVENTS:
            del _events[: len(_events) - _MAX_EVENTS]
        _save()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not record savings event: %s", e)


def totals() -> dict:
    """Aggregates for the dashboard. camelCase on the wire — this is a plain
    dict, so FastAPI won't apply the CamelModel alias generator for us."""
    counts = {k: 0 for k in _KINDS}
    tokens = 0
    by_project: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "calls": 0})
    by_day: dict[str, int] = defaultdict(int)
    sessions: set[str] = set()
    reasoning = optimization = 0

    for e in _events:
        if e.kind in counts:
            counts[e.kind] += 1
        tokens += e.tokens
        if e.mode == "forge" or e.kind == "brief_built":
            optimization += 1
        elif e.mode == "answer":
            reasoning += 1
        if e.kind != "brief_built":
            name = e.project or "unknown"
            by_project[name]["tokens"] += e.tokens
            by_project[name]["calls"] += 1
            if e.date:
                by_day[e.date] += e.tokens
        if e.session:
            sessions.add(e.session)

    earlier_calls = (
        _earlier.answered_locally + _earlier.refined_locally + _earlier.cache_hits
    )
    total_tokens = tokens + _earlier.context_tokens
    return {
        "answeredLocally": counts["answered_locally"] + _earlier.answered_locally,
        "refinedLocally": counts["refined_locally"] + _earlier.refined_locally,
        "cacheHits": counts["cache_hits"] + _earlier.cache_hits,
        "contextTokens": total_tokens,
        # brief_built is deliberately excluded — a brief is sent to Copilot.
        "copilotCallsAvoided": (
            sum(counts[k] for k in _AVOIDED_KINDS) + earlier_calls
        ),
        "byType": {"reasoning": reasoning, "promptOptimization": optimization},
        # A hypothetical, not a saving you booked: GitHub Copilot is a flat
        # subscription, so avoided tokens cost $0 there. This is what the same
        # context would have cost had it gone to a per-token API instead of the
        # local model. Rate and model name travel with the number so it can be
        # checked rather than believed.
        "costEstimate": {
            "usd": round(total_tokens / 1_000_000 * settings.price_per_mtok, 2),
            "perMTok": settings.price_per_mtok,
            "basis": settings.price_basis,
        },
        "chats": len(sessions),
        "since": _earlier.since or (min((e.date for e in _events if e.date), default="")),
        "byProject": sorted(
            ({"name": n, **v} for n, v in by_project.items()),
            key=lambda r: r["tokens"],
            reverse=True,
        ),
        "byDay": [{"date": d, "tokens": t} for d, t in sorted(by_day.items())][-30:],
        # Kept visibly separate: real counts, but no project/chat attribution.
        "earlier": {"tokens": _earlier.context_tokens, "calls": earlier_calls},
    }


def reset() -> None:
    global _events, _earlier
    _events = []
    _earlier = Earlier()
    _save()
