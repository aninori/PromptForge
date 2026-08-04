"""backend/app/turn_store.py — addressable per-turn state for the review gate.

Lets refine / "just explain it" / "make a brief" regenerate content from
already-retrieved chunks without re-running retrieval. Keyed by a per-turn id
(NOT session_id — several turns in one chat transcript can have pending
review-gate buttons simultaneously; session_id alone can't disambiguate which
one a button click refers to).

Single-process, but mirrored to a JSON file so a backend restart doesn't strand
every pending review-gate button with "that brief has expired".
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import settings
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_TTL_SECONDS = 2 * 60 * 60   # 2 hours of inactivity
_MAX_TURNS = 200
_STORE_NAME = "turns.json"


@dataclass
class TurnRecord:
    id: str
    session_id: str | None
    original_query: str            # raw user text, display only
    task_text: str                 # EXACT string passed to build()/build_forge() originally —
                                    # frozen so refine can't silently pick up a different
                                    # conversation-history prefix than the brief actually saw
    optimized_query: str
    chunks: list[RetrievedChunk]    # the exact kept chunks from the original retrieval
    total_chunks: int               # snapshot for budget/token-savings math parity
    model: str
    mode: str                       # current concrete mode: "answer" | "forge" — mutates on convert
    text: str                       # current generated text — mutates on refine/convert
    history: list[str] = field(default_factory=list)  # capped 2-entry prior-version ring
    updated_ts: float = field(default_factory=time.time)


_turns: dict[str, TurnRecord] = {}


# ---------- persistence ----------
# Best-effort throughout: a corrupt or unwritable turns.json must degrade to the
# old in-memory behavior, never fail the query that triggered the write.

def _store_path() -> Path:
    return Path(settings.chroma_dir) / _STORE_NAME


def _to_dict(t: TurnRecord) -> dict:
    # asdict() only recurses into dataclasses; RetrievedChunk is a pydantic
    # model, so it would survive as a non-JSON-serializable object.
    d = asdict(t)
    d["chunks"] = [c.model_dump() for c in t.chunks]
    return d


def _save() -> None:
    try:
        p = _store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Explicit utf-8: Path.write_text defaults to the Windows ANSI codepage,
        # which would choke the moment json.dumps stops escaping non-ASCII.
        p.write_text(json.dumps([_to_dict(t) for t in _turns.values()]), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not persist turn store: %s", e)


def _load() -> None:
    p = _store_path()
    if not p.exists():
        return
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read turn store (%s); starting empty", e)
        return
    now = time.time()
    for d in raw:
        try:
            # Apply the same TTL to whatever was on disk, so a stale file can't
            # resurrect turns that would have expired while the backend was down.
            if now - d.get("updated_ts", 0) > _TTL_SECONDS:
                continue
            d["chunks"] = [RetrievedChunk(**c) for c in d.get("chunks", [])]
            t = TurnRecord(**d)
            _turns[t.id] = t
        except Exception as e:
            logger.warning("Skipping unreadable stored turn: %s", e)
    if _turns:
        logger.info("Restored %d stored turn(s) from %s", len(_turns), p)


_load()


def create(
    session_id: str | None,
    original_query: str,
    task_text: str,
    optimized_query: str,
    chunks: list[RetrievedChunk],
    total_chunks: int,
    model: str,
    mode: str,
    text: str,
) -> TurnRecord:
    t = TurnRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        original_query=original_query,
        task_text=task_text,
        optimized_query=optimized_query,
        chunks=chunks,
        total_chunks=total_chunks,
        model=model,
        mode=mode,
        text=text,
    )
    _turns[t.id] = t
    _evict()
    _save()
    return t


def get(turn_id: str) -> TurnRecord | None:
    return _turns.get(turn_id)


def update(turn_id: str, *, mode: str | None = None, text: str | None = None) -> TurnRecord | None:
    t = _turns.get(turn_id)
    if t is None:
        return None
    if text is not None:
        t.history.append(t.text)
        del t.history[:-2]  # keep only the last 2 prior versions
        t.text = text
    if mode is not None:
        t.mode = mode
    t.updated_ts = time.time()
    _save()
    return t


def _evict() -> None:
    """TTL pass + max-entries pass, mirroring cache.py's _evict_oldest."""
    now = time.time()
    stale = [tid for tid, t in _turns.items() if now - t.updated_ts > _TTL_SECONDS]
    for tid in stale:
        del _turns[tid]
    if len(_turns) > _MAX_TURNS:
        overflow = len(_turns) - _MAX_TURNS
        for t in sorted(_turns.values(), key=lambda t: t.updated_ts)[:overflow]:
            del _turns[t.id]
