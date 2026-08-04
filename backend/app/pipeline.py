"""The single RAG pipeline, owned once and consumed by both /query and /query/stream.

Previously this sequence (cache -> intent guard -> complexity tier -> optimize+expand
-> retrieve -> assemble -> generate -> account -> cache-save) was
implemented twice: semantic_router.route() + main.py's blocking query(), and again
inline in main.py's _stream_pipeline() so it could yield SSE progress. The two copies
drifted apart in small ways (only the blocking path stripped whitespace off the final
answer; only the streaming path could report stage progress). run_pipeline() is now
the one place this logic lives — both endpoints just consume it differently.

Yields (event, data) pairs:
  ("stage", name)           routing | optimizing | retrieving | assembling | generating
  ("token", text)           one generated token
  ("done", PipelineResult)  final result
  ("error", message)

/query/stream forwards these directly as SSE lines. /query drains the generator,
discards "stage"/"token", and returns the "done" payload.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Iterator

from . import (
    cache, generation, indexing, optimizer, prompt_builder, retrieval, savings,
    semantic_router, turn_store,
)
from .config import settings
from .conversation import ConversationMemory
from .schemas import PipelineResult, RetrievedChunk, TokenStats
from .store import collection, count_tokens

PipelineEvent = tuple[str, Any]


def _current_project() -> str:
    """Repo basename for savings attribution, or "" if nothing is indexed yet."""
    try:
        p = indexing.indexed_repo()
        return os.path.basename(os.path.normpath(p)) if p else ""
    except Exception:
        return ""


def _naive_baseline(kept: list[RetrievedChunk]) -> int:
    """Tokens a developer would plausibly have pasted by hand: the full text of
    the files retrieval selected.

    The previous baseline was every chunk in the repo, which measured the saving
    against something nobody would ever do and inflated the number roughly 15x.
    Comparing against "I'd have pasted these files" is a claim that survives
    scrutiny.
    """
    paths = {c.path for c in kept}
    if not paths:
        return 0
    try:
        got = collection(settings.code_collection).get(
            where={"path": {"$in": list(paths)}}, include=["documents"]
        )
        return sum(count_tokens(d) for d in (got.get("documents") or []))
    except Exception:
        # Metrics must never sink a query — fall back to the kept chunks alone.
        return sum(count_tokens(c.snippet) for c in kept)


def _cached_result(
    hit: dict, model_override: str | None, started: float, session_id: str | None = None
) -> PipelineResult:
    """Rebuild a result from a cache hit, including a fresh review-gate turn.

    Entries saved before chunks were cached (or written by an older build) carry
    no `chunks_json`; those degrade to the previous behavior — approvable but not
    refinable — rather than failing the hit.
    """
    answer = hit.get("answer", "")
    mode = hit.get("mode", "answer")
    model = hit.get("model", model_override or settings.large_model)

    chunks: list[RetrievedChunk] = []
    try:
        raw = hit.get("chunks_json")
        if raw:
            chunks = [RetrievedChunk(**c) for c in json.loads(raw)]
    except Exception:
        chunks = []  # malformed payload must not sink an otherwise valid hit

    turn_id = None
    if chunks:
        turn = turn_store.create(
            session_id,
            hit.get("query", ""),
            hit.get("task_text", "") or hit.get("query", ""),
            hit.get("optimized", ""),
            chunks,
            int(hit.get("total_chunks", 0) or len(chunks)),
            model,
            mode,
            answer,
        )
        turn_id = turn.id

    return PipelineResult(
        optimized_query=hit.get("optimized", ""),
        expansions=[],
        chunks=chunks,
        assembled_prompt="(served from cache)",
        answer=answer,
        tokens=TokenStats(naive_baseline=0, optimized=0, saved=int(hit.get("tokens_saved", 0)), saved_pct=100.0),
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        cached=True,
        mode=mode,
        turn_id=turn_id,
    )


def _off_topic_result(query_text: str, started: float) -> PipelineResult:
    return PipelineResult(
        optimized_query=query_text,
        expansions=[],
        chunks=[],
        assembled_prompt="(off-topic guardrail)",
        answer=semantic_router.OFF_TOPIC_REPLY,
        tokens=TokenStats(naive_baseline=0, optimized=0, saved=0, saved_pct=0.0),
        model=settings.small_model,
        latency_ms=int((time.time() - started) * 1000),
        mode="answer",  # a guardrail reply is never a brief, regardless of what was requested
    )


def run_pipeline(
    query_text: str,
    mode: str,
    model_override: str | None,
    top_k: int,
    use_expansion: bool,
    session_id: str | None = None,
) -> Iterator[PipelineEvent]:
    """Full RAG pipeline for query/forge mode. `mode="auto"` defers the answer-vs-
    brief decision to the router (classify_request_type); `"answer"`/`"forge"`
    force that decision explicitly, skipping the extra classification call."""
    started = time.time()

    # 1. Semantic cache — cheapest path. Scoped to mode so a forge run never
    # serves a cached answer-mode essay (and vice versa). An "auto" request
    # hasn't resolved a concrete mode yet — check both scopes at once rather
    # than paying for classification before every cache hit.
    hit = cache.lookup(query_text, ["answer", "forge"] if mode == "auto" else mode)
    if hit:
        # No model ran at all — the strongest form of avoided cost. Credit the
        # context the original run needed, since that's what a fresh answer
        # (local or Copilot) would have had to read again.
        savings.record(
            "cache_hits",
            int(hit.get("context_tokens", 0) or 0),
            today=datetime.now().strftime("%Y-%m-%d"),
            project=_current_project(),
            session=session_id or "",
            mode=hit.get("mode", "answer"),
        )
        yield ("token", hit.get("answer", ""))
        yield ("done", _cached_result(hit, model_override, started, session_id))
        return

    try:
        # 2. Routing — intent guard + complexity tier are independent small-model
        # calls; run them concurrently instead of back-to-back. Request-type
        # classification joins the pool too, but only when the caller left the
        # mode decision to the router — an explicit answer/forge override
        # already knows its mode and shouldn't pay for a call it'll discard.
        yield ("stage", "routing")
        with ThreadPoolExecutor(max_workers=3) as pool:
            intent_future = pool.submit(semantic_router._classify, query_text)
            complexity_future = pool.submit(semantic_router.classify_complexity, query_text)
            request_type_future = (
                pool.submit(semantic_router.classify_request_type, query_text)
                if mode == "auto" else None
            )
            intent = intent_future.result()
            complexity = complexity_future.result()
            request_type = request_type_future.result() if request_type_future else None

        if intent == "off_topic":
            yield ("token", semantic_router.OFF_TOPIC_REPLY)
            yield ("done", _off_topic_result(query_text, started))
            return

        # Resolve "auto" to a concrete mode now, before assembly branches on it.
        resolved_mode = mode if mode != "auto" else ("forge" if request_type == "task" else "answer")

        model = model_override or (
            settings.large_model if complexity == "complex" else settings.small_model
        )

        # 3. Rewrite -> expand (sequential: expand takes the rewritten query, not
        # raw). Simple-tier queries skip expansion entirely — one direct question
        # doesn't need three sub-queries and the extra embeds/searches they cost.
        yield ("stage", "optimizing")
        optimized = optimizer.optimize_query(query_text, model)
        # ponytail: simple-tier skips expansion; revisit if simple queries start
        # missing context that expansion would have caught.
        expansions = (
            optimizer.expand_query(optimized, model)
            if use_expansion and complexity == "complex"
            else []
        )

        # 4. Retrieve top-K (original + expansions).
        yield ("stage", "retrieving")
        chunks = retrieval.retrieve([optimized, *expansions], top_k)

        # 5. Assemble within token budget. Prepend conversation history when present.
        yield ("stage", "assembling")
        code_col = collection(settings.code_collection)
        history_prefix = ConversationMemory.get_context_prefix(session_id) if session_id else ""
        if resolved_mode == "forge":
            # Forge seeds the Task section from the user's real request, not the
            # retrieval-optimized search blob (which reads vague). The optimized
            # + expanded queries are still used for retrieval above.
            task = history_prefix + query_text
            prompt, kept = prompt_builder.build_forge(task, chunks, code_col.count() or len(chunks))
            system_prompt = prompt_builder.FORGE_SYSTEM_PROMPT
        else:
            task = history_prefix + optimized
            prompt, kept = prompt_builder.build(task, chunks, code_col.count() or len(chunks))
            system_prompt = None
    except RuntimeError as e:
        yield ("error", str(e))
        return

    # 6. Generate — always via the streaming client; the blocking /query endpoint
    # just drains this generator and joins the tokens itself.
    yield ("stage", "generating")
    full_answer_parts: list[str] = []
    try:
        for token in generation.generate_stream(prompt, model, system_prompt=system_prompt):
            full_answer_parts.append(token)
            yield ("token", token)
    except RuntimeError as e:
        yield ("error", str(e))
        return

    answer = "".join(full_answer_parts).strip()

    # 7. Token accounting. Count only what the developer actually pays for — the
    # text they receive and paste onward. The assembled prompt is local Ollama
    # compute: it costs nothing, so charging it against the saving both
    # understates the benefit and makes `saved` stop equalling
    # `naive_baseline - optimized`.
    optimized_tokens = count_tokens(answer)
    naive_baseline = _naive_baseline(kept)
    saved = max(0, naive_baseline - optimized_tokens)
    saved_pct = round(saved / naive_baseline * 1000) / 10 if naive_baseline else 0.0

    # 8. Register this turn for the review gate — refine / "just explain it" /
    # "make a brief" regenerate from these exact chunks + task_text, never
    # re-retrieving. `task` here is the frozen history_prefix + query_text (or
    # + optimized) this exact prompt was built from.
    turn = turn_store.create(
        session_id, query_text, task, optimized, kept,
        code_col.count() or len(kept), model, resolved_mode, answer,
    )

    result = PipelineResult(
        optimized_query=optimized,
        expansions=expansions,
        chunks=kept,
        assembled_prompt=prompt,
        answer=answer,
        tokens=TokenStats(
            naive_baseline=naive_baseline,
            optimized=optimized_tokens,
            saved=saved,
            saved_pct=saved_pct,
        ),
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        mode=resolved_mode,
        turn_id=turn.id,
    )

    # 9a. Record the turn. Only answer mode credits an avoided Copilot call — a
    # forged brief is *meant* to be sent to Copilot, so it saves nothing. It's
    # still logged (as brief_built, tokens=0) so the reasoning-vs-prompt-
    # optimization split covers every query rather than only the free ones.
    context_tokens = count_tokens(prompt)
    savings.record(
        "answered_locally" if resolved_mode == "answer" else "brief_built",
        context_tokens,
        today=datetime.now().strftime("%Y-%m-%d"),
        project=_current_project(),
        session=session_id or "",
        mode=resolved_mode,
    )

    # 9b. Store turn in session memory, then persist to history / cache.
    if session_id:
        ConversationMemory.add_turn(session_id, query_text, answer)
    try:
        # Save under the resolved concrete mode, never "auto" — that's not a
        # real cache partition, and step 1's lookup already knows to check
        # both "answer" and "forge" for an "auto" caller.
        cache.save(
            query_text, optimized, answer, model, saved, resolved_mode,
            chunks=kept, task_text=task, total_chunks=turn.total_chunks,
            context_tokens=context_tokens,
        )
    except RuntimeError:
        pass  # don't fail the request if caching the result fails

    yield ("done", result)


def run_review_action(
    turn_id: str,
    action: str,
    note: str | None,
    model_override: str | None,
) -> Iterator[PipelineEvent]:
    """Regenerate a stored turn's content — refine a brief, or convert between
    answer and brief — from its ALREADY-retrieved chunks. Never re-retrieves.

    action:
      "refine"    — rewrite a forge brief given a free-text note (forge-only)
      "to_answer" — "just explain it instead" escape hatch (any mode -> answer)
      "to_brief"  — "make a brief" escape hatch (any mode -> forge)
    """
    started = time.time()
    turn = turn_store.get(turn_id)
    if turn is None:
        yield ("error", f"That brief has expired or is no longer available (id {turn_id}).")
        return
    model = model_override or turn.model

    yield ("stage", "assembling")
    if action == "refine":
        if turn.mode != "forge":
            yield ("error", "Refine is only available for forged briefs.")
            return
        prompt, kept = prompt_builder.refine_forge(
            turn.task_text, turn.text, note or "", turn.chunks, turn.total_chunks
        )
        system_prompt, new_mode = prompt_builder.FORGE_SYSTEM_PROMPT, "forge"
    elif action == "to_answer":
        prompt, kept = prompt_builder.build(turn.task_text, turn.chunks, turn.total_chunks)
        system_prompt, new_mode = None, "answer"
    elif action == "to_brief":
        prompt, kept = prompt_builder.build_forge(turn.task_text, turn.chunks, turn.total_chunks)
        system_prompt, new_mode = prompt_builder.FORGE_SYSTEM_PROMPT, "forge"
    else:
        yield ("error", f"Unknown review action: {action}")
        return

    yield ("stage", "generating")
    full_answer_parts: list[str] = []
    try:
        for token in generation.generate_stream(prompt, model, system_prompt=system_prompt):
            full_answer_parts.append(token)
            yield ("token", token)
    except RuntimeError as e:
        yield ("error", str(e))
        return

    answer = "".join(full_answer_parts).strip()

    # Same accounting rule as run_pipeline: only the delivered text counts.
    optimized_tokens = count_tokens(answer)
    naive_baseline = _naive_baseline(kept)
    saved = max(0, naive_baseline - optimized_tokens)
    saved_pct = round(saved / naive_baseline * 1000) / 10 if naive_baseline else 0.0

    # Fixing the brief here means not fixing it by arguing with Copilot — that
    # correction would otherwise be a second billed round-trip.
    # No session_id on the review path — the turn record doesn't carry one, so
    # this counts toward totals but not the distinct-chat tally.
    savings.record(
        "refined_locally",
        count_tokens(prompt),
        today=datetime.now().strftime("%Y-%m-%d"),
        project=_current_project(),
        session=turn.session_id or "",
        mode=new_mode,
    )

    # Deliberately not cache.save()'d / ConversationMemory.add_turn()'d — a
    # refine/convert result is a pre-approval draft iteration, not a completed
    # top-level answer.
    turn_store.update(turn_id, mode=new_mode, text=answer)

    yield ("done", PipelineResult(
        optimized_query=turn.optimized_query,
        expansions=[],
        chunks=kept,
        assembled_prompt=prompt,
        answer=answer,
        tokens=TokenStats(
            naive_baseline=naive_baseline,
            optimized=optimized_tokens,
            saved=saved,
            saved_pct=saved_pct,
        ),
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        mode=new_mode,
        turn_id=turn_id,
    ))
