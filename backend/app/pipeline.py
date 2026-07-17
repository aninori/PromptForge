"""The single RAG pipeline, owned once and consumed by both /query and /query/stream.

Previously this sequence (cache -> intent guard -> complexity tier -> optimize+expand
-> retrieve -> agent select -> assemble -> generate -> account -> cache-save) was
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

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

from . import cache, generation, optimizer, prompt_builder, retrieval, semantic_router
from .config import settings
from .conversation import ConversationMemory
from .schemas import PipelineResult, TokenStats
from .store import collection, count_tokens

PipelineEvent = tuple[str, Any]


def _cached_result(hit: dict, model_override: str | None, started: float) -> PipelineResult:
    return PipelineResult(
        optimized_query=hit.get("optimized", ""),
        expansions=[],
        chunks=[],
        assembled_prompt="(served from cache)",
        answer=hit.get("answer", ""),
        tokens=TokenStats(naive_baseline=0, optimized=0, saved=int(hit.get("tokens_saved", 0)), saved_pct=100.0),
        model=hit.get("model", model_override or settings.large_model),
        latency_ms=int((time.time() - started) * 1000),
        cached=True,
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
    )


def run_pipeline(
    query_text: str,
    mode: str,
    model_override: str | None,
    top_k: int,
    use_expansion: bool,
    session_id: str | None = None,
) -> Iterator[PipelineEvent]:
    """Full RAG pipeline for query/forge mode. Agent mode has its own runner (agent_runner.py)."""
    started = time.time()

    # 1. Semantic cache — cheapest path. Scoped to mode so a forge run never
    # serves a cached answer-mode essay (and vice versa).
    hit = cache.lookup(query_text, mode)
    if hit:
        yield ("token", hit.get("answer", ""))
        yield ("done", _cached_result(hit, model_override, started))
        return

    try:
        # 2. Routing — intent guard + complexity tier are independent small-model
        # calls; run them concurrently instead of back-to-back.
        yield ("stage", "routing")
        with ThreadPoolExecutor(max_workers=2) as pool:
            intent_future = pool.submit(semantic_router._classify, query_text)
            complexity_future = pool.submit(semantic_router.classify_complexity, query_text)
            intent = intent_future.result()
            complexity = complexity_future.result()

        if intent == "off_topic":
            yield ("token", semantic_router.OFF_TOPIC_REPLY)
            yield ("done", _off_topic_result(query_text, started))
            return

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

        # 5. Agent selection — pick the best-matching agent (if any) by keyword
        # scoring. Forge mode always uses its fixed system prompt instead.
        agent_id = semantic_router.select_agent(query_text, chunks) if mode == "answer" else None

        # 6. Assemble within token budget. Prepend conversation history when present.
        yield ("stage", "assembling")
        code_col = collection(settings.code_collection)
        history_prefix = ConversationMemory.get_context_prefix(session_id) if session_id else ""
        if mode == "forge":
            # Forge seeds the Task section from the user's real request, not the
            # retrieval-optimized search blob (which reads vague). The optimized
            # + expanded queries are still used for retrieval above.
            task = history_prefix + query_text
            prompt, kept = prompt_builder.build_forge(task, chunks, code_col.count() or len(chunks))
            system_prompt = prompt_builder.FORGE_SYSTEM_PROMPT
        else:
            task = history_prefix + optimized
            prompt, kept = prompt_builder.build(task, chunks, code_col.count() or len(chunks))
            if agent_id:
                from .agent_registry import AGENTS
                ag = AGENTS[agent_id]
                system_prompt = f"{ag.system_prompt}\n\n{ag.output_format}"
            else:
                system_prompt = None
    except RuntimeError as e:
        yield ("error", str(e))
        return

    # 7. Generate — always via the streaming client; the blocking /query endpoint
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

    # 8. Token accounting.
    optimized_tokens = count_tokens(prompt) + count_tokens(answer)
    naive_baseline = (code_col.count() or len(kept)) * settings.avg_tokens_per_chunk
    saved = max(0, naive_baseline - optimized_tokens)
    saved_pct = round(saved / naive_baseline * 1000) / 10 if naive_baseline else 0.0

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
    )

    # 9. Store turn in session memory, then persist to history / cache.
    if session_id:
        ConversationMemory.add_turn(session_id, query_text, answer)
    try:
        cache.save(query_text, optimized, answer, model, saved, mode)
    except RuntimeError:
        pass  # don't fail the request if caching the result fails

    yield ("done", result)
