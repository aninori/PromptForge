"""Smoke check for pipeline.py — the single generator /query and /query/stream both
consume. No pytest (none of this repo's dependencies pull it in yet); run directly:

    python -m app.test_pipeline

Monkeypatches every external call (Ollama, Chroma, cache) with canned values so this
runs in milliseconds with no live services. Verifies the two things that broke when
this logic lived in three separate places: stage events fire in order, and the
simple-tier query-expansion skip actually skips (Step 4 of the consolidation).
"""
from __future__ import annotations

import json

from . import cache, generation, optimizer, pipeline, prompt_builder, retrieval, semantic_router, turn_store
from .schemas import PipelineResult, RetrievedChunk

_CHUNKS = [RetrievedChunk(id="a.py:0", path="a.py", lines="1-10", score=0.9, snippet="def a(): ...")]


class _FakeCollection:
    def count(self) -> int:
        return 1


def _run_uncached_case() -> None:
    expand_calls: list[str] = []

    def fake_expand(*_a, **_kw):
        expand_calls.append("called")
        return ["should not run for simple tier"]

    patches = {
        (cache, "lookup"): lambda *_a, **_kw: None,
        (semantic_router, "_classify"): lambda *_a, **_kw: "codebase",
        (semantic_router, "classify_complexity"): lambda *_a, **_kw: "simple",
        (optimizer, "optimize_query"): lambda *_a, **_kw: "optimized query",
        (optimizer, "expand_query"): fake_expand,
        (retrieval, "retrieve"): lambda *_a, **_kw: list(_CHUNKS),
        (prompt_builder, "build"): lambda *_a, **_kw: ("assembled prompt", list(_CHUNKS)),
        (generation, "generate_stream"): lambda *_a, **_kw: iter(["hel", "lo "]),
        (cache, "save"): lambda *_a, **_kw: None,
        (pipeline, "collection"): lambda *_a, **_kw: _FakeCollection(),
    }
    originals = {(mod, name): getattr(mod, name) for mod, name in patches}
    for (mod, name), fn in patches.items():
        setattr(mod, name, fn)
    try:
        events = list(pipeline.run_pipeline("how does auth work?", "answer", None, 4, True))
    finally:
        for (mod, name), fn in originals.items():
            setattr(mod, name, fn)

    stages = [data for ev, data in events if ev == "stage"]
    assert stages == ["routing", "optimizing", "retrieving", "assembling", "generating"], stages

    assert not expand_calls, "expand_query ran for a simple-tier query — Step 4 regression"

    done = [data for ev, data in events if ev == "done"]
    assert len(done) == 1, events
    result = done[0]
    assert isinstance(result, PipelineResult)
    assert result.answer == "hello", result.answer
    assert result.expansions == []
    assert [c.id for c in result.chunks] == [c.id for c in _CHUNKS]
    assert not result.cached


def _run_auto_case() -> None:
    """mode="auto" must resolve via classify_request_type and assemble a forge
    brief (not an answer) when the classifier says "task" — and must NOT call
    classify_request_type at all for an explicit mode (proven implicitly by
    _run_uncached_case above, which never patches it and still passes)."""
    build_forge_calls: list[str] = []

    def fake_build_forge(task, *_a, **_kw):
        build_forge_calls.append(task)
        return ("assembled forge prompt", list(_CHUNKS))

    patches = {
        (cache, "lookup"): lambda *_a, **_kw: None,
        (semantic_router, "_classify"): lambda *_a, **_kw: "codebase",
        (semantic_router, "classify_complexity"): lambda *_a, **_kw: "simple",
        (semantic_router, "classify_request_type"): lambda *_a, **_kw: "task",
        (optimizer, "optimize_query"): lambda *_a, **_kw: "optimized query",
        (optimizer, "expand_query"): lambda *_a, **_kw: [],
        (retrieval, "retrieve"): lambda *_a, **_kw: list(_CHUNKS),
        (prompt_builder, "build_forge"): fake_build_forge,
        (generation, "generate_stream"): lambda *_a, **_kw: iter(["a", " brief"]),
        (cache, "save"): lambda *_a, **_kw: None,
        (pipeline, "collection"): lambda *_a, **_kw: _FakeCollection(),
    }
    originals = {(mod, name): getattr(mod, name) for mod, name in patches}
    for (mod, name), fn in patches.items():
        setattr(mod, name, fn)
    try:
        events = list(pipeline.run_pipeline("fix the login bug", "auto", None, 4, True))
    finally:
        for (mod, name), fn in originals.items():
            setattr(mod, name, fn)

    assert build_forge_calls, "auto+task should assemble via build_forge, not build"
    result = [data for ev, data in events if ev == "done"][0]
    assert result.mode == "forge", result.mode
    assert result.answer == "a brief", result.answer
    assert result.turn_id, "a completed run_pipeline call must register a turn_store record"
    assert turn_store.get(result.turn_id) is not None


def _run_review_action_case() -> None:
    """refine / to_answer / to_brief must regenerate from the stored turn's
    ALREADY-retrieved chunks and never call retrieval.retrieve() — that's the
    hard "never re-retrieve" requirement, enforced here by making a call raise."""
    turn = turn_store.create(
        session_id="s1", original_query="fix the bug", task_text="fix the bug",
        optimized_query="fix the bug", chunks=list(_CHUNKS), total_chunks=1,
        model="mistral:7b", mode="forge", text="=== Task ===\noriginal brief\n",
    )

    def retrieve_must_not_run(*_a, **_kw):
        raise AssertionError("refine/convert must never call retrieval.retrieve()")

    patches = {
        (retrieval, "retrieve"): retrieve_must_not_run,
        (generation, "generate_stream"): lambda *_a, **_kw: iter(["revised brief"]),
    }
    originals = {(mod, name): getattr(mod, name) for mod, name in patches}
    for (mod, name), fn in patches.items():
        setattr(mod, name, fn)
    try:
        events = list(pipeline.run_review_action(turn.id, "refine", "make it shorter", None))
    finally:
        for (mod, name), fn in originals.items():
            setattr(mod, name, fn)

    result = [data for ev, data in events if ev == "done"][0]
    assert result.answer == "revised brief", result.answer
    assert result.mode == "forge", result.mode
    assert turn_store.get(turn.id).text == "revised brief"
    assert turn_store.get(turn.id).history == ["=== Task ===\noriginal brief\n"]

    # to_answer converts the same turn to a plain answer, still no retrieval.
    for (mod, name), fn in patches.items():
        setattr(mod, name, fn)
    try:
        events = list(pipeline.run_review_action(turn.id, "to_answer", None, None))
    finally:
        for (mod, name), fn in originals.items():
            setattr(mod, name, fn)
    result = [data for ev, data in events if ev == "done"][0]
    assert result.mode == "answer", result.mode

    # Unknown turn id -> a clean error event, not an exception.
    events = list(pipeline.run_review_action("does-not-exist", "refine", None, None))
    assert events[0][0] == "error", events


def _run_cached_case() -> None:
    hit = {"answer": "cached answer", "optimized": "opt", "model": "mistral:7b", "tokens_saved": 42}
    saved = cache.lookup
    cache.lookup = lambda *_a, **_kw: hit
    try:
        events = list(pipeline.run_pipeline("how does auth work?", "answer", None, 4, True))
    finally:
        cache.lookup = saved

    assert [ev for ev, _ in events] == ["token", "done"], events
    result = events[-1][1]
    assert result.cached is True
    assert result.answer == "cached answer"
    # Legacy entry (no chunks_json) stays approvable but not refinable.
    assert result.turn_id is None, result.turn_id


def _run_cached_refinable_case() -> None:
    """A cache hit that stored its chunks must come back refinable — otherwise
    asking the same thing twice silently loses the Refine button."""
    chunk = RetrievedChunk(id="a.py:0", path="a.py", lines="L1-9", score=0.9, snippet="def f(): ...")
    hit = {
        "answer": "cached brief",
        "optimized": "opt",
        "model": "m",
        "tokens_saved": 42,
        "mode": "forge",
        "query": "raw question",
        "task_text": "frozen task text",
        "total_chunks": 7,
        "chunks_json": json.dumps([chunk.model_dump()]),
    }
    saved = cache.lookup
    cache.lookup = lambda *_a, **_kw: hit
    try:
        events = list(pipeline.run_pipeline("how does auth work?", "auto", None, 4, True))
    finally:
        cache.lookup = saved

    result = events[-1][1]
    assert result.cached is True and result.mode == "forge", (result.cached, result.mode)
    assert result.turn_id, "cache hit with stored chunks must expose a turn_id"
    assert [c.id for c in result.chunks] == ["a.py:0"], result.chunks

    # And that turn must actually be refinable from the restored chunks.
    turn = turn_store.get(result.turn_id)
    assert turn is not None and turn.task_text == "frozen task text", turn
    assert turn.total_chunks == 7, turn.total_chunks

    # A malformed payload must degrade, not explode.
    bad = dict(hit, chunks_json="{not json")
    cache.lookup = lambda *_a, **_kw: bad
    try:
        events = list(pipeline.run_pipeline("how does auth work?", "auto", None, 4, True))
    finally:
        cache.lookup = saved
    result = events[-1][1]
    assert result.cached is True and result.turn_id is None, result.turn_id


def demo() -> None:
    _run_uncached_case()
    _run_auto_case()
    _run_review_action_case()
    _run_cached_case()
    _run_cached_refinable_case()
    print("pipeline.py: all checks passed")


if __name__ == "__main__":
    demo()
