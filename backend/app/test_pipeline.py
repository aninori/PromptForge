"""Smoke check for pipeline.py — the single generator /query and /query/stream both
consume. No pytest (none of this repo's dependencies pull it in yet); run directly:

    python -m app.test_pipeline

Monkeypatches every external call (Ollama, Chroma, cache) with canned values so this
runs in milliseconds with no live services. Verifies the two things that broke when
this logic lived in three separate places: stage events fire in order, and the
simple-tier query-expansion skip actually skips (Step 4 of the consolidation).
"""
from __future__ import annotations

from . import cache, generation, optimizer, pipeline, prompt_builder, retrieval, semantic_router
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
        (semantic_router, "select_agent"): lambda *_a, **_kw: None,
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


def demo() -> None:
    _run_uncached_case()
    _run_cached_case()
    print("pipeline.py: all checks passed")


if __name__ == "__main__":
    demo()
