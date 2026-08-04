"""Cross-encoder reranker — re-scores (query, chunk) pairs jointly.

Unlike bi-encoder embeddings which score query and document independently,
a cross-encoder reads both together and understands relevance rather than
just similarity. This eliminates "close but wrong" chunks that score high
on cosine distance but don't actually answer the query.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~22 MB, CPU-friendly, <100ms for 20 candidates.

Lazy-loaded on first call so startup time is unaffected. Falls back to the
original RRF ordering if sentence-transformers is not installed.
"""
from __future__ import annotations

from .config import settings
from .schemas import RetrievedChunk

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_SNIPPET_CHARS = 1500  # truncate to keep inference fast

_model = None  # None = not loaded yet; False = unavailable


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _model = CrossEncoder(_RERANK_MODEL, max_length=512)
        except ImportError:
            _model = False
    return _model


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Return top_k chunks ranked by a blend of cross-encoder relevance and the
    incoming vector/RRF score.

    Blended, not replaced. This model is trained on MS MARCO web-search
    passages, so it scores fluent English above code: letting it decide alone
    put a 2-line README (vector 0.569) ahead of Sidebar.tsx (0.575) on a query
    about click behaviour. Giving the retrieval score an equal vote keeps the
    cross-encoder's judgement without letting prose bias override it outright.

    Falls back to original order[:top_k] if the model is unavailable.
    """
    if not chunks:
        return chunks
    model = _get_model()
    if not model:
        return chunks[:top_k]

    pairs = [(query, c.snippet[:_MAX_SNIPPET_CHARS]) for c in chunks]
    raw = [float(s) for s in model.predict(pairs)]

    # Cross-encoder output is an unbounded logit (often negative) — not on the
    # same 0-1 scale as chunk.score, so min-max it across this candidate set
    # before the two can be averaged meaningfully.
    lo, hi = min(raw), max(raw)
    span = (hi - lo) or 1.0
    w = settings.rerank_weight

    for c, s in zip(chunks, raw):
        # Write the blended value back so the score shown in the brief matches
        # the ordering that actually produced it.
        c.score = round(w * ((s - lo) / span) + (1.0 - w) * c.score, 3)

    return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
