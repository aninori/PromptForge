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
    """Return top_k chunks reordered by cross-encoder relevance score.

    Falls back to original order[:top_k] if the model is unavailable.
    """
    if not chunks:
        return chunks
    model = _get_model()
    if not model:
        return chunks[:top_k]

    pairs = [(query, c.snippet[:_MAX_SNIPPET_CHARS]) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
    return [c for c, _ in ranked[:top_k]]
