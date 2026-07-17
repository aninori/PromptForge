"""BM25 sparse index — runs alongside the Chroma vector store.

Handles exact identifier matches (function names, variable names, error codes)
that embedding similarity can miss. Built over all chunk documents at index
time and serialized to disk next to graph.json so it survives restarts.

At query time retrieval.py feeds the BM25 ranked list into RRF alongside
the vector search results — no score normalization needed since RRF only
uses rank positions, not raw scores.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import settings

_index: BM25Okapi | None = None
_ids: list[str] = []
_corpus_map: dict[str, list[str]] = {}  # chunk_id → tokenized tokens


def _tokenize(text: str) -> list[str]:
    """Split on non-alphanumeric, keep identifiers and numbers."""
    return re.findall(r"[a-zA-Z_]\w*|\d+", text.lower())


def _cid_to_path(cid: str) -> str:
    """Extract file rel-path from chunk ID (format: rel/path/file.py:N)."""
    return cid.rsplit(":", 1)[0]


def build(ids: list[str], docs: list[str]) -> None:
    """Full rebuild from scratch (used on first index or forced re-index)."""
    global _index, _ids, _corpus_map
    _corpus_map = {cid: _tokenize(doc) for cid, doc in zip(ids, docs)}
    _ids = list(ids)
    _index = BM25Okapi(list(_corpus_map.values())) if _corpus_map else None


def update(changed_paths: set[str], new_ids: list[str], new_docs: list[str]) -> None:
    """Incremental update: drop stale chunks for changed/deleted files, add new ones.

    Avoids the 500k-document Chroma fetch that build() requires; only re-tokenizes
    chunks from files that actually changed.
    # ponytail: BM25Okapi still rebuilt from full corpus on every call (it's
    # a statistical model, no incremental API). Cost = O(total_chunks), but
    # avoids the Chroma round-trip for unchanged docs.
    """
    global _index, _ids, _corpus_map
    for cid in [c for c in list(_corpus_map) if _cid_to_path(c) in changed_paths]:
        del _corpus_map[cid]
    for cid, doc in zip(new_ids, new_docs):
        _corpus_map[cid] = _tokenize(doc)
    _ids = list(_corpus_map.keys())
    _index = BM25Okapi(list(_corpus_map.values())) if _corpus_map else None


def query(text: str, n: int) -> list[tuple[str, float]]:
    """Return up to n (chunk_id, score) pairs, highest score first."""
    if _index is None or not _ids:
        return []
    tokens = _tokenize(text)
    scores = _index.get_scores(tokens)
    ranked = sorted(zip(_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [(cid, score) for cid, score in ranked[:n] if score > 0.0]


def save() -> None:
    path = Path(settings.bm25_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"ids": _ids, "index": _index, "corpus_map": _corpus_map}, f)


def load() -> None:
    global _index, _ids, _corpus_map
    path = Path(settings.bm25_path)
    if not path.exists():
        return
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        _ids = data["ids"]
        _index = data["index"]
        _corpus_map = data.get("corpus_map", {})  # backward compat — old pickle lacks it
    except (OSError, KeyError, pickle.UnpicklingError):
        _index = None
        _ids = []
        _corpus_map = {}


# Populate from disk when the module is first imported.
load()
