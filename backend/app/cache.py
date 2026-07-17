"""Prompt history as a semantic cache.

Every completed run is embedded and stored. Before doing the work for a new
query, we check whether a near-identical question was already answered; if the
similarity clears `cache_threshold`, we return the cached answer and skip
retrieval + generation entirely — the cheapest possible path.

TTL: entries older than _CACHE_TTL_DAYS are ignored on lookup.
Eviction: after every save, oldest entries are pruned when count > _MAX_CACHE_ENTRIES.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime

from . import ollama_client
from .config import settings
from .schemas import HistoryItem
from .store import collection

_CACHE_TTL_DAYS = 30
_MAX_CACHE_ENTRIES = 500


def lookup(raw_query: str, mode: str = "answer") -> dict | None:
    col = collection(settings.history_collection)
    if col.count() == 0:
        return None
    vec = ollama_client.embed(raw_query)
    # Scope the lookup to the same mode — an answer-mode essay must never be
    # served for a forge-mode request (or vice versa).
    res = col.query(query_embeddings=[vec], n_results=1, where={"mode": mode})
    dists = res.get("distances", [[]])[0]
    if not dists:
        return None
    similarity = 1.0 - float(dists[0])
    if similarity < settings.cache_threshold:
        return None
    meta = res.get("metadatas", [[]])[0][0]
    # TTL check — expired entries are ignored and will be pruned on the next save.
    age_seconds = time.time() - float(meta.get("ts", 0))
    if age_seconds > _CACHE_TTL_DAYS * 86_400:
        return None
    return meta  # contains the stored result payload (JSON-serialized fields)


def _evict_oldest(col) -> None:
    """Delete the oldest entries when the cache exceeds _MAX_CACHE_ENTRIES."""
    count = col.count()
    if count <= _MAX_CACHE_ENTRIES:
        return
    got = col.get(include=["metadatas"])
    ids: list[str] = got.get("ids") or []
    metas: list[dict] = got.get("metadatas") or []
    if not ids:
        return
    pairs = sorted(zip(ids, metas), key=lambda x: float(x[1].get("ts", 0)))
    to_delete = [cid for cid, _ in pairs[: count - _MAX_CACHE_ENTRIES]]
    if to_delete:
        col.delete(ids=to_delete)


def save(raw_query: str, optimized: str, answer: str, model: str, tokens_saved: int, mode: str = "answer") -> None:
    col = collection(settings.history_collection)
    col.add(
        ids=[str(uuid.uuid4())],
        documents=[raw_query],
        embeddings=[ollama_client.embed(raw_query)],
        metadatas=[{
            "query": raw_query,
            "optimized": optimized,
            "answer": answer,
            "model": model,
            "tokens_saved": tokens_saved,
            "mode": mode,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ts": time.time(),
        }],
    )
    _evict_oldest(col)


def list_history(limit: int = 50) -> list[HistoryItem]:
    col = collection(settings.history_collection)
    if col.count() == 0:
        return []
    got = col.get(include=["metadatas"])
    metas = got.get("metadatas", []) or []
    items = [
        HistoryItem(
            id=str(i),
            query=m.get("query", ""),
            optimized_query=m.get("optimized", ""),
            date=m.get("date", ""),
            model=m.get("model", ""),
            tokens_saved=int(m.get("tokens_saved", 0)),
            cached=False,
        )
        for i, m in enumerate(metas)
    ]
    items.sort(key=lambda x: x.date, reverse=True)
    return items[:limit]
