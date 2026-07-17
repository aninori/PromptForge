"""Retrieval — embed the (optimized) query, search Chroma, return top-K chunks.

Pipeline per query call:
  1. Vector search  — dense cosine similarity via Chroma (multi-query)
  2. BM25 search    — sparse exact-match on the primary (optimized) query
  3. RRF merge      — Reciprocal Rank Fusion across all ranked lists
  4. Graph expansion — import/call neighbors of the top seeds
  5. Cross-encoder rerank — (query, chunk) joint scoring for final precision

Feedback boost: chunks with net positive votes receive up to +15% score boost,
closing the RLHF loop started by the /feedback endpoint.
"""
from __future__ import annotations

from . import bm25_index, ollama_client, reranker
from .config import settings
from .graph_memory import get_graph
from .schemas import RetrievedChunk
from .store import collection

_RRF_K = 60  # standard RRF constant — higher = gentler rank differences


def _file_prefilter(query_vec: list[float], k: int = 10) -> list[str] | None:
    """Return top-k file paths from the summary collection, or None when below threshold.

    Only activates when chunk count exceeds settings.file_summary_threshold so small
    codebases keep flat search behaviour.
    """
    scol = collection(settings.summary_collection)
    if scol.count() == 0 or collection(settings.code_collection).count() <= settings.file_summary_threshold:
        return None
    res = scol.query(query_embeddings=[query_vec], n_results=k)
    metas = res.get("metadatas", [[]])[0]
    return [m["path"] for m in metas if m.get("path")] or None


def _search(query: str, k: int, col, file_filter: list[str] | None = None,
            vec: list[float] | None = None) -> list[RetrievedChunk]:
    """Vector search for a single query; returns chunks ordered by cosine similarity."""
    if col.count() == 0:
        return []
    if vec is None:
        vec = ollama_client.embed(query)
    kwargs: dict = {"query_embeddings": [vec], "n_results": k}
    if file_filter:
        kwargs["where"] = {"path": {"$in": file_filter}}
    res = col.query(**kwargs)
    chunks: list[RetrievedChunk] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        base_score = max(0.0, 1.0 - float(dist))
        boosted = _apply_feedback_boost(base_score, meta)
        chunks.append(RetrievedChunk(
            id=cid,
            path=meta.get("path", "?"),
            lines=meta.get("lines", ""),
            score=round(boosted, 3),
            snippet=doc,
        ))
    return chunks


def _apply_feedback_boost(base_score: float, meta: dict) -> float:
    """Boost score by up to ±15% based on accumulated user votes."""
    net_votes = int(meta.get("votes_up", 0)) - int(meta.get("votes_down", 0))
    if net_votes == 0:
        return base_score
    factor = 1.0 + max(-0.15, min(0.15, net_votes * 0.05))
    return min(1.0, base_score * factor)


def _rrf_merge(ranked_lists: list[list[RetrievedChunk]]) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple query result lists.

    Returns {chunk_id: rrf_score}. Higher is better.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (_RRF_K + rank)
    return scores


def retrieve(queries: list[str], top_k: int) -> list[RetrievedChunk]:
    col = collection(settings.code_collection)

    # 1a. File-level prefilter — narrows chunk search to the most relevant files
    #     when the codebase is large (above settings.file_summary_threshold chunks).
    #     Embed the primary query once and reuse for the prefilter to avoid a
    #     redundant embed call inside _search on the first iteration.
    primary_vec = ollama_client.embed(queries[0])
    file_filter = _file_prefilter(primary_vec)

    # 1b. Vector search across all queries, collect per-query ranked lists.
    chunk_by_id: dict[str, RetrievedChunk] = {}
    ranked_lists: list[list[RetrievedChunk]] = []
    for i, q in enumerate(queries):
        ranked = _search(q, top_k, col, file_filter=file_filter,
                         vec=primary_vec if i == 0 else None)
        ranked_lists.append(ranked)
        for chunk in ranked:
            if chunk.id not in chunk_by_id or chunk.score > chunk_by_id[chunk.id].score:
                chunk_by_id[chunk.id] = chunk

    # 2. BM25 hybrid: sparse search on the primary (optimized) query.
    #    BM25 results are added as an extra ranked list into RRF — no score
    #    normalization needed since RRF only cares about rank positions.
    bm25_hits = bm25_index.query(queries[0], top_k * 2)
    if bm25_hits:
        max_bm25 = bm25_hits[0][1] or 1.0
        bm25_ranked: list[RetrievedChunk] = []
        for cid, raw_score in bm25_hits:
            if cid in chunk_by_id:
                bm25_ranked.append(chunk_by_id[cid])
            else:
                got = col.get(ids=[cid], include=["documents", "metadatas"])  # type: ignore[arg-type]
                if got and got.get("ids"):
                    ndoc = str((got.get("documents") or [""])[0])
                    nmeta: dict = (got.get("metadatas") or [{}])[0]  # type: ignore[assignment]
                    norm = _apply_feedback_boost(raw_score / max_bm25, nmeta)
                    c = RetrievedChunk(
                        id=cid,
                        path=str(nmeta.get("path", "?")),
                        lines=str(nmeta.get("lines", "")),
                        score=round(norm, 3),
                        snippet=ndoc,
                    )
                    chunk_by_id[cid] = c
                    bm25_ranked.append(c)
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)

    # 3. RRF merge to get a consensus ranking across vector + BM25 lists.
    rrf_scores = _rrf_merge(ranked_lists)

    # 4. Graph expansion: pull in import neighbors of top-scoring chunks.
    graph = get_graph()
    top_seeds = sorted(chunk_by_id.values(), key=lambda c: rrf_scores.get(c.id, 0.0), reverse=True)
    for seed in top_seeds[: max(2, top_k // 2)]:
        for nid in graph.neighbors(seed.id, depth=1):
            if nid in chunk_by_id:
                continue
            got = col.get(ids=[nid], include=["documents", "metadatas"])  # type: ignore[arg-type]
            if got and got.get("ids"):
                ndoc = str((got.get("documents") or [""])[0])
                nmeta: dict = (got.get("metadatas") or [{}])[0]  # type: ignore[assignment]
                neighbor_score = _apply_feedback_boost(seed.score * 0.85, nmeta)
                chunk_by_id[nid] = RetrievedChunk(
                    id=nid,
                    path=str(nmeta.get("path", "?")),
                    lines=str(nmeta.get("lines", "")),
                    score=round(neighbor_score, 3),
                    snippet=ndoc,
                )
                rrf_scores[nid] = seed.score * 0.85 / _RRF_K

    # 5. Sort by RRF score, then cross-encoder rerank on expanded candidate pool.
    ranked_final = sorted(
        chunk_by_id.values(),
        key=lambda c: rrf_scores.get(c.id, c.score),
        reverse=True,
    )
    return reranker.rerank(queries[0], ranked_final[: top_k * 2], top_k)
