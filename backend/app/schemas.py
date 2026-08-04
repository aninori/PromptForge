"""Request/response models.

The response shapes mirror the TypeScript interfaces in
`src/lib/mockData.ts` exactly (camelCase), so the frontend can swap mocks for
`fetch()` without touching any component.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class IndexRequest(CamelModel):
    path: str = Field(..., description="Local folder or git URL to index")
    name: str | None = None


class IndexResponse(CamelModel):
    name: str
    files: int
    chunks: int
    indexed_at: str
    embed_model: str
    vector_db: str = "Chroma"
    size_mb: float


class RepoEntry(CamelModel):
    name: str
    path: str
    type: str


class RepoTreeResponse(CamelModel):
    root_path: str
    current_path: str
    parent_path: str | None = None
    entries: list[RepoEntry]


class QueryRequest(CamelModel):
    query: str
    # None = use the configured value. A literal default here shadowed
    # settings.top_k entirely, so /config advertised one number while every
    # query used another. Callers can still override per-request.
    top_k: int | None = None
    use_expansion: bool | None = None
    model: str | None = None
    session_id: str | None = None
    mode: str = "answer"  # "answer" | "forge" | "auto" (router decides answer vs. forge)


class RetrievedChunk(CamelModel):
    id: str
    path: str
    lines: str
    score: float
    snippet: str


class TokenStats(CamelModel):
    naive_baseline: int
    optimized: int
    saved: int
    saved_pct: float


class PipelineResult(CamelModel):
    optimized_query: str
    expansions: list[str]
    chunks: list[RetrievedChunk]
    assembled_prompt: str
    answer: str
    tokens: TokenStats
    model: str
    latency_ms: int
    cached: bool = False
    mode: str = "answer"  # resolved concrete mode: "answer" | "forge" (never "auto")
    turn_id: str | None = None


class HistoryItem(CamelModel):
    id: str
    query: str
    optimized_query: str
    date: str
    model: str
    tokens_saved: int
    cached: bool


class FeedbackRequest(CamelModel):
    query_id: str = Field(..., description="ID of the query that was answered")
    feedback: str = Field(..., description="'up' or 'down' — relevancy signal")
    chunk_ids: list[str] | None = None
    """Optional list of chunk IDs included in the prompt. When present, the
    feedback signal is attributed to individual chunks for per-file re-ranking."""


class ReviewActionRequest(CamelModel):
    turn_id: str
    action: str  # "refine" | "to_answer" | "to_brief"
    note: str | None = None
    model: str | None = None


