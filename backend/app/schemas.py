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
    top_k: int = 4
    use_expansion: bool = True
    model: str | None = None
    session_id: str | None = None
    mode: str = "answer"  # "answer" | "forge"


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


class AgentCard(CamelModel):
    id: str
    name: str
    role: str
    purpose: str
    model: str
    focus: str
    color: str
    default_top_k: int = 6
    use_expansion: bool = True


class AgentRunRequest(CamelModel):
    agent_id: str = "auto"
    repo_path: str = "."
    target_path: str
    user_request: str
    logs: str = ""
    attachments: list[str] = []
    attachment_names: list[str] = []
    task_type: str | None = None
    top_k: int = 6
    use_expansion: bool = False
    model: str | None = None


class CodebaseSummary(CamelModel):
    root_path: str
    file_count: int
    languages: list[str]
    frameworks: list[str]
    services: list[str]
    storage: list[str]
    signals: list[str]
    risk_flags: list[str]


class AgentFinding(CamelModel):
    severity: str
    title: str
    detail: str
    files: list[str] = []


class AgentLogItem(CamelModel):
    stage: str
    message: str


class AgentRunResult(CamelModel):
    agent_id: str
    agent_name: str
    model: str
    target_path: str
    summary: str
    architecture: CodebaseSummary
    relevant_files: list[RetrievedChunk]
    findings: list[AgentFinding]
    plan: list[str]
    prompt: str
    answer: str
    patch_diff: str = ""
    patch_files: list[str] = []
    logs: list[AgentLogItem]
    latency_ms: int
