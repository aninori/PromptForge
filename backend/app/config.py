"""Central configuration. Reads from environment with sensible local defaults.

Every value here has a matching control in the frontend Settings page, so the UI
and the backend stay in lockstep.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Generation
    provider: str = os.getenv("CF_PROVIDER", "ollama")          # ollama | groq
    gen_model: str = os.getenv("CF_GEN_MODEL", "deepseek-coder-v2:16b")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("CF_GROQ_MODEL", "qwen/qwen3-32b")

    # Embeddings
    embed_model: str = os.getenv("CF_EMBED_MODEL", "nomic-embed-text")
    chunk_size: int = int(os.getenv("CF_CHUNK_SIZE", "800"))     # tokens
    chunk_overlap: int = int(os.getenv("CF_CHUNK_OVERLAP", "120"))
    # Simultaneous embed requests to Ollama. Ollama serializes work per model, so
    # oversubscribing just deepens its queue until individual requests time out.
    # Keep at or below the physical core count on CPU-only hosts.
    embed_concurrency: int = int(os.getenv("CF_EMBED_CONCURRENCY", "4"))
    # Per-request read timeout (seconds). A queued request on a CPU-only box can
    # legitimately wait minutes before Ollama gets to it.
    embed_timeout: int = int(os.getenv("CF_EMBED_TIMEOUT", "300"))

    # Retrieval
    # top_k is the real lever on how much code reaches the model — token_budget
    # never binds at these sizes (4 chunks assembled to ~4,400 tokens against a
    # 6,000 budget). Raised to 8 to roughly double the context the brief is
    # grounded in, at a measured ~20 tokens/sec of CPU prompt processing.
    top_k: int = int(os.getenv("CF_TOP_K", "8"))
    token_budget: int = int(os.getenv("CF_TOKEN_BUDGET", "12000"))
    # Context window requested from Ollama. MUST exceed token_budget with room to
    # spare for the generated reply — Ollama defaults to 4096 regardless of what
    # the model supports, so a 6,000-token prompt was rejected with a bare
    # 400 ("the prompt is longer than the context length currently available").
    # Bigger costs RAM (KV cache) and CPU time per token, linearly: 32768 was
    # measured at 837s for a 17k-token prompt on this host — usable, but barely.
    num_ctx: int = int(os.getenv("CF_NUM_CTX", "16384"))
    use_expansion: bool = os.getenv("CF_USE_EXPANSION", "true").lower() == "true"
    # How much the cross-encoder gets to override vector similarity. It used to
    # get 100% — which let a 2-line README (0.569) displace Sidebar.tsx (0.575),
    # because ms-marco is trained on prose passages and scores English above code.
    # 0.5 = the two scorers share the decision.
    rerank_weight: float = float(os.getenv("CF_RERANK_WEIGHT", "0.5"))
    # Max .md/.txt chunks allowed in a result set. Docs took 4 of 8 slots on a
    # "why does clicking X do Y" query, crowding out every component. A cap, not
    # an exclusion — for "how do I set this up", the README really is the answer.
    max_doc_chunks: int = int(os.getenv("CF_MAX_DOC_CHUNKS", "2"))
    # Rate used for the dashboard's cost estimate. GitHub Copilot is a flat
    # subscription, so avoided tokens genuinely cost $0 there — this prices the
    # hypothetical "what if this context had gone to a per-token API instead of
    # the local model". Default is Claude Sonnet 5 input ($3/M as of 2026-08).
    # The basis string is displayed alongside the figure so it stays checkable.
    price_per_mtok: float = float(os.getenv("CF_PRICE_PER_MTOK", "3.00"))
    price_basis: str = os.getenv("CF_PRICE_BASIS", "Claude Sonnet 5 input rates")
    # A per-file chunk quota was tried here and reverted: it did not recover the
    # missing components on the query that motivated it, and cost the video query
    # 5 components -> 3 by stripping legitimate second chunks from files that
    # genuinely deserved two. Measured, not assumed — don't re-add without data.

    # Infra
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chroma_dir: str = os.getenv("CF_CHROMA_DIR", "./.chroma")
    graph_path: str = os.getenv("CF_GRAPH_PATH", "./.chroma/graph.json")
    bm25_path: str = os.getenv("CF_BM25_PATH", "./.chroma/bm25.pkl")
    code_collection: str = "codebase"
    history_collection: str = "prompt_history"
    summary_collection: str = "file_summaries"
    # Only prefilter by file when chunk count exceeds this — flat search is fine below.
    file_summary_threshold: int = 500

    # Model tiers for semantic routing.
    # small_model handles classification + simple queries; large_model handles
    # complex queries.
    # small_model runs 3-4 times per query (intent, complexity, request-type,
    # query optimization), so on CPU-only hosts its size dominates latency far
    # more than its quality affects output — keep it small.
    small_model: str = os.getenv("CF_SMALL_MODEL", "llama3.2:3b")
    # The intent guard gets its own, larger model. It's the one classifier whose
    # mistakes are user-visible and blocking — a false "off_topic" refuses to
    # answer a legitimate question — and small models are measurably noisy on it
    # (llama3.2:3b wrongly rejected 2/12 real questions; qwen2.5-coder:7b, 0/13).
    # Runs concurrently with the small-model classifiers, so it's off the
    # critical path for everything except its own latency.
    guard_model: str = os.getenv("CF_GUARD_MODEL", "qwen2.5-coder:7b")
    large_model: str = os.getenv("CF_LARGE_MODEL", "deepseek-coder-v2:16b")

    # Below this cosine similarity, a history hit is NOT treated as a cache hit.
    # Must be strictly between 0 and 1; 0.85 balances recall and precision.
    cache_threshold: float = 0.85

    # GitHub integration.
    #   Core (clone/index) works with just a PAT (github_token) or on public repos.
    #   OAuth activates only when client_id + client_secret are set (register a
    #   GitHub OAuth App; callback URL = <backend>/github/callback — localhost is OK).
    #   Webhook activates only when webhook_secret is set AND the backend is
    #   reachable by GitHub (public URL / tunnel).
    github_token: str = os.getenv("CF_GITHUB_TOKEN", "")            # PAT fallback (private repos)
    github_client_id: str = os.getenv("CF_GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("CF_GITHUB_CLIENT_SECRET", "")
    github_webhook_secret: str = os.getenv("CF_GITHUB_WEBHOOK_SECRET", "")
    github_oauth_callback: str = os.getenv("CF_GITHUB_CALLBACK", "http://localhost:8000/github/callback")
    github_workspace: str = os.getenv("CF_GITHUB_WORKSPACE", "./.github_repos")


settings = Settings()
