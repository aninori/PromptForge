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

    # Retrieval
    top_k: int = int(os.getenv("CF_TOP_K", "4"))
    token_budget: int = int(os.getenv("CF_TOKEN_BUDGET", "6000"))
    use_expansion: bool = os.getenv("CF_USE_EXPANSION", "true").lower() == "true"

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

    # Heuristic: tokens an average chunk would occupy if you dumped the whole
    # repo into a frontier model. Used only to compute the "naive baseline".
    avg_tokens_per_chunk: int = 85

    # Model tiers for semantic routing.
    # small_model handles classification + simple queries; large_model handles
    # complex queries and all agent runs.
    small_model: str = os.getenv("CF_SMALL_MODEL", "mistral:7b")
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
