# PromptForge — Architecture Overview

> A "vibe coding" RAG system that indexes local codebases, optimizes natural-language queries, retrieves relevant code chunks via semantic search + graph memory, and generates answers (or agent-guided patches) using local LLMs (Ollama) or cloud providers (Groq).

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React / Vite)                       │
│  Repository.tsx  │  Workbench.tsx  │  History.tsx  │  Agents Page    │
└────────────────────────┬────────────────────────────────────────────┘
                         │  REST / SSE (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI / Python)                      │
│                                                                       │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Indexing │  │ Optimizer │  │ Retrieval  │  │ Agent Runner     │  │
│  │  .py     │  │  .py      │  │  .py       │  │  .py             │  │
│  └────┬─────┘  └─────┬─────┘  └─────┬──────┘  └────────┬─────────┘  │
│       │               │              │                   │           │
│       ▼               ▼              ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Ollama Client (ollama_client.py)           │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │    │
│  │  │ Embeddings │  │ Chat (local) │  │ Chat Stream (SSE)  │   │    │
│  │  └────────────┘  └──────────────┘  └────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│       │               │              │                   │           │
│       ▼               ▼              ▼                   ▼           │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  Chroma  │  │  Cache    │  │Graph Memory│  │  Prompt Builder  │  │
│  │ (Vector) │  │ (Semantic)│  │ (Adjacency)│  │  (Token Budget)  │  │
│  └──────────┘  └───────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Local File System  │
              │  (indexed repos)    │
              └─────────────────────┘
```

### Architectural Style: **RAG Pipeline + Agent Orchestration**

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline with three distinct modes, all fronted by a **semantic router** (`semantic_router.py`) that classifies intent and picks the model tier before any expensive work runs:

1. **Quick Query Mode** (`/query` & `/query/stream`) — Route → Optimize → Retrieve → Assemble → Generate
2. **Forge Mode** (`/query` & `/query/stream` with `mode=forge`) — Route → Optimize → Retrieve → Assemble (6-section template) → Generate a paste-ready prompt for Copilot/Claude (never answers the task itself)
3. **Agent Mode** (`/agents/run` & `/agents/stream`) — Intake → Analyze → Query-Optimize → Retrieve → Plan → Generate → Patch

> **Not multi-agent.** Agent Mode selects **one** agent per request by keyword score — there is no orchestrator, no parallel agents, and no aggregator. The extra "agent" steps are additional *pipeline stages*, not additional agents.

### Semantic Router & Model Tiers (`semantic_router.py`)

Every query passes through the router before the RAG pipeline:

1. **Cache check** — return immediately on a near-identical prior question.
2. **Intent guard** — classify `codebase` vs `off_topic` (uses `small_model`); off-topic queries get a guardrail reply in **all** modes, before any generation.
3. **Complexity tier** — classify `simple` vs `complex` (uses `small_model`) to pick the generation model:

| Signal | Model chosen |
|--------|--------------|
| Off-topic (any mode) | Guardrail reply — no generation |
| Agent mode | `large_model` (`deepseek-coder-v2:16b`) |
| Complex query (query/forge) | `large_model` |
| Simple query (query/forge) | `small_model` (`mistral:7b`) |
| Explicit `req.model` override | that model, tier ignored |

Classification always uses `small_model` — fast binary decisions that don't need the larger model's reasoning depth.

---

## 2. Core Pipeline Stages

### Stage 1: Indexing (`indexing.py`)
- **Offline process** run once per repo (or on-demand via the Repository page)
- Walks the file system, skipping `node_modules`, `.git`, `dist`, etc.
- Splits each file into **overlapping chunks** (~800 tokens each, 120-token overlap)
- Embeds each chunk using `nomic-embed-text` (Ollama)
- Stores embeddings + metadata in **Chroma** vector DB
- Builds a **graph memory** by parsing import statements across files

### Stage 2: Query Optimization (`optimizer.py`)
- Runs a **local LLM** (e.g., `deepseek-coder-v2:16b`) to:
  - **Rewrite** the user's rough request into a rich, self-contained search query
  - **Expand** into 3 sub-queries capturing different angles of the request
- Costs nothing on paid APIs — runs entirely on the local machine
- **Semantic cache** (`cache.py`) checks prior queries first (cosine similarity ≥ 0.85, with a 30-day TTL and 500-entry cap)

### Stage 3: Retrieval (`retrieval.py`)
- **File-level prefilter** (large repos only): on codebases above `file_summary_threshold` chunks, first searches a `file_summaries` collection to narrow to the most relevant files, then restricts chunk search to those files
- Embeds each (optimized) query via Ollama and searches Chroma for top-K chunks per query (**dense** vector search)
- **BM25 sparse search** (`bm25_index.py`): exact-identifier matching that embeddings miss; corpus is updated incrementally (only changed files re-tokenized)
- **RRF merge** (Reciprocal Rank Fusion, k=60): fuses dense + sparse ranked lists by rank position
- **Graph memory expansion** (`graph_memory.py`): traverses import/call graph neighbors of top chunks (with a 0.85× score penalty for indirect matches)
- **Feedback boost**: chunks with net positive votes get up to ±15% score adjustment
- **Cross-encoder rerank** (`reranker.py`, `ms-marco-MiniLM-L-6-v2`): joint (query, chunk) scoring for final precision, returns the best chunks

### Stage 4: Prompt Assembly (`prompt_builder.py`)
- Builds the final LLM prompt within a **configurable token budget** (default 6000)
- Adds chunks highest-relevancy first, dropping once the budget is exceeded
- Controls actual token spend — the only real cost optimization step

### Stage 5: Generation (`generation.py`)
- Sends the assembled prompt to the model (Ollama local or Groq cloud)
- Supports both:
  - **Blocking**: returns the full answer at once
  - **Streaming (SSE)**: yields tokens progressively to the frontend

### Stage 6: Agent Orchestration (`agent_runner.py` + `agent_registry.py`)
- Sits **on top** of the RAG pipeline for specialized workflows
- Four agents defined in `agent_registry.py`:

| Agent | Role | Purpose |
|-------|------|---------|
| 🔍 **Debug Agent** | Specialist | Bug triage, stack traces, surgical fixes |
| 🏗️ **Refactor Pro** | Architect | Legacy cleanup, modularization, safe refactors |
| 📝 **Doc Bot** | Writer | Doc generation, API references, onboarding notes |
| 🔒 **Security Auditor** | Security | Vulnerability detection, auth review, injection risks |

- Each agent has:
  - A **system prompt** and **output format**
  - **Retrieval bias** keywords for boosting relevant chunks
  - Configurable `top_k` and `use_expansion` flags
  - A **plan** of 3 steps it follows
  - **Patch generation** — produces a unified diff for the user to apply

### Stage 7: Feedback Loop (`main.py` → `/feedback`)
- Thumbs up/down per query result
- Votes stored as metadata on individual Chroma chunks (`votes_up`, `votes_down`)
- Foundation for future per-file re-ranking

---

## 3. Data Flow (Quick Query)

```
User: "Add rate limiting to the API"

┌─ Semantic Router ───────────────────────┐
│  Cache check → intent guard (codebase?) │
│  → complexity tier → pick model         │
│  (simple→mistral:7b, complex→deepseek)  │
└─────────────────────────────────────────┘
                    │
┌─ Optimizer ───────▼──────────────────────┐
│  "rewrite... Add rate limiting to the   │
│   FastAPI endpoints in src/routes/..."  │ (rich contextual query)
│  "expand..." → 3 sub-queries (chained   │
│   off the rewrite, not the raw query)   │
└─────────────────────────────────────────┘
                    │
┌─ Retrieval ───────▼──────────────────────┐
│  (file prefilter on large repos)         │
│  Dense (Chroma) + BM25 sparse → RRF      │
│  Graph memory → neighbor expansion       │
│  Cross-encoder rerank → top-K chunks     │
└─────────────────────────────────────────┘
                    │
┌─ Prompt Builder ─▼───────────────────────┐
│  header + relevant files + footer        │
│  (within token budget)                   │
└─────────────────────────────────────────┘
                    │
┌─ Generation ─────▼───────────────────────┐
│  Ollama / Groq → answer (or SSE stream)  │
└─────────────────────────────────────────┘
                    │
┌─ Cache ──────────▼───────────────────────┐
│  Save query + result to history          │
│  (semantic cache for future lookups)     │
└─────────────────────────────────────────┘
```

---

## 4. Key Design Decisions

### 4.1 Local-First Optimization
- Query optimization and embedding run **locally on Ollama** — zero API cost
- The expensive generation step can use either local models or Groq
- Token budget in `prompt_builder.py` prevents runaway context windows

### 4.2 Graph Memory (Import/Call Graph)
- Built during indexing by parsing imports from Python, JS/TS, Go, Java, Rust, Ruby, PHP, C/C++
- Adds **cross-file context** that pure embedding similarity would miss
- Configurable traversal depth (default: 1 hop)
- Indirect neighbor chunks get a 0.85× score penalty to prioritize direct matches

### 4.3 Agent System
- Agents are **prompt templates + retrieval biases** — no fine-tuning required
- `_infer_agent_id()` auto-selects the right agent based on keywords in the request/logs
- Each agent produces a structured output: summary, relevant files, findings, plan, answer, patch diff
- Streaming agent runs use a **background thread** + queue to yield progress events

### 4.4 Frontend-Backend Contract
- Schemas (`schemas.py`) use camelCase aliases to match TypeScript interfaces (`mockData.ts`)
- Two transport modes:
  - **REST**: `/query`, `/agents/run` for simple request/response
  - **SSE (Server-Sent Events)**: `/query/stream`, `/agents/stream` for progressive UI updates
- Config endpoint (`/config`) lets the frontend adapt to backend settings dynamically

### 4.5 Caching Strategy
- **Semantic cache**: looks up past queries by cosine similarity (threshold: 0.85); entries expire after 30 days and the oldest are evicted past 500
- Cache entry records: original query, optimized query, answer, model used, tokens saved
- Cache is also the **history store** (serves both `/query` fast-path and `/history` page)

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + TypeScript + Vite + Tailwind CSS |
| **Backend** | Python 3.11+ / FastAPI |
| **Vector DB** | Chroma (`chromadb`) |
| **Local LLM** | Ollama — `deepseek-coder-v2:16b` (large tier), `mistral:7b` (small tier + classification), `nomic-embed-text` (embeddings) |
| **Cloud LLM** | Groq (`qwen/qwen3-32b`) — optional |
| **Sparse search** | BM25 (`rank-bm25`) |
| **Reranker** | Cross-encoder (`sentence-transformers`, `ms-marco-MiniLM-L-6-v2`) |
| **File watcher** | `watchfiles` (auto re-index on change) |
| **SQL/History** | Chroma collections (codebase + prompt_history) |
| **Configuration** | Environment variables (`CF_*` prefix) |
| **Streaming** | Server-Sent Events (SSE) |

---

## 6. Project Structure

```
PromptForge/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes (HTTP endpoints)
│   │   ├── config.py            # Environment-based configuration
│   │   ├── schemas.py           # Pydantic models (matching TS interfaces)
│   │   ├── semantic_router.py   # Intent guard + complexity tier + model selection
│   │   ├── ollama_client.py     # Ollama embedding + chat + stream
│   │   ├── optimizer.py         # Query rewriting + expansion (chained, local LLM)
│   │   ├── retrieval.py         # Dense + BM25 + RRF + graph + rerank
│   │   ├── bm25_index.py        # BM25 sparse index (incremental corpus map)
│   │   ├── reranker.py          # Cross-encoder (query, chunk) reranking
│   │   ├── graph_memory.py      # Import/call graph (networkx)
│   │   ├── ast_chunker.py       # AST-aware code chunking
│   │   ├── indexing.py          # Offline: files → chunks → embed → store (incremental)
│   │   ├── watcher.py           # Background file watcher → auto re-index
│   │   ├── prompt_builder.py    # Token-budget-aware prompt assembly (+ forge template)
│   │   ├── generation.py        # Blocking + streaming generation
│   │   ├── cache.py             # Semantic cache (cosine similarity, TTL + eviction)
│   │   ├── conversation.py      # Per-session conversation memory
│   │   ├── agent_runner.py      # Agent orchestration workflow
│   │   ├── agent_registry.py    # Agent definitions (Debug, Refactor, Docs, Security)
│   │   └── store.py             # Chroma collection helpers + token counting
│   ├── requirements.txt
│   └── run.sh
├── src/
│   ├── pages/
│   │   ├── Workbench.tsx        # RAG query + answer panel
│   │   ├── Repository.tsx       # Repo browser + indexer
│   │   ├── History.tsx          # Past queries + saved results
│   │   └── Settings/            # Configuration UI
│   ├── lib/
│   │   ├── api.ts               # Live backend client (REST + SSE)
│   │   └── mockData.ts          # Mock data for development
│   └── components/
└── ARCHITECTURE.md
```

---

## 7. Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CF_PROVIDER` | `ollama` | LLM provider (`ollama` / `groq`) |
| `CF_GEN_MODEL` | `deepseek-coder-v2:16b` | Default generation model |
| `CF_SMALL_MODEL` | `mistral:7b` | Small tier — classification + simple queries |
| `CF_LARGE_MODEL` | `deepseek-coder-v2:16b` | Large tier — complex queries + all agent runs |
| `CF_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CF_GROQ_MODEL` | `qwen/qwen3-32b` | Groq cloud model (when `CF_PROVIDER=groq`) |
| `CF_TOP_K` | `4` | Default number of chunks to retrieve |
| `CF_TOKEN_BUDGET` | `6000` | Max tokens for assembled prompt |
| `CF_USE_EXPANSION` | `true` | Enable multi-query expansion |
| `CF_CHUNK_SIZE` | `800` | Tokens per chunk during indexing |
| `CF_CHUNK_OVERLAP` | `120` | Overlap tokens between chunks |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GROQ_API_KEY` | — | Groq API key (for cloud generation) |

---

## 8. API Endpoints   

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + indexed chunk count |
| `GET` | `/config` | Runtime configuration for frontend |
| `PATCH` | `/config` | Update runtime settings in-memory |
| `GET` | `/repo/tree` | Browse local file system |
| `POST` | `/index` | Index a repo/folder into Chroma (starts the file watcher) |
| `GET` | `/watcher/status` | Background watcher state (`active`, `path`) |
| `POST` | `/cache/clear` | Clear the semantic cache |
| `POST` | `/query` | Full RAG pipeline (blocking) — Query & Forge modes |
| `POST` | `/query/stream` | Full RAG pipeline (SSE streaming) — Query & Forge modes |
| `POST` | `/feedback` | Record relevancy feedback (up/down) |
| `GET` | `/history` | Past query history |
| `GET` | `/agents` | List available agents |
| `POST` | `/agents` | Create a custom agent |
| `POST` | `/agents/run` | Run an agent workflow (blocking) |
| `POST` | `/agents/stream` | Run an agent workflow (SSE streaming) |

---

## 9. Running the System

```bash
# 1. Start Ollama (must have models pulled)
ollama pull deepseek-coder-v2:16b   # large tier
ollama pull mistral:7b              # small tier + classification
ollama pull nomic-embed-text        # embeddings

# 2. Start the backend
cd backend
pip install -r requirements.txt
bash run.sh           # or: uvicorn app.main:app --reload --port 8000

# 3. Start the frontend (separate terminal)
npm install
npm run dev           # → http://localhost:3000

# 4. Open the Repository page → index a local codebase
# 5. Use the Workbench page to query your indexed code