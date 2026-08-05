# PromptForge — Architecture Overview

> A "vibe coding" RAG system that indexes local codebases, optimizes natural-language queries, retrieves relevant code chunks via semantic search + graph memory, and generates answers or paste-ready prompts for Copilot using local LLMs (Ollama) or cloud providers (Groq).

---

## 1. High-Level Architecture

> **No web frontend exists in this repo.** Earlier revisions of this doc described
> a planned React/Vite UI (`Repository.tsx`, `Workbench.tsx`, `History.tsx`); that
> was never built (or was removed) — there is no `src/` and no root `package.json`.
> The only client today is the VS Code chat extension below.

```
┌─────────────────────────────────────────────────────────────────────┐
│              CLIENT — VS Code Chat Participant (vscode-extension/)   │
│  @promptforge <message> — no command needed; /query /forge /history  │
│  /model stay as hidden power-user overrides. extension.ts spawns +   │
│  health-checks uvicorn at activation, consumes SSE, renders streamed │
│  tokens as chat markdown, and gates task briefs behind Approve /     │
│  Refine buttons — nothing reaches Copilot without a click.           │
└────────────────────────┬────────────────────────────────────────────┘
                         │  REST / SSE (JSON, camelCase)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI / Python)                      │
│                                                                       │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Indexing │  │ Optimizer │  │ Retrieval  │  │ Semantic Router  │  │
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

### Architectural Style: **RAG Pipeline + Review Gate**

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline with two modes, both fronted by a **semantic router** (`semantic_router.py`) that classifies intent and picks the model tier before any expensive work runs. The user never chooses a mode: a plain `@promptforge <message>` sends `mode="auto"`, and the router's third classifier decides answer vs. brief.

1. **Quick Query Mode** (`mode="answer"`, or `"auto"` classified as a question) — Route → Optimize → Retrieve → Assemble → Generate, streamed straight to the chat.
2. **Forge Mode** (`mode="forge"`, or `"auto"` classified as a task) — Route → Optimize → Retrieve → Assemble (6-section template) → Generate a paste-ready prompt for Copilot. The result is held behind a **review gate** (Approve / Refine / "just explain it") — nothing is dispatched to Copilot until the user clicks Approve.

`/query` and `/forge` remain in the chat command list as explicit overrides — a power user forcing a concrete mode skips the third classifier entirely (see below).

### Semantic Router & Model Tiers (`semantic_router.py`)

Every query passes through the router before the RAG pipeline:

1. **Cache check** — return immediately on a near-identical prior question. An `"auto"` request checks both the `answer` and `forge` cache scopes at once (via Chroma's `$in`), since it hasn't resolved a concrete mode yet.
2. **Intent guard** — classify `codebase` vs `off_topic` (uses `guard_model`, **not** `small_model`); off-topic queries get a guardrail reply in **all** modes, before any generation. This is the one classifier whose errors are user-visible and blocking — a false `off_topic` refuses a legitimate question — and small models are measurably noisy on it (`llama3.2:3b` wrongly rejected 2/12 real questions; `qwen2.5-coder:7b`, 0/13). Its prompt is deliberately biased toward `codebase`: it's told every message comes from a developer about their own project, so "why are the videos not working?" is a bug report, not generic tech support. Measured cost of the larger model: none — it runs concurrently with the small-model classifiers.
3. **Complexity tier** — classify `simple` vs `complex` (uses `small_model`) to pick the generation model.
4. **Request-type classifier** (`classify_request_type`, `mode="auto"` only) — classify `question` vs `task` (uses `small_model`); defaults to `task` when ambiguous, since briefs are the product's main purpose and a wrong guess is cheap to correct via the review gate's escape hatches. Skipped entirely when the caller already forced `"answer"`/`"forge"` explicitly.

| Signal | Model chosen |
|--------|--------------|
| Off-topic (any mode) | Guardrail reply — no generation |
| Complex query (query/forge) | `large_model` |
| Simple query (query/forge) | `small_model` (`llama3.2:3b`) |
| Explicit `req.model` override | that model, tier ignored |

Classification always uses `small_model` — fast binary decisions that don't need the larger model's reasoning depth. All three classifiers on an `"auto"` request run concurrently in the same thread pool as the intent/complexity calls, since none of them need the optimized query or retrieved chunks.

---

## 2. Core Pipeline Stages

### Stage 1: Indexing (`indexing.py`)
- **Automatic** — the extension kicks off indexing at activation (no chat command); `watcher.py` then handles incremental re-indexing on file changes. `POST /index` also stays available directly for scripting/GitHub-integration use.
- **One repo at a time.** The Chroma collections are global and chunk ids are repo-relative paths, so two codebases cannot coexist in the index. `index_repo()` records the indexed repo's absolute path in `.chroma/indexed_repo.json`; when asked to index a *different* root it resets the code + summary collections, the BM25 corpus, and the import graph before re-indexing, and forces a non-incremental pass (the per-file hashes belong to the old repo). `/health` reports `indexedRepo` so the extension can detect a workspace switch and re-index automatically.
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
- **Cross-encoder rerank** (`reranker.py`, `ms-marco-MiniLM-L-6-v2`): joint (query, chunk) scoring for final precision. **Blended, not substituted** — the cross-encoder score is min-max normalised (it's an unbounded logit, not comparable to the 0–1 vector score) and averaged with the incoming score via `settings.rerank_weight` (0.5). Letting it decide alone put a 2-line `README.md` (0.569) above `Sidebar.tsx` (0.575): the model is trained on MS MARCO *web-search prose*, so it rates fluent English above code.
- **Doc quota** (`_cap_docs`, `settings.max_doc_chunks` = 2): both scorers share that prose bias, so documentation crowded out implementation — docs took 4 of 8 slots on a "why does clicking X do Y" query, returning zero components. A cap, not an exclusion: for "how do I set this project up" the README genuinely is the answer.
  - Measured on a video-playback query: components went 0 → 5 and `depth.ts` (data-modelling prose) fell from rank 1 (0.94) to rank 4.
  - **A per-file chunk quota was tried and reverted.** It recovered no components on the query it targeted and cost the video query 5 components → 3 by stripping legitimate second chunks. Don't re-add without measurements.
  - **Known unfixed:** a query repeating a domain word ("module" ×3) still lets the file named for it (`modules.ts`, 0.945) outrank the component implementing the behaviour. Quotas can't fix this — filtering only reshuffles what already ranked. The real cause is representational: the file prefilter summarises each file by its **first chunk** (imports/boilerplate for a `.tsx`), and code embeds weakly against natural-language questions.

### Stage 4: Prompt Assembly (`prompt_builder.py`)
- Builds the final LLM prompt within a **configurable token budget** (`CF_TOKEN_BUDGET`, default 12000)
- Adds chunks highest-relevancy first, dropping once the budget is exceeded
- Controls actual token spend — the only real cost optimization step
- **Anti-fabrication rules.** A confidently wrong brief is worse than no brief: the
  developer trusts it and the coding agent acts on it. Three rules, carried in
  `FORGE_SYSTEM_PROMPT`, `_FORGE_HEADER`, and the trailing `_FORGE_TEMPLATE`
  (belt-and-braces — the template is the last text the model reads):
  1. If the user reported a problem without naming its cause, the Task states the
     **symptom** and asks for investigation. It never asserts a cause. (The paired
     rule lives in `optimizer.py`'s `_REWRITE`, which otherwise turns "why are the
     videos not working?" into a fabricated "fix the video source URL format".)
  2. Tech Stack / Constraints / Do NOT must be traceable to the user's message or
     the retrieved context; otherwise they read `"None specified."`
  3. Never name a language, library, or tool unless that name **literally appears**
     in the context. The abstract "only state what you can point to" phrasing was
     measurably insufficient — the model still inferred Redux from the presence of
     React in a project with no Redux dependency. Naming the failure mode fixed it.
- These rules apply to refine automatically: `refine_forge()` reuses
  `_FORGE_TEMPLATE`, and `_REFINE_HEADER` carries the same grounding clause (refine
  previously *amplified* fabrication rather than correcting it).
- **Relevant Files points, it does not copy.** The section lists `path (Lstart-Lend)`
  plus a sub-10-word verb-first note, and explicitly forbids pasting code: the
  agent receiving the brief already has the repo open, so a copied snippet only
  adds tokens and can *contradict* the real file if it is stale or truncated
  mid-expression. The word limit is not decoration — without it the model replaced
  the code with padding ("This file matters because it contains the code that…")
  and the brief grew 36% instead of shrinking.
- **Do not confuse the two prompts.** Retrieved chunk bodies still go into the
  prompt sent to the *local* model (`_fit_chunks` → `# Codebase context`); that is
  the evidence the anti-fabrication rules check against. Only the model's *written
  output* is snippet-free. Stripping the context body would silently reintroduce
  invented libraries.

### Stage 5: Generation (`generation.py`)
- Sends the assembled prompt to the model (Ollama local or Groq cloud)
- Supports both:
  - **Blocking**: returns the full answer at once
  - **Streaming (SSE)**: yields tokens progressively to the client

### Stage 6: Feedback Loop (`main.py` → `/feedback`)
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
│  (simple→llama3.2:3b, complex→deepseek) │
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

### 4.3 Client-Backend Contract
- Schemas (`schemas.py`) use camelCase aliases; `extension.ts` mirrors them with hand-written TS interfaces (`PipelineResult`, `HistoryItem`) — there's no shared/generated type layer
- Two transport modes:
  - **REST**: `/query` for simple request/response
  - **SSE (Server-Sent Events)**: `/query/stream` for progressive UI updates (framed as `event: name\ndata: json\n\n`; the extension POSTs and reads the stream manually since `EventSource` can't POST)
- Config endpoint (`/config`) lets the extension adapt to backend settings dynamically
- The extension owns backend lifecycle: at activation (not on first message — see `onStartupFinished`) it health-checks `/health`, spawns `uvicorn app.main:app` itself if not running (locating `backend/` via the `promptforge.backendDir` setting or workspace scan), and auto-triggers indexing if the index is empty **or belongs to a different workspace** (comparing the open folder against `/health`'s `indexedRepo`). A shared in-flight promise means a chat request arriving mid-index waits on the same work rather than double-spawning the backend.

### 4.3b Token Accounting (`TokenStats`)
- `optimized` counts **only the delivered text** — the answer or brief the developer
  receives and pastes onward. The assembled prompt is local Ollama compute: it costs
  nothing, so charging it against the saving understated the benefit *and* broke the
  invariant `saved == naive_baseline - optimized`.
- `naive_baseline` is the real token count of the files retrieval selected
  (`pipeline._naive_baseline()`), i.e. what a developer would plausibly have pasted
  by hand. It replaced `every chunk in the repo × 85`, which compared against
  something nobody would ever do and inflated the claim ~15x.
- The extension renders this under the file references, **always naming what it is
  measured against** ("vs. pasting these 3 files"). An unqualified "26,400 saved" is
  the exact failure mode the old metric had — impressive and uncheckable.
- Two guards in `renderStats()`: a **cache hit** shows "served from cache" (its
  stored `saved` value is stale and was never earned by that turn — the `cached`
  check must run *before* the baseline check), and a **zero baseline** (off-topic
  guardrail, or an older backend) renders nothing rather than "0 saved".

### 4.4 Caching Strategy
- **Semantic cache**: looks up past queries by cosine similarity (threshold: 0.85); entries expire after 30 days and the oldest are evicted past 500
- Cache entry records: original query, optimized query, answer, model used, tokens saved
- Cache is also the **history store** (serves both `/query` fast-path and `/history` page)

### 4.5 Review Gate (`turn_store.py` + `POST /query/review`)
- Every completed (non-cached) `run_pipeline()` call registers a **`TurnRecord`** (`turn_store.py`) — the exact retrieved chunks, the frozen `task_text` the prompt was built from, and the current answer/brief text — keyed by a generated `turn_id`, **not** `session_id` (a chat transcript can have several pending briefs at once; session alone can't disambiguate which one a button refers to). TTL (2h) + max-entries (200) eviction, mirroring `cache.py`'s pattern. Turns are mirrored to `.chroma/turns.json` on every write and reloaded (TTL re-applied) at import, so a backend restart doesn't strand every on-screen Refine button with "that brief has expired". Persistence is best-effort — an unreadable or unwritable file logs a warning and degrades to in-memory behavior rather than failing the query.
- **Refine** / **"just explain it"** / **"make a brief"** all regenerate from that stored turn via `POST /query/review` (`pipeline.run_review_action`) — reusing `prompt_builder.build()`/`build_forge()`/`refine_forge()` on the already-retrieved chunks. **Retrieval never re-runs** for any of these; that's a hard invariant, not an optimization.
- **Refine** collects its note in a native input box (`showInputBox`) opened by the `promptforge.refine` command, then prefills that note into the chat input. A command handler has no `ChatResponseStream`, so submitting the prefilled note re-enters the participant handler as an ordinary turn, where the armed `pendingRefine` reroutes it to `/query/review` with a live stream to render into.
- Refine keeps the last 2 prior brief versions (`TurnRecord.history`) and is only offered on a forge-mode brief; the two convert actions ("explain instead" / "make a brief") work either direction and are surfaced as VS Code chat **followups** (mirrored, one per mode) rather than buttons, since a followup click resubmits through the same handler with a fresh stream — buttons can't do that.
- Regenerated results are deliberately **not** cache-saved or added to conversation history — they're pre-approval draft iterations, not completed top-level answers.
- **Cache hits are refinable too.** `cache.save()` stores the run's chunks (JSON-encoded in Chroma metadata, which only accepts scalars) alongside its `task_text` and chunk total, so `_cached_result` can rebuild a full `TurnRecord` and hand back a `turn_id`. Without this, asking the same question twice would silently drop the Refine button on the second ask — the cache is mode-scoped and long-lived (30 days), so a repeated task request hits it often. Retrieval still never re-runs: the chunks come from the cache entry, not a fresh search. Entries written before this existed carry no `chunks_json` and degrade to approve-only rather than failing the hit.
- **Approve renders for every forge result** regardless, since dispatch needs only the text. This keeps "nothing reaches Copilot without approval" true on every code path.

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| **Client** | VS Code Chat Participant extension (TypeScript, `vscode-extension/`) — no web frontend exists |
| **Backend** | Python 3.11+ / FastAPI |
| **Vector DB** | Chroma (`chromadb`) |
| **Local LLM** | Ollama — `deepseek-coder-v2:16b` (large tier), `llama3.2:3b` (small tier + classification), `nomic-embed-text` (embeddings) |
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
│   │   ├── pipeline.py          # Single RAG pipeline generator, shared by /query and /query/stream
│   │   ├── config.py            # Environment-based configuration
│   │   ├── schemas.py           # Pydantic models (camelCase aliases)
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
│   │   ├── github_integration.py# Clone/checkout by branch, OAuth login, webhook auto re-index
│   │   ├── prompt_builder.py    # Token-budget-aware prompt assembly (+ forge template)
│   │   ├── generation.py        # Blocking + streaming generation
│   │   ├── cache.py             # Semantic cache (cosine similarity, TTL + eviction)
│   │   ├── conversation.py      # Per-session conversation memory
│   │   ├── turn_store.py        # Review-gate state: chunks + text per turn_id, TTL + eviction
│   │   └── store.py             # Chroma collection helpers + token counting
│   ├── archive/
│   │   ├── agent_registry.py    # Archived: specialist agent definitions (Debug/Refactor/Docs/Security)
│   │   └── README.md
│   ├── requirements.txt
│   └── run.sh
├── vscode-extension/
│   ├── package.json             # Chat participant + commands (query/forge/history/model)
│   └── src/
│       └── extension.ts         # Chat handler, SSE client, backend lifecycle management
└── ARCHITECTURE.md
```

---

## 7. Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CF_PROVIDER` | `ollama` | LLM provider (`ollama` / `groq`) |
| `CF_GEN_MODEL` | `deepseek-coder-v2:16b` | Default generation model |
| `CF_SMALL_MODEL` | `llama3.2:3b` | Small tier — complexity + request-type classification, query optimization |
| `CF_GUARD_MODEL` | `qwen2.5-coder:7b` | Intent guard only — bigger on purpose; a wrong call here blocks a real question |
| `CF_LARGE_MODEL` | `deepseek-coder-v2:16b` | Large tier — complex queries |
| `CF_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CF_GROQ_MODEL` | `qwen/qwen3-32b` | Groq cloud model (when `CF_PROVIDER=groq`) |
| `CF_NUM_CTX` | `16384` | Context window requested from Ollama. **Must exceed `CF_TOKEN_BUDGET`** with room for the reply — Ollama defaults to 4096 whatever the model supports, so a larger prompt is rejected with a bare `400` ("the prompt is longer than the context length currently available"). Bigger costs RAM and CPU linearly |
| `CF_TOP_K` | `8` | Chunks retrieved. The real lever on how much code reaches the model — `CF_TOKEN_BUDGET` rarely binds at these sizes |
| `CF_TOKEN_BUDGET` | `12000` | Max tokens for assembled prompt |
| `CF_RERANK_WEIGHT` | `0.5` | Cross-encoder's share of the final rank; the rest is vector similarity. `1.0` restores the old override-everything behaviour |
| `CF_MAX_DOC_CHUNKS` | `2` | Max `.md`/`.txt`/`.rst` chunks per result — a cap, not an exclusion |
| `CF_USE_EXPANSION` | `true` | Enable multi-query expansion |
| `CF_CHUNK_SIZE` | `800` | Tokens per chunk during indexing |
| `CF_CHUNK_OVERLAP` | `120` | Overlap tokens between chunks |
| `CF_EMBED_CONCURRENCY` | `4` | Simultaneous embed requests during indexing. Ollama serializes work per model, so oversubscribing only deepens its queue until individual requests time out — keep at or below the physical core count |
| `CF_EMBED_TIMEOUT` | `300` | Per-request embed read timeout (seconds) |
| `CF_PRICE_PER_MTOK` | `3.00` | USD per million tokens, used for the dashboard's cost estimate |
| `CF_PRICE_BASIS` | `Claude Sonnet 5 input rates` | Label displayed beside that estimate so the figure stays checkable |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GROQ_API_KEY` | — | Groq API key (for cloud generation). Read from the environment only — never commit a value |

---

## 8. API Endpoints   

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health, indexed chunk count, and `indexedRepo` (path of the repo currently indexed) |
| `GET` | `/config` | Runtime configuration for the client |
| `PATCH` | `/config` | Update runtime settings in-memory |
| `GET` | `/repo/tree` | Browse local file system |
| `POST` | `/index` | Index a repo/folder into Chroma (starts the file watcher) |
| `GET` | `/watcher/status` | Background watcher state (`active`, `path`) |
| `POST` | `/cache/clear` | Clear the semantic cache |
| `GET` | `/dashboard` | Self-contained HTML savings dashboard (inline CSS/JS, no CDN); fetches `/savings` same-origin |
| `GET` | `/savings` | Cumulative turns that never became a Copilot call, by project / day / chat, plus the query-type split and a cost estimate |
| `POST` | `/savings/reset` | Zero the savings counters |
| `POST` | `/query` | Full RAG pipeline (blocking) — Query & Forge modes |
| `POST` | `/query/stream` | Full RAG pipeline (SSE streaming) — Query & Forge modes |
| `POST` | `/query/review` | Review-gate regeneration (SSE streaming) — refine / to_answer / to_brief, from a stored turn's chunks, never re-retrieves |
| `POST` | `/feedback` | Record relevancy feedback (up/down) |
| `GET` | `/history` | Past query history |
| `GET` | `/github/status` | OAuth-enabled / authenticated / connected-repos state |
| `GET` | `/github/login` / `/github/callback` | OAuth sign-in flow (redirects) |
| `POST` | `/github/logout` | Clear the in-memory OAuth token |
| `GET` | `/github/repos` / `/github/branches` | List repos / branches for the signed-in user |
| `POST` | `/github/connect` | Clone a repo at a branch and index it |
| `POST` | `/github/switch-branch` | Re-checkout a connected repo at a different branch, re-index |
| `POST` | `/github/sync` | Pull latest on the connected branch, re-index |
| `POST` | `/github/webhook` | GitHub push webhook → auto re-index (HMAC-verified) |

---

## 9. Running the System

```bash
# 1. Start Ollama (must have models pulled)
ollama pull deepseek-coder-v2:16b   # large tier
ollama pull llama3.2:3b             # small tier + classification
ollama pull qwen2.5-coder:7b        # intent guard
ollama pull nomic-embed-text        # embeddings

# 2. Start the backend (or let the extension auto-start it — see below)
cd backend
pip install -r requirements.txt
bash run.sh           # or: uvicorn app.main:app --reload --port 8000

# 3. Build and run the VS Code extension
cd vscode-extension
npm install
npm run compile        # or: npm run watch
# Press F5 in VS Code (uses vscode-extension/.vscode/launch.json) to launch
# an Extension Development Host with the extension loaded.

# 4. Opening a workspace in the dev host starts the backend + indexing
#    automatically (no command needed). Then just type into @promptforge —
#    the router decides question vs. task:
#      @promptforge what does parse_x12 do        -> streamed answer
#      @promptforge the upload times out           -> brief + Approve/Refine buttons
#    Explicit overrides and utilities stay available:
#    /query   — force answer mode
#    /forge   — force brief mode
#    /history — list past queries
#    /model   — pick the Ollama generation model
# If the backend isn't already running, the extension health-checks it and
# spawns uvicorn itself (see `promptforge.backendDir` / `promptforge.pythonPath`).