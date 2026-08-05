# PromptForge — RAG Code Assistant

A full-stack developer tool that answers questions about your codebase using a **hybrid RAG pipeline** running entirely on local, open-source models via Ollama. No API keys required, no code leaves your machine.

The core insight: instead of dumping your whole repo into a frontier model's context window (expensive, slow, leaks code), PromptForge retrieves only the top-K relevant chunks, assembles a tight prompt, and generates locally. The headline metric is **tokens saved** vs. a naive whole-repo dump.

---

## Demo

> PromptForge runs locally — it needs the Python backend + Ollama on your own machine,
> so there is no public hosted demo. The screenshots below show it running end-to-end.

<!--
  Add screenshots / a short GIF here so visitors can see it without running anything.
  1. Take screenshots of each page (Dashboard, Architecture, History, Settings).
  2. Drop the image files in a `docs/` folder.
  3. Reference them like:
       ![Workbench](docs/workbench.png)
       ![Architecture](docs/architecture.png)
  A 10–20s screen-recording GIF of a query streaming in is the single most
  effective thing you can add for a portfolio.
-->

| Dashboard (Workbench) | Architecture |
|---|---|
| _add `docs/workbench.png`_ | _add `docs/architecture.png`_ |

| History | Settings |
|---|---|
| _add `docs/history.png`_ | _add `docs/settings.png`_ |

---

## Benchmark numbers

Measured on the PromptForge codebase itself (~50 source files, ~18k lines):

| Metric | Value |
|---|---|
| Files indexed | 52 |
| Total chunks | ~480 |
| Naive baseline (whole repo) | ~41,000 tokens |
| Optimized prompt (top-6 chunks) | ~2,800 tokens |
| **Tokens saved** | **~93%** |
| Index time (first run, async batch embed) | ~28s |
| Index time (re-run, incremental SHA-256) | ~1s (unchanged files skipped) |
| Query latency (cache miss, p50) | ~4–8s on DeepSeek-Coder 16B |
| Query latency (cache hit) | <100ms |

---

## RAG pipeline

### Query path

```
User query
    │
    ▼
Semantic cache lookup  ──hit──▶  Stream cached answer (< 100 ms)
    │ miss
    ▼
┌─────────────────────────────────────────────┐
│  Parallel (ThreadPoolExecutor × 2)          │
│  optimize_query → 1 rich search query       │
│  expand_query   → 3 sub-queries             │
└─────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐   ┌──────────────────────┐
│  Dense vector search          │   │  BM25 sparse search  │
│  nomic-embed-text → Chroma    │   │  exact identifiers   │
│  cosine similarity, per query │   │  rank-bm25 index     │
└───────────────────────────────┘   └──────────────────────┘
    │                                         │
    └──────────────┬──────────────────────────┘
                   ▼
         Reciprocal Rank Fusion
         RRF(k=60) across all lists
                   │
                   ▼
         Graph expansion
         networkx DiGraph (import edges)
         BFS depth=1 from top seeds
                   │
                   ▼
         Cross-encoder rerank
         ms-marco-MiniLM-L-6-v2
         joint (query, chunk) scoring
                   │
                   ▼
         Token-budget prompt assembly
         model-aware char-ratio token counting
         conversation history prefix injected
                   │
                   ▼
         Generation (SSE stream)
         Ollama /api/chat  OR  Groq API
                   │
                   ▼
         Save to history + session memory
```

### Index path

```
File walker (os.walk, skip node_modules / .git / dist)
    │
    ▼
SHA-256 hash check → skip unchanged files (incremental)
    │ changed/new
    ▼
AST-aware chunker
  .py  → stdlib ast  (FunctionDef / ClassDef boundaries)
  .js/.ts/.tsx → tree-sitter-languages
  else → line-window with overlap
    │
    ▼
Async batch embeddings
  httpx.AsyncClient + asyncio.Semaphore(16)
  asyncio.gather over all chunks
    │
    ▼
Chroma upsert (batches of 128)
    │
    ├──▶ BM25 rebuild + pickle (.chroma/bm25.pkl)
    ├──▶ Import graph rebuild + JSON (.chroma/graph.json)
    ├──▶ SHA-256 hash store (.chroma/file_hashes.json)
    └──▶ Semantic cache invalidation
```

---

## Backend module map

```
backend/app/
├── main.py           POST /index /query /query/stream
│                     GET  /health /config /history /repo/tree
│                     POST /feedback /cache/clear
│                     PATCH /config
├── config.py         Settings dataclass, all CF_* env vars
├── schemas.py        Pydantic CamelModel request/response shapes
├── store.py          Chroma client + model-aware token counter
├── indexing.py       File walker, incremental hash check, chunk→embed→upsert
├── ast_chunker.py    Python AST + tree-sitter JS/TS + line-window fallback
├── ollama_client.py  /api/chat (sync + stream), /api/embeddings (async batch)
├── optimizer.py      optimize_query + expand_query via /api/chat
├── bm25_index.py     BM25Okapi build/query/save/load
├── retrieval.py      Vector search + BM25 + RRF + graph expansion + rerank
├── reranker.py       CrossEncoder ms-marco-MiniLM-L-6-v2 (lazy load)
├── graph_memory.py   networkx DiGraph, import extraction, JSON persistence
├── prompt_builder.py Token-budget prompt assembly
├── generation.py     Blocking + streaming generation (Ollama / Groq)
├── cache.py          Semantic cache lookup + save, history listing
├── conversation.py   Per-session turn memory (deque, last 5 turns)
├── turn_store.py     Per-turn state for the review gate (persisted to .chroma/turns.json)
├── savings.py        Counts turns that never became a Copilot call
└── dashboard.py      Self-contained HTML savings dashboard served at /dashboard
```

Agent mode was removed: `agent_runner.py` is deleted and `agent_registry.py` now sits
unimported in `backend/archive/` as reference for possible future per-task templates.

---

## Frontend pages

| Page | What it does |
|---|---|
| **Workbench** | Agent mode (auto-routed) + Query mode (direct RAG). Streaming answer panel, session memory, thumbs up/down feedback. |
| **Repository** | Index a local folder by path, live chunk count from `/health`, architecture flow diagram. |
| **History** | All past queries from ChromaDB. Copy to clipboard, Reuse in Workbench. |
| **Settings** | Load and save runtime config via `PATCH /config`. Provider, model, embedding model, top-k, token budget, Ollama URL. |

---

## Quick start (Docker — recommended)

```bash
git clone <repo>
cd PromptForge

# Start everything: Ollama + model pull + backend + frontend
docker compose up

# First run pulls deepseek-coder-v2:16b and nomic-embed-text (~9 GB total).
# Open http://localhost:3000
```

**No GPU?** Remove the `deploy.resources` block from `docker-compose.yml` — Ollama runs on CPU (slower but functional).

**Want a faster/smaller model?**
```bash
# In docker-compose.yml, change CF_GEN_MODEL= to one of:
# qwen2.5-coder:7b       (~4 GB, good at code, much faster on CPU)
# deepseek-coder-v2:16b  (~9 GB, best quality, default)
#
# On a CPU-only host the generation model dominates latency: a 16B model can
# take minutes per answer where a 7B takes well under one.
```

---

## Quick start (manual)

### Backend

```bash
# 1. Pull models
ollama pull nomic-embed-text
ollama pull deepseek-coder-v2:16b

# 2. Install and start
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
npm install
npm run dev   # http://localhost:3000
```

### Index a repo

```bash
# Via curl
curl -X POST http://localhost:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"path": "/absolute/path/to/your/repo"}'

# Or use the Repository page in the UI — paste the folder path and click "Index folder"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CF_PROVIDER` | `ollama` | `ollama` or `groq` |
| `CF_GEN_MODEL` | `deepseek-coder-v2:16b` | Generation model |
| `CF_SMALL_MODEL` | `llama3.2:3b` | Complexity + request-type classifiers, query optimization |
| `CF_GUARD_MODEL` | `qwen2.5-coder:7b` | Intent guard only — bigger on purpose; a wrong call here refuses a real question |
| `CF_LARGE_MODEL` | `deepseek-coder-v2:16b` | Complex queries |
| `CF_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CF_NUM_CTX` | `16384` | Context window requested from Ollama. **Must exceed `CF_TOKEN_BUDGET`** — Ollama defaults to 4096 regardless of model, which rejects larger prompts with a bare 400 |
| `CF_EMBED_CONCURRENCY` | `4` | Simultaneous embed requests. Ollama serializes per model, so oversubscribing just deepens its queue until requests time out — keep at or below physical core count |
| `CF_EMBED_TIMEOUT` | `300` | Per-request embed timeout (seconds) |
| `CF_TOP_K` | `8` | Chunks per query — the real lever on how much code reaches the model |
| `CF_TOKEN_BUDGET` | `12000` | Max prompt tokens |
| `CF_RERANK_WEIGHT` | `0.5` | How much the cross-encoder overrides vector similarity (1.0 = full override) |
| `CF_MAX_DOC_CHUNKS` | `2` | Max `.md`/`.txt` chunks in a result — stops docs crowding out code |
| `CF_USE_EXPANSION` | `true` | Multi-query expansion |
| `CF_PRICE_PER_MTOK` | `3.00` | Rate for the dashboard's cost estimate (USD per million tokens) |
| `CF_PRICE_BASIS` | `Claude Sonnet 5 input rates` | Label shown beside that estimate |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `CF_CHROMA_DIR` | `./.chroma` | Vector store path |
| `GROQ_API_KEY` | — | Required if `CF_PROVIDER=groq` |
| `CF_GROQ_MODEL` | `qwen/qwen3-32b` | Groq model ID |

All settings can also be changed at runtime via `PATCH /config` (Settings page).

---

## Key design decisions

**Why BM25 + vector search?** Embedding similarity misses exact identifier matches (`get_embedding_by_id`, specific error codes). BM25 catches those. RRF merges both ranked lists without needing score normalization.

**Why a cross-encoder reranker?** Bi-encoder cosine similarity scores query and chunk independently. A cross-encoder reads both together — it understands relevance, not just similarity. `ms-marco-MiniLM-L-6-v2` is 22 MB and adds <100ms on CPU for 20 candidates.

**Why AST-aware chunking?** Token-window chunking splits functions in the middle, discarding the signature context the model needs. AST chunking emits whole functions/classes as chunks, so the model always sees complete, meaningful units.

**Why import graph expansion?** If you ask about a function in `retrieval.py`, the graph expansion also pulls in `bm25_index.py` and `reranker.py` because they're imported there — even if they didn't score highly on their own. Exposes cross-file dependencies automatically.

**Why a semantic cache?** The most expensive operation is generation. If two questions have cosine similarity ≥ 0.92 in embedding space, the second one gets the cached answer instantly. Token savings: 100%.

---

## How retrieval scoring actually works (read before touching `retrieval.py`)

`retrieval.retrieve()` is the most counter-intuitive code in the repo, because it runs **three different scoring systems side by side**, and the one that looks most important — the `score` field on each chunk — is *not* what decides the ranking. If you change one score without knowing which system it belongs to, retrieval will silently degrade.

### The three score spaces

| Score | Lives in | Typical range | What it's actually used for |
|---|---|---|---|
| **Cosine + feedback boost** | `chunk.score` | ~0.3 – 1.0 | Display in the UI; ordering *within* one vector result list; seeding graph-neighbor scores |
| **BM25** | `bm25_index.query()` return | 0 – ~30 (unbounded) | Only the *rank order* of the BM25 list; the raw value is discarded |
| **RRF** | local `rrf_scores` dict | ~0.016 per list (`1/(60+rank)`) | The real ranking key that picks the candidate pool |

These are **not comparable to each other**. A great RRF score (~0.03) is numerically tiny next to a mediocre cosine score (~0.5). Never mix them in a comparison — see the "gotchas" below for the one line that appears to.

### Step by step

1. **Vector search, one list per query.** The optimized query plus up to 3 expansions each get their own Chroma search. Each hit gets `chunk.score` = `1 − cosine_distance`, then a **feedback boost**: net thumbs-up/down votes adjust it by up to ±15% (`_apply_feedback_boost`). Important subtlety: the boost re-sorts chunks *within* each list, which changes their **rank**, which is the only thing the next stage looks at.

2. **BM25 sparse search** on the primary query catches exact identifiers (`get_embedding_by_id`, `E504`) that embeddings blur. Its results become one more ranked list. The `raw_score / max_bm25` normalization you see in the code exists **only** to fill a plausible display value into `chunk.score` for chunks the vector search never saw — it plays no part in ranking.

3. **Reciprocal Rank Fusion** (`_rrf_merge`) collapses all lists (4–5 of them) into one consensus ranking. RRF ignores every raw score and uses only rank positions: each chunk earns `1/(60 + rank)` per list it appears in. A chunk ranked #1 in a single list scores `1/61 ≈ 0.0164`; a chunk appearing mid-list in *several* lists beats a chunk that topped just one. That's the point — consensus over confidence, and no cross-system score normalization needed.

4. **Graph expansion — the trick that trips everyone up.** Import-graph neighbors of the top seeds are pulled in even though no search matched them. They need an entry in `rrf_scores` to participate in the final sort, but they were never in any ranked list. So the code *manufactures* one:

   ```python
   rrf_scores[nid] = seed.score * 0.85 / _RRF_K
   ```

   Decoded: take the seed's cosine score (≤ 1.0), apply the 0.85 indirect-match penalty, then divide by 60 to **shrink it onto the RRF scale**. The result (≈ 0.014 for a strong seed) lands just *below* a first-place single-list finish (0.0164). That's deliberate: neighbors ride along in the candidate pool but can never outrank a chunk that actually matched a search. The neighbor's *display* score is set separately (`seed.score * 0.85`, boost-adjusted) — same formula, different scale, different purpose.

5. **Final sort + cross-encoder rerank — the punchline.** The RRF sort does **not** produce the final answer order. It selects the top `2 × top_k` *candidates*, and then the cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores every (query, chunk) pair jointly and imposes its own order. So the entire scoring machinery above answers only one question: *which ~2K chunks deserve a seat in front of the reranker?* The reranker decides who actually gets into the prompt. (If `sentence-transformers` isn't installed, the RRF order is used as-is — that's the fallback in `reranker.rerank`.)

### Gotchas for contributors

- **`rrf_scores.get(c.id, c.score)` in the final sort looks like a bug — it's a dormant fallback.** Every chunk that reaches this line already has an RRF entry (searched chunks via `_rrf_merge`, neighbors via the manufactured score), so the `c.score` default is effectively unreachable today. But if you add a new chunk source and forget to give it an `rrf_scores` entry, this fallback will compare a ~0.8 cosine score against ~0.016 RRF scores and your new chunks will **always rank first**. Add an RRF-scale entry for any new source.
- **The `score` users see in the UI is the cosine/boost score, not the ranking key.** A chunk displayed with 0.91 can legitimately rank below one showing 0.74 — RRF consensus and the reranker outvote single-list similarity. This is expected behavior, not a sorting bug.
- **Feedback votes influence ranking indirectly.** The ±15% boost only re-orders chunks within their own result list before RRF; it cannot push a chunk into a list it didn't match. Don't expect thumbs-up to force a chunk into unrelated queries.
- **`top_k * 2` appears twice with different meanings**: BM25 fetches `top_k * 2` hits (a wider sparse net), and the reranker receives `top_k * 2` candidates (headroom to reorder). They're independent knobs that happen to share a formula.
