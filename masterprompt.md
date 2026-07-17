# PromptForge — Master Context for AI Assistants

You are working on **PromptForge**, a local-first RAG code assistant. Read this entire document before writing or changing any code. It encodes the architecture, the style rules, and — most importantly — the non-obvious invariants that look like bugs but are deliberate, and the known debt that looks deliberate but is a bug.

---

## 1. What this project is

A full-stack developer tool that indexes a local codebase (or a connected GitHub repo) into a vector store and answers questions about it using local LLMs via Ollama (optionally Groq for cloud generation). The headline metric is **tokens saved** vs. naively dumping the whole repo into a frontier model. No code leaves the machine by default.

- **Frontend:** React 18 + TypeScript + Vite + Tailwind, in `src/` (pages: Workbench, Repository, History, Settings)
- **Backend:** Python 3.11 / FastAPI, in `backend/app/` — this is where nearly all logic lives
- **Storage:** Chroma (embedded, persistent at `./.chroma`) for both code chunks and the semantic cache/history; BM25 pickle and import-graph JSON live alongside it
- **Models:** `nomic-embed-text` (embeddings), `mistral:7b` (small tier: classification + simple queries), `deepseek-coder-v2:16b` (large tier: complex queries + agents), Groq `qwen/qwen3-32b` (optional cloud)
- **Transport:** REST for request/response, SSE for streaming (`/query/stream`, `/agents/stream`)

## 2. Architecture — one pipeline, three modes

Every query flows through the same linear RAG pipeline:

```
cache lookup → intent guard → complexity tier → optimize (rewrite) → expand (3 sub-queries)
→ hybrid retrieval → prompt assembly (token budget) → generation → cache save
```

Three modes reuse this pipeline; they are **not** separate systems:

1. **Query mode** (`/query`, `mode=answer`) — answers directly.
2. **Forge mode** (`mode=forge`) — same retrieval, but assembles a paste-ready prompt for an external tool (Copilot/Claude) instead of answering. Uses a fixed 6-section template and never answers the task itself.
3. **Agent mode** (`/agents/run`, `/agents/stream`) — one agent per request (Debug / Refactor / Docs / Security, selected by keyword score in `_infer_agent_id`), which is a system prompt + retrieval-bias keywords + extra stages (plan, findings, patch diff). There is no orchestrator and no multi-agent execution — "agents" are prompt templates.

**Hybrid retrieval** (the heart of the system, `retrieval.py`): dense vector search per query variant + BM25 sparse search → Reciprocal Rank Fusion (rank-only, k=60) → import-graph neighbor expansion (0.85× penalty) → cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) picks the final top-K from a 2×top_k candidate pool.

**Model tiering** (`semantic_router.py`): both classifiers (intent: codebase/off_topic; complexity: simple/complex) always use `small_model`. Generation uses `small_model` for simple queries, `large_model` for complex queries and all agent runs. An explicit `req.model` override bypasses tiering. Off-topic queries get a canned guardrail reply before any generation, in all modes.

**Indexing** (`indexing.py`): os.walk with skip-dirs → SHA-256 per file for incremental re-index (unchanged files keep their chunk IDs) → AST-aware chunking (`ast_chunker.py`: Python stdlib `ast`, JS/TS tree-sitter, line-window fallback) → async batch embedding → Chroma upsert in batches of 128 → BM25 corpus update → import-graph rebuild → file-summary collection (reuses first-chunk embeddings; used as a file-level prefilter only above 500 chunks). A `watchfiles` background thread re-indexes incrementally on file changes.

## 3. Module map (backend/app/)

| Module | Owns |
|---|---|
| `main.py` | All FastAPI routes; `_stream_pipeline` (the SSE copy of the pipeline); GitHub endpoints |
| `config.py` | `Settings` dataclass singleton, all `CF_*` env vars; mutated in-place by `PATCH /config` |
| `schemas.py` | Pydantic models with **camelCase aliases** matching the TS interfaces in `src/lib/mockData.ts` |
| `semantic_router.py` | Intent + complexity classifiers, agent keyword scorer, `route()` (blocking pipeline front half) |
| `optimizer.py` | Query rewrite + expansion (expansion is chained off the rewrite — deliberate, see §5) |
| `retrieval.py` | Dense + BM25 + RRF + graph expansion + rerank (see §5 before touching) |
| `bm25_index.py` | BM25Okapi with incremental corpus map; full model rebuild each update (known ceiling) |
| `reranker.py` | Lazy-loaded cross-encoder; falls back to RRF order if sentence-transformers is missing |
| `graph_memory.py` | networkx DiGraph of import edges, JSON persistence |
| `indexing.py` | File walk, SHA-256 incremental diff, chunk→embed→upsert, summary collection |
| `prompt_builder.py` | Token-budget assembly; `build()` for answers, `build_forge()` + `FORGE_SYSTEM_PROMPT` for forge mode |
| `generation.py` | Blocking + streaming generation, Ollama or Groq |
| `cache.py` | Semantic cache = history store (one Chroma collection serves both); 0.85 cosine threshold, 30-day TTL, 500-entry cap |
| `conversation.py` | `ConversationMemory`: per-session deque of last 5 turns, prepended to prompts |
| `agent_runner.py` | Agent workflow (third pipeline copy); streaming via background thread + queue |
| `agent_registry.py` | Agent definitions + custom-agent creation |
| `store.py` | Chroma client singleton + heuristic token counter (chars-per-token per model family) |
| `github_integration.py` | OAuth / PAT auth, clone/pull/checkout, webhook HMAC verification |
| `watcher.py` | Background re-index thread |

## 4. Style guide — match this exactly

- **Module-level functions, not classes.** State is module-level globals (`_index`, `_connected`, `_model`). The only classes are dataclasses, Pydantic schemas, and `ConversationMemory` (a namespace of classmethods). Do not introduce service classes, dependency injection, factories, or interfaces.
- **Every module opens with a docstring header** explaining what it owns and its strategy, often with a short numbered pipeline list. Maintain this when creating modules.
- **`# ponytail:` comments mark deliberate shortcuts** with a known ceiling and upgrade path (e.g. "single-process in-memory token store — swap for a per-session table when multi-user"). Preserve them; add one when you take a deliberate shortcut; never "fix" one without being asked.
- **Private helpers are `_underscore`-prefixed**, module constants are `_UPPER_SNAKE`. Section dividers are `# ---` comment rules.
- **Comments explain *why*, not *what*** — constraints and non-obvious decisions only.
- **camelCase at the API boundary, snake_case internally.** Pydantic aliases handle the conversion; keep the frontend TS interfaces and `schemas.py` in lockstep.
- **Errors:** internal code raises `ValueError` (client fault) or `RuntimeError` (upstream/infra fault); routes translate to 400/422 vs 502. Cache-save failures are swallowed — never fail a request because persistence hiccupped.
- **Deletion over addition; stdlib over dependencies.** The repo minimizes files and abstraction; the shortest working diff in the existing idiom wins.
- **Config:** every new tunable is a `CF_*` env var with a sensible local default in `Settings`, and (if user-facing) a matching control in the Settings page + `PATCH /config` mapping + README env-var table.

## 5. Invariants that look wrong but are deliberate — DO NOT "fix"

1. **Expansion chains off the rewrite, not the raw query** (`optimizer`): `expand_query(optimize_query(q))` is sequential by design — sub-queries should diversify the *enriched* query. Do not parallelize these two. (The two *classifier* calls, by contrast, are independent and safe to parallelize.)
2. **Three score spaces coexist in `retrieval.py`** and are not comparable: `chunk.score` (cosine + feedback boost, ~0.3–1.0, display + intra-list order), BM25 raw scores (rank-only, value discarded), and `rrf_scores` (~0.016/list, the real ranking key). Graph neighbors get a *manufactured* RRF entry: `seed.score * 0.85 / 60`, deliberately tuned to sit just below a first-place single-list finish so neighbors never outrank real matches. The final RRF sort only selects the reranker's candidate pool — the cross-encoder owns final order. **Any new chunk source must add an RRF-scale entry to `rrf_scores`**, or the `rrf_scores.get(c.id, c.score)` fallback will compare cosine (~0.8) vs RRF (~0.016) and your chunks will always rank first. Full explainer: README §"How retrieval scoring actually works".
3. **The semantic cache is mode-scoped.** Lookups filter `where={"mode": mode}` so a forge-mode prompt is never served for an answer-mode query. Preserve this in any cache change.
4. **Forge mode seeds its Task section from the user's raw query**, not the optimized one (the optimized blob reads vague to humans); the optimized/expanded queries are still used for retrieval. Answer mode uses the optimized query as the task.
5. **Token counting is a calibrated chars-per-token heuristic** (`store.py`), not a real tokenizer — deliberate, to avoid loading vocabs per Ollama model. It over-counts slightly on the fallback, which is the safe direction for budget enforcement.
6. **The "naive baseline" token stat** (`chunk_count × avg_tokens_per_chunk`) is a marketing-style comparison metric, not a measurement. Don't try to make it precise.
7. **The webhook HMAC check fails closed** (unconfigured secret ⇒ reject). Keep it that way.

## 6. Known debt — real problems, acknowledged, unfixed

Do not silently work around these; if your change touches one, flag it.

1. **The pipeline exists in three copies**: `semantic_router.route()` + `/query` (blocking), `_stream_pipeline` in `main.py` (streaming re-implementation that even calls the private `semantic_router._classify`), and `agent_runner._run_agent_internal`. The ~40 lines of token accounting + cache-save are near-identical in the first two. Any pipeline change must currently be made 2–3 times. The agreed fix (planned, not yet built): one event-yielding generator (`("stage"|"token"|"done"|"error", payload)`) that `/query` drains and `/query/stream` forwards as SSE.
2. **No trust boundary**: no auth on any endpoint; CORS is wildcard-with-credentials (Starlette echoes any origin); `/repo/tree` walks arbitrary absolute paths; `/index` indexes any directory; `PATCH /config` can repoint `ollama_url` (SSRF/exfiltration). Acceptable only as a localhost single-user tool; must be fixed before any deployment. Do not add new endpoints that widen this surface.
3. **GitHub token leaks to disk**: `clone_or_update` embeds the OAuth/PAT token in the remote URL and `git remote set-url` persists it in `.git/config` plaintext (also visible in process argv). The error-message scrubber in `_run_git` does not cover these sinks.
4. **Single-process global mutable state + fully sync handlers**: every route is sync `def` (shares the ~40-thread default pool; SSE generators pin a thread for the whole generation); `PATCH /config` mutates the live singleton read by in-flight requests; `ConversationMemory._sessions` grows unbounded; `/feedback` does unlocked read-modify-write; concurrent `index_repo` runs (watcher + webhook + manual) can corrupt `file_hashes.json` / BM25 pickle / graph. Multiple uvicorn workers are NOT safe.
5. **N+1 Chroma gets** in retrieval (per-BM25-hit and per-graph-neighbor `col.get(ids=[cid])`); full-collection scan in `cache._evict_oldest` on every save.
6. **`_link_graph` matches imports by bidirectional substring** across path variants and adds complete bipartite chunk-edge sets per matched file pair — combinatorial `graph.json` growth and false neighbors on large repos.
7. **Docs drift**: README's query diagram claims optimize/expand run in parallel (they're sequential — see invariant 1) and says the cache threshold is 0.92 (config says 0.85). ARCHITECTURE.md is the more accurate document.

## 7. Roadmap (agreed direction)

1. Unify the pipeline into one generator (kills debt #1); make `/query` and `/query/stream` thin consumers; then parallelize the two classifier calls (`ThreadPoolExecutor`, already the codebase idiom) and skip expansion for simple-tier queries.
2. Add the trust boundary: CORS pinned to the frontend origin, bearer token on mutating endpoints, allowlisted root for `/repo/tree` and `/index`.
3. Stop writing the GitHub token into remote URLs (per-invocation credential helper / `http.extraHeader`).
4. Longer term: retire `agent_runner`'s third pipeline copy by having it consume the shared generator.

## 8. Working rules

- Bug fixes go to the root cause — grep all callers before patching one path.
- Non-trivial logic ships with one minimal runnable check (a small `test_*.py` or `assert`-based `__main__`), no frameworks or fixtures.
- Never add a dependency for what stdlib or an installed package already does; check `requirements.txt` / `package.json` first.
- Endpoints map 1:1 to frontend data needs — don't add speculative API surface.
- When you change behavior described in README.md or ARCHITECTURE.md, update the doc in the same change (see debt #7 for what drift costs).
