# PromptForge — Master Context for AI Assistants

You are working on **PromptForge**, a local-first RAG code assistant. Read this entire document before writing or changing any code. It encodes the architecture, the style rules, and — most importantly — the non-obvious invariants that look like bugs but are deliberate, and the known debt that looks deliberate but is a bug.

---

## 1. What this project is

A full-stack developer tool that indexes a local codebase (or a connected GitHub repo) into a vector store and answers questions about it using local LLMs via Ollama (optionally Groq for cloud generation). The headline metric is **tokens saved** vs. naively dumping the whole repo into a frontier model. No code leaves the machine by default.

- **Client:** VS Code chat participant (TypeScript, `vscode-extension/`). **There is no web frontend** — earlier revisions of this doc described a planned React/Vite UI in `src/` that was never built. The backend also serves one small HTML page at `/dashboard`.
- **Backend:** Python 3.11 / FastAPI, in `backend/app/` — this is where nearly all logic lives
- **Storage:** Chroma (embedded, persistent at `./.chroma`) for both code chunks and the semantic cache/history; BM25 pickle and import-graph JSON live alongside it
- **Models:** `nomic-embed-text` (embeddings), `llama3.2:3b` (small tier: complexity + request-type classifiers, query optimization), `qwen2.5-coder:7b` (intent guard only — its errors are user-visible and blocking, so it gets a bigger model than the other classifiers), `deepseek-coder-v2:16b` (large tier: generation), Groq `qwen/qwen3-32b` (optional cloud)
- **Transport:** REST for request/response, SSE for streaming (`/query/stream`, `/query/review`)

## 2. Architecture — one pipeline, two modes

Every query flows through the same linear RAG pipeline:

```
cache lookup → intent guard → complexity tier → optimize (rewrite) → expand (3 sub-queries)
→ hybrid retrieval → prompt assembly (token budget) → generation → cache save
```

Two modes reuse this pipeline; they are **not** separate systems. The user never
picks one — a plain `@promptforge <message>` sends `mode="auto"` and a third
classifier decides:

1. **Query mode** (`mode=answer`) — answers directly.
2. **Forge mode** (`mode=forge`) — same retrieval, but assembles a paste-ready prompt for an external tool (Copilot/Claude) instead of answering. Uses a fixed 6-section template and never answers the task itself. Held behind a **review gate** (Approve / Refine / "just explain it") — nothing reaches Copilot without a click.

**Agent mode was removed.** `agent_runner.py` is deleted; `agent_registry.py` sits
unimported in `backend/archive/`. Do not reintroduce it — it duplicated what
Copilot does and competed with the product's actual purpose.

**Hybrid retrieval** (the heart of the system, `retrieval.py`): dense vector search per query variant + BM25 sparse search → Reciprocal Rank Fusion (rank-only, k=60) → import-graph neighbor expansion (0.85× penalty) → cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) **blended 50/50 with the vector score**, then a doc quota caps `.md`/`.txt` at 2 slots. Both mitigations exist because the embedding model and the cross-encoder each favour fluent prose over code — un-blended, a 2-line `README.md` outranked a real component on a question about click behaviour.

**Model tiering** (`semantic_router.py`): the complexity and request-type classifiers use `small_model`; the **intent guard uses `guard_model`** (larger) because a false `off_topic` refuses a legitimate question, and 3B models are measurably noisy on it. Generation uses `small_model` for simple queries and `large_model` for complex ones. An explicit `req.model` override bypasses tiering. Off-topic queries get a canned guardrail reply before any generation, in all modes.

**Indexing** (`indexing.py`): os.walk with skip-dirs → SHA-256 per file for incremental re-index (unchanged files keep their chunk IDs) → AST-aware chunking (`ast_chunker.py`: Python stdlib `ast`, JS/TS tree-sitter, line-window fallback) → async batch embedding → Chroma upsert in batches of 128 → BM25 corpus update → import-graph rebuild → file-summary collection (reuses first-chunk embeddings; used as a file-level prefilter only above 500 chunks). A `watchfiles` background thread re-indexes incrementally on file changes.

## 3. Module map (backend/app/)

| Module | Owns |
|---|---|
| `main.py` | All FastAPI routes; `_to_sse()` frames `pipeline.run_pipeline()`'s events (no longer a second copy of the pipeline); GitHub endpoints; root logging config, which must run **before** the package imports below it or import-time logs vanish |
| `pipeline.py` | The single RAG pipeline as an event-yielding generator, plus `run_review_action()` for the review gate |
| `config.py` | `Settings` dataclass singleton, all `CF_*` env vars; mutated in-place by `PATCH /config` |
| `schemas.py` | Pydantic models with **camelCase aliases** matching the TS interfaces in `vscode-extension/src/extension.ts` |
| `semantic_router.py` | Three classifiers — intent guard (`guard_model`), complexity, request-type (answer vs brief) |
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
| `turn_store.py` | Per-turn state for the review gate (chunks + frozen task text), persisted to `.chroma/turns.json` so Refine survives a restart |
| `savings.py` | Counts turns that never became a Copilot call; persisted to `.chroma/savings.json` |
| `dashboard.py` | Self-contained HTML page served at `/dashboard` (inline CSS/JS, no CDN) |
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
6. **The "naive baseline" token stat** is now a real measurement, not a marketing figure: `pipeline._naive_baseline()` sums the actual token count of every indexed chunk belonging to the files retrieval selected — i.e. what a developer would have pasted by hand. It replaced `chunk_count × avg_tokens_per_chunk` (every chunk in the repo), which inflated the claimed saving roughly 15x by comparing against something nobody would ever do.
7. **The webhook HMAC check fails closed** (unconfigured secret ⇒ reject). Keep it that way.

## 6. Known debt — real problems, acknowledged, unfixed

Do not silently work around these; if your change touches one, flag it.

1. ~~**The pipeline exists in three copies**~~ — **FIXED.** `pipeline.run_pipeline()` is now the single event-yielding generator (`("stage"|"token"|"done"|"error", payload)`); `/query` drains it and `/query/stream` forwards it as SSE. `run_review_action()` mirrors its tail for the review gate. The third copy went away with agent mode.
2. **No trust boundary**: no auth on any endpoint; CORS is wildcard-with-credentials (Starlette echoes any origin); `/repo/tree` walks arbitrary absolute paths; `/index` indexes any directory; `PATCH /config` can repoint `ollama_url` (SSRF/exfiltration). Acceptable only as a localhost single-user tool; must be fixed before any deployment. Do not add new endpoints that widen this surface.
3. **GitHub token leaks to disk**: `clone_or_update` embeds the OAuth/PAT token in the remote URL and `git remote set-url` persists it in `.git/config` plaintext (also visible in process argv). The error-message scrubber in `_run_git` does not cover these sinks.
4. **Single-process global mutable state + fully sync handlers**: every route is sync `def` (shares the ~40-thread default pool; SSE generators pin a thread for the whole generation); `PATCH /config` mutates the live singleton read by in-flight requests; `ConversationMemory._sessions` grows unbounded; `/feedback` does unlocked read-modify-write; concurrent `index_repo` runs (watcher + webhook + manual) can corrupt `file_hashes.json` / BM25 pickle / graph. Multiple uvicorn workers are NOT safe.
5. **N+1 Chroma gets** in retrieval (per-BM25-hit and per-graph-neighbor `col.get(ids=[cid])`); full-collection scan in `cache._evict_oldest` on every save.
6. **`_link_graph` matches imports by bidirectional substring** across path variants and adds complete bipartite chunk-edge sets per matched file pair — combinatorial `graph.json` growth and false neighbors on large repos.
7. **Docs drift**: README's query diagram claims optimize/expand run in parallel (they're sequential — see invariant 1) and says the cache threshold is 0.92 (config says 0.85). ARCHITECTURE.md is the more accurate document. `notes/PromptForge_Documentation.md` and `codebase.md` still describe agent mode and a React frontend, neither of which exists — treat them as historical.

## 7. Roadmap (agreed direction)

1. ~~Unify the pipeline into one generator~~ — **done.** Classifiers now run concurrently in a `ThreadPoolExecutor`.
2. Add the trust boundary: CORS pinned to the client origin, bearer token on mutating endpoints, allowlisted root for `/repo/tree` and `/index`. **Still open — the top blocker for any multi-user deployment.**
3. Stop writing the GitHub token into remote URLs (per-invocation credential helper / `http.extraHeader`). **Still open.**
4. **Per-repo indexing.** The index is global — one repo at a time — so a second developer on a different project wipes the first's index. Blocks any team rollout; see `COST-ANALYSIS.md`.
5. **Retrieval quality.** Blended reranking + a doc quota fixed one of two regression queries; a question about click behaviour still returns data files instead of the component that handles the click. Root cause is representational: the file prefilter summarises each file by its *first chunk* (imports/boilerplate for a `.tsx`).

## 8. Working rules

- Bug fixes go to the root cause — grep all callers before patching one path.
- Non-trivial logic ships with one minimal runnable check (a small `test_*.py` or `assert`-based `__main__`), no frameworks or fixtures.
- Never add a dependency for what stdlib or an installed package already does; check `requirements.txt` / `package.json` first.
- Endpoints map 1:1 to frontend data needs — don't add speculative API surface.
- When you change behavior described in README.md or ARCHITECTURE.md, update the doc in the same change (see debt #7 for what drift costs).
