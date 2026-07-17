# PromptForge VS Code Chat Extension — v1 (shipped)

## Context

The manager's feedback: nobody has time to open the PromptForge portal to run queries or grab forged prompts. The fix is to bring the pipeline into the IDE — a native VS Code **chat participant** (`@promptforge` in the built-in chat panel, like Copilot) that talks to the existing FastAPI backend. The React frontend (`src/`) stays untouched for now; the extension is a new sibling package. **Zero backend changes** — every needed feature maps to an existing endpoint (verified in `backend/app/main.py`).

**Status: shipped and working (v1 above).** Everything below is a new, second phase.

User-confirmed decisions:
- Chat participant, not a webview. Modes via slash commands (Copilot-style picker): default = answer, `/forge`, `/agent`, plus `/index`, `/history`, `/model`.
- Extension auto-starts the backend (spawn uvicorn) if `/health` is down.
- v1 scope: query, forge, agent, auto-index open workspace, history, feedback, Ollama model selection.

## Files (all new, nothing else changes)

```
vscode-extension/
  package.json           # manifest + contributes (chatParticipants, configuration)
  tsconfig.json          # tsc-only build — zero runtime deps, nothing to bundle
  .vscode/launch.json    # F5 Extension Development Host
  src/extension.ts       # everything: participant, SSE client, backend spawn, model picker (~350 lines)
```

No file splitting, no esbuild, no README/icon/tests (not publishing; the one non-trivial unit — the SSE parser — is a direct port of the working parser in `src/lib/api.ts:216-274`).

## Step 1 — package.json

- `engines.vscode: ^1.95.0`, `main: ./out/extension.js`, `activationEvents: []` (chat contribution auto-activates).
- devDeps only: `typescript`, `@types/vscode`, `@types/node`. `fetch` is global in the extension host.
- `contributes.chatParticipants`: id `promptforge.chat`, name `promptforge`, `isSticky: true`, commands: `forge`, `agent`, `index`, `history`, `model`.
- `contributes.configuration`: `promptforge.backendUrl` (default `http://localhost:8000`), `promptforge.backendDir` (empty = scan workspace for `backend/app/main.py`), `promptforge.pythonPath` (empty = `<backendDir>/.venv/Scripts/python.exe`), `promptforge.model` (empty = backend default).

## Step 2 — src/extension.ts

**Wire contract** (authoritative: `backend/app/schemas.py`, camelCase on the wire):
- `POST /query/stream` `{query, topK:4, model?, useExpansion:true, sessionId, mode:'answer'|'forge'}` — SSE events: `stage` (routing|optimizing|retrieving|assembling|generating), `token` (JSON string), `done` (PipelineResult), `error`.
- `POST /agents/stream` `{agentId:'auto', repoPath, targetPath, userRequest, logs:'', attachments:[], topK:6, useExpansion:false, model?}` — SSE: `stage`, `log` `{stage,message}`, `artifact`, `done` (AgentRunResult), `error`.
- `POST /index {path}`, `GET /health`, `GET /history`, `POST /feedback {queryId, feedback, chunkIds?}`.

**2a. SSE client** — one generic `async function* sse(url, body, signal)` over `fetch` + `res.body.getReader()`, splitting on `\n`, tracking `event:` / `data:` lines, `JSON.parse` each data payload (port of `src/lib/api.ts`). Both query and agent streams consume it. Wire `token.onCancellationRequested` → `AbortController.abort()`.

**2b. Backend auto-start** — `ensureBackend(stream)` at the top of every handler:
1. `GET /health` (1.5s timeout) → ok? done.
2. Resolve `backendDir` (setting, else scan `workspaceFolders` for `backend/app/main.py`); missing → markdown error telling user to set `promptforge.backendDir`.
3. Python: setting or `.venv/Scripts/python.exe` (win) / `.venv/bin/python`.
4. `spawn(python, ['-m','uvicorn','app.main:app','--port','8000'], {cwd: backendDir})` — **no `--reload`** (reloader parent orphans the worker on Windows kill).
5. `stream.progress('Starting backend…')`, poll `/health` 1s × 60 (model warmup is slow).
6. Keep `ChildProcess` in a module var; `deactivate()` kills it. `// ponytail: kill() only; tree-kill if uvicorn ever spawns workers`.

**2c. Auto-index** — in `ensureBackend`: if `health.indexedChunks === 0` and a workspace folder is open, `POST /index {path: workspaceFolders[0].uri.fsPath}` with progress, report `{files, chunks}`. `/index` command does the same unconditionally (the re-index / re-point escape hatch — index is server-global, no staleness detection). `// ponytail: no repo-identity check on the global index; /index to re-point it`.

**2d. Handler dispatch** — `vscode.chat.createChatParticipant('promptforge.chat', handler)`, switch on `request.command`:
- **default / `forge`** → `/query/stream` with `mode: 'forge'` when forging. `stage`→`stream.progress`, `token`→`stream.markdown`, `done`→`stream.reference(Uri.file(...))` per retrieved chunk; forge additionally gets `stream.button({command:'promptforge.copy', arguments:[answer]})` (runtime-registered command → `env.clipboard.writeText`).
- **`agent`** → `/agents/stream`; `log`→`stream.progress("stage: message")`; ignore intermediate `artifact` events (`done` is a superset); `done`→markdown: summary, `### Findings` (severity/title/detail bullets), `### Plan` numbered, patch diff in ```diff fence + Copy button, `relevantFiles` as references.
- **`history`** → `GET /history`, markdown table (query | date | model | tokens saved).
- **`index`** → 2c unconditionally.
- **`model`** → `GET http://localhost:11434/api/tags` (direct to Ollama — backend has no model-list endpoint; no CORS in Node), `showQuickPick` of model names, save to `promptforge.model`, pass as `model` on subsequent requests. QuickPick from a chat handler is fine; no separate palette command.

**2e. Session + feedback** — module-level `sessionId = randomUUID()`, regenerated when `context.history.length === 0` (new chat). Handler returns `{metadata: {sessionId, chunkIds}}`; `participant.onDidReceiveFeedback` maps `Helpful/Unhelpful` → `POST /feedback {queryId: sessionId, feedback: 'up'|'down', chunkIds}` (backend records votes on chunk metadata — verified `main.py:616-637`). `// ponytail: one live session id; per-session Map if concurrent chats matter`.

## Step 3 — Build

`tsconfig.json`: `module: commonjs`, `target: es2022`, `outDir: out`, `strict: true`. Scripts: `compile`/`watch` = `tsc`. `launch.json`: standard `extensionHost` with `--extensionDevelopmentPath=${workspaceFolder}/vscode-extension`.

## Verification (manual, F5)

1. `cd vscode-extension && npm i && npm run compile`.
2. F5 → Extension Development Host → open the PromptForge folder (or any repo).
3. With backend stopped: `@promptforge how does retrieval work?` → "Starting backend…" → auto-index if 0 chunks → staged progress → streamed answer → clickable file references.
4. `@promptforge /forge add rate limiting` → streamed 6-section prompt + working Copy button.
5. `@promptforge /agent fix the index endpoint validation` → live stage/log progress → findings/plan/diff.
6. `/model` → QuickPick shows Ollama tags; next query's result reports that model.
7. `/history` renders; 👍/👎 lands in uvicorn log as `POST /feedback`.
8. Close dev host → uvicorn process gone (Task Manager).

## Explicitly NOT doing

- No marketplace publish, icon, CI, tests, webview, JetBrains, monorepo tooling.
- React frontend untouched; backend untouched.
- No per-request repo switching for `/query` (server-global index stays; `/index` re-points it).
- Not rendering intermediate agent `artifact` SSE events (`done` payload contains everything).

---
---

# Phase 2 — Scale up: multi-repo isolation, central backend, company rollout

**Deliverable for this phase: a single markdown document, not code changes.** Everything below was researched and verified against the actual code (file:line references throughout), but the ask right now is to write it down as `SCALE-UP.md` at the repo root — a design doc the user can act on later or hand to whoever owns infra/rollout. No source files get touched in this phase.

## Context

v1 works but exposed a real bug during testing: the backend has exactly **one global index**. Asking about "Swift" while PromptForge's own Python code was indexed produced a garbled forged prompt — the intent guard correctly saw "codebase question" and let it through, but retrieval had nothing to scope to, so it silently served the closest (wrong) chunks. That's the "clobbering" problem: indexing repo B mixes into (and for the import graph, **wipes**) repo A's context, because nothing in the backend carries a repo identity.

The user now wants to go from "works on my machine via F5" to a standing company tool: (1) many repos indexed side-by-side without clobbering, (2) headroom for bigger codebases, (3) one shared central backend instead of per-laptop Ollama+Chroma, (4) distribution via a private extension registry (Open VSX / internal gallery) instead of F5.

User-confirmed decisions:
- **Central backend**: one shared server (reuse `docker-compose.yml`), not per-developer local. Needs a lightweight shared-secret auth since it's now reachable over the network.
- **Distribution**: private extension registry (Open VSX or an internal gallery), not a shared `.vsix` file — auto-updates like the real Marketplace.

## Root cause (verified in code, not guessed)

Three independent global stores collide on a second `index_repo()` call, confirmed in `backend/app/`:
1. **Chroma** (`indexing.py:159`) — chunk IDs are `f"{rel}:{j}"`, no repo prefix → indexing repo B **overwrites** repo A's chunk if a relative path matches (e.g. both have `src/index.ts`).
2. **`graph_memory.py`** (`indexing.py:90`, `_link_graph` → `graph_memory.reset_graph()`) — called **unconditionally on every index run**, so indexing repo B **deletes repo A's entire import graph**, not just mixes it.
3. **`bm25_index.py`** (`_index`/`_ids`/`_corpus_map` module globals) — one in-memory corpus, one pickle file (`settings.bm25_path`), same path-based collision risk as Chroma.

Also global/unscoped: `watcher.py` (one watched path at a time — `start()` stops any previous watcher), `cache.py`/`/history` (scoped by `mode` only, never by repo), and `retrieval.py`'s `file_summary_threshold` check (counts chunks across *all* indexed repos combined, not one).

`AgentRunRequest` already carries `repo_path`/`target_path` (confirmed `agent_runner.py` `_run_agent_internal`), but today it's only used for auto-indexing and a **soft +0.18 score boost** on path-prefix matches (`_boost_chunks`/`_path_matches`, `agent_runner.py:121-126, 218-239`) — retrieval itself is never hard-filtered by it. Fixing repo scoping fixes this pre-existing bug for free.

## Design: repo_id via Chroma metadata filtering, not separate collections

Cheapest correct lever (confirmed feasible — `retrieval.py:38-48` already merges a `where={"path": {"$in": file_filter}}` filter into Chroma queries, so adding a second filter key is the same pattern, not new machinery): tag every chunk with a **`repo_id`** derived deterministically from the repo's absolute path, and filter every read by it. No new Chroma collections per repo (that would multiply file handles for no benefit over metadata filtering at this scale).

```python
# backend/app/store.py — new helper, next to the existing client()/collection() singletons
def repo_id_for(path: str) -> str:
    norm = str(Path(path).resolve()).lower()
    return hashlib.sha1(norm.encode()).hexdigest()[:12]
```

**Repo resolution for `/query` and `/query/stream`** (which — unlike `/agents/*` — carry no repo field today): resolve as `req.repo_id or store.get_last_repo_id()`, where `store.py` tracks the most-recently-indexed repo id as a module global (mirrors the existing `_client` singleton pattern in that file, updated by `indexing.index_repo()` on success). This means **zero-config callers keep working exactly as today** (whatever was indexed last), while the updated extension can pass an explicit `repoId` to target a specific repo. Deliberately *not* building a "search across all repos merged" mode — nobody asked for that, and it would require double-writing every index (global + scoped) into BM25/graph for no confirmed use case.

## The repeated pattern (applies to 3 files, same shape each time)

Each of these is today a **module-level singleton** with no repo key; the fix is the same shape in all three — swap the singleton for a `dict[str, State]` keyed by `repo_id`, and swap the single on-disk file for a directory of `<repo_id>.{json,pkl}` files:

- **`backend/app/graph_memory.py`** — `_digraph`/`_adapter` → `_graphs: dict[str, nx.DiGraph]`. `get_graph(repo_id)`, `reset_graph(repo_id)` (only clears *that* repo's graph — this is the fix for the "wipes on every index" bug), `save_graph(repo_id)` writes `<graph_dir>/<repo_id>.json`, `load_graph()` at startup globs `<graph_dir>/*.json` into the dict.
- **`backend/app/bm25_index.py`** — `_index`/`_ids`/`_corpus_map` → keyed by `repo_id` the same way. `build(repo_id, ids, docs)`, `update(repo_id, changed_paths, new_ids, new_docs)`, `query(repo_id, text, n)`, `save(repo_id)` writes `<bm25_dir>/<repo_id>.pkl`, `load()` globs `<bm25_dir>/*.pkl` at startup.
- **`backend/app/watcher.py`** — `_thread`/`_stop_event`/`_watched_path` → `_watchers: dict[str, WatcherState]`. `start(path, repo_id)` only stops/replaces `_watchers[repo_id]`, so watching repo B no longer kills repo A's watcher. `status()` returns a list, one entry per active repo.

`config.py` changes needed to support the directories above: replace `graph_path`/`bm25_path` (single-file settings, nothing external depends on their exact old paths — these are generated cache data, not user data) with `graph_dir: str = os.getenv("CF_GRAPH_DIR", "./.chroma/graphs")` and `bm25_dir: str = os.getenv("CF_BM25_DIR", "./.chroma/bm25")`. Also wrap `file_summary_threshold` with a `CF_FILE_SUMMARY_THRESHOLD` env override (currently a bare literal, no override — worth fixing while touching this file since it's directly relevant to "bigger codebases"), and add `api_key: str = os.getenv("CF_API_KEY", "")` for the central-backend auth below.

## Other backend files touched

- **`backend/app/indexing.py`** (`index_repo`, lines 119-231) — compute `repo_id = store.repo_id_for(path)` once at the top; add `"repo_id": repo_id` into every chunk/summary metadata dict (lines 162, 192); scope the existing-chunk lookups and deletion queries (`col.get(where={"path": rel}, ...)` at lines 151, 200) to `where={"$and": [{"path": rel}, {"repo_id": repo_id}]}`; pass `repo_id` into `bm25_index.update(...)`, `_link_graph(...)`/`graph_memory.reset_graph(...)`/`save_graph(...)`; call `store.set_last_repo_id(repo_id)` on success. Fix the chunk-count bug at line 223 (`col.count()` is *global* across all repos) — replace with `sum(len(v) for v in chunk_ids_by_file.values())`, which is already the correct per-repo live count from data already in hand, no extra Chroma round-trip. Add a small `list_repos()` / `_register_repo(...)` pair backed by a flat `<chroma_dir>/repos.json` registry (id → {path, name, indexedAt, files, chunks}), updated each successful index — powers the new `/repos` endpoint below.
- **`backend/app/retrieval.py`** — `retrieve(queries, top_k, repo_id=None)`, `_search(..., repo_id=None)` merges `{"repo_id": repo_id}` into the existing `where` dict (same `$and` merge as above when a `file_filter` is also present), `_file_prefilter(query_vec, repo_id, k=10)` scopes the summary-collection query and the threshold check (a repo-scoped count via `col.get(where={"repo_id": repo_id}, include=[])` length — acceptable cost at these repo sizes; `# ponytail:` comment noting this over a scoped-count API Chroma doesn't currently expose).
- **`backend/app/cache.py`** — add `repo_id` alongside the existing `mode` key already used in `lookup()`'s `where` filter; `list_history(repo_id=None)` gains an optional filter param.
- **`backend/app/agent_runner.py`** — `_run_agent_internal` derives `repo_id = store.repo_id_for(request.repo_path)` and passes it into the `retrieval.retrieve(...)` call (currently unscoped at line ~608) — this is the free fix for the pre-existing soft-boost-only bug.
- **`backend/app/schemas.py`** — `IndexResponse` gains `repo_id: str`; `QueryRequest` gains `repo_id: str | None = None`; new `RepoInfo(CamelModel)`: `id, name, path, indexed_at, files, chunks`.
- **`backend/app/main.py`**:
  - `/index` (line 139): thread `repo_id` through, scope the semantic-cache clear to that repo (currently `_clear_semantic_cache()` wipes everything — should only wipe the re-indexed repo's cached answers), `watcher.start(req.path, repo_id)`.
  - `/query`, `/query/stream` (lines 364-517): resolve `repo_id = req.repo_id or store.get_last_repo_id()`, thread into `retrieval.retrieve(...)`.
  - `/agents/run`, `/agents/stream`: derive `repo_id` from `req.repo_path`, thread into `agent_runner`.
  - New `GET /repos` → `indexing.list_repos()`.
  - `GET /health` (line 52): accept optional `?repoId=` → scoped chunk count when given, existing global total otherwise.
  - `/cache/clear`: accept optional `?repoId=` → scoped clear; omitted = today's full-wipe behavior (kept as the explicit "nuke everything" escape hatch).
  - `/watcher/status`: reflects the new list-of-watchers shape.
  - **Auth** (new, needed once this is reachable over the network): a `require_api_key` dependency checking `Authorization: Bearer <token>` against `settings.api_key`, wired via `FastAPI(dependencies=[Depends(require_api_key)])`. No-ops when `CF_API_KEY` is unset, so today's zero-config local dev experience is unchanged — it only activates once the shared-server `.env` sets it.

## Extension changes (`vscode-extension/src/extension.ts`)

- **Auto-index on first invocation, per-repo (not per-backend).** v1's `ensureBackend` auto-indexes only when `health.indexedChunks === 0` — a *global* count. Under multi-repo that check breaks: once *any* repo has ever been indexed on that backend, the global count is nonzero forever, so a second or third project opened later would never trigger auto-index and `@promptforge` would silently answer from the wrong (or no) repo. Fix: the invoke-time check moves from "is the backend's index empty?" to "does *this* workspace have a cached `repoId` yet?"
  1. On every `@promptforge` invocation, before dispatching to a handler: if `context.workspaceState.get('promptforge.repoId')` is unset for this workspace, compute the expected id client-side (mirror `store.repo_id_for` — same `sha1(resolve(path).lower())[:12]`, a few lines in TS, no backend round trip needed just to check) and call `GET /health?repoId=<id>`.
  2. If that reports chunks > 0, the repo was already indexed — by this user previously, or (on a shared central backend) by a teammate — so just cache the id and proceed. No redundant re-index.
  3. If it reports 0 (or 404s), this is genuinely new: `stream.progress('Indexing workspace…')`, call `POST /index {path: workspaceRoot()}`, cache the returned `repoId`.
  4. Either way, the user never has to type `/index` manually the first time — asking a question in a fresh repo just works, matching the original ask ("automatically indexed when the chat is invoked").
- Cache the active `repoId` in `context.workspaceState` (auto-scoped per opened folder by VS Code); read it before every `/query/stream` and `/history` call and send it as `repoId` (currently omitted entirely from the query body — this is the actual client-side fix). Agent mode already sends `repoPath`, so it needs no change once the server derives `repo_id` from it.
- New `/repos` slash command: `GET /repos` → render as clickable chat buttons (same idiom already built for `/model`) → clicking sets `workspaceState`'s active `repoId` without re-indexing, so a previously-indexed repo can be queried without having it open.
- New `promptforge.apiKey` setting; a small `authHeaders()` helper merged into every `fetch`/SSE call as `Authorization: Bearer <key>` when set.
- `ensureBackend`'s auto-spawn guard: only attempt to spawn a local `uvicorn` when `new URL(backendUrl()).hostname` is `localhost`/`127.0.0.1`. Against a configured remote/shared URL, a failed health check should show "can't reach the shared PromptForge server — check VPN / `promptforge.backendUrl`" instead of trying (and failing) to spawn one locally.
- Rollout note (docs, not code): the shared backend's URL and API key get set once per teammate via User-level settings (or a company-managed settings profile) — not something to hardcode into the extension, since the real address isn't known here.

## Central backend deployment

Reuse the existing `docker-compose.yml` (already has `ollama` + `ollama-init` + `backend` services, GPU passthrough, and a `promptforge_data` volume at `/data` — confirmed present):
- Drop the `frontend` service — the React portal is frozen (per earlier decision), not needed on a shared server.
- Add `CF_API_KEY` to the `backend` service's environment, sourced from an untracked `.env` (add `.env.example` with `CF_API_KEY=` as a placeholder if one doesn't already exist).
- The existing single `promptforge_data` volume needs no structural change — it now just holds the new per-repo subdirectories (`graphs/`, `bm25/`, `repos.json`) alongside the existing Chroma data.
- Flag, don't build: concurrent-user capacity (Ollama serializes generation per model by default) is a real scaling wall once more than a few people hit it at once. Out of scope until it's an observed problem — `# ponytail:` the compose file noting "single Ollama instance; add replicas/queueing if concurrent load becomes a bottleneck."

## Distribution via private extension registry

What's buildable from inside this repo: `vscode-extension/package.json` gains `publisher` (a real namespace registered on whichever registry is chosen), `repository`, `license`, and a minimal required `README.md` (registries generally require one); add `vsce`/`ovsx` devDependencies and `"package": "vsce package"` / `"publish": "ovsx publish -p $env:OVSX_PAT --registryUrl $env:OVSX_URL"` scripts (PowerShell env-var syntax, matching the user's shell). Run `npx vsce package` early to surface whatever else it demands rather than guessing every required field up front.

**Explicitly out of scope for this plan**: standing up the actual private Open VSX instance (or an Azure DevOps Artifacts extension feed) is real infrastructure — a hosted service the user's org needs to decide on and provision. This plan wires the extension to be publishable to one once it exists; it does not create the registry itself.

## Deliverable

Write `SCALE-UP.md` at the repo root (`c:\Users\nsani\Desktop\PromptForge\SCALE-UP.md`), containing everything above: the root-cause analysis, the `repo_id` design, the per-file change list (backend + extension + deploy + distribution), and a "when you're ready to build this" checklist. Structure it so each section is independently actionable later — someone should be able to pick up just the "central backend" section and act on it without needing the multi-repo section done first, since multi-repo isolation is the only true prerequisite the other two depend on.

No code, config, or docker-compose changes in this phase — documentation only.
