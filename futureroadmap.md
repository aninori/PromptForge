# PromptForge — Future Roadmap

Four items, roughly in dependency order. Item 1 is a pure backend refactor with no external dependents. Item 2 is the prerequisite for items 3 and 4 — there is no point centralizing a backend, or shipping it to a whole company, while indexing a second repo silently corrupts the first one's context (the exact bug that surfaced this session: asking about "Swift" against a Python-only index returned a garbled answer instead of "nothing indexed here," because retrieval had no repo boundary to respect).

---

## 1. Consolidate the triplicated pipeline

The natural next move: collapse the three pipeline copies into one event-yielding generator, and use that single owner to cut pre-flight latency. The codebase is visibly straining toward this — it's half-built already — and it's the prerequisite the earlier findings kept circling: no optimization can land once while the pipeline exists in three places.

Why this is the native next step rather than an imposed one:

The event vocabulary already exists. _stream_pipeline emits stage / token / done / error SSE events, and agent_runner.py:543 already uses a stage(name, message) callback pattern with a queue-backed worker. The paradigm is established; the query path just never adopted it.
The seams are already cut. semantic_router.route() exists precisely because someone started extracting the pipeline — but the streaming path couldn't use it (it can't yield stage events mid-flight), so it re-inlined everything and reaches into the private _classify. A generator resolves the tension that created the fork.
The tools are already imported. ThreadPoolExecutor sits unused at main.py:17 and is already the concurrency idiom in agent_runner and ollama_client. The README's query-path diagram even advertises a "Parallel (ThreadPoolExecutor × 2)" step that the code doesn't actually perform yet.
Implementation plan
Step 1 — Create pipeline.py with one generator owning the full flow. A module-level function (matching the codebase's no-classes style, docstring header on top): run_pipeline(query, mode, model_override, top_k, use_expansion, session_id) -> Iterator[tuple[str, Any]] yielding ("stage", name), ("token", text), ("done", PipelineResult), ("error", msg). Move the cache-hit path, off-topic guard, optimize→expand, retrieve, agent-select, assemble, generate, token accounting, and cache-save into it — each currently duplicated across main.py:360-499 and main.py:516-609.

Step 2 — Make both endpoints thin consumers. /query/stream maps events to SSE lines (a ~10-line adapter). /query drains the generator, discards tokens, returns the done payload. Delete _stream_pipeline and the body of route()'s duplication; semantic_router keeps only the classifiers and select_agent. Net deletion should be ~150 lines — consistent with this repo's deletion-over-addition habit.

Step 3 — Parallelize the two classifier calls. Intent guard and complexity tier are independent small-model round trips made sequentially today (main.py:396 then 414). Submit both to a two-worker ThreadPoolExecutor and join. Zero semantic change, removes one full small-model round trip (~0.5–2s on local Ollama) from every uncached query. Note: do not parallelize optimize+expand — ARCHITECTURE.md documents that expansion deliberately chains off the rewritten query, so that ordering is a feature.

Step 4 — Skip expansion for simple-tier queries. The router already computes the tier; simple one-liner questions don't need three sub-queries and the extra embeds/searches they trigger. expansions = expand_query(...) if use_expansion and tier == "complex" else [], with a one-line # ponytail: comment naming the assumption — matching the existing convention for deliberate shortcuts.

Step 5 — Leave one runnable check behind. A small test_pipeline.py: monkeypatch ollama_client.chat/embed with canned responses, assert that draining run_pipeline yields the stage names in order and that /query and /query/stream produce the same PipelineResult for the same input. That single test is the guard that prevents the blocking/streaming paths from ever diverging again — the exact failure mode the current structure invites.

Expected payoff: every future pipeline change (including the auth/CORS work from the security review) lands in one place; uncached simple queries lose two LLM round trips (~2–4s of the 4–8s p50 the README reports); and the agent runner becomes the obvious next candidate to consume the same generator, retiring the third copy when you're ready.

---

## 2. Multi-repo isolation — stop indexing runs from clobbering each other

The frontend is now a VS Code chat extension (`vscode-extension/`), not the old React app (archived to `_archive/react-frontend/`, no longer part of the running system) — which means the natural usage pattern shifted from "one person, one browser tab, one repo" to "one backend, opened against whatever repo the developer currently has open in their editor." The backend was never built for that: it has exactly one global index, and switching repos doesn't scope, it *destroys*.

Three independent global stores collide the moment a second repo is indexed, confirmed in `backend/app/`:
- **Chroma** (`indexing.py:159`) — chunk IDs are `f"{rel}:{j}"`, no repo prefix. Indexing repo B silently overwrites repo A's chunk wherever a relative path collides (e.g. both have `src/index.ts`).
- **`graph_memory.py`** (`indexing.py:90`, `_link_graph` → `graph_memory.reset_graph()`) — called unconditionally on *every* index run, so indexing repo B deletes repo A's entire import graph, not just mixes it in.
- **`bm25_index.py`** — one in-memory corpus, one pickle file (`settings.bm25_path`), same path-collision risk as Chroma.

Also unscoped: `watcher.py` (only one watched path at a time — starting a second stops the first), `cache.py`/`/history` (scoped by `mode` only, never by repo, so a cache hit from repo A can be served verbatim for the same question asked against repo B), and `retrieval.py`'s `file_summary_threshold` check (counts chunks across *every* indexed repo combined, not the one being searched). `AgentRunRequest` already carries `repo_path`/`target_path`, but today it only drives a soft +0.18 score boost (`agent_runner.py:121-126, 218-239`) — retrieval itself is never hard-filtered by it, so even agent mode isn't actually repo-scoped despite looking like it should be.

Implementation plan

Step 1 — Add `repo_id_for(path)` to `store.py`, next to the existing `client()`/`collection()` singletons: `hashlib.sha1(str(Path(path).resolve()).lower().encode()).hexdigest()[:12]`. Deterministic — same repo, same id, no client-side bookkeeping required. Also add `get_last_repo_id()`/`set_last_repo_id()` module globals there (same singleton pattern the file already uses for `_client`).

Step 2 — Tag every chunk with `repo_id` in `indexing.py` (`all_metas` at line 162, summary metas at line 192), and scope the existing-chunk/deletion lookups (lines 151, 200) to `where={"$and": [{"path": rel}, {"repo_id": repo_id}]}`. Fix the chunk-count bug at line 223 while in there — `col.count()` is global across all repos; replace with `sum(len(v) for v in chunk_ids_by_file.values())`, which is already the correct per-repo count from data already in hand.

Step 3 — Apply the same shape to the three singleton modules: `graph_memory.py`'s `_digraph`/`_adapter` becomes `_graphs: dict[str, nx.DiGraph]` keyed by `repo_id`, persisted as `<graph_dir>/<repo_id>.json` instead of one file; `bm25_index.py`'s `_index`/`_ids`/`_corpus_map` becomes the same dict-per-repo shape, persisted as `<bm25_dir>/<repo_id>.pkl`; `watcher.py`'s single `_thread`/`_watched_path` becomes `_watchers: dict[str, WatcherState]` so watching repo B no longer kills repo A's watcher. `config.py` gains `graph_dir`/`bm25_dir` (replacing the old single-file `graph_path`/`bm25_path` — generated cache data, nothing external depends on the old paths) plus a `CF_FILE_SUMMARY_THRESHOLD` env override (currently a bare literal with no override, worth fixing here since it's directly relevant to bigger codebases).

Step 4 — Thread `repo_id` through `retrieval.retrieve(queries, top_k, repo_id=None)` and `agent_runner.py`'s `_run_agent_internal` (derive `repo_id` from `request.repo_path`, pass into the currently-unscoped `retrieval.retrieve()` call at line ~608 — this is a free fix for the soft-boost-only bug above). For `/query`/`/query/stream`, which carry no repo field at all today, resolve as `req.repo_id or store.get_last_repo_id()` — zero-config callers keep working exactly as today (whatever was indexed last), while an updated client can pass an explicit `repoId`. Add `QueryRequest.repo_id: str | None`, `IndexResponse.repo_id: str`, and a new `RepoInfo` schema + `GET /repos` endpoint (backed by a flat `<chroma_dir>/repos.json` registry written on each index) so a client can list and switch between previously-indexed repos.

Step 5 — Update `vscode-extension/src/extension.ts` to cache the active `repoId` in `context.workspaceState` after `/index`, send it on every `/query/stream` and `/history` call, and add a `/repos` slash command (same button-list idiom already built for `/model`) for switching without re-indexing.

Expected payoff: the exact failure from this session — asking about a language/framework that isn't in the currently-open repo — either scopes correctly to what's actually indexed or clearly reports nothing found, instead of silently answering from whatever unrelated repo happens to share the global collection. Also the prerequisite for items 3 and 4 below: a shared backend serving multiple developers *is* multiple repos indexed concurrently, so this has to land first.

---

## 3. Central backend for company-wide use

Right now every developer runs their own backend + Ollama locally (the VS Code extension's `ensureBackend` auto-spawns `uvicorn` if it's not already up). That's fine for one person; it means N developers each need a GPU-capable machine with the full model set pulled locally, which doesn't scale to "the whole team uses this."

`docker-compose.yml` already has the shape of a shared deployment — `ollama` + `ollama-init` + `backend` services, GPU passthrough, a `promptforge_data` volume at `/data` (the `frontend` service was removed this session along with the React app it served). What's missing is auth: `main.py:43-49` combines `allow_origins=["*"]` with `allow_credentials=True` and has no authentication on any endpoint, which `TechnicalDebt.md` #1 already flags as a live vulnerability on localhost — the moment this is reachable over a network instead of just 127.0.0.1, `POST /index` and `GET /repo/tree` become an unauthenticated arbitrary-file-read/exfiltration primitive.

Implementation plan

Step 1 — Do the auth fix `TechnicalDebt.md` already specced: pin CORS to known origins, add a `CF_API_KEY` env var and a `require_api_key` FastAPI dependency checking `Authorization: Bearer <token>`, wired at the app level so it covers every route. No-op when `CF_API_KEY` is unset, so local solo-developer usage stays zero-config.

Step 2 — In the extension, add a `promptforge.apiKey` setting and attach it as a header on every request; guard `ensureBackend`'s auto-spawn to only fire when `backendUrl`'s hostname is `localhost`/`127.0.0.1` — against a real shared server, a failed health check should say "can't reach the shared backend," not try to spawn a local uvicorn that has nothing to spawn.

Step 3 — Stand up the shared instance from the trimmed `docker-compose.yml` on a host with a GPU sized for concurrent generation. Point the team's `promptforge.backendUrl`/`promptforge.apiKey` settings at it (User-level settings or a company-managed profile, not hardcoded into the extension since the real address isn't known until this is deployed).

Expected payoff: onboarding a new developer becomes "install the extension, set two settings" instead of "install Ollama, pull two multi-GB models, run the backend locally." Flagged, not built yet: concurrent-request capacity — Ollama serializes generation per model by default, so this is a real wall once more than a handful of people query simultaneously; add replicas/queuing if that becomes an observed problem rather than guessing at load now.

---

## 4. Distribution via a private extension registry

Once the extension is worth using company-wide, F5-from-source stops being a reasonable install path. The user chose a private registry (Open VSX or an internal gallery) over a passed-around `.vsix` specifically for auto-updates.

What's buildable from inside this repo: `vscode-extension/package.json` needs `publisher` (a real namespace registered on whichever registry gets chosen), `repository`, `license`, and a minimal `README.md` — most registries require these. Add `vsce`/`ovsx` as devDependencies and `"package": "vsce package"` / `"publish": "ovsx publish -p $env:OVSX_PAT --registryUrl $env:OVSX_URL"` scripts. Run `npx vsce package` early to see what else it demands rather than guessing every required field up front.

**Out of scope for this repo:** provisioning the actual private Open VSX instance (or an Azure DevOps Artifacts extension feed) is real infrastructure — a hosted service the org needs to decide on and stand up. This item makes the extension publishable to one once it exists; it doesn't create the registry itself.