"""PromptForge backend — FastAPI.

Endpoints map 1:1 to the frontend's data layer:
  POST /index    -> Repository page (index a repo/folder)
  POST /query    -> Workbench (the full RAG run)
  GET  /history  -> History page
  GET  /health   -> sanity check

Run:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

# uvicorn configures handlers only for its own loggers, so without this every
# logger.info/error in this package (indexing progress, watcher failures) is
# dropped on the floor. Attach a root handler unless the host already set one up.
# Must run BEFORE the package imports below: they execute module-level code that
# logs (turn_store restoring persisted turns), which would otherwise vanish.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(name)s - %(message)s",
    )

from . import cache, dashboard, github_integration, indexing, pipeline, savings, watcher  # noqa: E402
from .config import settings  # noqa: E402
from .schemas import (
    FeedbackRequest,
    IndexRequest,
    IndexResponse,
    PipelineResult,
    QueryRequest,
    RepoEntry,
    RepoTreeResponse,
    ReviewActionRequest,
)
from .store import collection  # noqa: E402

app = FastAPI(title="PromptForge RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
def health():
    code = collection(settings.code_collection)
    return {
        "ok": True,
        "indexedChunks": code.count(),
        "provider": settings.provider,
        # Absolute path of the repo in the index — lets a client notice the
        # index belongs to a different workspace and re-index.
        "indexedRepo": indexing.indexed_repo(),
    }


@app.get("/config")
def runtime_config():
    return {
        "provider": settings.provider,
        "generationModel": settings.gen_model,
        "embeddingModel": settings.embed_model,
        "topK": settings.top_k,
        "tokenBudget": settings.token_budget,
        "useExpansion": settings.use_expansion,
    }


@app.patch("/config")
def update_config(body: dict):
    """Update runtime settings in-memory. Changes persist until server restart;
    set the corresponding CF_* env vars for permanent changes."""
    # Each entry: camelCase key → (attr_name, coerce_fn)
    mapping: dict[str, tuple[str, type]] = {
        "provider":       ("provider",      str),
        "generationModel":("gen_model",      str),
        "embeddingModel": ("embed_model",    str),
        "topK":           ("top_k",         int),
        "tokenBudget":    ("token_budget",  int),
        "useExpansion":   ("use_expansion", bool),
        "ollamaUrl":      ("ollama_url",    str),
    }
    errors: list[str] = []
    for key, value in body.items():
        entry = mapping.get(key)
        if not entry:
            continue
        attr, coerce = entry
        try:
            setattr(settings, attr, coerce(value))
            # ponytail: Settings page has one model field; keep the "large" tier
            # in sync so complex/agent queries don't silently use the old default.
            if attr == "gen_model":
                settings.large_model = settings.gen_model
        except (ValueError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))
    return runtime_config()


@app.get("/repo/tree", response_model=RepoTreeResponse)
def repo_tree(path: str = Query(".", description="Folder to inspect")):
    current = Path(path)
    if not current.is_absolute():
        current = (Path.cwd() / current).resolve()
    if not current.exists() or not current.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {path}")

    entries: list[RepoEntry] = []
    skip = {"node_modules", ".git", "dist", "build", ".venv", "__pycache__", ".chroma"}
    try:
        children = sorted(
            [child for child in current.iterdir() if child.name not in skip],
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for child in children[:150]:
        entries.append(
            RepoEntry(
                name=child.name,
                path=str(child),
                type="folder" if child.is_dir() else "file",
            )
        )

    parent = current.parent if current.parent != current else None
    return RepoTreeResponse(
        root_path=str(Path.cwd()),
        current_path=str(current),
        parent_path=str(parent) if parent else None,
        entries=entries,
    )


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest):
    try:
        result = indexing.index_repo(req.path, req.name)
        _clear_semantic_cache()
        watcher.start(req.path)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    """Self-contained savings dashboard. Served from the backend so its numbers
    are live — it fetches /savings same-origin."""
    return HTMLResponse(dashboard.PAGE)


@app.get("/savings")
def savings_totals():
    """Cumulative count of turns that never became a Copilot call.

    Deliberately an under-estimate — input context only, no model replies and
    no allowance for the extra files Copilot reads while hunting. See
    savings.py for why avoided *pasting* is not counted at all.
    """
    return savings.totals()


@app.post("/savings/reset")
def savings_reset():
    savings.reset()
    return {"ok": True}


@app.get("/watcher/status")
def watcher_status():
    return watcher.status()


# ---------------------------------------------------------------------------
# GitHub integration
# ---------------------------------------------------------------------------

# Where to send the browser back after OAuth (the frontend origin).
_FRONTEND_URL = "http://localhost:3000"


def _clone_and_index(full_name: str, url: str, branch: str) -> dict:
    """Clone/checkout the repo at `branch`, then incrementally (re)index it.

    Shared by connect / switch-branch / sync / webhook. Branch-specific:
    index_repo(incremental=True) purges files gone from the new branch and
    re-embeds changed ones, so answers never go stale across branch moves.
    """
    path = github_integration.clone_or_update(full_name, url, branch)
    result = indexing.index_repo(str(path), name=full_name)
    _clear_semantic_cache()
    return {
        "repo": full_name,
        "branch": branch,
        "sha": github_integration.current_sha(full_name),
        "files": result.files,
        "chunks": result.chunks,
        "sizeMb": result.size_mb,
    }


def _repo_url(full_name: str, url: str | None) -> str:
    return url or f"https://github.com/{full_name}.git"


@app.get("/github/status")
def github_status():
    return {
        "oauthEnabled": github_integration.oauth_enabled(),
        "authenticated": github_integration.is_authenticated(),
        "connected": github_integration.connected(),
    }


@app.get("/github/login")
def github_login():
    try:
        return RedirectResponse(github_integration.oauth_login_url())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/github/callback")
def github_callback(code: str = Query(...), state: str = Query(...)):
    try:
        github_integration.oauth_exchange(code, state)
    except ValueError as e:
        return RedirectResponse(f"{_FRONTEND_URL}/?github=error&reason={e}")
    return RedirectResponse(f"{_FRONTEND_URL}/?github=connected")


@app.post("/github/logout")
def github_logout():
    github_integration.logout()
    return {"ok": True}


@app.get("/github/repos")
def github_repos():
    try:
        return github_integration.list_repos()
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/github/branches")
def github_branches(repo: str = Query(..., description="owner/name")):
    try:
        return github_integration.list_branches(repo)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/github/connect")
def github_connect(body: dict):
    full_name = body.get("fullName", "").strip()
    branch = body.get("branch", "").strip()
    if not full_name or not branch:
        raise HTTPException(status_code=422, detail="fullName and branch are required.")
    try:
        return _clone_and_index(full_name, _repo_url(full_name, body.get("url")), branch)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/github/switch-branch")
def github_switch_branch(body: dict):
    full_name = body.get("fullName", "").strip()
    branch = body.get("branch", "").strip()
    info = github_integration.connected().get(full_name)
    if not info or not branch:
        raise HTTPException(status_code=422, detail="Repo not connected or branch missing.")
    try:
        return _clone_and_index(full_name, info["url"], branch)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/github/sync")
def github_sync(body: dict):
    full_name = body.get("fullName", "").strip()
    info = github_integration.connected().get(full_name)
    if not info:
        raise HTTPException(status_code=422, detail="Repo not connected.")
    try:
        return _clone_and_index(full_name, info["url"], info["branch"])
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/github/webhook")
async def github_webhook(request: Request):
    """Auto re-index on push. Needs CF_GITHUB_WEBHOOK_SECRET + a public URL."""
    payload = await request.body()
    if not github_integration.verify_signature(payload, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="Invalid signature.")
    if request.headers.get("X-GitHub-Event") != "push":
        return {"ok": True, "ignored": "not a push event"}
    data = await request.json()
    full_name = (data.get("repository") or {}).get("full_name", "")
    pushed_branch = (data.get("ref") or "").replace("refs/heads/", "")
    info = github_integration.connected().get(full_name)
    if not info or info["branch"] != pushed_branch:
        return {"ok": True, "ignored": "repo/branch not connected"}
    try:
        return _clone_and_index(full_name, info["url"], pushed_branch)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=str(e))


def _clear_semantic_cache() -> int:
    col = collection(settings.history_collection)
    ids = col.get(include=[], limit=100_000).get("ids") or []  # type: ignore[arg-type]
    if ids:
        col.delete(ids=ids)
    return len(ids)


@app.post("/cache/clear")
def clear_cache():
    removed = _clear_semantic_cache()
    return {"ok": True, "removed": removed}


# ---------------------------------------------------------------------------
# Query pipeline — /query and /query/stream are thin consumers of the single
# generator in pipeline.py. /query/stream forwards its events as SSE lines;
# /query drains it and returns only the final "done" payload.
# ---------------------------------------------------------------------------


def _to_sse(events):
    for event, data in events:
        if event == "done":
            yield f"event: done\ndata: {data.model_dump_json(by_alias=True)}\n\n"
        elif event == "stage":
            yield f"event: stage\ndata: {data}\n\n"
        else:  # token | error — plain-text payloads, JSON-encoded
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Streaming variant — uses SSE to progressively push tokens to the UI."""
    events = pipeline.run_pipeline(
        req.query, req.mode, req.model,
        # Omitted fields fall back to config, so /config is authoritative rather
        # than decorative. `is None` (not `or`) — use_expansion=False is a real
        # choice, and `or` would silently flip it back to the configured value.
        settings.top_k if req.top_k is None else req.top_k,
        settings.use_expansion if req.use_expansion is None else req.use_expansion,
        req.session_id,
    )
    return StreamingResponse(
        _to_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/query", response_model=PipelineResult)
def query(req: QueryRequest):
    events = pipeline.run_pipeline(
        req.query, req.mode, req.model,
        # Omitted fields fall back to config, so /config is authoritative rather
        # than decorative. `is None` (not `or`) — use_expansion=False is a real
        # choice, and `or` would silently flip it back to the configured value.
        settings.top_k if req.top_k is None else req.top_k,
        settings.use_expansion if req.use_expansion is None else req.use_expansion,
        req.session_id,
    )
    for event, data in events:
        if event == "done":
            return data
        if event == "error":
            raise HTTPException(status_code=502, detail=data)
    raise HTTPException(status_code=502, detail="Pipeline ended without a result.")


@app.post("/query/review")
def query_review(req: ReviewActionRequest):
    """Review-gate regeneration — refine a brief, or convert answer<->brief —
    from a stored turn's already-retrieved chunks. Never re-runs retrieval.
    Streaming only: the extension always consumes this via SSE."""
    events = pipeline.run_review_action(req.turn_id, req.action, req.note, req.model)
    return StreamingResponse(
        _to_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Record relevancy feedback (thumbs up/down) for a previous query.

    In production this would update a per-file relevance score in Chroma
    metadata, then let retrieval.py bias toward highly-rated chunks.
    Currently records the signal; full re-ranking integration is the next step.
    """
    try:
        if req.chunk_ids:
            code_col = collection(settings.code_collection)
            for cid in req.chunk_ids:
                existing = code_col.get(ids=[cid], include=["metadatas"])  # type: ignore[arg-type]
                ids_found = existing.get("ids") if existing else None
                if ids_found:
                    metas = existing.get("metadatas") or [{}]  # type: ignore[union-attr]
                    meta: dict = dict(metas[0]) if metas else {}
                    if req.feedback == "up":
                        meta["votes_up"] = int(meta.get("votes_up", 0)) + 1
                    else:
                        meta["votes_down"] = int(meta.get("votes_down", 0)) + 1
                    code_col.update(ids=[cid], metadatas=[meta])
        return {"ok": True, "recorded": req.feedback}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/history")
def history():
    return cache.list_history()
