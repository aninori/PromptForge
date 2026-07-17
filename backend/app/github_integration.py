"""GitHub integration — clone a repo at a branch, keep its index fresh.

Three layers, each activates only when its config is present:

  1. Core (always on): clone/pull/checkout a repo into a local workspace, then
     reuse indexing.index_repo() on that path. Branch-specific and incremental —
     switching branches or syncing purges deleted files and re-embeds changed
     ones (via the existing SHA-256 diff in indexing.py).
  2. OAuth (needs github_client_id + github_client_secret): "Sign in with GitHub"
     to pull a user token for private repos + repo listing. localhost callback OK.
  3. Webhook (needs github_webhook_secret + a public URL): auto re-index on push.

Auth precedence for git/API calls: OAuth token (if signed in) → PAT (github_token)
→ none (public repos only).

# ponytail: single-process in-memory token store — one signed-in user at a time.
# Swap for a per-session table when this becomes a multi-user portal.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import subprocess
from pathlib import Path

import httpx

from .config import settings

_API = "https://api.github.com"

# In-memory state (single-user local portal).
_oauth_token: str | None = None
_oauth_states: set[str] = set()
# repo_full_name -> {"path": str, "branch": str, "url": str}
_connected: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _active_token() -> str:
    """OAuth token if signed in, else the configured PAT, else empty (public)."""
    return _oauth_token or settings.github_token or ""


def oauth_enabled() -> bool:
    return bool(settings.github_client_id and settings.github_client_secret)


def is_authenticated() -> bool:
    return bool(_active_token())


# ---------------------------------------------------------------------------
# OAuth (layer 2)
# ---------------------------------------------------------------------------

def oauth_login_url() -> str:
    """Build the GitHub authorize URL; caller redirects the browser to it."""
    if not oauth_enabled():
        raise ValueError("GitHub OAuth is not configured (set CF_GITHUB_CLIENT_ID/SECRET).")
    state = secrets.token_urlsafe(16)
    _oauth_states.add(state)
    return (
        "https://github.com/login/oauth/authorize"
        f"?client_id={settings.github_client_id}"
        f"&redirect_uri={settings.github_oauth_callback}"
        "&scope=repo"
        f"&state={state}"
    )


def oauth_exchange(code: str, state: str) -> None:
    """Exchange the callback code for an access token; store it in memory."""
    global _oauth_token
    if state not in _oauth_states:
        raise ValueError("Invalid OAuth state.")
    _oauth_states.discard(state)
    resp = httpx.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code": code,
            "redirect_uri": settings.github_oauth_callback,
        },
        timeout=15,
    )
    token = resp.json().get("access_token")
    if not token:
        raise ValueError("GitHub did not return an access token.")
    _oauth_token = token


def logout() -> None:
    global _oauth_token
    _oauth_token = None


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = _active_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def list_repos() -> list[dict]:
    """List repos the authenticated user can access (name, private, default branch)."""
    if not is_authenticated():
        raise ValueError("Not authenticated — sign in with GitHub or set CF_GITHUB_TOKEN.")
    repos: list[dict] = []
    resp = httpx.get(
        f"{_API}/user/repos?per_page=100&sort=updated",
        headers=_headers(), timeout=20,
    )
    resp.raise_for_status()
    for r in resp.json():
        repos.append({
            "fullName": r["full_name"],
            "url": r["clone_url"],
            "private": r["private"],
            "defaultBranch": r["default_branch"],
        })
    return repos


def list_branches(full_name: str) -> list[str]:
    resp = httpx.get(f"{_API}/repos/{full_name}/branches?per_page=100",
                     headers=_headers(), timeout=20)
    resp.raise_for_status()
    return [b["name"] for b in resp.json()]


# ---------------------------------------------------------------------------
# Git operations (layer 1 core)
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _local_path(full_name: str) -> Path:
    safe = _SAFE.sub("_", full_name)
    return Path(settings.github_workspace) / safe


def _auth_url(clone_url: str) -> str:
    """Inject the token into an https clone URL for private-repo access."""
    token = _active_token()
    if token and clone_url.startswith("https://"):
        return clone_url.replace("https://", f"https://x-access-token:{token}@", 1)
    return clone_url


def _run_git(args: list[str], cwd: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        # Never surface the token if it leaked into an error message.
        msg = (proc.stderr or proc.stdout).strip()
        token = _active_token()
        if token:
            msg = msg.replace(token, "***")
        raise RuntimeError(f"git {args[0]} failed: {msg}")
    return proc.stdout.strip()


def clone_or_update(full_name: str, clone_url: str, branch: str) -> Path:
    """Clone the repo at `branch` if missing; otherwise fetch + checkout + pull.

    Returns the local path. Idempotent — safe to call for connect, switch, and sync.
    """
    path = _local_path(full_name)
    auth_url = _auth_url(clone_url)
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--branch", branch, auth_url, str(path)])
    else:
        _run_git(["remote", "set-url", "origin", auth_url], cwd=str(path))
        _run_git(["fetch", "origin", branch], cwd=str(path))
        _run_git(["checkout", branch], cwd=str(path))
        _run_git(["reset", "--hard", f"origin/{branch}"], cwd=str(path))
    _connected[full_name] = {"path": str(path), "branch": branch, "url": clone_url}
    return path


def current_sha(full_name: str) -> str:
    info = _connected.get(full_name)
    if not info:
        return ""
    return _run_git(["rev-parse", "HEAD"], cwd=info["path"])


def connected() -> dict[str, dict]:
    return dict(_connected)


# ---------------------------------------------------------------------------
# Webhook (layer 3)
# ---------------------------------------------------------------------------

def verify_signature(payload: bytes, signature: str | None) -> bool:
    """Verify GitHub's X-Hub-Signature-256 HMAC. Fails closed when unconfigured."""
    secret = settings.github_webhook_secret
    if not secret or not signature:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
