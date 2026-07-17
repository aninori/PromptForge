"""Background file watcher — auto re-indexes when source files change.

Started by /index; watches the last indexed path using watchfiles
(already a transitive dep via uvicorn[standard]). Changes are debounced
before triggering an incremental re-index.
"""
from __future__ import annotations

import logging
import threading

from .indexing import CODE_EXTENSIONS

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_watched_path: str | None = None
_lock = threading.Lock()


def start(path: str) -> None:
    """Start watching `path`; stops any previous watcher first."""
    global _thread, _stop_event, _watched_path
    with _lock:
        _stop()
        _watched_path = path
        _stop_event = threading.Event()
        _thread = threading.Thread(target=_loop, args=(path, _stop_event),
                                   daemon=True, name="fs-watcher")
        _thread.start()
        logger.info("File watcher started: %s", path)


def stop() -> None:
    with _lock:
        _stop()


def _stop() -> None:
    global _thread
    if _thread and _thread.is_alive():
        _stop_event.set()
        _thread.join(timeout=3)
    _thread = None


def status() -> dict:
    return {"active": bool(_thread and _thread.is_alive()), "path": _watched_path}


def _loop(path: str, stop_event: threading.Event) -> None:
    try:
        from watchfiles import watch
    except ImportError:
        logger.warning("watchfiles not available — background watcher disabled")
        return

    def _code_filter(change, p: str) -> bool:  # noqa: ANN001
        from pathlib import Path
        return Path(p).suffix.lower() in CODE_EXTENSIONS

    try:
        for changes in watch(path, watch_filter=_code_filter, debounce=2000,
                             rust_timeout=1000, yield_on_timeout=True):
            if stop_event.is_set():
                break
            if not changes:
                continue
            from . import indexing
            try:
                indexing.index_repo(path, incremental=True)
                logger.info("Auto re-indexed %s (%d file(s) changed)", path, len(changes))
            except Exception as exc:
                logger.error("Auto re-index failed: %s", exc)
    except Exception as exc:
        logger.error("Watcher loop crashed: %s", exc)
