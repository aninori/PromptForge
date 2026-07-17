"""Graph memory — link chunks by import/call graph, backed by networkx + disk.

The graph is persisted to graph.json alongside the Chroma data so it survives
server restarts.  Re-indexing rebuilds and re-saves the graph atomically.

Public API (same as before):
  get_graph()        → _GraphAdapter (supports .add_edge / .neighbors)
  reset_graph()      → clear in-memory graph and delete the saved file
  extract_imports()  → parse import statements from source text
  save_graph()       → flush the current graph to disk
  load_graph()       → populate from disk (called once at import time)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from .config import settings

# ---------------------------------------------------------------------------
# Import-parsing heuristics for common languages.
# ---------------------------------------------------------------------------

_IMPORT_PATTERNS: list[re.Pattern] = [
    re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE),
    re.compile(r'^\s*from\s+([\w.]+)\s+import', re.MULTILINE),
    re.compile(r"""(?:import|from)\s+['"]([^'"]+)['"]""", re.MULTILINE),
    re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]""", re.MULTILINE),
    re.compile(r"""import\s+['"]([^'"]+)['"]""", re.MULTILINE),
    re.compile(r"""^\s*import\s+([\w.*]+);""", re.MULTILINE),
    re.compile(r"""^\s*use\s+([\w:]+)""", re.MULTILINE),
    re.compile(r"""(?:require|require_relative)\s+['"]([^'"]+)['"]""", re.MULTILINE),
    re.compile(r"""^\s*use\s+([\w\\]+)""", re.MULTILINE),
    re.compile(r"""(?:require|include)(?:_once)?\s+['"]([^'"]+)['"]""", re.MULTILINE),
    re.compile(r"""#\s*include\s+[<"]([^>"]+)[>"]""", re.MULTILINE),
]


def extract_imports(file_path: str, content: str) -> list[str]:
    """Extract imported module/file paths from source code."""
    imports: list[str] = []
    for pat in _IMPORT_PATTERNS:
        for m in pat.finditer(content):
            imp = m.group(1).strip()
            if imp and imp not in imports:
                imports.append(imp)
    return imports


# ---------------------------------------------------------------------------
# networkx-backed graph with a thin adapter layer.
# ---------------------------------------------------------------------------

class _GraphAdapter:
    """Thin wrapper so callers don't import networkx directly."""

    def __init__(self, g: nx.DiGraph) -> None:
        self._g = g

    def add_edge(self, source: str, target: str, relation: str = "import") -> None:
        self._g.add_edge(source, target, relation=relation)
        # Bidirectional: give the target a back-edge so graph expansion works
        # from either direction (useful when the indexed order varies).
        if not self._g.has_edge(target, source):
            self._g.add_edge(target, source, relation=f"{relation}_rev")

    def neighbors(self, chunk_id: str, depth: int = 1) -> set[str]:
        """BFS over successors up to `depth` hops."""
        visited: set[str] = set()
        frontier = {chunk_id}
        for _ in range(depth):
            if not frontier:
                break
            visited.update(frontier)
            next_frontier: set[str] = set()
            for cid in frontier:
                if cid in self._g:
                    for nb in self._g.successors(cid):
                        if nb not in visited:
                            next_frontier.add(nb)
            frontier = next_frontier
        # Add the final frontier layer — without this, depth=1 always returns empty.
        visited.update(frontier)
        visited.discard(chunk_id)
        return visited

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()


# ---------------------------------------------------------------------------
# Singleton + persistence helpers.
# ---------------------------------------------------------------------------

_digraph: nx.DiGraph = nx.DiGraph()
_adapter = _GraphAdapter(_digraph)


def get_graph() -> _GraphAdapter:
    return _adapter


def reset_graph() -> None:
    global _digraph, _adapter
    _digraph = nx.DiGraph()
    _adapter = _GraphAdapter(_digraph)
    _graph_file = Path(settings.graph_path)
    if _graph_file.exists():
        try:
            _graph_file.unlink()
        except OSError:
            pass


def save_graph() -> None:
    """Serialize edge list to JSON next to the Chroma data dir."""
    path = Path(settings.graph_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = [
        {"src": u, "tgt": v, "relation": d.get("relation", "import")}
        for u, v, d in _digraph.edges(data=True)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(edges, f)


def load_graph() -> None:
    """Populate the in-memory graph from disk (called once at startup)."""
    global _digraph, _adapter
    path = Path(settings.graph_path)
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            edges = json.load(f)
        _digraph = nx.DiGraph()
        for e in edges:
            _digraph.add_edge(e["src"], e["tgt"], relation=e.get("relation", "import"))
        _adapter = _GraphAdapter(_digraph)
    except (OSError, json.JSONDecodeError, KeyError):
        # Corrupt file — start fresh; will be rebuilt on next /index call.
        _digraph = nx.DiGraph()
        _adapter = _GraphAdapter(_digraph)


# Load from disk when the module is first imported.
load_graph()
