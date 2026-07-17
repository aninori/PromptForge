"""AST-aware chunker — emits whole functions/classes instead of token windows.

Routing by file extension:
  .py               → stdlib ast (zero extra deps, perfect Python support)
  .js .jsx .ts .tsx → tree-sitter via tree_sitter_languages (optional)
  everything else   → line-window fallback (original behavior)

If tree_sitter_languages is not installed, JS/TS files silently fall back to
the line-window splitter — behavior is identical to Phase 2 for those files.
"""
from __future__ import annotations

import ast

from .store import count_tokens

# ---------------------------------------------------------------------------
# Line-window splitter (original Phase 2 logic, used as fallback everywhere)
# ---------------------------------------------------------------------------

def _line_window(text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    start = 1
    tok = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        buf.append(line)
        tok += count_tokens(line)
        if tok >= chunk_size:
            chunks.append(("\n".join(buf), f"L{start}-{i + 1}"))
            if overlap > 0:
                overlap_buf: list[str] = []
                overlap_tok = 0
                for prev in reversed(buf):
                    lt = count_tokens(prev)
                    if overlap_buf and overlap_tok + lt > overlap:
                        break
                    overlap_buf.append(prev)
                    overlap_tok += lt
                    if overlap_tok >= overlap:
                        break
                buf = list(reversed(overlap_buf))
                start = (i + 1) - len(buf) + 1
                tok = sum(count_tokens(ln) for ln in buf)
            else:
                buf, tok, start = [], 0, i + 2
        i += 1
    if buf:
        chunks.append(("\n".join(buf), f"L{start}-{len(lines)}"))
    return chunks


# ---------------------------------------------------------------------------
# Python chunker — stdlib ast, no extra dependencies
# ---------------------------------------------------------------------------

def _chunk_python(text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _line_window(text, chunk_size, overlap)

    top_nodes = [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not top_nodes:
        return _line_window(text, chunk_size, overlap)

    chunks: list[tuple[str, str]] = []
    covered_end = 0  # 0-indexed, exclusive

    for node in top_nodes:
        start0 = node.lineno - 1   # 0-indexed inclusive
        end0 = node.end_lineno     # 0-indexed exclusive (== 1-indexed inclusive)

        # Emit any module-level code that sits before this definition.
        if start0 > covered_end:
            gap = "\n".join(lines[covered_end:start0])
            if gap.strip():
                for sub, _ in _line_window(gap, chunk_size, overlap):
                    chunks.append((sub, f"L{covered_end + 1}-{start0}"))

        snippet = "\n".join(lines[start0:end0])
        if count_tokens(snippet) > chunk_size * 1.5:
            # Node is oversized — split it but keep the line prefix for context.
            for sub, sub_r in _line_window(snippet, chunk_size, overlap):
                chunks.append((sub, f"L{node.lineno}:{sub_r}"))
        else:
            chunks.append((snippet, f"L{node.lineno}-{node.end_lineno}"))

        covered_end = end0

    # Emit any trailing module-level code after the last definition.
    if covered_end < len(lines):
        tail = "\n".join(lines[covered_end:])
        if tail.strip():
            for sub, _ in _line_window(tail, chunk_size, overlap):
                chunks.append((sub, f"L{covered_end + 1}-{len(lines)}"))

    return chunks if chunks else _line_window(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# JS / TS chunker — tree-sitter (optional)
# ---------------------------------------------------------------------------

_TS_BOUNDARY_TYPES = {
    "function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
    "class_declaration",
    "class_expression",
    "export_statement",
}


def _chunk_treesitter(lang: str, text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except ImportError:
        return _line_window(text, chunk_size, overlap)

    try:
        parser = get_parser(lang)
        tree = parser.parse(bytes(text, "utf-8"))
    except Exception:
        return _line_window(text, chunk_size, overlap)

    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    visited: set[tuple[int, int]] = set()

    def walk(node) -> None:
        if node.type in _TS_BOUNDARY_TYPES:
            s = node.start_point[0]       # 0-indexed
            e = node.end_point[0] + 1     # exclusive
            if (s, e) not in visited:
                visited.add((s, e))
                snippet = "\n".join(lines[s:e])
                if count_tokens(snippet) > chunk_size * 1.5:
                    for sub, sub_r in _line_window(snippet, chunk_size, overlap):
                        chunks.append((sub, f"L{s + 1}:{sub_r}"))
                else:
                    chunks.append((snippet, f"L{s + 1}-{e}"))
        else:
            for child in node.children:
                walk(child)

    walk(tree.root_node)
    return chunks if chunks else _line_window(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Public routing entry point
# ---------------------------------------------------------------------------

_EXT_TO_TS_LANG: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


def chunk_file(ext: str, text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    """Route to the best available chunker for the given file extension."""
    if ext == ".py":
        return _chunk_python(text, chunk_size, overlap)
    lang = _EXT_TO_TS_LANG.get(ext)
    if lang:
        return _chunk_treesitter(lang, text, chunk_size, overlap)
    return _line_window(text, chunk_size, overlap)
