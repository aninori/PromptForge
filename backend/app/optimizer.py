"""Query-side optimization — runs on the local model before retrieval.

Two techniques:
  1. Rewrite  (optimize_query): raw ask → one precise, self-contained instruction.
  2. Expand   (expand_query):   rewritten instruction → 3 short search queries.

Chaining contract:
  optimized  = optimize_query(raw)
  expansions = expand_query(optimized)   # takes the REWRITE output, not raw

Both prompts are format strings keyed on {q} so callers can inspect or log the
exact text sent to the model:
  ollama_client.chat(_REWRITE.format(q=raw), model=None)
  ollama_client.chat(_EXPAND.format(q=optimized), model=None)
"""
from __future__ import annotations

from . import ollama_client

_REWRITE = (
    "You rewrite a developer's rough request into one precise, self-contained "
    "engineering instruction. Output only the rewritten instruction — no explanation.\n\n"
    "{q}"
)

_EXPAND = (
    "Generate 3 short search queries that would help find the code relevant to this task. "
    "Return one query per line with no numbering, bullets, or extra commentary.\n\n"
    "{q}"
)


def optimize_query(raw: str, model: str | None = None) -> str:
    out = ollama_client.chat(_REWRITE.format(q=raw.strip()), model=model)
    return out.strip() if out else raw.strip()


def expand_query(optimized: str, model: str | None = None) -> list[str]:
    """Takes the REWRITE output (not the raw query) and returns 3 search queries."""
    out = ollama_client.chat(_EXPAND.format(q=optimized.strip()), model=model)
    lines = [ln.strip("-• \t") for ln in out.splitlines() if ln.strip()]
    return lines[:3]
