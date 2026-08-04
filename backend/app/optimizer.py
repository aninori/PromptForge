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

# "Be precise" alone makes this fabricate. Asked "why are the videos not
# working?", a model with no way to say "cause unknown" invents a plausible cause
# and the downstream template renders the guess as a requirement. Precision is
# right for a known change ("add pagination"); for a diagnosis the honest
# rewrite keeps the symptom and lets the downstream agent do the diagnosing.
_REWRITE = (
    "You rewrite a developer's rough request into one precise, self-contained "
    "engineering instruction. Output only the rewritten instruction — no explanation.\n\n"
    "If the user is reporting a problem and has not identified its cause, keep "
    "the task as a symptom. Do not guess or assert a cause.\n"
    'Bad:  "Fix the video source URL format"\n'
    'Good: "Videos do not play in VideoPlayer. Investigate the cause and fix it."\n'
    "Only name a cause if the user named it.\n"
    "When the user HAS said what they want done, stay precise and specific — "
    "this rule applies to unexplained problems, not to clear change requests.\n\n"
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
