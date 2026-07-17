"""Semantic router — intent classification, complexity scoring, agent selection.

Classifiers used by pipeline.py's single RAG pipeline:
  - _classify()          — "codebase" or "off_topic" (intent guard).
  - classify_complexity() — "simple" or "complex" (drives model tier).
  - select_agent()        — score each AgentDefinition's retrieval_bias keywords
                            against the query + top chunks; best match wins when
                            the signal is strong enough (>=2 keyword hits).

Classification calls always use settings.small_model (mistral:7b) — fast binary
decisions that don't need the larger model's reasoning depth.
"""
from __future__ import annotations

from . import ollama_client
from .agent_registry import AGENTS, AgentDefinition
from .config import settings
from .schemas import RetrievedChunk

# ---------------------------------------------------------------------------
# Classifier prompts
# ---------------------------------------------------------------------------

_SYS_CLASSIFY = (
    "You are a query classifier for a code-focused retrieval assistant. "
    "Classify the user's question into exactly one of two categories:\n"
    "- codebase: The question is about code, software, programming, debugging, "
    "architecture, APIs, frameworks, files, or anything answerable from a code repository.\n"
    "- off_topic: The question is completely unrelated to code or software "
    "(e.g. general trivia, weather, cooking, history, sports).\n"
    "Reply with exactly one word: codebase or off_topic. No explanation."
)

_SYS_COMPLEXITY = (
    "Classify this developer query as 'simple' or 'complex'.\n"
    "- simple: one direct question about one error, one function, or one concept\n"
    "- complex: multi-step, architectural, comparative, or requires understanding "
    "multiple files, systems, or workflows\n"
    "Reply with exactly one word: simple or complex. No explanation."
)

OFF_TOPIC_REPLY = (
    "I'm specialized for your indexed codebase and can only answer questions about it. "
    "Try asking about your code — debugging, architecture, documentation, security, "
    "or how specific components work."
)

# Minimum keyword hits for an agent to be selected over the default pipeline.
_AGENT_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Classifiers — always use small_model for speed
# ---------------------------------------------------------------------------

def _classify(query: str) -> str:
    """Returns 'codebase' or 'off_topic'. Always uses settings.small_model."""
    raw = ollama_client.chat(
        query.strip(), model=settings.small_model, system_prompt=_SYS_CLASSIFY
    )
    first_token = (raw or "").strip().lower().split()[0] if raw else ""
    return "off_topic" if first_token == "off_topic" else "codebase"


def classify_complexity(query: str) -> str:
    """Returns 'simple' or 'complex'. Always uses settings.small_model."""
    raw = ollama_client.chat(
        query.strip(), model=settings.small_model, system_prompt=_SYS_COMPLEXITY
    )
    first_token = (raw or "").strip().lower().split()[0] if raw else ""
    return "complex" if first_token == "complex" else "simple"


# ---------------------------------------------------------------------------
# Agent keyword scorer
# ---------------------------------------------------------------------------

def _agent_bias_score(query_lower: str, snippet_blob: str, agent: AgentDefinition) -> int:
    """Count how many of the agent's bias keywords appear in the query + top chunk text."""
    combined = query_lower + " " + snippet_blob
    return sum(1 for kw in agent.retrieval_bias if kw in combined)


def select_agent(query: str, chunks: list[RetrievedChunk]) -> str | None:
    """Return the agent_id whose bias keywords best match this query+chunks.

    Returns None when no agent clears the threshold — the default pipeline runs.
    """
    q = query.lower()
    snippet_blob = " ".join(c.snippet for c in chunks[:4]).lower()
    scores = {
        aid: _agent_bias_score(q, snippet_blob, agent)
        for aid, agent in AGENTS.items()
    }
    best_id, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_id if best_score >= _AGENT_THRESHOLD else None
