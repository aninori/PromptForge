"""Semantic router — intent classification, complexity scoring, request-type routing.

Classifiers used by pipeline.py's single RAG pipeline:
  - _classify()            — "codebase" or "off_topic" (intent guard).
  - classify_complexity()  — "simple" or "complex" (drives model tier).
  - classify_request_type() — "question" or "task" (drives answer vs. brief when
    the caller didn't force a mode explicitly).

Complexity and request-type classification use settings.small_model — fast binary
decisions where a wrong call degrades quality quietly. The intent guard uses the
larger settings.guard_model instead: its errors are user-visible and blocking, so
accuracy outranks speed there.
"""
from __future__ import annotations

from . import ollama_client
from .config import settings

# ---------------------------------------------------------------------------
# Classifier prompts
# ---------------------------------------------------------------------------

# Deliberately biased toward "codebase". Judging a message in a vacuum made this
# reject real work: "why are the videos not working?" reads as generic tech
# support unless the classifier is told the asker is a developer describing their
# own app. A guardrail that blocks legitimate questions is far worse than one
# that occasionally lets a trivia question through — the cost of a leak is one
# wasted generation, the cost of a false reject is the product refusing to work.
_SYS_CLASSIFY = (
    "You are the guardrail for an assistant that answers questions about the "
    "developer's own software project. Every message comes from a developer "
    "working on that project. Vague words like 'it', 'the videos', 'login', or "
    "'the page' refer to features of their application.\n"
    "Reply off_topic ONLY when the message clearly has nothing to do with "
    "software or their project - general trivia, weather, cooking, history, "
    "sports, celebrities.\n"
    "Everything else is codebase, including vague bug reports, questions about "
    "features not working, and requests to build or change something.\n"
    "Reply with exactly one word: codebase or off_topic. No explanation."
)

_SYS_COMPLEXITY = (
    "Classify this developer query as 'simple' or 'complex'.\n"
    "- simple: one direct question about one error, one function, or one concept\n"
    "- complex: multi-step, architectural, comparative, or requires understanding "
    "multiple files, systems, or workflows\n"
    "Reply with exactly one word: simple or complex. No explanation."
)

_SYS_REQUEST_TYPE = (
    "Classify the developer's message as exactly one of two categories:\n"
    "- question: they want information, an explanation, or an answer about the codebase.\n"
    "- task: they want something built, changed, fixed, generated, or otherwise "
    "produced — actionable engineering work.\n"
    "When genuinely ambiguous, prefer task.\n"
    "Reply with exactly one word: question or task. No explanation."
)

OFF_TOPIC_REPLY = (
    "I'm specialized for your indexed codebase and can only answer questions about it. "
    "Try asking about your code — debugging, architecture, documentation, security, "
    "or how specific components work."
)


# ---------------------------------------------------------------------------
# Classifiers — small_model for speed, except the intent guard (see below)
# ---------------------------------------------------------------------------

def _classify(query: str) -> str:
    """Returns 'codebase' or 'off_topic'. Uses settings.guard_model — see the
    config comment: this classifier's errors block real work, so it doesn't
    share the small_model the other two use."""
    raw = ollama_client.chat(
        query.strip(), model=settings.guard_model, system_prompt=_SYS_CLASSIFY
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


def classify_request_type(query: str) -> str:
    """Returns 'question' or 'task'. Always uses settings.small_model.

    Defaults to 'task' on an unparseable reply — the opposite convention from
    the other two classifiers (which default to their majority case): briefs
    are this product's main purpose, and a wrong task-guess is cheap to correct
    via the review gate's "just explain it" escape hatch, per product spec.
    """
    raw = ollama_client.chat(
        query.strip(), model=settings.small_model, system_prompt=_SYS_REQUEST_TYPE
    )
    first_token = (raw or "").strip().lower().split()[0] if raw else ""
    return "question" if first_token == "question" else "task"
