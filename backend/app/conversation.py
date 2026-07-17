"""Session-scoped conversation memory.

Keeps the last N turns (query + answer) per session_id in memory so that
follow-up questions like "now add tests for it" have the context they need.

The context prefix is injected into the assembled prompt before the code
chunks, so the generation model sees prior turns without the optimizer
needing to know about them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

_MAX_TURNS = 5
_ANSWER_PREVIEW_CHARS = 300


@dataclass
class _Turn:
    query: str
    answer: str


class ConversationMemory:
    _sessions: dict[str, deque[_Turn]] = {}

    @classmethod
    def add_turn(cls, session_id: str, query: str, answer: str) -> None:
        if session_id not in cls._sessions:
            cls._sessions[session_id] = deque(maxlen=_MAX_TURNS)
        cls._sessions[session_id].append(_Turn(query=query, answer=answer))

    @classmethod
    def get_context_prefix(cls, session_id: str) -> str:
        """Return a formatted history block to prepend to the prompt task.

        Returns empty string when there is no prior history (first turn or
        unknown session_id), so callers don't need to check.
        """
        turns = cls._sessions.get(session_id)
        if not turns:
            return ""
        lines = ["## Conversation history (most recent last)"]
        for i, t in enumerate(turns, 1):
            preview = t.answer[:_ANSWER_PREVIEW_CHARS].replace("\n", " ")
            if len(t.answer) > _ANSWER_PREVIEW_CHARS:
                preview += "…"
            lines.append(f"[{i}] User: {t.query}")
            lines.append(f"     Assistant: {preview}")
        lines.append("")
        return "\n".join(lines) + "\n\n"
