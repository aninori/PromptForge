"""Prompt builder — assemble the final prompt while respecting the token budget.

Chunks are added highest-relevancy first and dropped once the budget is hit, so
the model never receives more context than it needs. This is the step that
actually controls token spend.
"""
from __future__ import annotations

from .config import settings
from .schemas import RetrievedChunk
from .store import count_tokens

_HEADER = "# Task\n{task}\n\n# Relevant context (top {n} of {total} chunks)\n"
_FOOTER = (
    "\n# Instructions\n"
    "Answer using only the context above. "
    "Be specific and cite file paths when relevant. "
    "If the context is insufficient, say so clearly rather than guessing."
)

FORGE_SYSTEM_PROMPT = (
    "You are a prompt engineer. Your ONLY output is a structured prompt that a "
    "developer pastes into a coding agent. Use EXACTLY the six section headers "
    "given in the user message (=== Task === … === Expected Output ===) and write "
    "nothing outside them. Never answer, solve, or explain the task; never write "
    "prose, advice, or numbered tips. Ground every detail in the supplied codebase "
    "context — do not invent files."
)

_FORGE_HEADER = (
    "# PROMPT FORGE — READ THIS FIRST\n"
    "You are a prompt engineer. Your ONLY job is to output a structured prompt "
    "that a developer can paste into GitHub Copilot or Claude. "
    "Do NOT solve the task. Do NOT explain anything. Do NOT write prose.\n"
    "Output the prompt using EXACTLY these six section headers and no other text:\n\n"
    "=== Task ===\n"
    "=== Tech Stack ===\n"
    "=== Relevant Files ===\n"
    "=== Constraints ===\n"
    "=== Do NOT ===\n"
    "=== Expected Output ===\n\n"
    "Rules:\n"
    "- Derive ALL content from the codebase context below — do not invent files or code.\n"
    "- Quote actual file paths and code verbatim from the context.\n"
    "- Task section: rewrite the user request as a precise engineering instruction.\n"
    "- Relevant Files: list each path, one-line role annotation, then the key snippet.\n"
    "- Do NOT section: guardrails specific to this codebase (not generic advice).\n"
    "- Expected Output: describe which files to edit and what done looks like.\n\n"
    "# Codebase context\n"
)


def build(task: str, chunks: list[RetrievedChunk], total_chunks: int) -> tuple[str, list[RetrievedChunk]]:
    header = _HEADER.format(task=task, n=len(chunks), total=total_chunks)
    budget = settings.token_budget - count_tokens(header) - count_tokens(_FOOTER)

    kept: list[RetrievedChunk] = []
    body_parts: list[str] = []
    used = 0
    for c in chunks:
        block = f"// {c.path} ({c.lines}) — relevancy {int(c.score * 100)}%\n{c.snippet}"
        t = count_tokens(block)
        if used + t > budget and kept:
            break
        body_parts.append(block)
        used += t
        kept.append(c)

    prompt = header + "\n\n".join(body_parts) + _FOOTER
    return prompt, kept


_FORGE_TASK_PREFIX = "\n\n# User's task (what the forged prompt must accomplish)\n"

# Recency anchor: the LAST thing the model reads is the empty skeleton it must
# fill. On smaller local models (e.g. deepseek-coder-v2:16b) the top-of-prompt
# instruction alone is not enough — the model answers the task it just read.
# Ending with the template to complete forces it back into prompt-engineer mode.
_FORGE_TEMPLATE = (
    "\n\n# OUTPUT NOW\n"
    "Reply with ONLY the following template, filling each section from the "
    "codebase context above. No preamble, no prose, no code fences around the "
    "whole reply.\n\n"
    "=== Task ===\n"
    "<rewrite the user's task as one precise engineering instruction>\n\n"
    "=== Tech Stack ===\n"
    "<languages, frameworks, and tools evident in the context>\n\n"
    "=== Relevant Files ===\n"
    "<for each file: path — one-line role, then the key snippet>\n\n"
    "=== Constraints ===\n"
    "<guardrails specific to THIS codebase>\n\n"
    "=== Do NOT ===\n"
    "<things the agent must avoid here>\n\n"
    "=== Expected Output ===\n"
    "<which files to edit and what 'done' looks like>\n"
)

def build_forge(task: str, chunks: list[RetrievedChunk], total_chunks: int) -> tuple[str, list[RetrievedChunk]]:
    """Forge mode: instruction-first layout so the model is in prompt-engineer mode
    before it reads any context, plus a trailing skeleton (recency anchor) so the
    last thing it sees is the template to fill — not the task to answer."""
    task_block = _FORGE_TASK_PREFIX + task
    budget = (
        settings.token_budget
        - count_tokens(_FORGE_HEADER)
        - count_tokens(task_block)
        - count_tokens(_FORGE_TEMPLATE)
    )

    kept: list[RetrievedChunk] = []
    body_parts: list[str] = []
    used = 0
    for c in chunks:
        block = f"// {c.path} ({c.lines}) — relevancy {int(c.score * 100)}%\n{c.snippet}"
        t = count_tokens(block)
        if used + t > budget and kept:
            break
        body_parts.append(block)
        used += t
        kept.append(c)

    context_body = "\n\n".join(body_parts)
    prompt = _FORGE_HEADER + context_body + task_block + _FORGE_TEMPLATE
    return prompt, kept
