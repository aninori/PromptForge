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
    "context — do not invent files, and do not invent constraints, requirements, "
    'or causes. If the context does not support a section, write "None specified."'
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
    "- Task section: rewrite the user request as a precise engineering instruction. "
    "If the user reported a problem without naming its cause, state the symptom "
    "and ask for investigation — never assert a cause the user did not give.\n"
    "- Relevant Files: list each path with its line range, then a terse note on what "
    "it does. Do NOT paste the code itself — the developer's agent already has these "
    "files open, so a copied snippet only adds tokens and risks contradicting the "
    "real file. Point at the code; don't reproduce it.\n"
    "  Keep each note under 10 words and start it with a verb. Never pad with "
    '"This file matters because it contains the code that…" — write '
    '"uploads videos and refreshes the list", not a sentence about the file.\n'
    "- Do NOT section: guardrails specific to this codebase (not generic advice).\n"
    "- Tech Stack, Constraints and Do NOT: only state something you can point to in "
    'the user\'s message or the context below. If you cannot, write "None specified." '
    "Never invent a technical requirement — an empty section is safe, a fabricated "
    "one makes the agent break working code.\n"
    "- Never name a language, framework, library, tool, or pattern unless that name "
    "literally appears in the context above. Do not infer it from what projects "
    "like this usually use: if the context shows React but never mentions Redux, "
    "Redux does not exist here. Listing a library the project does not use makes "
    "the agent write code against an API that isn't installed.\n"
    "- Expected Output: describe which files to edit and what done looks like.\n\n"
    "# Codebase context\n"
)


def _fit_chunks(chunks: list[RetrievedChunk], budget: int) -> tuple[str, list[RetrievedChunk]]:
    """Greedily keep highest-relevancy-first chunks until the token budget is hit.
    Shared by build(), build_forge(), and refine_forge()."""
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
    return "\n\n".join(body_parts), kept


def build(task: str, chunks: list[RetrievedChunk], total_chunks: int) -> tuple[str, list[RetrievedChunk]]:
    header = _HEADER.format(task=task, n=len(chunks), total=total_chunks)
    budget = settings.token_budget - count_tokens(header) - count_tokens(_FOOTER)

    body, kept = _fit_chunks(chunks, budget)
    prompt = header + body + _FOOTER
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
    "<rewrite the user's task as one precise engineering instruction; if they "
    "reported a problem without naming its cause, state the symptom and ask for "
    "investigation instead of asserting a cause>\n\n"
    "=== Tech Stack ===\n"
    '<languages, frameworks, and tools visible in the context above, or "None specified">\n\n'
    "=== Relevant Files ===\n"
    "<for each file, exactly two lines and no code, e.g.:\n"
    "app/components/VideoPlayer.tsx (L91-190)\n"
    "  - renders the video element and playback controls\n"
    "Keep the note under 10 words, verb-first, no preamble.>\n\n"
    "=== Constraints ===\n"
    '<guardrails you can point to in the user\'s message or the context, or "None specified">\n\n'
    "=== Do NOT ===\n"
    '<things the agent must avoid here, evidenced by the context, or "None specified">\n\n'
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

    context_body, kept = _fit_chunks(chunks, budget)
    prompt = _FORGE_HEADER + context_body + task_block + _FORGE_TEMPLATE
    return prompt, kept


_REFINE_HEADER = (
    "# PROMPT FORGE — REFINE\n"
    "You already produced a structured prompt for a developer (shown below as "
    "the previous version). Revise it according to the user's refinement note. "
    "Keep EXACTLY the six section headers (=== Task === … === Expected Output ===) "
    "and write nothing outside them. Ground every detail in the supplied codebase "
    "context — do not invent files, and do not invent constraints, requirements, "
    'or causes. If the context does not support a section, write "None specified."\n\n'
    "# Codebase context\n"
)
_REFINE_PREVIOUS_PREFIX = "\n\n# Previous forged prompt (revise this)\n"
_REFINE_NOTE_PREFIX = "\n\n# User's refinement note\n"


def refine_forge(
    task_text: str,
    previous_text: str,
    note: str,
    chunks: list[RetrievedChunk],
    total_chunks: int,
) -> tuple[str, list[RetrievedChunk]]:
    """Rebuild a forge brief from the SAME retrieved chunks plus the previous
    brief and a free-text note. Never re-retrieves — chunks/total_chunks come
    from the stored TurnRecord, not from retrieval.retrieve()."""
    task_block = _FORGE_TASK_PREFIX + task_text
    previous_block = _REFINE_PREVIOUS_PREFIX + previous_text
    note_block = _REFINE_NOTE_PREFIX + note
    fixed_cost = count_tokens(
        _REFINE_HEADER + task_block + previous_block + note_block + _FORGE_TEMPLATE
    )
    budget = settings.token_budget - fixed_cost

    body, kept = _fit_chunks(chunks, budget)
    prompt = _REFINE_HEADER + body + task_block + previous_block + note_block + _FORGE_TEMPLATE
    return prompt, kept
