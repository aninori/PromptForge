"""Generation — final answer from the configured model.

Supports both blocking and streaming (SSE) responses via Ollama and Groq.
An optional system_prompt is forwarded to the model's system role slot.
"""
from __future__ import annotations

from typing import Generator

from . import ollama_client


def generate(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    return ollama_client.chat(prompt, model=model, system_prompt=system_prompt)


def generate_stream(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
) -> Generator[str, None, None]:
    """Yield tokens one by one as the model generates them."""
    yield from ollama_client.chat_stream(prompt, model=model, system_prompt=system_prompt)
