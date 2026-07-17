"""Thin client over Ollama (and optionally Groq) for embeddings + chat.

Ollama now uses /api/chat (messages array) instead of /api/generate so that
system prompts are properly separated from user content — the model sees them
in their intended roles rather than as a flat concatenated string.

Groq already used the messages format; system_prompt is prepended there too.
"""
from __future__ import annotations

import asyncio
import json
from typing import Generator

import httpx
import requests

from .config import settings

_EMBED_CONCURRENCY = 16  # max simultaneous embed requests to Ollama


def embed(text: str) -> list[float]:
    """Return an embedding vector for a single piece of text."""
    try:
        r = requests.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except requests.RequestException as e:
        raise RuntimeError(
            f"Embedding failed. Is Ollama running at {settings.ollama_url} "
            f"with `{settings.embed_model}` pulled? ({e})"
        ) from e


async def _embed_batch_async(texts: list[str]) -> list[list[float]]:
    url = settings.ollama_url
    model = settings.embed_model
    sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def _one(client: httpx.AsyncClient, text: str) -> list[float]:
        async with sem:
            r = await client.post(
                f"{url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]

    async with httpx.AsyncClient(timeout=60) as client:
        return list(await asyncio.gather(*[_one(client, t) for t in texts]))


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts concurrently. Faster than calling embed() in a loop."""
    if not texts:
        return []
    return asyncio.run(_embed_batch_async(texts))


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    system_prompt: str | None = None,
) -> str:
    """Generate a completion. Routes to Groq when provider=groq, else Ollama."""
    model = model or settings.gen_model

    if settings.provider == "groq":
        return _groq_chat(prompt, temperature, system_prompt=system_prompt)

    messages = _build_messages(prompt, system_prompt)
    try:
        r = requests.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Generation failed. Is Ollama running with `{model}` pulled? ({e})"
        ) from e


def chat_stream(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    system_prompt: str | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from the model (SSE-style).

    Yields plain token strings. Caller wraps in FastAPI StreamingResponse
    with media_type="text/event-stream".
    """
    model = model or settings.gen_model

    if settings.provider == "groq":
        yield from _groq_chat_stream(prompt, temperature, system_prompt=system_prompt)
        return

    messages = _build_messages(prompt, system_prompt)
    try:
        r = requests.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done", False):
                break
    except requests.RequestException as e:
        raise RuntimeError(
            f"Streaming generation failed. Is Ollama running with `{model}` pulled? ({e})"
        ) from e


def _groq_chat(
    prompt: str,
    temperature: float,
    system_prompt: str | None = None,
) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    messages = _build_messages(prompt, system_prompt)
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _groq_chat_stream(
    prompt: str,
    temperature: float,
    system_prompt: str | None = None,
) -> Generator[str, None, None]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    messages = _build_messages(prompt, system_prompt)
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Accept": "text/event-stream",
        },
        json={
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        choices = data.get("choices", [])
        if choices:
            token = choices[0].get("delta", {}).get("content", "")
            if token:
                yield token
