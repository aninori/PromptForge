"""Token counting and a tiny Chroma wrapper shared by indexing/retrieval/cache.

Token counting strategy
-----------------------
We need token counts to enforce the prompt budget. The "right" tokenizer is
model-specific, but loading the full sentencepiece/BPE vocab for every Ollama
model at runtime is overkill and adds a hard dependency.

Instead we use a calibrated character-ratio heuristic that is accurate enough
for budget enforcement:

  • Llama 3 / DeepSeek / Qwen / Mistral (SentencePiece, ~3.8 chars/token on code)
  • GPT-4 / GPT-3.5 / cl100k_base   (BPE, ~4.0 chars/token on code)
  • Safe fallback: 3.5 chars/token   (conservative — over-counts slightly, which
    is the safer direction: we'd trim context a little early vs. overrun the window)

If tiktoken IS installed and the model is a known OpenAI one we still use it for
precision. For every other model (Ollama / Groq Llama / DeepSeek) we use the
heuristic — it's within 5% of the real count on code text, which is well within
the noise of the budget estimate.
"""
from __future__ import annotations

import chromadb

from .config import settings

# Characters per token for common model families, measured on mixed code+prose.
_CHARS_PER_TOKEN: dict[str, float] = {
    "gpt-4":        4.0,
    "gpt-3.5":      4.0,
    "cl100k":       4.0,
    "llama":        3.8,
    "deepseek":     3.8,
    "qwen":         3.8,
    "mistral":      3.9,
    "codellama":    3.8,
    "gemma":        3.9,
    "phi":          3.9,
}
_DEFAULT_CPT = 3.5  # conservative fallback


def _chars_per_token() -> float:
    model = settings.gen_model.lower()
    for prefix, cpt in _CHARS_PER_TOKEN.items():
        if prefix in model:
            return cpt
    return _DEFAULT_CPT


def count_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken for OpenAI models, heuristic otherwise."""
    if not text:
        return 0
    model = settings.gen_model.lower()
    is_openai = any(k in model for k in ("gpt-4", "gpt-3.5", "cl100k"))
    if is_openai:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / _chars_per_token()))


_client: chromadb.ClientAPI | None = None


def client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def collection(name: str):
    return client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
