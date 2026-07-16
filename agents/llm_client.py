"""Shared Groq LLM client with graceful fallback."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from utils.env import ENV_FILE, get_groq_api_key, load_project_env

load_project_env()

GROQ_MODEL = "llama-3.3-70b-versatile"
_client = None
_cached_key: str | None = None


@dataclass
class LLMUsage:
    """Provider-reported token usage collected within one logical operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    model: str = GROQ_MODEL

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_active_usage: ContextVar[LLMUsage | None] = ContextVar("active_llm_usage", default=None)


@contextmanager
def capture_llm_usage() -> Iterator[LLMUsage]:
    """Collect usage without changing the established chat_completion contract."""

    usage = LLMUsage()
    token = _active_usage.set(usage)
    try:
        yield usage
    finally:
        _active_usage.reset(token)


def _record_usage(response: object) -> None:
    active = _active_usage.get()
    if active is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    active.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
    active.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
    active.calls += 1


def get_groq_client():
    global _client, _cached_key
    api_key = get_groq_api_key()
    if not api_key:
        return None, (
            f"GROQ_API_KEY is not set. Add your key to {ENV_FILE} "
            f"(replace your_key_here with your real Groq API key)."
        )
    if _client is None or _cached_key != api_key:
        try:
            from groq import Groq
            _client = Groq(api_key=api_key)
            _cached_key = api_key
        except Exception as e:
            return None, f"Failed to initialize Groq client: {e}"
    return _client, None


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str | None, str | None]:
    """Return (response_text, error_message)."""
    client, err = get_groq_client()
    if err:
        return None, err
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        _record_usage(response)
        return response.choices[0].message.content or "", None
    except Exception as e:
        return None, f"LLM request failed: {e}"
