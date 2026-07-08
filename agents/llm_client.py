"""Shared Groq LLM client with graceful fallback."""

from __future__ import annotations

from utils.env import ENV_FILE, get_groq_api_key, load_project_env

load_project_env()

GROQ_MODEL = "llama-3.3-70b-versatile"
_client = None
_cached_key: str | None = None


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
        return response.choices[0].message.content or "", None
    except Exception as e:
        return None, f"LLM request failed: {e}"