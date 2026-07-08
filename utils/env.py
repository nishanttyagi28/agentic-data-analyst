"""Load .env from the project root regardless of working directory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

PLACEHOLDER_KEYS = {
    "",
    "your_key_here",
    "your_groq_api_key_here",
    "changeme",
    "replace_me",
}


def _read_key_from_env_file() -> str:
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GROQ_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key.lower() not in PLACEHOLDER_KEYS:
                return key
    return ""


def load_project_env() -> Path:
    """Load .env from project root into os.environ."""
    load_dotenv(ENV_FILE, override=True)
    return ENV_FILE


def _read_streamlit_secret() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def _valid_key(key: str) -> bool:
    return bool(key) and key.lower() not in PLACEHOLDER_KEYS


def get_groq_api_key() -> str:
    """Return GROQ_API_KEY — shell env, Streamlit secrets, then .env file."""
    shell_key = os.getenv("GROQ_API_KEY", "").strip()
    if _valid_key(shell_key):
        return shell_key
    secret_key = _read_streamlit_secret()
    if _valid_key(secret_key):
        return secret_key
    return _read_key_from_env_file()