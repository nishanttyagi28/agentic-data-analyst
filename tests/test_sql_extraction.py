from types import SimpleNamespace

from agents import llm_client
from agents.sql_agent import extract_sql_from_response


def test_plain_fenced_sql():
    assert extract_sql_from_response("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_unlabelled_and_sqlite_fences():
    assert extract_sql_from_response("```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql_from_response("```sqlite\nSELECT 3\n```") == "SELECT 3"


def test_inline_reasoning_is_dropped():
    text = "<think>maybe SELECT * FROM t? no.</think>\n```sql\nSELECT COUNT(*) FROM user_data\n```"
    assert extract_sql_from_response(text) == "SELECT COUNT(*) FROM user_data"


def test_last_sql_block_wins_over_drafts_and_other_fences():
    text = (
        "First try:\n```sql\nSELECT age FROM user_data\n```\n"
        "Better:\n```sql\nSELECT AVG(age) FROM user_data;\n```\n"
        "Columns used:\n```\nage\n```"
    )
    assert extract_sql_from_response(text) == "SELECT AVG(age) FROM user_data"


def test_unfenced_cte_after_prose():
    text = "Here you go:\nWITH t AS (SELECT 1 AS x) SELECT x FROM t;"
    assert extract_sql_from_response(text) == "WITH t AS (SELECT 1 AS x) SELECT x FROM t"


def test_413_over_token_budget_is_retried():
    class TooLarge(Exception):
        status_code = 413

    err = TooLarge("Request too large ... Please try again in 1.5s")
    assert llm_client._rate_limit_delay(err, 0) == 1.5
    assert llm_client._rate_limit_delay(ValueError("bad request"), 0) is None


def test_model_default_and_env_override(monkeypatch):
    import importlib

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert importlib.reload(llm_client).GROQ_MODEL == "openai/gpt-oss-120b"
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert importlib.reload(llm_client).GROQ_MODEL == "qwen/qwen3.8-27b"
    monkeypatch.delenv("GROQ_MODEL")
    importlib.reload(llm_client)


def test_none_content_becomes_empty_string(monkeypatch):
    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
                usage=None,
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(llm_client, "get_groq_client", lambda: (fake, None))
    assert llm_client.chat_completion([]) == ("", None)
