"""MCP tool functions called directly, with a fake LLM (no network, no API key)."""

import pandas as pd
import pytest

from ada_mcp import tools
from ada_mcp.workspace import Workspace
from agents import sql_agent


@pytest.fixture
def ws():
    w = Workspace.in_memory()
    tools.load_dataset(w, "customer_churn")
    return w


def fake_llm(sql):
    """Return the given SQL for generation calls and a fixed sentence for explanations."""
    calls = []

    def chat(messages, **_kwargs):
        calls.append(messages)
        if "Explain SQL query results" in messages[0]["content"]:
            return "There are 30 customers.", None
        return f"```sql\n{sql}\n```", None

    chat.calls = calls
    return chat


def test_load_sample_by_name_and_with_extension():
    w = Workspace.in_memory()
    out = tools.load_dataset(w, "customer_churn")
    assert out["table"] == "customer_churn"
    assert out["row_count"] == 30
    assert "churn" in out["columns"]
    assert tools.load_dataset(w, "house_prices.csv")["table"] == "house_prices"


def test_load_csv_path_cleans_columns_and_honours_table_name(tmp_path):
    path = tmp_path / "orders.csv"
    pd.DataFrame({"Order ID": [1, 2], "Total $": [9.5, 3.0]}).to_csv(path, index=False)
    w = Workspace.in_memory()
    out = tools.load_dataset(w, str(path), table_name="My Orders")
    assert out["table"] == "my_orders"
    assert out["columns"] == ["order_id", "total"]


def test_load_rejects_unknown_source_and_non_csv(tmp_path):
    w = Workspace.in_memory()
    with pytest.raises(ValueError, match="not a sample"):
        tools.load_dataset(w, "no_such_sample")
    with pytest.raises(ValueError, match="File not found"):
        tools.load_dataset(w, str(tmp_path / "missing.csv"))


def test_list_tables_reports_tables_and_samples(ws):
    out = tools.list_tables(ws)
    assert out["tables"] == [{"name": "customer_churn", "row_count": 30, "column_count": 9}]
    assert set(out["samples"]) >= {"customer_churn", "house_prices"}


def test_describe_table(ws):
    out = tools.describe_table(ws, "customer_churn", preview_rows=3)
    assert out["row_count"] == 30
    assert {c["name"] for c in out["columns"]} >= {"customer_id", "churn"}
    assert len(out["preview"]) == 3
    with pytest.raises(ValueError, match="Unknown table"):
        tools.describe_table(ws, "nope")


def test_run_sql_select_and_truncation(ws):
    out = tools.run_sql(ws, "SELECT customer_id FROM customer_churn ORDER BY customer_id", max_rows=5)
    assert out["status"] == "ok"
    assert out["row_count"] == 30
    assert len(out["rows"]) == 5
    assert out["truncated"] is True
    assert out["rows"][0] == {"customer_id": 1001}


def test_run_sql_allows_cte(ws):
    sql = "WITH c AS (SELECT churn FROM customer_churn) SELECT COUNT(*) AS n FROM c WHERE churn = 'Yes'"
    out = tools.run_sql(ws, sql)
    expected = int((pd.read_csv("sample_data/customer_churn.csv")["churn"] == "Yes").sum())
    assert out["rows"] == [{"n": expected}]


@pytest.mark.parametrize("sql", [
    "DELETE FROM customer_churn",
    "DROP TABLE customer_churn",
    "UPDATE customer_churn SET churn = 'No'",
    "SELECT 1; DROP TABLE customer_churn",
    "ATTACH DATABASE 'x.db' AS x",
])
def test_run_sql_write_needs_approval_and_is_not_executed(ws, sql):
    out = tools.run_sql(ws, sql)
    assert out["status"] == "needs_human_approval"
    assert out["executed"] is False
    assert tools.run_sql(ws, "SELECT COUNT(*) AS n FROM customer_churn")["rows"] == [{"n": 30}]


def test_run_sql_rejects_non_select_without_write_keywords(ws):
    with pytest.raises(ValueError, match="Multiple statements"):
        tools.run_sql(ws, "SELECT 1; SELECT 2")


def test_ask_runs_sql_agent_with_fake_llm(ws, monkeypatch):
    chat = fake_llm("SELECT COUNT(*) AS n FROM customer_churn")
    monkeypatch.setattr(sql_agent, "chat_completion", chat)
    out = tools.ask(ws, "How many customers are there?")
    assert out["status"] == "ok"
    assert out["sql"] == "SELECT COUNT(*) AS n FROM customer_churn"
    assert out["rows"] == [{"n": 30}]
    assert out["explanation"] == "There are 30 customers."
    # schema of the loaded table is what the agent sees
    assert '"customer_churn"' in chat.calls[0][1]["content"]


def test_ask_write_sql_from_model_needs_approval(ws, monkeypatch):
    monkeypatch.setattr(sql_agent, "chat_completion", fake_llm("DELETE FROM customer_churn"))
    out = tools.ask(ws, "Remove all customers")
    assert out["status"] == "needs_human_approval"
    assert tools.run_sql(ws, "SELECT COUNT(*) AS n FROM customer_churn")["rows"] == [{"n": 30}]


def test_ask_needs_a_table_and_a_question(monkeypatch):
    monkeypatch.setattr(sql_agent, "chat_completion", fake_llm("SELECT 1"))
    with pytest.raises(ValueError, match="load_dataset"):
        tools.ask(Workspace.in_memory(), "anything")
    w = Workspace.in_memory()
    tools.load_dataset(w, "customer_churn")
    with pytest.raises(ValueError, match="empty"):
        tools.ask(w, "  ")


def test_ask_surfaces_llm_error(ws, monkeypatch):
    monkeypatch.setattr(sql_agent, "chat_completion", lambda *a, **k: (None, "GROQ_API_KEY is not set."))
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        tools.ask(ws, "How many customers?")


def test_search_history_is_off_by_default(ws, monkeypatch):
    monkeypatch.delenv("ADA_MCP_RAG", raising=False)
    assert tools.search_history(ws, "customers")["status"] == "disabled"


class FakeRAG:
    def __init__(self):
        self.docs = []

    def index_exchange(self, question, answer, source_type, extra=""):
        self.docs.append(f"Q: {question}\nA: {answer}\n{extra}")
        return 1

    def retrieve(self, query, top_k=5):
        return [{"text": d, "source_type": "sql", "distance": 0.1} for d in self.docs][:top_k]


def test_ask_indexes_answers_for_search_history(ws, monkeypatch):
    monkeypatch.setenv("ADA_MCP_RAG", "1")
    ws.rag = FakeRAG()
    monkeypatch.setattr(sql_agent, "chat_completion", fake_llm("SELECT COUNT(*) AS n FROM customer_churn"))
    tools.ask(ws, "How many customers are there?")
    out = tools.search_history(ws, "customers")
    assert out["status"] == "ok"
    assert "How many customers are there?" in out["results"][0]["text"]


def test_file_workspace_reloads_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("ADA_MCP_DB", str(tmp_path / "ada.db"))
    tools.load_dataset(Workspace.from_env(), "house_prices")
    again = Workspace.from_env()
    assert "house_prices" in again.tables
    expected = len(pd.read_csv("sample_data/house_prices.csv"))
    assert tools.run_sql(again, "SELECT COUNT(*) AS n FROM house_prices")["rows"] == [{"n": expected}]
