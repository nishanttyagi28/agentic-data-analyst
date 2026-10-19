"""The MCP server through an in-process client session, same as a real client sees it."""

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from agents import sql_agent
from ada_mcp.server import build_server
from ada_mcp.workspace import Workspace

EXPECTED_TOOLS = {"load_dataset", "list_tables", "describe_table", "run_sql", "ask", "search_history"}


def call(server, steps):
    """Run (tool, args) steps in one client session and return the results."""

    async def go():
        async with create_connected_server_and_client_session(server) as client:
            return [await client.call_tool(name, args) for name, args in steps]

    return anyio.run(go)


def test_server_lists_all_tools():
    server = build_server(Workspace.in_memory())

    async def go():
        async with create_connected_server_and_client_session(server) as client:
            return await client.list_tools()

    listed = anyio.run(go)
    names = {t.name for t in listed.tools}
    assert names == EXPECTED_TOOLS
    by_name = {t.name: t for t in listed.tools}
    assert by_name["run_sql"].annotations.readOnlyHint is True
    assert by_name["load_dataset"].annotations.readOnlyHint is False


def test_load_then_query_over_the_protocol():
    server = build_server(Workspace.in_memory())
    load, tables, desc, rows = call(server, [
        ("load_dataset", {"source": "customer_churn"}),
        ("list_tables", {}),
        ("describe_table", {"table": "customer_churn", "preview_rows": 2}),
        ("run_sql", {"sql": "SELECT COUNT(*) AS n FROM customer_churn"}),
    ])
    assert not load.isError
    assert load.structuredContent["row_count"] == 30
    assert tables.structuredContent["tables"][0]["name"] == "customer_churn"
    assert len(desc.structuredContent["preview"]) == 2
    assert rows.structuredContent["rows"] == [{"n": 30}]


def test_write_sql_over_the_protocol_needs_approval():
    server = build_server(Workspace.in_memory())
    _, drop, count = call(server, [
        ("load_dataset", {"source": "customer_churn"}),
        ("run_sql", {"sql": "DROP TABLE customer_churn"}),
        ("run_sql", {"sql": "SELECT COUNT(*) AS n FROM customer_churn"}),
    ])
    assert drop.structuredContent["status"] == "needs_human_approval"
    assert count.structuredContent["rows"] == [{"n": 30}]


def test_bad_input_comes_back_as_tool_error():
    server = build_server(Workspace.in_memory())
    (res,) = call(server, [("describe_table", {"table": "missing"})])
    assert res.isError
    assert "Unknown table" in res.content[0].text


def test_ask_over_the_protocol_with_fake_llm(monkeypatch):
    def chat(messages, **_kwargs):
        if "Explain SQL query results" in messages[0]["content"]:
            return "Count of all houses.", None
        return "```sql\nSELECT COUNT(*) AS n FROM house_prices\n```", None

    monkeypatch.setattr(sql_agent, "chat_completion", chat)
    server = build_server(Workspace.in_memory())
    _, answer = call(server, [
        ("load_dataset", {"source": "house_prices"}),
        ("ask", {"question": "How many houses are there?"}),
    ])
    assert not answer.isError
    assert answer.structuredContent["sql"] == "SELECT COUNT(*) AS n FROM house_prices"
    assert answer.structuredContent["explanation"] == "Count of all houses."
