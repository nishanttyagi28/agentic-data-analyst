"""FastMCP server over stdio. Run with `ada-mcp` or `python -m ada_mcp`."""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ada_mcp import tools
from ada_mcp.workspace import Workspace

INSTRUCTIONS = """Tools for analysing CSV data with SQL.
Start with load_dataset (a sample name or a .csv path), then list_tables / describe_table.
Use run_sql for read-only SELECT/WITH queries, or ask for a plain-English question.
Write or schema-changing SQL is never executed; it comes back as needs_human_approval."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def build_server(ws: Workspace | None = None) -> FastMCP:
    ws = ws or Workspace.from_env()
    mcp = FastMCP("agentic-data-analyst", instructions=INSTRUCTIONS)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    def load_dataset(source: str, table_name: str | None = None) -> dict[str, Any]:
        """Load a CSV into a SQLite table.

        source: a bundled sample name (see list_tables -> samples) or a path to a .csv file.
        table_name: optional; defaults to the file name. Existing tables are never overwritten
        with different data, a suffix is added instead.
        """
        return tools.load_dataset(ws, source, table_name)

    @mcp.tool(annotations=READ_ONLY)
    def list_tables() -> dict[str, Any]:
        """List loaded tables with row/column counts, likely join keys, and available samples."""
        return tools.list_tables(ws)

    @mcp.tool(annotations=READ_ONLY)
    def describe_table(table: str, preview_rows: int = 5) -> dict[str, Any]:
        """Column names and types, row count, and the first few rows of a loaded table."""
        return tools.describe_table(ws, table, preview_rows)

    @mcp.tool(annotations=READ_ONLY)
    def run_sql(sql: str, max_rows: int = tools.DEFAULT_ROWS) -> dict[str, Any]:
        """Run one read-only SQLite query (SELECT or WITH ... SELECT).

        Write/DDL statements are not executed and return status needs_human_approval.
        """
        return tools.run_sql(ws, sql, max_rows)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
    def ask(question: str, max_rows: int = 20) -> dict[str, Any]:
        """Answer a plain-English question about the loaded tables.

        Uses the app's SQL agent (Groq, GROQ_API_KEY / GROQ_MODEL): writes SQL, self-checks it,
        runs it read-only and explains the result. Returns the SQL, rows and explanation.
        """
        return tools.ask(ws, question, max_rows)

    @mcp.tool(annotations=READ_ONLY)
    def search_history(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search earlier ask() answers (needs ADA_MCP_RAG=1 and the rag extra)."""
        return tools.search_history(ws, query, top_k)

    return mcp


def main() -> None:
    # stdout belongs to the protocol on stdio, so logs go to stderr
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
