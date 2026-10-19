"""Plain functions behind each MCP tool.

Kept free of MCP types so they can be tested directly. Analysis logic lives in
`agents/` and `db/`; this module only validates input and shapes the output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ada_mcp.workspace import SAMPLE_DIR, Workspace, list_samples
from agents.multitable import detect_join_keys, register_dataframe
from agents.sql_agent import FORBIDDEN_KEYWORDS, is_safe_select, run_sql_query
from db.database import execute_select_query, get_row_count, get_table_schema

log = logging.getLogger("ada_mcp")

MAX_ROWS = 1000
DEFAULT_ROWS = 100


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # to_json handles numpy types, NaN and timestamps; to_dict does not
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _needs_approval(sql: str, reason: str) -> dict[str, Any]:
    return {
        "status": "needs_human_approval",
        "executed": False,
        "sql": sql,
        "message": (
            "This looks like a write or schema change. The MCP server only runs "
            "read-only SELECT/WITH queries, so it was not executed. A person has to "
            "review and run it outside this server."
        ),
        "reason": reason,
    }


def _is_write_like(sql: str) -> bool:
    return bool(FORBIDDEN_KEYWORDS.search(sql or ""))


def _require_table(ws: Workspace, table: str) -> str:
    if table not in ws.tables:
        loaded = ", ".join(ws.tables) or "none"
        raise ValueError(f"Unknown table '{table}'. Loaded tables: {loaded}")
    return table


def _resolve_source(source: str) -> tuple[Path, str]:
    name = source.strip()
    stem = name[:-4] if name.lower().endswith(".csv") else name
    if stem in list_samples():
        return SAMPLE_DIR / f"{stem}.csv", stem
    path = Path(name).expanduser()
    if path.suffix.lower() != ".csv":
        samples = ", ".join(list_samples()) or "none"
        raise ValueError(f"'{source}' is not a sample ({samples}) or a .csv path")
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path, path.stem


def load_dataset(ws: Workspace, source: str, table_name: str | None = None) -> dict[str, Any]:
    path, default_name = _resolve_source(source)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} has no rows")
    result = register_dataframe(df, ws.engine, table_name or default_name, ws.tables)
    return {
        "table": result["table_name"],
        "row_count": result["row_count"],
        "columns": result["columns"],
        "source": str(path),
    }


def list_tables(ws: Workspace) -> dict[str, Any]:
    return {
        "tables": [
            {"name": name, "row_count": len(df), "column_count": len(df.columns)}
            for name, df in ws.tables.items()
        ],
        "join_hints": [j["message"] for j in detect_join_keys(ws.tables)] if len(ws.tables) > 1 else [],
        "samples": list_samples(),
    }


def describe_table(ws: Workspace, table: str, preview_rows: int = 5) -> dict[str, Any]:
    name = _require_table(ws, table)
    n = max(0, min(int(preview_rows), 50))
    preview = execute_select_query(ws.engine, f'SELECT * FROM "{name}" LIMIT {n}')
    return {
        "table": name,
        "row_count": get_row_count(ws.engine, name),
        "columns": get_table_schema(ws.engine, name),
        "preview": _records(preview),
    }


def run_sql(ws: Workspace, sql: str, max_rows: int = DEFAULT_ROWS) -> dict[str, Any]:
    safe, reason = is_safe_select(sql)
    if not safe:
        if _is_write_like(sql):
            return _needs_approval(sql, reason)
        raise ValueError(f"Query rejected: {reason}")
    limit = max(1, min(int(max_rows), MAX_ROWS))
    df = execute_select_query(ws.engine, sql)
    return {
        "status": "ok",
        "columns": list(df.columns),
        "row_count": len(df),
        "rows": _records(df.head(limit)),
        "truncated": len(df) > limit,
    }


def ask(ws: Workspace, question: str, max_rows: int = 20) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question is empty")
    if not ws.tables:
        raise ValueError("No tables loaded. Call load_dataset first.")

    result = run_sql_query(question, ws.engine, tables=ws.tables)
    sql = result.get("sql") or ""
    if not result.get("success"):
        if sql and _is_write_like(sql):
            return _needs_approval(sql, result.get("error", ""))
        raise ValueError(result.get("error") or "SQL agent failed")

    df = result["result"]
    limit = max(1, min(int(max_rows), MAX_ROWS))
    out = {
        "status": "ok",
        "sql": sql,
        "explanation": result.get("explanation", ""),
        "row_count": result.get("row_count", len(df)),
        "rows": _records(df.head(limit)),
        "truncated": len(df) > limit,
        "self_check_retried": bool((result.get("self_check") or {}).get("retried")),
    }
    _index_answer(ws, question, out)
    return out


def _index_answer(ws: Workspace, question: str, answer: dict[str, Any]) -> None:
    try:
        rag = ws.get_rag()
        if rag is not None:
            rag.index_exchange(question, answer["explanation"], "sql", extra=f"SQL: {answer['sql']}")
    except Exception as e:  # history is optional, never fail the answer over it
        log.warning("could not index answer for search_history: %s", e)


def search_history(ws: Workspace, query: str, top_k: int = 5) -> dict[str, Any]:
    try:
        rag = ws.get_rag()
    except ImportError as e:
        raise ValueError(f"search_history needs chromadb and sentence-transformers: {e}") from e
    if rag is None:
        return {
            "status": "disabled",
            "message": "Set ADA_MCP_RAG=1 to index ask() answers and search them.",
            "results": [],
        }
    k = max(1, min(int(top_k), 20))
    chunks = rag.retrieve(query, top_k=k)
    return {
        "status": "ok",
        "results": [
            {"text": c["text"], "source_type": c["source_type"], "distance": c["distance"]}
            for c in chunks
        ],
    }
