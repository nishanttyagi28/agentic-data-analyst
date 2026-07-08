"""SQL Agent — natural language to SQL with SELECT-only guard."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.engine import Engine

from agents.llm_client import chat_completion
from db.database import TABLE_NAME, execute_select_query, get_table_schema


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|"
    r"GRANT|REVOKE|EXEC|EXECUTE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Validate that SQL is a read-only SELECT query."""
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    cleaned = sql.strip().rstrip(";")
    cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if FORBIDDEN_KEYWORDS.search(cleaned):
        return False, "Query contains forbidden keywords (only SELECT is allowed)"

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        return False, "Only SELECT queries are allowed"

    if ";" in cleaned:
        return False, "Multiple statements are not allowed"

    return True, ""


def extract_sql_from_response(response: str) -> str:
    """Extract SQL from LLM response, handling markdown code blocks."""
    code_block = re.search(r"```(?:sql)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    select_match = re.search(r"(SELECT\b.+)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip().rstrip(";")
    return response.strip().rstrip(";")


def build_schema_context(engine: Engine) -> str:
    schema = get_table_schema(engine, TABLE_NAME)
    if not schema:
        return "No table loaded."
    lines = [f'Table: "{TABLE_NAME}"']
    for col in schema:
        lines.append(f"  - {col['name']} ({col['type']})")
    return "\n".join(lines)


def generate_sql(question: str, engine: Engine) -> tuple[str | None, str | None]:
    schema_ctx = build_schema_context(engine)
    system_prompt = """You are a SQL expert. Generate a single SQLite SELECT query to answer the user's question.
Rules:
- Only output ONE SELECT statement
- Use double quotes around table and column names if they contain special characters
- Table name is "user_data"
- Do not use INSERT, UPDATE, DELETE, DROP, or any DDL/DML
- Return ONLY the SQL query in a ```sql code block"""

    user_prompt = f"""Database schema:
{schema_ctx}

User question: {question}

Generate the SQL query:"""

    response, err = chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    if err:
        return None, err
    sql = extract_sql_from_response(response or "")
    return sql, None


def explain_results(question: str, sql: str, result_preview: str) -> tuple[str | None, str | None]:
    system_prompt = "You are a data analyst. Explain SQL query results in plain English. Be concise and insightful."
    user_prompt = f"""Question: {question}
SQL: {sql}
Results (preview):
{result_preview}

Provide a clear plain-English explanation of these results."""

    return chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])


def run_sql_query(
    question: str,
    engine: Engine,
) -> dict[str, Any]:
    """Full SQL agent pipeline: generate, validate, execute, explain."""
    sql, gen_err = generate_sql(question, engine)
    if gen_err:
        return {"success": False, "error": gen_err, "agent": "sql"}

    if not sql:
        return {"success": False, "error": "Failed to generate SQL", "agent": "sql"}

    safe, safety_err = is_safe_select(sql)
    if not safe:
        return {
            "success": False,
            "error": f"SQL safety check failed: {safety_err}",
            "sql": sql,
            "agent": "sql",
        }

    try:
        df = execute_select_query(engine, sql)
        preview = df.head(20)
        preview_str = preview.to_string(index=False) if not preview.empty else "(no rows)"
        explanation, exp_err = explain_results(question, sql, preview_str)

        return {
            "success": True,
            "agent": "sql",
            "question": question,
            "sql": sql,
            "result": df,
            "row_count": len(df),
            "explanation": explanation or "Results retrieved successfully.",
            "explanation_error": exp_err,
            "summary_for_rag": (
                f"SQL Query: {sql}\n"
                f"Rows returned: {len(df)}\n"
                f"Explanation: {explanation or preview_str[:500]}"
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Query execution failed: {e}",
            "sql": sql,
            "agent": "sql",
        }