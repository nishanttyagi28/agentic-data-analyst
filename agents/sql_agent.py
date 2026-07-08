"""SQL Agent — natural language to SQL with SELECT-only guard and quality self-check."""

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

# Patterns that signal multi-clause / ranking / HAVING needs
_TOP_N_RE = re.compile(
    r"\btop\s+(\d+)\b|\b(?:first|best)\s+(\d+)\b|\brank(?:ed)?\s+(?:within|by|per)\b|"
    r"\bwithin\s+(?:each|every)\b|\bper\s+(?:group|season|category|region|country|competition)\b",
    re.IGNORECASE,
)
_TOP_N_NUM_RE = re.compile(r"\btop\s+(\d+)\b|\b(?:first|best)\s+(\d+)\b", re.IGNORECASE)
_AGG_THRESHOLD_RE = re.compile(
    r"\b(?:only\s+(?:for|where|show)|only\s+seasons?|with\s+more\s+than|with\s+at\s+least|"
    r"having\s+more\s+than|more\s+than\s+\d+|at\s+least\s+\d+|greater\s+than\s+\d+|"
    r"fewer\s+than\s+\d+|less\s+than\s+\d+\s+(?:matches|rows|records|games))\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"\bpercent(?:age)?\s+of\s+total\b|\b% of total\b|\bshare of\b|\bproportion of total\b",
    re.IGNORECASE,
)
_WINDOW_RE = re.compile(
    r"\b(ROW_NUMBER|RANK|DENSE_RANK)\s*\(|\bOVER\s*\(",
    re.IGNORECASE,
)
_HAVING_RE = re.compile(r"\bHAVING\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_RANK_FILTER_RE = re.compile(
    r"\b(?:rnk|rank|row_num|row_number|rn|rnum)\b\s*<=?\s*\d+|"
    r"\bWHERE\b[^;]*\b(?:rnk|rank|row_num|rn)\b",
    re.IGNORECASE,
)


SQL_SYSTEM_PROMPT = """You are an expert SQLite analyst. Generate ONE correct, complete SQLite query that fully answers the user's question.

## Hard rules
- Output ONLY the SQL in a ```sql code block (no prose).
- Use the exact table names from the schema (often "user_data"; multi-table sessions may have several).
- When multiple tables are listed, JOIN them on the suggested keys (or matching id columns) as needed.
- Read-only: SELECT or WITH ... SELECT only. No INSERT/UPDATE/DELETE/DROP/DDL.
- Prefer CTEs (WITH) for multi-step logic; they are allowed and preferred for clarity.
- Use double quotes for identifiers when names have special characters.
- Fully satisfy EVERY clause of the question (filters, grouping, ranking, thresholds, limits).

## Pattern guide (use these when the question matches)

### 1) Top N per group / ranked within X
When the user says "top N per …", "ranked within each …", "best N for each …":
- Aggregate to the grain first (e.g. season + competition).
- Use ROW_NUMBER() or RANK() OVER (PARTITION BY <group> ORDER BY <metric> DESC).
- Outer-filter WHERE rank_col <= N.
- A plain GROUP BY + LIMIT is WRONG for top-N-per-group (LIMIT applies globally, not per group).

### 2) Only groups where [aggregate condition]
When the user says "only seasons with more than 20 matches", "categories with at least 10 rows":
- Filter aggregates with HAVING (or a CTE then WHERE on the aggregate column).
- WHERE filters raw rows BEFORE aggregation; HAVING filters AFTER aggregation.
- Example intent "seasons with > 20 matches" → HAVING COUNT(*) > 20 (on season grain), or a CTE of seasons that qualify, then INNER JOIN / IN.

### 3) Percentage of total
Use SUM(x) OVER () or a subquery total: value * 100.0 / SUM(value) OVER ().

### 4) Multi-clause questions (combine patterns)
If the question stacks ranking + group filters + aggregates, implement ALL of them in one query via CTEs.

## Worked examples (follow these shapes)

Example A — top N per group + season match threshold:
Question: For each season, show the top 3 competitions by average goals per match, only for seasons with more than 20 matches.
```sql
WITH season_ok AS (
  SELECT season
  FROM "user_data"
  GROUP BY season
  HAVING COUNT(*) > 20
),
comp_stats AS (
  SELECT
    u.season,
    u.competition,
    AVG(u.goals) AS avg_goals,
    COUNT(*) AS matches
  FROM "user_data" u
  INNER JOIN season_ok s ON u.season = s.season
  GROUP BY u.season, u.competition
),
ranked AS (
  SELECT
    season,
    competition,
    avg_goals,
    matches,
    ROW_NUMBER() OVER (PARTITION BY season ORDER BY avg_goals DESC) AS rnk
  FROM comp_stats
)
SELECT season, competition, avg_goals, matches, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY season, rnk;
```

Example B — HAVING only (groups meeting a threshold):
Question: Which contract types have more than 5 customers?
```sql
SELECT contract_type, COUNT(*) AS customer_count
FROM "user_data"
GROUP BY contract_type
HAVING COUNT(*) > 5
ORDER BY customer_count DESC;
```

Example C — percentage of total per category:
Question: What percentage of total revenue does each region contribute?
```sql
SELECT
  region,
  SUM(revenue) AS region_revenue,
  ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 2) AS pct_of_total
FROM "user_data"
GROUP BY region
ORDER BY pct_of_total DESC;
```

## Checklist before you answer
1. Did the user ask for top/rank per group? → CTE + window + rank filter (not global LIMIT alone).
2. Did they filter on an aggregate (counts, averages of groups)? → HAVING or pre-aggregate CTE.
3. Did they ask for % of total? → window total or subquery total.
4. Does the SQL answer the full question, not a simplified subset?
"""


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Validate that SQL is a read-only SELECT or CTE (WITH ... SELECT) query."""
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    cleaned = sql.strip().rstrip(";")
    cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if FORBIDDEN_KEYWORDS.search(cleaned):
        return False, "Query contains forbidden keywords (only SELECT/WITH are allowed)"

    # Allow SELECT and CTE (WITH ... SELECT); window functions are fine.
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return False, "Only SELECT or WITH (CTE) queries are allowed"

    if ";" in cleaned:
        return False, "Multiple statements are not allowed"

    return True, ""


def extract_sql_from_response(response: str) -> str:
    """Extract SQL from LLM response, handling markdown code blocks and CTEs."""
    code_block = re.search(r"```(?:sql)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip().rstrip(";")

    # Prefer full WITH ... SELECT CTEs over the first inner SELECT
    with_match = re.search(r"(WITH\b.+)", response, re.DOTALL | re.IGNORECASE)
    if with_match:
        return with_match.group(1).strip().rstrip(";")

    select_match = re.search(r"(SELECT\b.+)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip().rstrip(";")
    return response.strip().rstrip(";")


def build_schema_context(engine: Engine, tables: dict | None = None) -> str:
    """Single-table (legacy) or multi-table schema text for the LLM."""
    if tables and len(tables) > 0:
        from agents.multitable import build_multi_schema_context
        return build_multi_schema_context(engine, tables)

    schema = get_table_schema(engine, TABLE_NAME)
    if not schema:
        return "No table loaded."
    lines = [f'Table: "{TABLE_NAME}"']
    for col in schema:
        lines.append(f"  - {col['name']} ({col['type']})")
    return "\n".join(lines)


def check_sql_covers_request(question: str, sql: str) -> list[str]:
    """
    Lightweight rule-based self-check: return list of missing elements vs the NL request.
    Empty list means no obvious gaps detected.
    """
    q = question or ""
    s = sql or ""
    issues: list[str] = []

    # Top-N / ranked within group
    if _TOP_N_RE.search(q):
        has_window = bool(_WINDOW_RE.search(s))
        has_rank_filter = bool(_RANK_FILTER_RE.search(s))
        # LIMIT alone is insufficient for "top N per group"
        per_group = bool(re.search(
            r"\bper\b|\bwithin\b|\beach\b|\bfor each\b|\bby season\b|\bby competition\b|"
            r"\bby region\b|\bby category\b|\bby country\b|\bby group\b",
            q,
            re.IGNORECASE,
        ))
        if per_group:
            if not has_window:
                issues.append(
                    "Question asks for top-N / ranking within groups, but SQL has no "
                    "ROW_NUMBER()/RANK()/OVER (PARTITION BY ...) window function."
                )
            if has_window and not has_rank_filter and not re.search(
                r"\bWHERE\b[\s\S]*<=\s*\d+", s, re.IGNORECASE
            ):
                # Allow WHERE rnk <= 3 style more loosely
                if not re.search(r"<=\s*\d+", s):
                    issues.append(
                        "Question asks for top N per group, but SQL never filters rank <= N "
                        "after the window function."
                    )
        else:
            # Global top N — window OR LIMIT acceptable
            num_m = _TOP_N_NUM_RE.search(q)
            if num_m and not has_window and not _LIMIT_RE.search(s):
                issues.append(
                    "Question asks for top N results, but SQL has neither a rank filter nor LIMIT."
                )

    # Aggregate threshold → HAVING (or filter on aggregated CTE)
    if _AGG_THRESHOLD_RE.search(q):
        has_having = bool(_HAVING_RE.search(s))
        # Also accept filtering a pre-aggregated count/sum column in outer WHERE
        has_agg_filter = bool(re.search(
            r"\b(COUNT|SUM|AVG)\s*\(|\bHAVING\b|"
            r"\b(matches|cnt|count|total|n_rows|num_)\b\s*[><=]{1,2}\s*\d+",
            s,
            re.IGNORECASE,
        ))
        # Require either HAVING or an obvious aggregate comparison
        if not has_having:
            # If they only used WHERE on a raw non-aggregate path without count CTE, flag
            if not re.search(
                r"\b(COUNT|SUM|AVG)\s*\([^)]*\)\s*[><=]|"
                r"\bAS\s+\w*(count|matches|total|cnt)\w*\b[\s\S]*\bWHERE\b[\s\S]*[><=]\s*\d+",
                s,
                re.IGNORECASE,
            ):
                issues.append(
                    "Question filters groups by an aggregate threshold (e.g. more than N matches), "
                    "but SQL has no HAVING clause and no clear filter on an aggregated count/sum."
                )
        elif not has_agg_filter:
            issues.append(
                "Question implies an aggregate threshold filter that is not reflected in the SQL."
            )

    # Percentage of total
    if _PERCENT_RE.search(q):
        if not re.search(r"\bOVER\s*\(|/\s*\(|\bTOTAL\b", s, re.IGNORECASE):
            issues.append(
                "Question asks for percentage/share of total, but SQL has no window total "
                "or total subquery divisor."
            )
        if not re.search(r"100|percent|pct|share|proportion", s, re.IGNORECASE):
            # soft check — still ok if they compute ratio without *100
            if "/" not in s and not re.search(r"\bOVER\s*\(", s, re.IGNORECASE):
                issues.append(
                    "Question asks for a percentage of total; SQL does not appear to compute a share."
                )

    return issues


def generate_sql(
    question: str,
    engine: Engine,
    feedback: str | None = None,
    tables: dict | None = None,
) -> tuple[str | None, str | None]:
    schema_ctx = build_schema_context(engine, tables=tables)

    user_prompt = f"""Database schema:
{schema_ctx}

User question: {question}
"""
    if feedback:
        user_prompt += f"""
## Self-check feedback (your previous SQL was incomplete — fix ALL of these)
{feedback}

Regenerate a COMPLETE SQL query that addresses every issue above while still answering the question.
"""
    else:
        user_prompt += "\nGenerate the complete SQL query now:"

    response, err = chat_completion([
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], temperature=0.1, max_tokens=1200)
    if err:
        return None, err
    sql = extract_sql_from_response(response or "")
    return sql, None


def generate_sql_with_self_check(
    question: str,
    engine: Engine,
    tables: dict | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """
    Generate SQL, rule-check coverage vs the NL request, retry once with feedback if needed.
    Returns (sql, error, meta).
    """
    meta: dict[str, Any] = {
        "attempts": 0,
        "issues_first": [],
        "issues_final": [],
        "retried": False,
        "sql_first": None,
    }

    sql, err = generate_sql(question, engine, tables=tables)
    meta["attempts"] = 1
    if err or not sql:
        return sql, err or "Failed to generate SQL", meta

    meta["sql_first"] = sql
    issues = check_sql_covers_request(question, sql)
    meta["issues_first"] = issues

    if not issues:
        meta["issues_final"] = []
        return sql, None, meta

    # One retry with specific mismatch feedback
    feedback = "\n".join(f"- {i}" for i in issues)
    sql2, err2 = generate_sql(question, engine, feedback=feedback, tables=tables)
    meta["attempts"] = 2
    meta["retried"] = True
    if err2 or not sql2:
        # Keep first SQL; caller may note residual issues
        meta["issues_final"] = issues
        return sql, None, meta

    issues2 = check_sql_covers_request(question, sql2)
    meta["issues_final"] = issues2
    # Prefer second attempt if it improved (fewer issues) or equal issues
    if len(issues2) <= len(issues):
        return sql2, None, meta
    return sql, None, meta


def explain_results(
    question: str,
    sql: str,
    result_preview: str,
    residual_issues: list[str] | None = None,
) -> tuple[str | None, str | None]:
    system_prompt = (
        "You are a data analyst. Explain SQL query results in plain English. "
        "Be concise and insightful. If residual gaps are listed, honestly say what the "
        "query still does not fully cover — never invent rankings or filters that are not in the SQL."
    )
    user_prompt = f"""Question: {question}
SQL: {sql}
Results (preview):
{result_preview}
"""
    if residual_issues:
        user_prompt += (
            "\nKnown gaps after self-check (mention honestly if still relevant):\n"
            + "\n".join(f"- {i}" for i in residual_issues)
        )

    user_prompt += "\nProvide a clear plain-English explanation of these results."

    return chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])


def run_sql_query(
    question: str,
    engine: Engine,
    tables: dict | None = None,
) -> dict[str, Any]:
    """Full SQL agent pipeline: generate, self-check/retry, validate, execute, explain."""
    sql, gen_err, check_meta = generate_sql_with_self_check(question, engine, tables=tables)
    if gen_err:
        return {"success": False, "error": gen_err, "agent": "sql", "self_check": check_meta}

    if not sql:
        return {
            "success": False,
            "error": "Failed to generate SQL",
            "agent": "sql",
            "self_check": check_meta,
        }

    safe, safety_err = is_safe_select(sql)
    if not safe:
        return {
            "success": False,
            "error": f"SQL safety check failed: {safety_err}",
            "sql": sql,
            "agent": "sql",
            "self_check": check_meta,
        }

    residual = check_meta.get("issues_final") or []

    try:
        df = execute_select_query(engine, sql)
        preview = df.head(20)
        preview_str = preview.to_string(index=False) if not preview.empty else "(no rows)"
        explanation, exp_err = explain_results(question, sql, preview_str, residual_issues=residual)

        # Honest note if self-check still finds gaps after retry
        if residual:
            gap_note = (
                "\n\n**Note:** After a self-check retry, this query may still not fully match "
                "every part of your request:\n"
                + "\n".join(f"- {i}" for i in residual)
            )
            explanation = (explanation or "Results retrieved.") + gap_note

        return {
            "success": True,
            "agent": "sql",
            "question": question,
            "sql": sql,
            "result": df,
            "row_count": len(df),
            "explanation": explanation or "Results retrieved successfully.",
            "explanation_error": exp_err,
            "self_check": check_meta,
            "summary_for_rag": (
                f"SQL Query: {sql}\n"
                f"Rows returned: {len(df)}\n"
                f"Self-check retried: {check_meta.get('retried')}\n"
                f"Explanation: {explanation or preview_str[:500]}"
            ),
        }
    except Exception as e:
        # If execution failed and we had a first attempt, try executing first SQL once
        first = check_meta.get("sql_first")
        if first and first != sql:
            safe1, _ = is_safe_select(first)
            if safe1:
                try:
                    df = execute_select_query(engine, first)
                    preview = df.head(20)
                    preview_str = preview.to_string(index=False) if not preview.empty else "(no rows)"
                    explanation, exp_err = explain_results(
                        question, first, preview_str,
                        residual_issues=check_meta.get("issues_first") or [],
                    )
                    return {
                        "success": True,
                        "agent": "sql",
                        "question": question,
                        "sql": first,
                        "result": df,
                        "row_count": len(df),
                        "explanation": (explanation or "") + (
                            f"\n\n(Used first SQL attempt after retry failed to execute: {e})"
                        ),
                        "explanation_error": exp_err,
                        "self_check": check_meta,
                        "summary_for_rag": f"SQL Query: {first}\nRows: {len(df)}\n{explanation}",
                    }
                except Exception:
                    pass
        return {
            "success": False,
            "error": f"Query execution failed: {e}",
            "sql": sql,
            "agent": "sql",
            "self_check": check_meta,
        }
