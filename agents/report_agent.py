"""Report Agent — compile session findings into a downloadable HTML report."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from agents.llm_client import chat_completion


def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


def _collect_findings(session: dict[str, Any], chat_history: list[dict] | None = None) -> dict[str, Any]:
    """Gather quality, EDA, ML, stats, forecast snippets from session state-like dict."""
    findings: dict[str, Any] = {
        "quality": session.get("quality_report"),
        "ml_results": [],
        "stats_results": [],
        "forecast_results": [],
        "sql_results": [],
        "dataset": {
            "rows": session.get("row_count"),
            "columns": session.get("column_count"),
            "column_names": session.get("columns") or [],
        },
    }

    history = chat_history or session.get("chat_history") or []
    for item in history:
        result = item.get("result") if isinstance(item, dict) and "result" in item else item
        if not isinstance(result, dict) or not result.get("success"):
            continue
        agent = result.get("agent") or result.get("route") or ""
        entry = {
            "query": item.get("query") or item.get("question") or "",
            "summary": result.get("summary") or result.get("explanation") or result.get("answer") or "",
            "metrics": result.get("metrics"),
            "sql": result.get("sql"),
        }
        if agent == "ml":
            findings["ml_results"].append(entry)
        elif agent == "stats":
            findings["stats_results"].append(entry)
        elif agent == "forecast":
            findings["forecast_results"].append(entry)
        elif agent == "sql":
            findings["sql_results"].append(entry)
        elif agent == "quality" and not findings["quality"]:
            findings["quality"] = result

    return findings


def generate_executive_summary(findings: dict[str, Any], use_llm: bool = True) -> str:
    parts = []
    ds = findings.get("dataset") or {}
    if ds.get("rows"):
        parts.append(f"Dataset has {ds['rows']} rows and {ds.get('columns', '?')} columns.")
    q = findings.get("quality") or {}
    if q.get("quality_score") is not None:
        parts.append(f"Data quality score is {q['quality_score']}/100 with {q.get('duplicate_count', 0)} duplicate rows.")
    if findings.get("ml_results"):
        last = findings["ml_results"][-1]
        parts.append(f"ML analysis: {(last.get('summary') or '')[:300]}")
    if findings.get("stats_results"):
        last = findings["stats_results"][-1]
        parts.append(f"Statistical finding: {(last.get('summary') or '')[:300]}")
    if findings.get("forecast_results"):
        last = findings["forecast_results"][-1]
        parts.append(f"Forecast: {(last.get('summary') or '')[:300]}")
    if findings.get("sql_results"):
        parts.append(f"{len(findings['sql_results'])} SQL insight(s) captured in this session.")

    fallback = " ".join(parts) if parts else (
        "Analysis is still early. Upload data and run queries, EDA, stats, or models to build a fuller report."
    )

    if not use_llm or not parts:
        return fallback[:800]

    prompt = f"""Write a 2-4 sentence executive summary for a business manager based on these analysis notes.
Use plain language, no jargon dumps, and do not overstate certainty.
Notes:
{fallback}"""
    text, err = chat_completion([
        {"role": "system", "content": "You write concise executive summaries for data analysis reports."},
        {"role": "user", "content": prompt},
    ], max_tokens=300)
    if err or not text:
        return fallback[:800]
    return text.strip()


def build_html_report(
    findings: dict[str, Any],
    executive_summary: str,
    title: str = "Agentic Data Analyst Report",
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = []

    sections.append(f"<h1>{_esc(title)}</h1>")
    sections.append(f"<p class='meta'>Generated { _esc(now) }</p>")
    sections.append("<h2>Executive Summary</h2>")
    sections.append(f"<p class='exec'>{_esc(executive_summary)}</p>")

    ds = findings.get("dataset") or {}
    sections.append("<h2>Dataset</h2>")
    sections.append(
        f"<p>Rows: <b>{_esc(ds.get('rows', 'n/a'))}</b> · "
        f"Columns: <b>{_esc(ds.get('columns', 'n/a'))}</b></p>"
    )
    cols = ds.get("column_names") or []
    if cols:
        sections.append("<p>Columns: " + ", ".join(f"<code>{_esc(c)}</code>" for c in cols[:40]) + "</p>")

    q = findings.get("quality")
    if q and q.get("success") is not False:
        sections.append("<h2>Data Quality</h2>")
        sections.append(f"<p>Score: <b>{_esc(q.get('quality_score', 'n/a'))}/100</b></p>")
        sections.append(f"<pre>{_esc(q.get('summary', ''))}</pre>")

    def _list_section(title: str, items: list[dict]):
        if not items:
            return
        sections.append(f"<h2>{_esc(title)}</h2>")
        for i, item in enumerate(items, 1):
            if item.get("query"):
                sections.append(f"<h3>Finding {i}: {_esc(item['query'][:120])}</h3>")
            sections.append(f"<p>{_esc(item.get('summary', ''))}</p>")
            if item.get("metrics"):
                sections.append("<ul>")
                for k, v in item["metrics"].items():
                    sections.append(f"<li><b>{_esc(k)}</b>: {_esc(v)}</li>")
                sections.append("</ul>")
            if item.get("sql"):
                sections.append(f"<pre>{_esc(item['sql'])}</pre>")

    _list_section("SQL Insights", findings.get("sql_results") or [])
    _list_section("Machine Learning", findings.get("ml_results") or [])
    _list_section("Statistical Analysis", findings.get("stats_results") or [])
    _list_section("Forecasts", findings.get("forecast_results") or [])

    sections.append(
        "<hr><p class='footer'>Generated by Agentic Data Analyst. "
        "Forecasts and statistical results are estimates; verify before business-critical decisions.</p>"
    )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_esc(title)}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1.25rem; color: #1a1a1a; line-height: 1.55; }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  h2 {{ color: #1565C0; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.35rem; margin-top: 1.75rem; }}
  h3 {{ font-size: 1.05rem; margin-top: 1rem; }}
  .meta {{ color: #666; font-size: 0.9rem; }}
  .exec {{ background: #E3F2FD; padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid #1565C0; }}
  pre {{ background: #f5f5f5; padding: 0.75rem 1rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; font-size: 0.88rem; }}
  code {{ background: #f0f0f0; padding: 0.1rem 0.35rem; border-radius: 3px; }}
  .footer {{ color: #888; font-size: 0.85rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def generate_report(
    dataframe: pd.DataFrame | None,
    quality_report: dict | None,
    chat_history: list[dict] | None,
    use_llm: bool = True,
) -> dict[str, Any]:
    session = {
        "quality_report": quality_report,
        "chat_history": chat_history or [],
        "row_count": len(dataframe) if dataframe is not None else None,
        "column_count": len(dataframe.columns) if dataframe is not None else None,
        "columns": list(dataframe.columns) if dataframe is not None else [],
    }
    findings = _collect_findings(session, chat_history)
    exec_summary = generate_executive_summary(findings, use_llm=use_llm)
    html_doc = build_html_report(findings, exec_summary)

    return {
        "success": True,
        "agent": "report",
        "executive_summary": exec_summary,
        "html": html_doc,
        "findings": findings,
        "summary": exec_summary,
        "summary_for_rag": f"Generated analysis report.\nExecutive summary: {exec_summary}",
    }
