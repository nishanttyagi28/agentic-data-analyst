"""End-to-end self-test for Agentic Data Analyst."""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from utils.env import get_groq_api_key, load_project_env

load_project_env()

import numpy as np
import pandas as pd

from agents.forecast_agent import run_forecast
from agents.ingestion import ingest_csv
from agents.ml_agent import run_eda, run_ml_analysis
from agents.orchestrator import Orchestrator, classify_query
from agents.quality_agent import analyze_data_quality, apply_auto_clean, format_quality_markdown
from agents.rag_agent import RAGAgent
from agents.report_agent import generate_report
from agents.sql_agent import is_safe_select, run_sql_query
from agents.stats_agent import run_stats_analysis
from db.database import get_engine

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
HAS_API_KEY = bool(get_groq_api_key())

passed = 0
failed = 0
skipped = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


def skip(name: str, reason: str):
    global skipped
    skipped += 1
    print(f"  SKIP: {name} — {reason}")


def test_sql_safety():
    print("\n=== SQL Safety Guards ===")
    check("allows SELECT", is_safe_select("SELECT * FROM user_data")[0])
    check("blocks INSERT", not is_safe_select("INSERT INTO user_data VALUES (1)")[0])
    check("blocks DROP", not is_safe_select("DROP TABLE user_data")[0])
    check("blocks DELETE", not is_safe_select("DELETE FROM user_data")[0])
    check("blocks multi-statement", not is_safe_select("SELECT 1; DROP TABLE user_data")[0])
    cte_window = (
        "WITH ranked AS (SELECT sales_person, RANK() OVER "
        "(PARTITION BY country ORDER BY SUM(amount) DESC) as rnk "
        "FROM sales GROUP BY sales_person, country) "
        "SELECT * FROM ranked WHERE rnk <= 2"
    )
    check("allows CTE + window functions", is_safe_select(cte_window)[0])
    check(
        "blocks chained DROP after SELECT",
        not is_safe_select("SELECT * FROM sales; DROP TABLE sales;")[0],
    )
    check(
        "allows trailing semicolon only",
        is_safe_select("SELECT * FROM sales;")[0],
    )
    check(
        "word-boundary: column name with 'update'",
        is_safe_select("SELECT last_update FROM sales")[0],
    )


def test_quality_agent():
    print("\n=== Phase A: Data Quality Agent ===")
    # Messy synthetic data: missing, dups, numbers as text, near-dup categories, outliers
    df = pd.DataFrame({
        "region": ["USA", "US", "United States", "USA", "Canada", "Canada"],
        "amount": [10.0, 12.0, 11.0, 10.0, 9.0, 500.0],  # 500 is outlier; one exact dup row later
        "qty_text": ["1", "2", "3", "1", "4", "5"],
        "note": ["a", None, "c", "a", "e", "f"],
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # exact duplicate

    report = analyze_data_quality(df)
    check("quality success", report["success"], report.get("error", ""))
    check("detects duplicates", report.get("duplicate_count", 0) >= 1)
    check("detects missing", "note" in (report.get("missing") or {}))
    check("detects type issues", any(t["column"] == "qty_text" for t in report.get("type_issues", [])))
    check("detects outliers", "amount" in (report.get("outliers") or {}))
    check("flags cat inconsistencies", "region" in (report.get("categorical_issues") or {}))
    check("has suggestions", len(report.get("suggestions") or []) > 0)
    check("markdown report", "Data Quality" in format_quality_markdown(report))

    cleaned = apply_auto_clean(df)
    check("auto-clean success", cleaned["success"], cleaned.get("error", ""))
    if cleaned["success"]:
        cdf = cleaned["dataframe"]
        check("duplicates removed", int(cdf.duplicated().sum()) == 0)
        check("missing imputed", int(cdf["note"].isnull().sum()) == 0)
        check("qty cast numeric", pd.api.types.is_numeric_dtype(cdf["qty_text"]))
        # Outliers must NOT be auto-removed
        check("outliers kept (no auto-drop)", float(cdf["amount"].max()) >= 500)

    # Sample CSVs still work
    churn = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    q2 = analyze_data_quality(churn)
    check("quality on churn sample", q2["success"])


def test_expanded_eda():
    print("\n=== Phase B: Expanded EDA ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    eda = run_eda(df)
    check("eda has numeric_stats", bool(eda.get("numeric_stats")))
    stats = next(iter(eda["numeric_stats"].values()))
    check("stats has median", "median" in stats)
    check("stats has quartiles", "q25" in stats and "q75" in stats)
    check("describe_table present", eda.get("describe_table") is not None and len(eda["describe_table"]) > 0)
    check("correlation matrix or heatmap", eda.get("correlation_matrix") is not None or eda["charts"].get("correlation") is not None)
    check("distribution charts", len(eda["charts"].get("distributions") or {}) > 0)
    check("groupby suggestions", isinstance(eda.get("groupings"), list))
    check("groupby charts when cats exist", len(eda["charts"].get("groupbys") or {}) >= 1)

    # Tiny edge case
    tiny = pd.DataFrame({"a": [1], "b": ["x"]})
    eda_tiny = run_eda(tiny)
    check("eda on tiny df", "summary_text" in eda_tiny)


def test_stats_agent():
    print("\n=== Phase C: Stats Agent ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    # Comparison / t-test style
    r1 = run_stats_analysis(df, "Is there a significant difference in monthly_charges between contract_type groups?")
    check("comparison success", r1["success"], r1.get("error", ""))
    if r1["success"]:
        check("has p_value", "p_value" in r1)
        check("plain english interpretation", len(r1.get("interpretation") or r1.get("summary") or "") > 40)
        check("states assumptions", len(r1.get("assumptions") or []) > 0 or "causation" in (r1.get("summary") or "").lower() or "assum" in (r1.get("summary") or "").lower())

    r2 = run_stats_analysis(df, "What's correlated with churn?")
    check("correlation success", r2["success"], r2.get("error", ""))
    if r2["success"]:
        check("rankings present", len(r2.get("rankings") or []) > 0)

    r3 = run_stats_analysis(df, "Are there any outliers I should know about?")
    check("outlier summary success", r3["success"], r3.get("error", ""))

    # Houses: price by neighborhood
    houses = pd.read_csv(os.path.join(SAMPLE_DIR, "house_prices.csv"))
    r4 = run_stats_analysis(houses, "Is there a significant difference in price between neighborhoods?")
    check("anova/ttest on houses", r4["success"], r4.get("error", ""))


def test_forecast_agent():
    print("\n=== Phase D: Forecasting ===")
    # Build a small dated series
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    revenue = np.linspace(100, 200, 12) + np.random.RandomState(0).normal(0, 5, 12)
    df = pd.DataFrame({"month": dates, "revenue": revenue})
    result = run_forecast(df, "forecast next month's revenue")
    check("forecast success", result["success"], result.get("error", ""))
    if result["success"]:
        check("forecast table rows", len(result.get("forecast_table") or []) >= 1)
        row = result["forecast_table"][0]
        check("has confidence band", "lower_95" in row and "upper_95" in row)
        check("band width positive", row["upper_95"] >= row["lower_95"])
        check("caveats present", len(result.get("caveats") or []) > 0)
        check("chart present", "forecast" in (result.get("charts") or {}))
        check("summary mentions estimate", "estimate" in result.get("summary", "").lower() or "trend" in result.get("summary", "").lower())

    # No date column — synthetic index fallback
    nodate = pd.DataFrame({"sales": [10, 12, 14, 13, 15, 16]})
    r2 = run_forecast(nodate, "forecast next period sales")
    check("forecast without datetime", r2["success"], r2.get("error", ""))

    # Sample churn has no dates — should still not crash
    churn = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    r3 = run_forecast(churn, "forecast next month's monthly_charges")
    check("forecast on churn sample", r3["success"], r3.get("error", ""))


def test_report_agent():
    print("\n=== Phase E: Report Export ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    quality = analyze_data_quality(df)
    ml = run_ml_analysis(df, "Train a model to predict churn")
    stats = run_stats_analysis(df, "What's correlated with churn?")
    history = [
        {"query": "predict churn", "result": ml},
        {"query": "correlated with churn", "result": stats},
    ]
    report = generate_report(df, quality, history, use_llm=False)
    check("report success", report["success"], report.get("error", ""))
    check("has html", bool(report.get("html")) and "<html" in report["html"].lower())
    check("executive summary", len(report.get("executive_summary") or "") > 20)
    check("html has exec section", "Executive Summary" in report["html"])
    check("html has quality", "Data Quality" in report["html"] or "quality" in report["html"].lower())


def test_orchestrator_routing():
    print("\n=== Phase F: Orchestrator Routing ===")
    check("sql route", classify_query("How many rows are there?", True, False)[0] == "sql")
    check("ml route", classify_query("Train a model to predict churn", True, False)[0] == "ml")
    check("rag route", classify_query("What were the key findings?", True, True)[0] == "rag")
    check("quality route", classify_query("Show me a data quality report", True, False)[0] == "quality")
    check("stats route", classify_query("Is there a significant difference in revenue between regions?", True, False)[0] == "stats")
    check("forecast route", classify_query("Forecast next month's revenue", True, False)[0] == "forecast")
    check("report route", classify_query("Generate a report of my analysis", True, False)[0] == "report")
    check("outliers stats route", classify_query("Are there any outliers I should know about?", True, False)[0] == "stats")
    check("correlation stats route", classify_query("What's correlated with churn?", True, False)[0] == "stats")


def test_ingestion(csv_path: str, label: str):
    print(f"\n=== Ingestion: {label} ===")
    engine = get_engine()
    result = ingest_csv(file_path=csv_path, engine=engine)
    check("ingestion success", result["success"], result.get("error", ""))
    if result["success"]:
        check("row count > 0", result["row_count"] > 0)
        check("schema available", len(result["schema"]) > 0)
        check("dataframe returned", result["dataframe"] is not None)
        return result
    return None


def test_ml(df, label: str, query: str):
    print(f"\n=== ML Agent: {label} ===")
    result = run_ml_analysis(df, query)
    check("ml success", result["success"], result.get("error", ""))
    if result["success"]:
        check("task detected", result.get("task") in ("classification", "regression", "clustering"))
        check("metrics present", len(result.get("metrics", {})) > 0)
        check("eda charts", "missing" in result.get("charts", {}))
        check("expanded describe", result.get("eda", {}).get("describe_table") is not None)
        check("summary generated", bool(result.get("summary")))
    return result


def test_sql_agent(engine, query: str):
    print(f"\n=== SQL Agent: {query[:50]} ===")
    if not HAS_API_KEY:
        skip("sql agent", "GROQ_API_KEY not set")
        return None
    result = run_sql_query(query, engine)
    check("sql success", result["success"], result.get("error", ""))
    if result["success"]:
        check("sql generated", bool(result.get("sql")))
        check("result returned", result.get("result") is not None)
        check("explanation", bool(result.get("explanation")))
    return result


def test_rag(session_id: str, texts: list[str], questions: list[str]):
    print(f"\n=== RAG Agent (session {session_id}) ===")
    rag = RAGAgent(session_id)
    total_chunks = 0
    for i, text in enumerate(texts):
        n = rag.index_text(text, f"test_{i}")
        total_chunks += n
    check("chunks indexed", total_chunks > 0, f"indexed {total_chunks}")

    for q in questions:
        if not HAS_API_KEY:
            skip(f"rag answer: {q[:40]}", "GROQ_API_KEY not set")
            continue
        result = rag.answer(q)
        check(f"rag answer: {q[:40]}", result["success"], result.get("error", ""))
        if result["success"]:
            check(f"  citations for: {q[:30]}", len(result.get("citations", [])) > 0)


def test_full_flow(csv_path: str, label: str, sql_q: str, ml_q: str, rag_qs: list[str]):
    print(f"\n{'='*60}")
    print(f"FULL FLOW: {label}")
    print(f"{'='*60}")

    result = test_ingestion(csv_path, label)
    if not result:
        return

    engine = get_engine()
    df = result["dataframe"]
    session_id = f"test_{label}_{uuid.uuid4().hex[:6]}"
    orch = Orchestrator(engine, session_id, df)

    # Quality via orchestrator
    q = orch.run_quality_scan()
    check(f"orch quality: {label}", q.get("success"), q.get("error", ""))
    orch.index_result("data quality", q)

    sql_result = test_sql_agent(engine, sql_q)
    if sql_result and sql_result.get("success"):
        orch.index_result(sql_q, sql_result)

    ml_result = test_ml(df, label, ml_q)
    if ml_result and ml_result.get("success"):
        orch.index_result(ml_q, ml_result)

    stats_result = run_stats_analysis(df, "Are there any outliers I should know about?")
    check(f"stats in flow: {label}", stats_result.get("success"), stats_result.get("error", ""))
    if stats_result.get("success"):
        orch.index_result("outliers", stats_result)

    forecast_result = run_forecast(df, "forecast next month's values")
    check(f"forecast in flow: {label}", forecast_result.get("success"), forecast_result.get("error", ""))
    if forecast_result.get("success"):
        orch.index_result("forecast", forecast_result)

    report = generate_report(df, q, orch.session_findings, use_llm=False)
    check(f"report in flow: {label}", report.get("success"))
    if report.get("success"):
        orch.index_result("generate report", report)

    texts = []
    for r in (sql_result, ml_result, stats_result, forecast_result, report):
        if r and r.get("summary_for_rag"):
            texts.append(r["summary_for_rag"])

    if texts:
        test_rag(session_id, texts, rag_qs)
    else:
        skip("rag tests", "no indexed content")

    if HAS_API_KEY:
        print(f"\n=== Orchestrator E2E: {label} ===")
        for qtext in [sql_q, ml_q, "Show data quality report", "What's correlated with the target?", "Generate a report"] + rag_qs[:1]:
            r = orch.handle_query(qtext)
            check(f"orchestrator: {qtext[:40]}", r.get("success") or "error" in r, r.get("error", ""))


def main():
    print("Agentic Data Analyst — Self Test (full analyst upgrade)")
    print(f"GROQ_API_KEY set: {HAS_API_KEY}")

    test_sql_safety()
    test_quality_agent()
    test_expanded_eda()
    test_stats_agent()
    test_forecast_agent()
    test_report_agent()
    test_orchestrator_routing()

    test_full_flow(
        os.path.join(SAMPLE_DIR, "customer_churn.csv"),
        "churn",
        sql_q="How many customers churned?",
        ml_q="Train a classification model to predict churn",
        rag_qs=[
            "What is the churn rate?",
            "What were the ML model metrics?",
            "Summarize the analysis findings",
        ],
    )

    test_full_flow(
        os.path.join(SAMPLE_DIR, "house_prices.csv"),
        "houses",
        sql_q="What is the average house price?",
        ml_q="Run regression analysis to predict price",
        rag_qs=[
            "What factors affect house prices?",
            "What was the model R2 score?",
            "Give me a summary of the EDA",
        ],
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
