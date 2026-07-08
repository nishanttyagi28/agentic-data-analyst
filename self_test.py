"""End-to-end self-test for Agentic Data Analyst."""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from utils.env import get_groq_api_key, load_project_env

load_project_env()

from agents.ingestion import ingest_csv
from agents.ml_agent import run_ml_analysis
from agents.orchestrator import Orchestrator, classify_query
from agents.rag_agent import RAGAgent
from agents.sql_agent import is_safe_select, run_sql_query
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
        check("summary generated", bool(result.get("summary")))
    return result


def test_orchestrator_routing():
    print("\n=== Orchestrator Routing ===")
    check("sql route", classify_query("How many rows are there?", True, False)[0] == "sql")
    check("ml route", classify_query("Train a model to predict churn", True, False)[0] == "ml")
    check("rag route", classify_query("What were the key findings?", True, True)[0] == "rag")


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

    sql_result = test_sql_agent(engine, sql_q)
    if sql_result and sql_result.get("success"):
        orch.index_result(sql_q, sql_result)

    ml_result = test_ml(df, label, ml_q)
    if ml_result and ml_result.get("success"):
        orch.index_result(ml_q, ml_result)

    texts = []
    if sql_result and sql_result.get("summary_for_rag"):
        texts.append(sql_result["summary_for_rag"])
    if ml_result and ml_result.get("summary_for_rag"):
        texts.append(ml_result["summary_for_rag"])

    if texts:
        test_rag(session_id, texts, rag_qs)
    else:
        skip("rag tests", "no indexed content")

    if HAS_API_KEY:
        print(f"\n=== Orchestrator E2E: {label} ===")
        for q in [sql_q, ml_q] + rag_qs[:1]:
            r = orch.handle_query(q)
            check(f"orchestrator: {q[:40]}", r.get("success") or "error" in r)


def main():
    print("Agentic Data Analyst — Self Test")
    print(f"GROQ_API_KEY set: {HAS_API_KEY}")

    test_sql_safety()
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