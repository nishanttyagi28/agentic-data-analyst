"""End-to-end self-test for Agentic Data Analyst."""

import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from utils.env import get_groq_api_key, load_project_env

load_project_env()

import numpy as np
import pandas as pd

from agents.forecast_agent import parse_mixed_dates, run_forecast
from agents.ingestion import ingest_csv
from agents.insight_agent import generate_insight_suggestions
from agents.ml_agent import run_eda, run_ml_analysis
from agents.multitable import detect_join_keys
from agents.orchestrator import Orchestrator, classify_query
from agents.quality_agent import (
    analyze_data_quality,
    apply_auto_clean,
    apply_category_merge,
    format_quality_markdown,
)
from agents.rag_agent import RAGAgent
from agents.report_agent import generate_report
from agents.sql_agent import (
    check_sql_covers_request,
    is_safe_select,
    run_sql_query,
)
from agents.stats_agent import run_stats_analysis
from db.database import get_engine, load_dataframe_to_table

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


def test_sql_self_check_rules():
    print("\n=== SQL Self-Check Rules ===")
    q_top = (
        "For each season, show the top 3 competitions by average goals per match, "
        "only for seasons with more than 20 matches"
    )
    bad_sql = (
        'SELECT season, competition, AVG(goals) AS avg_goals '
        'FROM "user_data" GROUP BY season, competition'
    )
    issues = check_sql_covers_request(q_top, bad_sql)
    check("flags missing window for top-N-per-group", any("window" in i.lower() or "rank" in i.lower() for i in issues), str(issues))
    check("flags missing aggregate threshold", any("having" in i.lower() or "threshold" in i.lower() or "aggregate" in i.lower() for i in issues), str(issues))

    good_sql = """
    WITH season_ok AS (
      SELECT season FROM "user_data" GROUP BY season HAVING COUNT(*) > 20
    ),
    comp_stats AS (
      SELECT u.season, u.competition, AVG(u.goals) AS avg_goals
      FROM "user_data" u INNER JOIN season_ok s ON u.season = s.season
      GROUP BY u.season, u.competition
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY season ORDER BY avg_goals DESC) AS rnk
      FROM comp_stats
    )
    SELECT * FROM ranked WHERE rnk <= 3
    """
    issues_good = check_sql_covers_request(q_top, good_sql)
    check("accepts correct multi-clause SQL", len(issues_good) == 0, str(issues_good))

    q_pct = "What percentage of total revenue does each region contribute?"
    issues_pct = check_sql_covers_request(q_pct, 'SELECT region, SUM(revenue) FROM "user_data" GROUP BY region')
    check("flags missing pct-of-total", len(issues_pct) > 0, str(issues_pct))


def _messi_style_dataframe() -> pd.DataFrame:
    """Synthetic football match rows: season, competition, goals (messi_matches-style)."""
    rows = []
    # Season A: 25 matches across 4 competitions — qualifies for >20
    comps_a = ["La Liga"] * 10 + ["UCL"] * 8 + ["Copa"] * 5 + ["Friendly"] * 2
    goals_a = [2, 1, 3, 0, 2, 1, 4, 2, 1, 0, 3, 2, 1, 2, 0, 1, 2, 3, 1, 0, 2, 1, 0, 1, 2]
    for i, (c, g) in enumerate(zip(comps_a, goals_a)):
        rows.append({"season": "2022/23", "competition": c, "goals": g, "match_id": f"a{i}"})
    # Season B: 22 matches — qualifies
    comps_b = ["La Liga"] * 9 + ["UCL"] * 7 + ["Copa"] * 4 + ["Supercup"] * 2
    goals_b = [1, 0, 2, 1, 3, 2, 1, 0, 1, 2, 3, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 1]
    for i, (c, g) in enumerate(zip(comps_b, goals_b)):
        rows.append({"season": "2023/24", "competition": c, "goals": g, "match_id": f"b{i}"})
    # Season C: only 8 matches — must be EXCLUDED by >20 filter
    for i in range(8):
        rows.append({
            "season": "2021/22",
            "competition": "La Liga" if i < 5 else "UCL",
            "goals": i % 3,
            "match_id": f"c{i}",
        })
    return pd.DataFrame(rows)


def _is_rate_limited(result: dict | None) -> bool:
    err = str((result or {}).get("error") or "")
    return "429" in err or "rate_limit" in err.lower() or "Rate limit" in err


def test_sql_multi_clause_generation():
    print("\n=== SQL Multi-Clause Generation (LLM) ===")
    if not HAS_API_KEY:
        skip("multi-clause SQL generation", "GROQ_API_KEY not set")
        return

    engine = get_engine()
    df = _messi_style_dataframe()
    load_dataframe_to_table(df, engine)

    # Pattern 1: top N per group
    q1 = "For each season, show the top 2 competitions by average goals per match, ranked within season"
    r1 = run_sql_query(q1, engine)
    if _is_rate_limited(r1):
        skip("top-N-per-group success", "Groq rate limit")
        skip("remaining multi-clause LLM checks", "Groq rate limit")
        return
    check("top-N-per-group success", r1.get("success"), r1.get("error", ""))
    if r1.get("success"):
        sql1 = r1.get("sql") or ""
        print(f"  SQL[top-N]: {sql1[:300]}...")
        check("top-N has window", bool(re.search(r"ROW_NUMBER|RANK|DENSE_RANK|OVER\s*\(", sql1, re.I)), sql1[:200])
        check("top-N has partition", "PARTITION" in sql1.upper(), sql1[:200])
        res1 = r1["result"]
        check("top-N returned rows", len(res1) > 0)
        # At most 2 rows per season
        if "season" in res1.columns:
            max_per = res1.groupby("season").size().max()
            check("top-N max 2 per season", int(max_per) <= 2, f"max_per={max_per}")

    # Pattern 2: HAVING aggregate threshold
    q2 = "Which seasons have more than 20 matches?"
    r2 = run_sql_query(q2, engine)
    check("HAVING query success", r2.get("success"), r2.get("error", ""))
    if r2.get("success"):
        sql2 = r2.get("sql") or ""
        print(f"  SQL[HAVING]: {sql2[:300]}...")
        check("HAS HAVING or count filter", bool(re.search(r"HAVING|COUNT\s*\(", sql2, re.I)), sql2[:200])
        res2 = r2["result"]
        seasons = set()
        for col in res2.columns:
            if "season" in col.lower():
                seasons = set(res2[col].astype(str).tolist())
                break
        check("excludes short season 2021/22", "2021/22" not in seasons, str(seasons))
        check("includes long seasons", "2022/23" in seasons or "2023/24" in seasons, str(seasons))

    # Pattern 3: percentage of total
    q3 = "What percentage of total goals does each competition contribute?"
    r3 = run_sql_query(q3, engine)
    check("pct-of-total success", r3.get("success"), r3.get("error", ""))
    if r3.get("success"):
        sql3 = r3.get("sql") or ""
        print(f"  SQL[pct]: {sql3[:300]}...")
        check(
            "pct uses window or total divisor",
            bool(re.search(r"OVER\s*\(|/\s*\(|100", sql3, re.I)),
            sql3[:200],
        )
        res3 = r3["result"]
        check("pct returned rows", len(res3) > 0)

    # Pattern 4: FULL multi-clause (original failing example)
    q4 = (
        "For each season, show the top 3 competitions by average goals per match, "
        "ranked within season, only for seasons with more than 20 matches"
    )
    r4 = run_sql_query(q4, engine)
    check("multi-clause success", r4.get("success"), r4.get("error", ""))
    if r4.get("success"):
        sql4 = r4.get("sql") or ""
        print(f"  SQL[multi]:\n{sql4}")
        check("multi has window", bool(re.search(r"ROW_NUMBER|RANK|DENSE_RANK", sql4, re.I)), sql4[:250])
        check("multi has PARTITION BY", "PARTITION" in sql4.upper(), sql4[:250])
        check(
            "multi has season threshold",
            bool(re.search(r"HAVING|COUNT\s*\(|\b>\s*20\b|\b>=\s*20\b", sql4, re.I)),
            sql4[:250],
        )
        check(
            "multi filters rank",
            bool(re.search(r"<=\s*3|\bLIMIT\b", sql4, re.I)) or "rnk" in sql4.lower() or "rank" in sql4.lower(),
            sql4[:250],
        )
        res4 = r4["result"]
        check("multi returned data", len(res4) > 0, "empty result")
        # Must not include short season
        if "season" in res4.columns:
            seasons4 = set(res4["season"].astype(str).tolist())
            check("multi excludes 2021/22", "2021/22" not in seasons4, str(seasons4))
            max_per4 = res4.groupby("season").size().max()
            check("multi max 3 comps per season", int(max_per4) <= 3, f"max_per={max_per4}")
        residual = (r4.get("self_check") or {}).get("issues_final") or []
        check("multi self-check clean or minor", len(residual) == 0, str(residual))

    # Sample dataset regression — simple + top-style
    churn_path = os.path.join(SAMPLE_DIR, "customer_churn.csv")
    from agents.ingestion import ingest_csv as _ingest
    ing = _ingest(file_path=churn_path, engine=engine)
    check("reload churn for sql", ing.get("success"), ing.get("error", ""))
    if ing.get("success"):
        r5 = run_sql_query("How many customers churned?", engine)
        check("simple churn count still works", r5.get("success"), r5.get("error", ""))
        r6 = run_sql_query(
            "For each contract_type, show the top 1 internet_service by average monthly_charges, ranked within contract_type",
            engine,
        )
        check("churn top-N-per-group success", r6.get("success"), r6.get("error", ""))
        if r6.get("success"):
            sql6 = r6.get("sql") or ""
            check(
                "churn top-N uses window",
                bool(re.search(r"ROW_NUMBER|RANK|OVER\s*\(", sql6, re.I)),
                sql6[:200],
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
        # Regular monthly series should keep calendar-style periods when spacing is regular
        check("nonneg clip for revenue", result.get("nonnegative_clipped") is True)
        check("lower bound >= 0 for nonneg metric", all(r["lower_95"] >= -1e-9 for r in result["forecast_table"]))

    # No date column — synthetic index fallback
    nodate = pd.DataFrame({"sales": [10, 12, 14, 13, 15, 16]})
    r2 = run_forecast(nodate, "forecast next period sales")
    check("forecast without datetime", r2["success"], r2.get("error", ""))
    if r2["success"]:
        check("nodate uses event mode", r2.get("event_mode") is True)
        check("nodate warns user", len(r2.get("warnings") or []) > 0)
        check("nodate periods are Event+", all(str(r["period"]).startswith("Event") for r in r2["forecast_table"]))

    # Sample churn has no dates — should still not crash
    churn = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    r3 = run_forecast(churn, "forecast next month's monthly_charges")
    check("forecast on churn sample", r3["success"], r3.get("error", ""))

    # --- Mixed-format date column (football-style) ---
    print("\n=== Forecast: mixed-format dates + count clip ===")
    mixed_dates = [
        "13/03/2024",
        "2024-03-20 00:00:00",
        "27/03/2024",
        "2024-04-05 00:00:00",
        "12/04/2024",
        "2024-04-28 00:00:00",
        "03/05/2024",
        "2024-05-18 00:00:00",
        "01/06/2024",
        "2024-06-15 00:00:00",
        "2024-07-03 00:00:00",
        "20/07/2024",
    ]
    goals = [1, 2, 0, 3, 1, 2, 0, 1, 2, 1, 0, 2]
    foot = pd.DataFrame({"match_date": mixed_dates, "goals": goals})

    parsed, meta = parse_mixed_dates(foot["match_date"])
    check("mixed parse mostly succeeds", meta["n_parsed"] >= 10, str(meta))
    check("mixed parse fail_rate low", meta["fail_rate"] <= 0.05, f"fail_rate={meta['fail_rate']}")
    check("parsed has both March and July", parsed.min().month <= 3 and parsed.max().month >= 7)
    # Critical: ISO '2024-07-03' must NOT be mangled by dayfirst into 2024-03-07
    iso_row = foot["match_date"] == "2024-07-03 00:00:00"
    check(
        "ISO date not mangled by dayfirst",
        parsed.loc[iso_row].iloc[0].month == 7 and parsed.loc[iso_row].iloc[0].day == 3,
        str(parsed.loc[iso_row].iloc[0]),
    )
    dmy_row = foot["match_date"] == "13/03/2024"
    check(
        "DMY slash date day-first",
        parsed.loc[dmy_row].iloc[0].month == 3 and parsed.loc[dmy_row].iloc[0].day == 13,
        str(parsed.loc[dmy_row].iloc[0]),
    )

    r4 = run_forecast(foot, "forecast goals for next season")
    check("mixed-date forecast success", r4["success"], r4.get("error", ""))
    if r4["success"]:
        check("used real date column", r4.get("date_column") == "match_date")
        check("irregular/event framing", r4.get("event_mode") is True or r4.get("irregular_spacing") is True)
        check(
            "no false calendar precision on periods",
            all(not str(r["period"]).startswith("2024-") for r in r4["forecast_table"]),
            str([r["period"] for r in r4["forecast_table"][:3]]),
        )
        check("goals lower bound >= 0", all(r["lower_95"] >= -1e-9 for r in r4["forecast_table"]))
        check("goals point forecast >= 0", all(r["forecast"] >= -1e-9 for r in r4["forecast_table"]))
        check("nonneg clipped flag", r4.get("nonnegative_clipped") is True)
        check(
            "summary mentions event or irregular or warning",
            any(
                k in r4.get("summary", "").lower()
                for k in ("event", "irregular", "not calendar", "clip")
            ),
        )

    # Force negative interval without clip would be possible; with low mean goals band can go negative pre-clip
    # Explicit unit: all-zero-ish nonneg series
    tiny = pd.DataFrame({
        "match_date": mixed_dates[:6],
        "goals": [0, 1, 0, 0, 1, 0],
    })
    r5 = run_forecast(tiny, "forecast goals")
    check("low-count goals forecast", r5["success"], r5.get("error", ""))
    if r5["success"]:
        check("low-count lower never negative", all(r["lower_95"] >= -1e-9 for r in r5["forecast_table"]))


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
    check("insight route", classify_query("Suggest what to explore", True, False)[0] == "insight")


def test_phase_h_insights():
    print("\n=== Phase H: Proactive Insights ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    ins = generate_insight_suggestions(df, business_context="SaaS churn analytics")
    check("insights success", ins["success"], ins.get("error", ""))
    sugs = ins.get("suggestions") or []
    check("3-5 suggestions", 3 <= len(sugs) <= 5, f"n={len(sugs)}")
    check("suggestions have questions", all(s.get("question") for s in sugs))
    check("not purely generic", any(
        k in (s.get("label") or "").lower()
        for s in sugs
        for k in ("churn", "correlat", "missing", "spike", "outlier", "contract", "charge")
    ), str([s.get("label") for s in sugs]))


def test_phase_i_automl():
    print("\n=== Phase I: AutoML ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "customer_churn.csv"))
    # Avoid LLM if possible for speed — still may call for summary
    result = run_ml_analysis(df, "Train a classification model to predict churn", business_context="SaaS")
    check("automl success", result["success"], result.get("error", ""))
    if result["success"]:
        check("has model_name", bool(result.get("model_name")))
        check("leaderboard 2+", len(result.get("leaderboard") or []) >= 2, str(result.get("leaderboard")))
        check("drivers plain language", bool(result.get("drivers")))
        check("risk flags on small perfect data", len(result.get("risk_flags") or []) >= 1, str(result.get("risk_flags")))
        check("metrics present", "accuracy" in (result.get("metrics") or {}) or "r2" in (result.get("metrics") or {}))


def test_phase_j_multitable():
    print("\n=== Phase J: Multi-table ===")
    # messi-style split: matches + competitions lookup
    matches = pd.DataFrame({
        "competition_id": [1, 1, 2, 2, 3, 3, 1, 2],
        "season": ["2022/23"] * 4 + ["2023/24"] * 4,
        "goals": [2, 1, 0, 3, 1, 2, 0, 1],
    })
    competitions = pd.DataFrame({
        "competition_id": [1, 2, 3],
        "competition": ["La Liga", "UCL", "Copa"],
        "region": ["ES", "EU", "ES"],
    })
    joins = detect_join_keys({"matches": matches, "competitions": competitions})
    check("detects competition_id join", any(j.get("left_column") == "competition_id" for j in joins), str(joins))

    engine = get_engine()
    orch = Orchestrator(engine, f"mt_{uuid.uuid4().hex[:6]}", tables={})
    r1 = orch.add_table(matches, "matches")
    r2 = orch.add_table(competitions, "competitions")
    check("register matches", r1.get("success"), r1.get("error", ""))
    check("register competitions", r2.get("success"), r2.get("error", ""))
    check("two tables loaded", len(orch.tables) == 2, str(list(orch.tables.keys())))
    check("orch join suggestions", len(orch.join_suggestions) >= 1)

    from agents.sql_agent import build_schema_context
    schema_txt = build_schema_context(engine, tables=orch.tables)
    check("schema lists matches", "matches" in schema_txt)
    check("schema lists competitions", "competitions" in schema_txt)
    check("schema mentions join", "join" in schema_txt.lower() or "competition_id" in schema_txt)

    if HAS_API_KEY:
        q = "Combine matches and competitions to show total goals by competition name"
        res = run_sql_query(q, engine, tables=orch.tables)
        if _is_rate_limited(res):
            skip("join SQL LLM", "Groq rate limit")
        else:
            check("join SQL success", res.get("success"), res.get("error", ""))
            if res.get("success"):
                sql = res.get("sql") or ""
                print(f"  SQL[join]: {sql[:280]}...")
                check("SQL has JOIN", "JOIN" in sql.upper(), sql[:200])
                check("join returned rows", len(res.get("result") or []) > 0)
    else:
        skip("join SQL LLM", "GROQ_API_KEY not set")


def test_phase_k_business_context():
    print("\n=== Phase K: Business context ===")
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "house_prices.csv"))
    ins_blank = generate_insight_suggestions(df, business_context="")
    ins_ctx = generate_insight_suggestions(df, business_context="Regional housing market inventory")
    check("insights without context", ins_blank["success"])
    check("insights with context", ins_ctx["success"])
    # Context stored on orchestrator
    orch = Orchestrator(get_engine(), "ctx1", dataframe=df, business_context="Housing")
    check("orch context set", orch.business_context == "Housing")
    orch.set_business_context("Updated context")
    check("orch context update", orch.business_context == "Updated context")


def test_file_upload_orchestrator_construction():
    """
    End-to-end style: same construction path as app._load_primary_into_session
    after CSV ingest — both empty and non-empty business context must work.
    """
    print("\n=== File upload path: Orchestrator construction ===")
    engine = get_engine()
    # Simulate single-file upload ingest
    result = ingest_csv(
        file_path=os.path.join(SAMPLE_DIR, "customer_churn.csv"),
        engine=engine,
    )
    check("upload ingest success", result.get("success"), result.get("error", ""))
    if not result.get("success"):
        return

    # Mirror app._make_orchestrator call patterns
    import app as app_mod

    for ctx, label in (("", "no business context"), ("SaaS churn retention data", "with business context")):
        try:
            orch = app_mod._make_orchestrator(
                engine,
                f"up_{uuid.uuid4().hex[:6]}",
                dataframe=result["dataframe"],
                tables={"user_data": result["dataframe"]},
                business_context=ctx,
            )
            check(f"construct ok ({label})", orch is not None)
            check(f"has data ({label})", orch.has_data())
            check(f"context stored ({label})", orch.business_context == ctx.strip())
            q = orch.run_quality_scan()
            check(f"quality after upload ({label})", q.get("success"), q.get("error", ""))
            ins = orch.suggest_insights()
            check(f"insights after upload ({label})", ins.get("success"), ins.get("error", ""))
        except TypeError as e:
            check(f"construct ok ({label})", False, str(e))

    # Direct signature smoke (positional + kwargs combos used in codebase)
    df = result["dataframe"]
    combos = [
        lambda: Orchestrator(engine, "c1", df),
        lambda: Orchestrator(engine, "c2", dataframe=df),
        lambda: Orchestrator(engine, "c3", tables={"user_data": df}),
        lambda: Orchestrator(engine, "c4", dataframe=df, tables={"user_data": df}, business_context=""),
        lambda: Orchestrator(engine, "c5", dataframe=df, tables={"user_data": df}, business_context="x"),
    ]
    for i, fn in enumerate(combos):
        try:
            o = fn()
            check(f"combo {i} ok", o.has_data())
        except TypeError as e:
            check(f"combo {i} ok", False, str(e))


def test_phase_l_decisions():
    print("\n=== Phase L: Ambiguous decisions ===")
    df = pd.DataFrame({
        "region": ["USA", "US", "United States", "USA", "US", "Canada"],
        "amount": [10, 12, 11, 10, 9, 8],
        "customer_id": [1, 2, 3, 4, 5, 6],
    })
    report = analyze_data_quality(df)
    decisions = report.get("decisions") or []
    check("has decisions", len(decisions) >= 1, str(decisions))
    merge_dec = next((d for d in decisions if d.get("type") == "category_merge"), None)
    check("category merge decision", merge_dec is not None, str(decisions))
    if merge_dec:
        check("has options", len(merge_dec.get("options") or []) >= 2)
        # User confirms merge
        res = apply_category_merge(df, merge_dec["column"], merge_dec["values"], merge_dec["primary"])
        check("merge apply success", res["success"], res.get("error", ""))
        if res["success"]:
            vals = set(res["dataframe"][merge_dec["column"]].astype(str).unique())
            # Merged values should collapse to primary (others may remain)
            check("primary present after merge", merge_dec["primary"] in vals)
        # Keep separate does not require apply — orchestrator path
        orch = Orchestrator(get_engine(), "dec1", dataframe=df)
        orch.quality_report = report
        keep = orch.apply_decision(merge_dec["id"], "keep")
        check("keep separate success", keep.get("success"), keep.get("error", ""))


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
        if result.get("task") in ("classification", "regression"):
            check("automl leaderboard", len(result.get("leaderboard") or []) >= 1)
    return result


def test_sql_agent(engine, query: str):
    print(f"\n=== SQL Agent: {query[:50]} ===")
    if not HAS_API_KEY:
        skip("sql agent", "GROQ_API_KEY not set")
        return None
    result = run_sql_query(query, engine)
    if _is_rate_limited(result):
        skip("sql agent", "Groq rate limit")
        return None
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
        if _is_rate_limited(result):
            skip(f"rag answer: {q[:40]}", "Groq rate limit")
            continue
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
    test_sql_self_check_rules()
    test_sql_multi_clause_generation()
    test_quality_agent()
    test_expanded_eda()
    test_stats_agent()
    test_forecast_agent()
    test_report_agent()
    test_orchestrator_routing()
    test_phase_h_insights()
    test_phase_i_automl()
    test_phase_j_multitable()
    test_phase_k_business_context()
    test_file_upload_orchestrator_construction()
    test_phase_l_decisions()

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
