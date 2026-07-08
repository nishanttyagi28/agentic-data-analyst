"""Orchestrator — route user queries to specialized analyst agents."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from agents.forecast_agent import run_forecast
from agents.llm_client import chat_completion
from agents.ml_agent import run_ml_analysis
from agents.quality_agent import analyze_data_quality, apply_auto_clean, format_quality_markdown
from agents.rag_agent import RAGAgent
from agents.report_agent import generate_report
from agents.sql_agent import run_sql_query
from agents.stats_agent import run_stats_analysis


def classify_query_rule_based(query: str, has_data: bool, has_rag_index: bool) -> str | None:
    q = query.lower().strip()
    if not q:
        return None

    # Order matters: more specific routes before broad SQL/ML
    report_keywords = [
        "generate report", "generate a report", "download report", "export report",
        "create report", "create a report", "executive summary", "shareable report",
        "html report", "pdf report", "analysis report", "write a report",
    ]
    quality_keywords = [
        "data quality", "clean the data", "clean data", "auto-clean", "autoclean",
        "missing values", "duplicates", "duplicate rows", "data cleaning",
        "quality report", "fix data types", "impute",
    ]
    forecast_keywords = [
        "forecast", "predict next", "next month", "next quarter", "next week",
        "projection", "time series forecast", "future sales", "future revenue",
    ]
    stats_keywords = [
        "significant difference", "statistically", "t-test", "anova", "p-value",
        "p value", "hypothesis", "correlated with", "correlation with",
        "are there any outliers", "outlier", "difference between",
        "compare regions", "compare groups", "what's correlated",
        "what is correlated", "associated with",
    ]
    ml_keywords = [
        "train", "model", "classification", "regression",
        "cluster", "eda", "analyze", "analysis", "machine learning",
        "feature importance", "ml ", " ml", "predict churn", "predict price",
        "run eda", "exploratory",
    ]
    sql_keywords = [
        "how many", "count", "average", "sum", "total", "list", "show me",
        "top", "bottom", "filter", "where", "group by", "select", "query",
        "rows", "columns", "maximum", "minimum", "mean", "median",
        "how much", "which customers", "which houses",
    ]
    rag_keywords = [
        "what did", "tell me about", "explain the previous", "summary of findings",
        "finding", "earlier", "previous analysis", "why did", "what was",
        "recall", "remember", "follow up", "follow-up", "based on prior",
    ]

    if any(kw in q for kw in report_keywords):
        return "report"
    if any(kw in q for kw in quality_keywords):
        return "quality"
    if any(kw in q for kw in forecast_keywords):
        return "forecast"
    if any(kw in q for kw in stats_keywords):
        return "stats"
    # "predict" alone often means ML; "forecast" handled above
    if "predict" in q and not any(kw in q for kw in forecast_keywords):
        return "ml"
    if any(kw in q for kw in ml_keywords):
        return "ml"
    if any(kw in q for kw in sql_keywords) and has_data:
        return "sql"
    if any(kw in q for kw in rag_keywords) and has_rag_index:
        return "rag"
    return None


def classify_query_llm(query: str, has_data: bool, has_rag_index: bool) -> tuple[str | None, str | None]:
    system_prompt = """Classify the user query into exactly one category:
- SQL: questions about querying/filtering/aggregating the raw dataset
- ML: requests for EDA charts, modeling, classification, regression, clustering
- STATS: statistical tests (t-test, ANOVA), correlation rankings, outlier analysis
- FORECAST: time-series / future value forecasting
- QUALITY: data quality, cleaning, missing values, duplicates
- REPORT: generate/export a shareable analysis report
- RAG: follow-up questions about prior analysis results, insights, or reports
- GENERAL: greetings or unclear requests

Respond with JSON only: {"route": "SQL"|"ML"|"STATS"|"FORECAST"|"QUALITY"|"REPORT"|"RAG"|"GENERAL", "reason": "brief reason"}"""

    context = f"Data loaded: {has_data}. Prior analysis indexed: {has_rag_index}."
    response, err = chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nQuery: {query}"},
    ], temperature=0.0, max_tokens=256)

    if err or not response:
        return None, err

    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            route = data.get("route", "").upper()
            mapping = {
                "SQL": "sql",
                "ML": "ml",
                "STATS": "stats",
                "FORECAST": "forecast",
                "QUALITY": "quality",
                "REPORT": "report",
                "RAG": "rag",
                "GENERAL": "general",
            }
            return mapping.get(route, "general"), None
    except (json.JSONDecodeError, KeyError):
        pass
    return None, None


def classify_query(query: str, has_data: bool, has_rag_index: bool) -> tuple[str, str | None]:
    rule_result = classify_query_rule_based(query, has_data, has_rag_index)
    if rule_result:
        return rule_result, None

    llm_result, err = classify_query_llm(query, has_data, has_rag_index)
    if llm_result:
        return llm_result, err

    if has_rag_index:
        return "rag", err
    if has_data:
        return "sql", err
    return "general", err


class Orchestrator:
    def __init__(
        self,
        engine: Engine,
        session_id: str,
        dataframe: pd.DataFrame | None = None,
    ):
        self.engine = engine
        self.session_id = session_id
        self.dataframe = dataframe
        self.rag = RAGAgent(session_id)
        self.chat_history: list[dict[str, Any]] = []
        self.quality_report: dict[str, Any] | None = None
        self.session_findings: list[dict[str, Any]] = []

    def has_data(self) -> bool:
        return self.dataframe is not None and not self.dataframe.empty

    def has_rag_index(self) -> bool:
        return self.rag.get_stats()["chunk_count"] > 0

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.dataframe = df

    def index_result(self, question: str, result: dict[str, Any]) -> None:
        if not result.get("success"):
            return
        agent = result.get("agent", "")
        summary = (
            result.get("summary_for_rag")
            or result.get("summary")
            or result.get("explanation")
            or result.get("answer", "")
        )
        if summary:
            self.rag.index_exchange(question, summary, agent, extra=result.get("sql", ""))
        self.session_findings.append({
            "query": question,
            "result": result,
            "agent": agent,
        })

    def run_quality_scan(self) -> dict[str, Any]:
        if not self.has_data():
            return {"success": False, "error": "No data loaded", "agent": "quality"}
        report = analyze_data_quality(self.dataframe)
        self.quality_report = report
        return report

    def apply_cleaning(self) -> dict[str, Any]:
        if not self.has_data():
            return {"success": False, "error": "No data loaded", "agent": "quality"}
        result = apply_auto_clean(self.dataframe, engine=self.engine)
        if result.get("success") and result.get("dataframe") is not None:
            self.dataframe = result["dataframe"]
            self.quality_report = result.get("quality_report")
        return result

    def handle_query(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"success": False, "error": "Please enter a question.", "route": "none"}

        route, classify_err = classify_query(query, self.has_data(), self.has_rag_index())

        if route == "general":
            if not self.has_data():
                return {
                    "success": False,
                    "error": "Please upload a CSV file first, then ask questions about your data.",
                    "route": "general",
                }
            response, err = chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data analyst assistant. Users can: run SQL queries, "
                        "EDA/ML models, data quality checks, statistical tests, forecasts, and generate reports."
                    ),
                },
                {"role": "user", "content": query},
            ])
            result = {
                "success": True,
                "route": "general",
                "agent": "general",
                "answer": response or (
                    "I can help with SQL queries, data cleaning, EDA, stats tests, forecasting, "
                    "ML modeling, and downloadable reports."
                ),
                "classify_error": classify_err,
            }
        elif route == "sql":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before running SQL queries.", "route": "sql"}
            result = run_sql_query(query, self.engine)
            result["route"] = "sql"
            self.index_result(query, result)
        elif route == "ml":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before running ML analysis.", "route": "ml"}
            result = run_ml_analysis(self.dataframe, query)
            result["route"] = "ml"
            self.index_result(query, result)
        elif route == "stats":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before statistical analysis.", "route": "stats"}
            result = run_stats_analysis(self.dataframe, query)
            result["route"] = "stats"
            self.index_result(query, result)
        elif route == "forecast":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before forecasting.", "route": "forecast"}
            result = run_forecast(self.dataframe, query)
            result["route"] = "forecast"
            self.index_result(query, result)
        elif route == "quality":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before quality checks.", "route": "quality"}
            q_lower = query.lower()
            if any(w in q_lower for w in ("auto-clean", "autoclean", "clean the data", "apply clean", "fix data")):
                result = self.apply_cleaning()
                result["markdown"] = result.get("summary", "")
            else:
                result = self.run_quality_scan()
                result["markdown"] = format_quality_markdown(result)
                result["summary"] = result.get("markdown") or result.get("summary", "")
            result["route"] = "quality"
            self.index_result(query, result)
        elif route == "report":
            result = generate_report(
                self.dataframe,
                self.quality_report,
                self.session_findings or self.chat_history,
                use_llm=True,
            )
            result["route"] = "report"
            self.index_result(query, result)
        elif route == "rag":
            result = self.rag.answer(query)
            result["route"] = "rag"
        else:
            result = {"success": False, "error": f"Unknown route: {route}", "route": route}

        self.chat_history.append({
            "query": query,
            "route": route,
            "result": result,
        })
        return result
