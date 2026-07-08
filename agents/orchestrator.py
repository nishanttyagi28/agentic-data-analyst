"""Orchestrator — route user queries to specialized analyst agents."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from agents.forecast_agent import run_forecast
from agents.insight_agent import generate_insight_suggestions
from agents.llm_client import chat_completion
from agents.ml_agent import run_ml_analysis
from agents.multitable import detect_join_keys, primary_dataframe, register_dataframe
from agents.quality_agent import (
    analyze_data_quality,
    apply_auto_clean,
    apply_category_merge,
    format_quality_markdown,
)
from agents.rag_agent import RAGAgent
from agents.report_agent import generate_report
from agents.sql_agent import run_sql_query
from agents.stats_agent import run_stats_analysis
from db.database import TABLE_NAME


def classify_query_rule_based(query: str, has_data: bool, has_rag_index: bool) -> str | None:
    q = query.lower().strip()
    if not q:
        return None

    report_keywords = [
        "generate report", "generate a report", "download report", "export report",
        "create report", "create a report", "executive summary", "shareable report",
        "html report", "pdf report", "analysis report", "write a report",
    ]
    insight_keywords = [
        "suggest what to explore", "what should i explore", "what should i ask",
        "proactive insight", "suggest questions", "what to explore", "recommend analyses",
        "suggest analyses", "exploration ideas",
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
        "run eda", "exploratory", "automl",
    ]
    sql_keywords = [
        "how many", "count", "average", "sum", "total", "list", "show me",
        "top", "bottom", "filter", "where", "group by", "select", "query",
        "rows", "columns", "maximum", "minimum", "mean", "median",
        "how much", "which customers", "which houses", "join", "combine",
    ]
    rag_keywords = [
        "what did", "tell me about", "explain the previous", "summary of findings",
        "finding", "earlier", "previous analysis", "why did", "what was",
        "recall", "remember", "follow up", "follow-up", "based on prior",
    ]

    if any(kw in q for kw in report_keywords):
        return "report"
    if any(kw in q for kw in insight_keywords):
        return "insight"
    if any(kw in q for kw in quality_keywords):
        return "quality"
    if any(kw in q for kw in forecast_keywords):
        return "forecast"
    if any(kw in q for kw in stats_keywords):
        return "stats"
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
- SQL: questions about querying/filtering/aggregating/joining the raw dataset
- ML: requests for EDA charts, modeling, classification, regression, clustering, AutoML
- STATS: statistical tests (t-test, ANOVA), correlation rankings, outlier analysis
- FORECAST: time-series / future value forecasting
- QUALITY: data quality, cleaning, missing values, duplicates
- INSIGHT: suggest what to explore, proactive analysis ideas
- REPORT: generate/export a shareable analysis report
- RAG: follow-up questions about prior analysis results, insights, or reports
- GENERAL: greetings or unclear requests

Respond with JSON only: {"route": "SQL"|"ML"|"STATS"|"FORECAST"|"QUALITY"|"INSIGHT"|"REPORT"|"RAG"|"GENERAL", "reason": "brief reason"}"""

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
                "INSIGHT": "insight",
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
        tables: dict[str, pd.DataFrame] | None = None,
        business_context: str | None = "",
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        engine, session_id
            Required session wiring.
        dataframe
            Optional primary DataFrame (stored as user_data when tables empty).
        tables
            Optional multi-table registry {name: DataFrame}.
        business_context
            Optional free-text domain description (Phase K). Defaults to "".
        **kwargs
            Ignored for forward/backward compatibility with call sites.
        """
        # Allow legacy callers that only knew (engine, session_id, dataframe)
        if "business_context" in kwargs and not business_context:
            business_context = kwargs.get("business_context") or ""

        self.engine = engine
        self.session_id = session_id
        self.tables: dict[str, pd.DataFrame] = dict(tables or {})
        if dataframe is not None:
            if not self.tables:
                self.tables[TABLE_NAME] = dataframe
            elif TABLE_NAME not in self.tables:
                # Keep provided multi-table map; also expose primary under user_data if missing
                self.tables[TABLE_NAME] = dataframe
        self.dataframe = primary_dataframe(self.tables)
        self.business_context = (business_context or "").strip()
        self.rag = RAGAgent(session_id)
        self.chat_history: list[dict[str, Any]] = []
        self.quality_report: dict[str, Any] | None = None
        self.session_findings: list[dict[str, Any]] = []
        self.insight_suggestions: list[dict[str, Any]] = []
        self.exclude_ml_cols: set[str] = set()
        self.join_suggestions: list[dict[str, Any]] = []

    def has_data(self) -> bool:
        return self.dataframe is not None and not self.dataframe.empty

    def has_rag_index(self) -> bool:
        return self.rag.get_stats()["chunk_count"] > 0

    def set_business_context(self, text: str) -> None:
        self.business_context = (text or "").strip()

    def set_dataframe(self, df: pd.DataFrame, table_name: str = TABLE_NAME) -> None:
        self.tables[table_name] = df
        self.dataframe = primary_dataframe(self.tables)

    def add_table(self, df: pd.DataFrame, table_name: str) -> dict[str, Any]:
        result = register_dataframe(df, self.engine, table_name, self.tables)
        if result.get("success"):
            self.dataframe = primary_dataframe(self.tables)
            self.join_suggestions = detect_join_keys(self.tables)
        return result

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
        primary = "user_data" if "user_data" in self.tables else next(iter(self.tables.keys()))
        result = apply_auto_clean(self.dataframe, engine=self.engine, table_name=primary)
        if result.get("success") and result.get("dataframe") is not None:
            self.set_dataframe(result["dataframe"], primary)
            self.quality_report = result.get("quality_report")
        return result

    def apply_decision(self, decision_id: str, choice: str) -> dict[str, Any]:
        """Apply a user-confirmed quality decision (never silent)."""
        if not self.has_data() or not self.quality_report:
            return {"success": False, "error": "No quality decisions available", "agent": "quality"}
        decisions = self.quality_report.get("decisions") or []
        dec = next((d for d in decisions if d.get("id") == decision_id), None)
        if not dec:
            return {"success": False, "error": f"Unknown decision {decision_id}", "agent": "quality"}

        primary = "user_data" if "user_data" in self.tables else next(iter(self.tables.keys()))

        if dec["type"] == "category_merge":
            if choice == "merge":
                result = apply_category_merge(
                    self.dataframe,
                    dec["column"],
                    dec["values"],
                    dec["primary"],
                    engine=self.engine,
                    table_name=primary,
                )
                if result.get("success"):
                    self.set_dataframe(result["dataframe"], primary)
                    self.quality_report = result.get("quality_report")
                return result
            if choice == "keep":
                return {
                    "success": True,
                    "agent": "quality",
                    "summary": f"Kept categories separate in `{dec['column']}`: {dec['values']}",
                    "summary_for_rag": f"User chose to keep separate: {dec}",
                }
            if choice == "show":
                vc = self.dataframe[dec["column"]].astype(str).value_counts().head(20).to_dict()
                return {
                    "success": True,
                    "agent": "quality",
                    "summary": f"Value counts for `{dec['column']}`:\n" + "\n".join(f"- {k}: {v}" for k, v in vc.items()),
                }

        if dec["type"] == "id_or_feature":
            col = dec["column"]
            if choice == "exclude_ml":
                self.exclude_ml_cols.add(col)
                return {
                    "success": True,
                    "agent": "quality",
                    "summary": f"`{col}` will be excluded from ML features (treated as ID).",
                    "summary_for_rag": f"Excluded {col} from ML features",
                }
            if choice == "keep_feature":
                self.exclude_ml_cols.discard(col)
                return {
                    "success": True,
                    "agent": "quality",
                    "summary": f"`{col}` will be kept as an ML feature.",
                }
            if choice == "show":
                sample = self.dataframe[col].dropna().head(15).tolist()
                return {
                    "success": True,
                    "agent": "quality",
                    "summary": f"Sample values for `{col}`: {sample}",
                }

        return {"success": False, "error": f"Unhandled choice {choice}", "agent": "quality"}

    def suggest_insights(self) -> dict[str, Any]:
        if not self.has_data():
            return {"success": False, "error": "No data loaded", "agent": "insight"}
        result = generate_insight_suggestions(
            self.dataframe,
            business_context=self.business_context,
            table_name="user_data" if "user_data" in self.tables else next(iter(self.tables.keys()), "data"),
        )
        if result.get("success"):
            self.insight_suggestions = result.get("suggestions") or []
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
            ctx = self.business_context
            response, err = chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data analyst assistant. Users can: run SQL queries, "
                        "EDA/ML models, data quality checks, statistical tests, forecasts, "
                        "proactive insight suggestions, multi-table joins, and generate reports."
                        + (f" Business context: {ctx}" if ctx else "")
                    ),
                },
                {"role": "user", "content": query},
            ])
            result = {
                "success": True,
                "route": "general",
                "agent": "general",
                "answer": response or (
                    "I can help with SQL, cleaning, EDA, stats, forecasts, AutoML, insights, and reports."
                ),
                "classify_error": classify_err,
            }
        elif route == "sql":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before running SQL queries.", "route": "sql"}
            result = run_sql_query(query, self.engine, tables=self.tables)
            result["route"] = "sql"
            self.index_result(query, result)
        elif route == "ml":
            if not self.has_data():
                return {"success": False, "error": "Upload a CSV first before running ML analysis.", "route": "ml"}
            result = run_ml_analysis(
                self.dataframe,
                query,
                business_context=self.business_context,
                exclude_feature_cols=self.exclude_ml_cols,
            )
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
        elif route == "insight":
            if not self.has_data():
                return {"success": False, "error": "Upload data first.", "route": "insight"}
            result = self.suggest_insights()
            result["route"] = "insight"
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
