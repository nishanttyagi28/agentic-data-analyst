"""Orchestrator — route user queries to SQL, ML, or RAG agents."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from agents.llm_client import chat_completion
from agents.ml_agent import run_ml_analysis
from agents.rag_agent import RAGAgent
from agents.sql_agent import run_sql_query


def classify_query_rule_based(query: str, has_data: bool, has_rag_index: bool) -> str | None:
    q = query.lower().strip()
    if not q:
        return None

    ml_keywords = [
        "train", "model", "predict", "classification", "regression",
        "cluster", "eda", "analyze", "analysis", "machine learning",
        "correlation", "distribution", "feature importance", "ml",
    ]
    sql_keywords = [
        "how many", "count", "average", "sum", "total", "list", "show",
        "top", "bottom", "filter", "where", "group by", "select", "query",
        "rows", "columns", "maximum", "minimum", "mean", "median",
    ]
    rag_keywords = [
        "what did", "tell me about", "explain", "summary", "insight",
        "finding", "earlier", "previous", "report", "why", "what was",
        "recall", "remember", "follow up", "follow-up",
    ]

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
- ML: requests for analysis, EDA, modeling, predictions, clustering
- RAG: follow-up questions about prior analysis results, insights, or reports
- GENERAL: greetings or unclear requests

Respond with JSON only: {"route": "SQL"|"ML"|"RAG"|"GENERAL", "reason": "brief reason"}"""

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
            mapping = {"SQL": "sql", "ML": "ml", "RAG": "rag", "GENERAL": "general"}
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

    def has_data(self) -> bool:
        return self.dataframe is not None and not self.dataframe.empty

    def has_rag_index(self) -> bool:
        return self.rag.get_stats()["chunk_count"] > 0

    def index_result(self, question: str, result: dict[str, Any]) -> None:
        if not result.get("success"):
            return
        agent = result.get("agent", "")
        summary = result.get("summary_for_rag") or result.get("summary") or result.get("explanation") or result.get("answer", "")
        if summary:
            self.rag.index_exchange(question, summary, agent, extra=result.get("sql", ""))

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
                {"role": "system", "content": "You are a helpful data analyst assistant. Guide the user on what they can do."},
                {"role": "user", "content": query},
            ])
            result = {
                "success": True,
                "route": "general",
                "agent": "general",
                "answer": response or "I can help you query data (SQL), run ML analysis, or answer follow-up questions about prior results.",
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