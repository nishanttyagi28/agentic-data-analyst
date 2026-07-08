"""Agentic Data Analyst — Streamlit entrypoint."""

from __future__ import annotations

# Load .env from project root before any agent imports
from utils.env import load_project_env

load_project_env()

import hashlib
import os
import uuid

import pandas as pd
import streamlit as st

from agents.ingestion import ingest_csv
from agents.llm_client import get_groq_client
from agents.orchestrator import Orchestrator
from agents.quality_agent import format_quality_markdown
from agents.report_agent import generate_report
from db.database import TABLE_NAME, get_engine

st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #666; margin-bottom: 1.5rem; }
    .route-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .route-sql { background: #E3F2FD; color: #1565C0; }
    .route-ml { background: #E8F5E9; color: #2E7D32; }
    .route-rag { background: #FFF3E0; color: #E65100; }
    .route-general { background: #F3E5F5; color: #7B1FA2; }
    .route-quality { background: #E0F7FA; color: #006064; }
    .route-stats { background: #FCE4EC; color: #880E4F; }
    .route-forecast { background: #E8EAF6; color: #283593; }
    .route-report { background: #FFF8E1; color: #F57F17; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "engine": None,
        "dataframe": None,
        "orchestrator": None,
        "ingestion_result": None,
        "chat_messages": [],
        "data_loaded": False,
        "quality_report": None,
        "quality_dismissed": False,
        "last_report_html": None,
        "cleaning_log": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.engine is None:
        st.session_state.engine = get_engine()


def route_badge(route: str) -> str:
    known = ("sql", "ml", "rag", "general", "quality", "stats", "forecast", "report")
    css = f"route-{route}" if route in known else "route-general"
    return f'<span class="route-badge {css}">{route.upper()}</span>'


def _load_dataframe_into_session(result: dict, upload_key: str) -> None:
    st.session_state.ingestion_result = result
    st.session_state.dataframe = result["dataframe"]
    st.session_state.data_loaded = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.orchestrator = Orchestrator(
        st.session_state.engine,
        st.session_state.session_id,
        st.session_state.dataframe,
    )
    st.session_state.chat_messages = []
    st.session_state.last_upload_key = upload_key
    st.session_state.quality_dismissed = False
    st.session_state.cleaning_log = None
    st.session_state.last_report_html = None
    # Auto quality scan after ingestion (helpful gate, not a blocker)
    q_report = st.session_state.orchestrator.run_quality_scan()
    st.session_state.quality_report = q_report
    st.session_state.orchestrator.quality_report = q_report


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

        st.markdown("**Or try a sample dataset:**")
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".csv")] if os.path.isdir(sample_dir) else []
        selected_sample = st.selectbox("Sample datasets", ["—"] + samples, label_visibility="collapsed")

        if uploaded is not None:
            file_key = hashlib.md5(uploaded.getvalue()).hexdigest()
            if st.session_state.get("last_upload_key") != file_key:
                with st.spinner("Ingesting data..."):
                    result = ingest_csv(file_bytes=uploaded.getvalue(), engine=st.session_state.engine)
                    if result["success"]:
                        _load_dataframe_into_session(result, file_key)
                        st.success(f"Loaded {result['row_count']} rows, {result['column_count']} columns")
                    else:
                        st.error(result["error"])

        elif selected_sample != "—":
            sample_path = os.path.join(sample_dir, selected_sample)
            sample_key = f"sample_{selected_sample}"
            if st.session_state.get("last_upload_key") != sample_key:
                with st.spinner(f"Loading {selected_sample}..."):
                    result = ingest_csv(file_path=sample_path, engine=st.session_state.engine)
                    if result["success"]:
                        _load_dataframe_into_session(result, sample_key)
                        st.success(f"Loaded {result['row_count']} rows")
                    else:
                        st.error(result["error"])

        st.markdown("---")
        st.markdown("### 🧭 What I can help with")
        st.markdown(
            """
- **SQL queries** — counts, filters, rankings
- **Data cleaning** — missing values, duplicates, types
- **EDA** — stats, correlations, distributions
- **Stats tests** — t-test, ANOVA, correlations, outliers
- **Forecasting** — trend estimates with uncertainty
- **ML modeling** — classification, regression, clustering
- **Reports** — downloadable HTML summary
- **Follow-ups** — RAG chat on prior findings
            """.strip()
        )

        if st.session_state.data_loaded and st.session_state.ingestion_result:
            result = st.session_state.ingestion_result
            st.markdown("---")
            st.markdown("### 📋 Schema")
            st.markdown(f"**Table:** `{TABLE_NAME}`")
            st.markdown(f"**Rows:** {len(st.session_state.dataframe) if st.session_state.dataframe is not None else result['row_count']}")
            st.markdown(f"**Columns:** {result['column_count']}")
            if st.session_state.quality_report and st.session_state.quality_report.get("quality_score") is not None:
                st.markdown(f"**Quality score:** {st.session_state.quality_report['quality_score']}/100")
            with st.expander("Column details"):
                dtypes = (
                    {c: str(st.session_state.dataframe[c].dtype) for c in st.session_state.dataframe.columns}
                    if st.session_state.dataframe is not None
                    else result.get("dtypes", {})
                )
                for col, dtype in dtypes.items():
                    st.text(f"{col}: {dtype}")

            with st.expander("Data preview"):
                if st.session_state.dataframe is not None:
                    st.dataframe(st.session_state.dataframe.head(10), use_container_width=True)
                else:
                    st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True)

        if st.session_state.orchestrator:
            stats = st.session_state.orchestrator.rag.get_stats()
            st.markdown("---")
            st.markdown(f"**RAG chunks:** {stats['chunk_count']}")

        st.markdown("---")
        client, api_err = get_groq_client()
        if api_err:
            st.warning(f"⚠️ {api_err}")
        else:
            st.success("✅ Groq API connected")


def render_quality_gate():
    """Show data quality card after load; skip-able, not a hard blocker."""
    if not st.session_state.data_loaded:
        return
    if st.session_state.quality_dismissed:
        return
    report = st.session_state.quality_report
    if not report or not report.get("success"):
        return

    with st.container():
        st.markdown("### 🧹 Data Quality Report")
        st.caption("Review data health before analysis. You can auto-clean with safe defaults or skip and proceed.")
        score = report.get("quality_score", "n/a")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quality score", f"{score}/100")
        c2.metric("Missing cols", len(report.get("missing") or {}))
        c3.metric("Duplicates", report.get("duplicate_count", 0))
        c4.metric("Outlier cols", len(report.get("outliers") or {}))

        with st.expander("Full quality details", expanded=score is not None and float(score) < 90):
            st.markdown(format_quality_markdown(report))

        b1, b2, b3 = st.columns([2, 2, 2])
        with b1:
            if st.button("✨ Auto-clean with defaults", use_container_width=True, type="primary"):
                with st.spinner("Applying safe defaults (median/mode impute, drop exact duplicates)..."):
                    result = st.session_state.orchestrator.apply_cleaning()
                if result.get("success"):
                    st.session_state.dataframe = result["dataframe"]
                    st.session_state.quality_report = result.get("quality_report")
                    st.session_state.orchestrator.set_dataframe(result["dataframe"])
                    st.session_state.cleaning_log = result.get("actions_log") or []
                    st.session_state.quality_dismissed = True
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "route": "quality",
                        "content": result.get("summary", "Auto-clean applied."),
                    })
                    st.success("Data cleaned. You can continue chatting.")
                    st.rerun()
                else:
                    st.error(result.get("error", "Cleaning failed"))
        with b2:
            if st.button("Skip cleaning — analyze as-is", use_container_width=True):
                st.session_state.quality_dismissed = True
                st.rerun()
        with b3:
            st.caption("Outliers & category merges are never auto-applied.")


def render_report_toolbar():
    if not st.session_state.data_loaded or not st.session_state.orchestrator:
        return
    st.markdown("---")
    cols = st.columns([2, 3, 3])
    with cols[0]:
        if st.button("📄 Generate Report", use_container_width=True):
            with st.spinner("Compiling session report..."):
                report = generate_report(
                    st.session_state.dataframe,
                    st.session_state.quality_report,
                    st.session_state.orchestrator.session_findings
                    or st.session_state.orchestrator.chat_history,
                    use_llm=True,
                )
            if report.get("success"):
                st.session_state.last_report_html = report["html"]
                st.session_state.orchestrator.index_result("Generate report", report)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "route": "report",
                    "content": "**Executive Summary**\n\n" + report.get("executive_summary", ""),
                    "report_html": report["html"],
                })
                st.rerun()
            else:
                st.error(report.get("error", "Report generation failed"))
    with cols[1]:
        if st.session_state.last_report_html:
            st.download_button(
                "⬇️ Download HTML report",
                data=st.session_state.last_report_html,
                file_name="analysis_report.html",
                mime="text/html",
                use_container_width=True,
            )


def _render_charts(charts: dict):
    if not charts:
        return
    with st.expander("Charts & visuals", expanded=True):
        for name, fig in charts.items():
            if name in ("distributions", "groupbys", "timeseries") and isinstance(fig, dict):
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                for col_name, col_fig in fig.items():
                    if hasattr(col_fig, "update_layout"):
                        st.plotly_chart(col_fig, use_container_width=True)
            elif hasattr(fig, "update_layout"):
                st.plotly_chart(fig, use_container_width=True)


def render_chat():
    st.markdown('<p class="main-header">Agentic Data Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Your AI data analyst — clean data, explore, test, forecast, model, and export reports '
        'in plain English.</p>',
        unsafe_allow_html=True,
    )

    render_quality_gate()
    render_report_toolbar()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                route = msg.get("route", "")
                if route:
                    st.markdown(route_badge(route), unsafe_allow_html=True)
                st.markdown(msg["content"])

                if msg.get("sql"):
                    with st.expander("Generated SQL"):
                        st.code(msg["sql"], language="sql")

                if msg.get("result_df") is not None:
                    st.dataframe(msg["result_df"], use_container_width=True)

                if msg.get("describe_table") is not None and not getattr(msg["describe_table"], "empty", True):
                    with st.expander("Descriptive statistics"):
                        st.dataframe(msg["describe_table"], use_container_width=True)

                if msg.get("metrics"):
                    cols = st.columns(min(len(msg["metrics"]), 4))
                    for i, (k, v) in enumerate(msg["metrics"].items()):
                        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                        cols[i % len(cols)].metric(k.replace("_", " ").title(), fmt)

                if msg.get("forecast_table"):
                    st.dataframe(pd.DataFrame(msg["forecast_table"]), use_container_width=True)

                if msg.get("charts"):
                    _render_charts(msg["charts"])

                if msg.get("report_html"):
                    st.download_button(
                        "⬇️ Download this report",
                        data=msg["report_html"],
                        file_name="analysis_report.html",
                        mime="text/html",
                        key=f"dl_{hash(msg['report_html'][:80])}",
                    )

                if msg.get("citations"):
                    with st.expander("Sources cited"):
                        for cite in msg["citations"]:
                            st.markdown(f"**Source {cite['index']}** ({cite['source_type']})")
                            st.caption(cite["excerpt"])

    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if not st.session_state.orchestrator:
            assistant_msg = {
                "role": "assistant",
                "content": "Please upload a CSV file or select a sample dataset from the sidebar first.",
                "route": "general",
            }
            st.session_state.chat_messages.append(assistant_msg)
            st.rerun()

        with st.spinner("Thinking..."):
            result = st.session_state.orchestrator.handle_query(prompt)

        # Keep quality report / dataframe in sync after clean-via-chat
        if result.get("route") == "quality" and result.get("dataframe") is not None:
            st.session_state.dataframe = result["dataframe"]
            st.session_state.quality_report = result.get("quality_report") or st.session_state.quality_report
        if result.get("route") == "quality" and result.get("quality_report"):
            st.session_state.quality_report = result["quality_report"]
        if result.get("route") == "report" and result.get("html"):
            st.session_state.last_report_html = result["html"]

        assistant_msg: dict = {"role": "assistant", "route": result.get("route", "")}

        if not result.get("success"):
            assistant_msg["content"] = f"❌ {result.get('error', 'An error occurred')}"
        elif result.get("route") == "sql":
            assistant_msg["content"] = result.get("explanation", "Query executed.")
            assistant_msg["sql"] = result.get("sql")
            if result.get("result") is not None:
                assistant_msg["result_df"] = result["result"]
        elif result.get("route") == "ml":
            assistant_msg["content"] = result.get("summary", "Analysis complete.")
            assistant_msg["metrics"] = result.get("metrics")
            assistant_msg["charts"] = result.get("charts")
            eda = result.get("eda") or {}
            if eda.get("describe_table") is not None:
                assistant_msg["describe_table"] = eda["describe_table"]
        elif result.get("route") == "stats":
            assistant_msg["content"] = result.get("summary") or result.get("interpretation", "Stats complete.")
            if result.get("rankings"):
                assistant_msg["result_df"] = pd.DataFrame(result["rankings"])
            if result.get("group_means"):
                assistant_msg["metrics"] = {
                    **{f"mean_{k}": v for k, v in list(result["group_means"].items())[:3]},
                    "p_value": result.get("p_value"),
                }
        elif result.get("route") == "forecast":
            assistant_msg["content"] = result.get("summary", "Forecast complete.")
            assistant_msg["charts"] = result.get("charts")
            assistant_msg["forecast_table"] = result.get("forecast_table")
        elif result.get("route") == "quality":
            assistant_msg["content"] = result.get("markdown") or result.get("summary", "Quality check complete.")
            if result.get("quality_report"):
                st.session_state.quality_report = result["quality_report"]
        elif result.get("route") == "report":
            assistant_msg["content"] = "**Executive Summary**\n\n" + result.get("executive_summary", result.get("summary", ""))
            assistant_msg["report_html"] = result.get("html")
        elif result.get("route") == "rag":
            assistant_msg["content"] = result.get("answer", "")
            assistant_msg["citations"] = result.get("citations")
        else:
            assistant_msg["content"] = result.get("answer", "")

        st.session_state.chat_messages.append(assistant_msg)
        st.rerun()


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
