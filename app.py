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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.engine is None:
        st.session_state.engine = get_engine()


def route_badge(route: str) -> str:
    css = f"route-{route}" if route in ("sql", "ml", "rag", "general") else "route-general"
    return f'<span class="route-badge {css}">{route.upper()}</span>'


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
                        st.session_state.last_upload_key = file_key
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
                        st.session_state.last_upload_key = sample_key
                        st.success(f"Loaded {result['row_count']} rows")
                    else:
                        st.error(result["error"])

        if st.session_state.data_loaded and st.session_state.ingestion_result:
            result = st.session_state.ingestion_result
            st.markdown("---")
            st.markdown("### 📋 Schema")
            st.markdown(f"**Table:** `{TABLE_NAME}`")
            st.markdown(f"**Rows:** {result['row_count']}")
            st.markdown(f"**Columns:** {result['column_count']}")
            with st.expander("Column details"):
                for col, dtype in result.get("dtypes", {}).items():
                    st.text(f"{col}: {dtype}")

            with st.expander("Data preview"):
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


def render_chat():
    st.markdown('<p class="main-header">Agentic Data Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your data in plain English — no SQL or ML expertise needed. '
        'Get instant answers, predictive insights, and a chat that remembers your analysis.</p>',
        unsafe_allow_html=True,
    )

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

                if msg.get("metrics"):
                    cols = st.columns(len(msg["metrics"]))
                    for i, (k, v) in enumerate(msg["metrics"].items()):
                        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                        cols[i].metric(k.replace("_", " ").title(), fmt)

                if msg.get("charts"):
                    with st.expander("Charts & EDA"):
                        for name, fig in msg["charts"].items():
                            if name == "distributions" and isinstance(fig, dict):
                                for col_name, col_fig in fig.items():
                                    st.plotly_chart(col_fig, use_container_width=True)
                            elif hasattr(fig, "update_layout"):
                                st.plotly_chart(fig, use_container_width=True)

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