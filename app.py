"""Agentic Data Analyst — Streamlit entrypoint."""

from __future__ import annotations

from utils.env import load_project_env

load_project_env()

import hashlib
import os
import uuid

import pandas as pd
import streamlit as st

from agents.ingestion import ingest_csv
from agents.llm_client import get_groq_client
from agents.multitable import detect_join_keys, register_dataframe, sanitize_table_name
from agents.orchestrator import Orchestrator
from agents.quality_agent import format_quality_markdown
from agents.report_agent import generate_report
from db.database import TABLE_NAME, get_engine, load_dataframe_to_table

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
    .route-insight { background: #E0F2F1; color: #004D40; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "engine": None,
        "dataframe": None,
        "tables": {},
        "orchestrator": None,
        "ingestion_result": None,
        "chat_messages": [],
        "data_loaded": False,
        "quality_report": None,
        "quality_dismissed": False,
        "last_report_html": None,
        "cleaning_log": None,
        "business_context": "",
        "insight_suggestions": [],
        "pending_suggestion": None,
        "join_suggestions": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.engine is None:
        st.session_state.engine = get_engine()


def route_badge(route: str) -> str:
    known = ("sql", "ml", "rag", "general", "quality", "stats", "forecast", "report", "insight")
    css = f"route-{route}" if route in known else "route-general"
    return f'<span class="route-badge {css}">{route.upper()}</span>'


def _sync_orch_context():
    if st.session_state.orchestrator:
        st.session_state.orchestrator.set_business_context(st.session_state.business_context)


def _make_orchestrator(engine, session_id, dataframe=None, tables=None, business_context=""):
    """
    Construct Orchestrator matching agents.orchestrator.Orchestrator.__init__.
    Tries full kwargs first, then degrades for partial deploys, then always
    applies business_context via setter/attribute so upload never crashes.
    """
    ctx = business_context if business_context is not None else ""
    attempts = [
        dict(dataframe=dataframe, tables=tables, business_context=ctx),
        dict(dataframe=dataframe, tables=tables),
        dict(dataframe=dataframe),
        {},
    ]
    orch = None
    last_err = None
    for kwargs in attempts:
        try:
            orch = Orchestrator(engine, session_id, **kwargs)
            break
        except TypeError as e:
            last_err = e
            continue
    if orch is None:
        raise TypeError(f"Failed to construct Orchestrator: {last_err}")

    # Ensure multi-table registry if constructor ignored tables=
    if tables and (not getattr(orch, "tables", None)):
        orch.tables = dict(tables)
        if hasattr(orch, "set_dataframe") and dataframe is not None:
            orch.set_dataframe(dataframe)
        elif dataframe is not None:
            orch.dataframe = dataframe
    elif dataframe is not None and getattr(orch, "dataframe", None) is None:
        if hasattr(orch, "set_dataframe"):
            orch.set_dataframe(dataframe)
        else:
            orch.dataframe = dataframe

    if hasattr(orch, "set_business_context"):
        orch.set_business_context(ctx)
    else:
        orch.business_context = ctx
    return orch


def _load_primary_into_session(result: dict, upload_key: str, replace: bool = True) -> None:
    st.session_state.ingestion_result = result
    st.session_state.dataframe = result["dataframe"]
    st.session_state.data_loaded = True
    if replace:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.tables = {TABLE_NAME: result["dataframe"]}
        st.session_state.chat_messages = []
        st.session_state.insight_suggestions = []
    else:
        st.session_state.tables[TABLE_NAME] = result["dataframe"]

    st.session_state.orchestrator = _make_orchestrator(
        st.session_state.engine,
        st.session_state.session_id,
        dataframe=st.session_state.dataframe,
        tables=st.session_state.tables,
        business_context=st.session_state.get("business_context") or "",
    )
    st.session_state.last_upload_key = upload_key
    st.session_state.quality_dismissed = False
    st.session_state.cleaning_log = None
    st.session_state.last_report_html = None
    q_report = st.session_state.orchestrator.run_quality_scan()
    st.session_state.quality_report = q_report
    st.session_state.orchestrator.quality_report = q_report
    # Proactive insights after load (no LLM)
    ins = st.session_state.orchestrator.suggest_insights()
    st.session_state.insight_suggestions = ins.get("suggestions") or []
    st.session_state.join_suggestions = st.session_state.orchestrator.join_suggestions or detect_join_keys(
        st.session_state.tables
    )


def _add_extra_table(df: pd.DataFrame, name: str) -> dict:
    if not st.session_state.orchestrator:
        st.session_state.tables = {}
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.orchestrator = _make_orchestrator(
            st.session_state.engine,
            st.session_state.session_id,
            dataframe=None,
            tables={},
            business_context=st.session_state.get("business_context") or "",
        )
    result = st.session_state.orchestrator.add_table(df, name)
    if result.get("success"):
        st.session_state.tables = st.session_state.orchestrator.tables
        st.session_state.dataframe = st.session_state.orchestrator.dataframe
        st.session_state.data_loaded = True
        st.session_state.join_suggestions = st.session_state.orchestrator.join_suggestions
        if st.session_state.dataframe is not None:
            q = st.session_state.orchestrator.run_quality_scan()
            st.session_state.quality_report = q
    return result


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)

        st.markdown("**Or try a sample dataset:**")
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".csv")] if os.path.isdir(sample_dir) else []
        selected_sample = st.selectbox("Sample datasets", ["—"] + samples, label_visibility="collapsed")

        if uploaded:
            # Stable key from all file names + sizes
            key_src = "|".join(f"{f.name}:{len(f.getvalue())}" for f in uploaded)
            file_key = hashlib.md5(key_src.encode()).hexdigest()
            if st.session_state.get("last_upload_key") != file_key:
                with st.spinner("Ingesting data..."):
                    if len(uploaded) == 1:
                        result = ingest_csv(file_bytes=uploaded[0].getvalue(), engine=st.session_state.engine)
                        if result["success"]:
                            _load_primary_into_session(result, file_key, replace=True)
                            st.success(f"Loaded {result['row_count']} rows as `{TABLE_NAME}`")
                        else:
                            st.error(result["error"])
                    else:
                        # First file -> user_data, rest -> named tables
                        first = uploaded[0]
                        result = ingest_csv(file_bytes=first.getvalue(), engine=st.session_state.engine)
                        if not result["success"]:
                            st.error(result["error"])
                        else:
                            _load_primary_into_session(result, file_key, replace=True)
                            for f in uploaded[1:]:
                                tname = sanitize_table_name(os.path.splitext(f.name)[0])
                                from io import BytesIO
                                df = pd.read_csv(BytesIO(f.getvalue()))
                                r2 = _add_extra_table(df, tname)
                                if r2.get("success"):
                                    st.success(f"Loaded `{r2['table_name']}` ({r2['row_count']} rows)")
                                else:
                                    st.error(r2.get("error", "Failed to load table"))

        elif selected_sample != "—":
            sample_path = os.path.join(sample_dir, selected_sample)
            sample_key = f"sample_{selected_sample}"
            if st.session_state.get("last_upload_key") != sample_key:
                with st.spinner(f"Loading {selected_sample}..."):
                    result = ingest_csv(file_path=sample_path, engine=st.session_state.engine)
                    if result["success"]:
                        _load_primary_into_session(result, sample_key, replace=True)
                        st.success(f"Loaded {result['row_count']} rows")
                    else:
                        st.error(result["error"])

        st.markdown("---")
        st.markdown("### 🏷️ Business context (optional)")
        ctx = st.text_area(
            "What's this data about?",
            value=st.session_state.business_context,
            placeholder="e.g. Football match-level data for a striker across competitions…",
            height=80,
            label_visibility="collapsed",
        )
        if ctx != st.session_state.business_context:
            st.session_state.business_context = ctx
            _sync_orch_context()

        st.markdown("---")
        st.markdown("### 🧭 What I can help with")
        st.markdown(
            """
- **SQL** — queries & multi-table joins
- **Cleaning** — quality + your decisions
- **Insights** — suggested questions
- **EDA / AutoML** — multi-model compare
- **Stats & forecasts**
- **Reports** — downloadable HTML
- **Follow-ups** — RAG on prior findings
            """.strip()
        )

        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📋 Tables")
            for tname, tdf in (st.session_state.tables or {}).items():
                st.markdown(f"- **`{tname}`**: {len(tdf)} rows × {len(tdf.columns)} cols")
            if st.session_state.join_suggestions:
                with st.expander("Likely join keys"):
                    for j in st.session_state.join_suggestions:
                        st.caption(j.get("message", str(j)))

            if st.session_state.ingestion_result:
                result = st.session_state.ingestion_result
                st.markdown(f"**Primary preview table:** `{TABLE_NAME}`")
                if st.session_state.quality_report and st.session_state.quality_report.get("quality_score") is not None:
                    st.markdown(f"**Quality score:** {st.session_state.quality_report['quality_score']}/100")
                with st.expander("Column details"):
                    if st.session_state.dataframe is not None:
                        for col in st.session_state.dataframe.columns:
                            st.text(f"{col}: {st.session_state.dataframe[col].dtype}")
                with st.expander("Data preview"):
                    if st.session_state.dataframe is not None:
                        st.dataframe(st.session_state.dataframe.head(10), use_container_width=True)

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
    if not st.session_state.data_loaded:
        return
    if st.session_state.quality_dismissed:
        return
    report = st.session_state.quality_report
    if not report or not report.get("success"):
        return

    with st.container():
        st.markdown("### 🧹 Data Quality Report")
        st.caption("Review data health. Auto-clean is optional; ambiguous merges need your decision.")
        score = report.get("quality_score", "n/a")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quality score", f"{score}/100")
        c2.metric("Missing cols", len(report.get("missing") or {}))
        c3.metric("Duplicates", report.get("duplicate_count", 0))
        c4.metric("Decisions", len(report.get("decisions") or {}))

        with st.expander("Full quality details", expanded=score is not None and float(score) < 90):
            st.markdown(format_quality_markdown(report))

        # Phase L — concrete decisions
        decisions = report.get("decisions") or []
        if decisions:
            st.markdown("#### Decisions needed (nothing auto-applied)")
            for dec in decisions[:6]:
                st.markdown(f"**{dec['prompt']}**")
                cols = st.columns(len(dec.get("options") or []) or 1)
                for i, opt in enumerate(dec.get("options") or []):
                    if cols[i].button(opt["label"], key=f"dec_{dec['id']}_{opt['id']}"):
                        with st.spinner("Applying your choice..."):
                            res = st.session_state.orchestrator.apply_decision(dec["id"], opt["id"])
                        if res.get("success"):
                            st.session_state.dataframe = st.session_state.orchestrator.dataframe
                            st.session_state.tables = st.session_state.orchestrator.tables
                            if res.get("quality_report"):
                                st.session_state.quality_report = res["quality_report"]
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "route": "quality",
                                "content": res.get("summary", "Decision recorded."),
                            })
                            st.rerun()
                        else:
                            st.error(res.get("error", "Failed"))

        b1, b2, b3 = st.columns([2, 2, 2])
        with b1:
            if st.button("✨ Auto-clean with defaults", use_container_width=True, type="primary"):
                with st.spinner("Applying safe defaults..."):
                    result = st.session_state.orchestrator.apply_cleaning()
                if result.get("success"):
                    st.session_state.dataframe = result["dataframe"]
                    st.session_state.tables = st.session_state.orchestrator.tables
                    st.session_state.quality_report = result.get("quality_report")
                    st.session_state.quality_dismissed = True
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "route": "quality",
                        "content": result.get("summary", "Auto-clean applied."),
                    })
                    # Refresh insights after clean
                    ins = st.session_state.orchestrator.suggest_insights()
                    st.session_state.insight_suggestions = ins.get("suggestions") or []
                    st.rerun()
                else:
                    st.error(result.get("error", "Cleaning failed"))
        with b2:
            if st.button("Skip cleaning — analyze as-is", use_container_width=True):
                st.session_state.quality_dismissed = True
                st.rerun()
        with b3:
            st.caption("Category merges only apply when you click an option above.")


def render_insights():
    if not st.session_state.data_loaded:
        return
    st.markdown("### 💡 Suggest what to explore")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Refresh suggestions", use_container_width=True):
            with st.spinner("Scanning dataset..."):
                _sync_orch_context()
                ins = st.session_state.orchestrator.suggest_insights()
            st.session_state.insight_suggestions = ins.get("suggestions") or []
            st.rerun()
    suggestions = st.session_state.insight_suggestions or []
    if not suggestions and st.session_state.orchestrator:
        ins = st.session_state.orchestrator.suggest_insights()
        suggestions = ins.get("suggestions") or []
        st.session_state.insight_suggestions = suggestions

    if suggestions:
        st.caption("Click a suggestion to run it through the normal agent pipeline.")
        for s in suggestions:
            if st.button(s["label"], key=f"sug_{s['id']}", use_container_width=True):
                st.session_state.pending_suggestion = s["question"]
                st.rerun()
    else:
        st.caption("No suggestions yet — load data or click Refresh.")


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


def _handle_prompt(prompt: str):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    if not st.session_state.orchestrator:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "Please upload a CSV file or select a sample dataset from the sidebar first.",
            "route": "general",
        })
        st.rerun()
        return

    _sync_orch_context()
    with st.spinner("Thinking..."):
        result = st.session_state.orchestrator.handle_query(prompt)

    if result.get("route") == "quality" and result.get("dataframe") is not None:
        st.session_state.dataframe = result["dataframe"]
        st.session_state.tables = st.session_state.orchestrator.tables
        st.session_state.quality_report = result.get("quality_report") or st.session_state.quality_report
    if result.get("route") == "quality" and result.get("quality_report"):
        st.session_state.quality_report = result["quality_report"]
    if result.get("route") == "report" and result.get("html"):
        st.session_state.last_report_html = result["html"]
    if result.get("route") == "insight" and result.get("suggestions"):
        st.session_state.insight_suggestions = result["suggestions"]

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
        if result.get("leaderboard"):
            assistant_msg["result_df"] = pd.DataFrame(result["leaderboard"])
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
    elif result.get("route") == "insight":
        lines = [result.get("summary", "Suggestions:")]
        for s in result.get("suggestions") or []:
            lines.append(f"- {s['label']}")
        assistant_msg["content"] = "\n".join(lines)
        st.session_state.insight_suggestions = result.get("suggestions") or []
    elif result.get("route") == "report":
        assistant_msg["content"] = "**Executive Summary**\n\n" + result.get(
            "executive_summary", result.get("summary", "")
        )
        assistant_msg["report_html"] = result.get("html")
    elif result.get("route") == "rag":
        assistant_msg["content"] = result.get("answer", "")
        assistant_msg["citations"] = result.get("citations")
    else:
        assistant_msg["content"] = result.get("answer", "")

    st.session_state.chat_messages.append(assistant_msg)
    st.rerun()


def render_chat():
    st.markdown('<p class="main-header">Agentic Data Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Your AI data analyst — clean data, explore proactively, model, forecast, join tables, and export reports.</p>',
        unsafe_allow_html=True,
    )

    render_quality_gate()
    render_insights()
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

    # Clicked insight suggestion
    if st.session_state.pending_suggestion:
        prompt = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None
        _handle_prompt(prompt)
        return

    if prompt := st.chat_input("Ask about your data..."):
        _handle_prompt(prompt)


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
