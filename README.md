# Agentic Data Analyst

A Streamlit app I built so you can upload CSV files, ask questions in plain English, and get cleaning help, SQL answers, charts, simple forecasts, AutoML, and a downloadable report — in one chat.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://agentic-data-analyst-uqjwnx2jwzd2pe9vosnffw.streamlit.app/)

Most teams have useful data sitting in CSV exports. Asking a simple question still often means waiting on someone who can write SQL, open a notebook, or build a chart.

I built this so a person can drop in a file, ask what they need, and keep going in the same conversation — including follow-ups about what the app already found.

**[Try the live demo](https://agentic-data-analyst-uqjwnx2jwzd2pe9vosnffw.streamlit.app/)** — pick a sample dataset in the sidebar and start chatting. A free [Groq API key](https://console.groq.com/) unlocks the full flow (text-to-SQL, summaries, follow-up answers).

---

## The problem

Business questions are usually plain English:

- “How many customers churned?”
- “What’s correlated with price?”
- “Train a simple model and tell me what mattered.”
- “Put this in a short report I can send.”

The data is usually a CSV. The bottleneck is the tooling gap between the question and the answer.

---

## In simple terms

If a recruiter asks “so what does this project actually do?”, here’s how I’d answer.

You upload a customer CSV. The app checks quality first (missing values, duplicates, odd types) and lets you clean the safe stuff or skip.

Then you ask: **“How many customers churned?”** It turns that into read-only SQL, runs it, and shows the answer.

You can keep going in the same chat: correlations, a quick forecast, a model that tries a few algorithms, suggested next questions, or a downloadable HTML report.

Later you can ask: **“What were the key findings?”** It looks back at what it already computed in this session and answers from that, instead of starting from zero.

That’s the whole idea: one upload, one chat, several kinds of analysis without hopping tools.

---

## What I built

### Quality check before analysis

After upload, you get a quality score out of 100, plus missing values, duplicates, type issues, and outliers. Safe auto-clean is optional (median/mode fill, drop exact duplicates, cast numbers stored as text). Ambiguous choices — like whether `US` and `USA` should merge, or whether a column looks like an ID — are shown for you to decide. Nothing ambiguous is auto-decided.

**Why it helps:** You see whether the file is trustworthy before you trust the answers.

### Plain-English questions → read-only SQL

Counts, filters, rankings, joins across multiple uploaded CSVs, including CTEs and window functions when needed. Only `SELECT` / `WITH … SELECT` runs; write/DDL-style statements are blocked before execution.

**Why it helps:** Someone who doesn’t write SQL can still get facts from the table.

### Charts, stats, and light forecasts

Descriptive stats, correlation views, group charts, Welch t-test / ANOVA with plain-English caveats, and simple linear-trend forecasts with uncertainty bands when dates exist.

**Why it helps:** Exploration and basic testing stay in the same place as the chat.

### AutoML on the session data

For classification and regression it tries a small set of models (for example logistic regression / Ridge, random forest, XGBoost), does a light search, and explains which features the winning model leaned on. Clustering is available when there’s no clear target. This is exploratory on the uploaded data — not a deployment pipeline.

**Why it helps:** You can ask “what predicts churn?” without standing up a separate ML project.

### Session memory for follow-ups

Successful results from cleaning, SQL, ML, stats, forecasts, and reports are indexed so later questions can cite prior findings from this session.

**Why it helps:** Analysis becomes a conversation instead of a one-shot export.

### Shareable HTML report

You can generate a downloadable HTML report with an executive summary of the session so far.

**Why it helps:** Handy when you need something to send, not just a chat transcript.

---

## A few numbers

Things you can verify in this repo:

| | |
| --- | --- |
| Specialist agents behind the chat | **8** (SQL, ML, quality, stats, forecast, insight, report, session follow-up) |
| Chat routes the orchestrator knows | **9** (those eight + a general fallback) |
| Sample CSVs | **2** (`customer_churn.csv` 30 rows / 9 columns; `house_prices.csv` 25 rows / 10 columns) |
| Named checks in `self_test.py` | **20** test functions |
| Pytest unit tests in CI | **4** (`tests/test_llm_usage.py`, run on PRs) |
| Documented self-test run (see `BUGFIX_LOG.md`) | **179** passed, **0** failed, **10** skipped (LLM paths without API key) |
| Forbidden SQL keyword patterns blocked | **17** |
| Classification AutoML candidate families | **3** (logistic regression, random forest, XGBoost) |
| Insight suggestions per request | **3–5** |
| Python | **3.11+** |
| License | **MIT** |
| Live demo | Streamlit Cloud link above |

Sample datasets are intentionally small so the full pipeline is easy to demo. Model metrics on 25–30 rows illustrate the workflow; they are not production-grade performance.

---

## Why this matters

The useful part isn’t another chatbot wrapped around a spreadsheet.

It’s reducing the gap between “I have a CSV” and “I got a careful answer, a chart, and something I can share” — without requiring SQL, notebook, or ML setup first.

---

## How it works

```text
Upload CSV(s)
  ↓
Quality gate (optional clean / decisions)
  ↓
Ask in plain English
  ↓
Orchestrator picks a route
  ↓
Specialist agent runs (SQL / ML / stats / …)
  ↓
Result shown in chat (+ charts / tables when relevant)
  ↓
Useful outputs indexed for later follow-ups
  ↓
Optional HTML report
```

Details for engineers are below (install, run, stack).

---

## Install and run

### Prerequisites

- Python 3.11 or newer
- A free [Groq API key](https://console.groq.com/) for text-to-SQL, LLM summaries, and follow-up answers (ingestion, EDA charts, and local model training still work without it)

### Setup

```bash
git clone https://github.com/nishanttyagi28/agentic-data-analyst.git
cd agentic-data-analyst
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
# edit .env and set GROQ_API_KEY=...
```

### Run

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload a CSV or pick a sample dataset, then ask questions in the chat.

### Tests

```bash
# CI-focused unit tests (no API key required)
python -m pytest -q tests/test_llm_usage.py

# Broader end-to-end script (LLM steps skip without GROQ_API_KEY)
python self_test.py
```

### Streamlit Community Cloud

1. Push the repo (do not commit `.env`).
2. Create an app pointed at `app.py`.
3. Add `GROQ_API_KEY` under Secrets.
4. First follow-up / embedding use downloads the local embedding model (~90 MB).

---

## Tech (lower half)

| Layer | Choice |
| --- | --- |
| UI | Streamlit |
| Database | SQLite + SQLAlchemy |
| Routing | Custom orchestrator (rules first, LLM fallback) — not LangGraph |
| LLM | Groq `llama-3.3-70b-versatile` |
| ML | pandas, scikit-learn, XGBoost, Plotly |
| Session follow-up index | ChromaDB + HuggingFace `all-MiniLM-L6-v2` |
| Config | `python-dotenv` (`.env` in project root) |

### Project layout

```text
agentic-data-analyst/
├── app.py                 # Streamlit entrypoint
├── agents/                # specialists + orchestrator
├── db/                    # SQLite helpers
├── utils/                 # env, chunking, charts
├── sample_data/           # demo CSVs
├── tests/                 # pytest unit tests (CI)
├── self_test.py           # broader self-check script
├── requirements.txt
├── DECISIONS.md           # design choices
├── BUGFIX_LOG.md          # verified fixes + self-test counts
└── FINAL_REPORT.md        # build notes
```

### Safety notes

- Only read-only SQL is executed.
- Missing or placeholder API key shows a sidebar warning; the app should not crash.
- Agent errors are surfaced in chat.
- AutoML here is exploratory on session data, not monitoring, fairness audits, or production training pipelines.
- Stats and correlations are association-based; they do not prove causation.

---

## Status

**WIP · portfolio project · actively refined.**

Useful today for demos and local exploration on your own CSVs. Sample rows are tiny on purpose. Schemas and agent behavior may still move. I’m not calling this a finished product or a replacement for a data team.

## License

MIT. See [LICENSE](LICENSE).
