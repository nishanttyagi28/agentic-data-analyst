# Agentic Data Analyst

A Streamlit app where you upload CSV files and ask questions about them in plain English. Behind the chat, a router sends each question to a specialist: data cleaning, read-only SQL, charts and statistics, a simple forecast, AutoML, or an HTML report.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Live demo](https://agentic-data-analyst-uqjwnx2jwzd2pe9vosnffw.streamlit.app/)** (Streamlit Community Cloud; it may take a moment to wake up). Pick a sample dataset in the sidebar and start asking.

Asking a simple question about a CSV export usually means opening a notebook or writing SQL. Here you upload the file, the app checks its quality first, and then you keep asking in one conversation: "How many customers churned?", "What is correlated with price?", "Train a model and tell me what mattered", "What were the key findings so far?". Later questions can refer back to results from earlier in the session.

## Install

Python 3.11 or newer. A free [Groq API key](https://console.groq.com/) is needed for text-to-SQL, summaries and follow-up answers. Upload, quality checks, charts, stats, forecasts and model training work without it.

```bash
git clone https://github.com/nishanttyagi28/agentic-data-analyst.git
cd agentic-data-analyst
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then set GROQ_API_KEY in .env
```

`requirements.txt` includes `sentence-transformers`, which pulls in PyTorch, so the first install is large.

## Quick start

```bash
python -m streamlit run app.py
```

Open http://localhost:8501, choose `customer_churn.csv` or `house_prices.csv` from the sidebar (or upload your own), and try:

- "How many customers churned?" (churn data)
- "What is correlated with price?" (house prices)
- "Train a model to predict churn"
- "Is there a significant difference in monthly charges between churned and retained customers?"
- "Generate a report"

## Features

- **Quality check on upload.** You get a score out of 100, plus missing values, duplicates, type problems and outliers. An optional auto-clean handles the safe cases: median/mode fill, dropping exact duplicates, and casting numbers stored as text. Ambiguous choices, like merging `US` and `USA`, are left to you.
- **Plain English to SQL.** Questions are turned into SQL with joins, CTEs and window functions, and run against SQLite, including joins across several uploaded files. Only `SELECT` and `WITH … SELECT` statements run. Anything with write or DDL keywords is rejected before execution.
- **Stats and charts.** Descriptive stats, correlations, group charts, and Welch t-tests and ANOVA with plain-language caveats (Plotly).
- **Forecasts.** A linear trend with uncertainty bands. It uses a date column when one can be parsed, and falls back to row order otherwise.
- **AutoML.** For classification or regression it compares a few models (linear/logistic, random forest, XGBoost) with a light hyperparameter search and shows feature importance. Clustering is offered when there's no target column.
- **Session follow-ups.** Results from earlier steps are embedded (`all-MiniLM-L6-v2`) into a ChromaDB collection for the session, so "what did we find?" is answered from them.
- **Report.** Download an HTML report with a summary of the session.

## How it works

```text
upload CSV(s) -> quality check (optional clean) -> question
  -> orchestrator picks a route (keyword rules first, LLM fallback)
  -> specialist agent in agents/ (sql, ml, quality, stats, forecast, insight, report, rag)
  -> answer in chat (+ table/chart) -> result indexed for later follow-ups
```

The orchestrator is plain Python, not a framework like LangGraph. LLM calls go to Groq (`llama-3.3-70b-versatile`). Design notes are in [DECISIONS.md](DECISIONS.md), and fixes made along the way are logged in [BUGFIX_LOG.md](BUGFIX_LOG.md).

```text
app.py          Streamlit entry point
agents/         orchestrator and specialist agents
db/             SQLite helpers
utils/          env loading, chunking, charts
sample_data/    two small demo CSVs
tests/          pytest unit tests
self_test.py    end-to-end check script
```

## Limitations

- This is a personal project, not a replacement for an analyst. Check SQL and model output before relying on them.
- The sample datasets are tiny (25–30 rows) so the demo runs fast. Model scores on them only illustrate the workflow.
- AutoML is exploratory. There is no model deployment, monitoring or fairness checking.
- Correlations and tests show association, not causation.
- The first follow-up question downloads the embedding model (about 90 MB).
- Without a Groq key, SQL generation and follow-up answers are disabled, and the sidebar shows a warning.

## Development

```bash
pip install pytest
python -m pytest -q tests/        # unit tests, no API key needed
python self_test.py               # end-to-end script; LLM steps skip without GROQ_API_KEY
```

CI runs the unit tests on pull requests.

To deploy on Streamlit Community Cloud, point a new app at `app.py` and add `GROQ_API_KEY` under Secrets. Don't commit `.env`.

## License

MIT. See [LICENSE](LICENSE).
