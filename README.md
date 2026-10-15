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


## Eval cases (pandas ground truth)

These cases live in `evals/cases.jsonl`. Expected answers are computed with pandas from the sample CSVs, so you can re-check them any time with `python evals/make_cases.py`.

Hand-written cases (ids `manual_01`–`manual_10`) go in `evals/cases_manual.jsonl` when you add them.

| id | dataset | question | expected |
| --- | --- | --- | --- |
| `auto_11` | `customer_churn.csv` | How many customers are in the dataset? | 30 |
| `auto_12` | `customer_churn.csv` | How many customers churned? | 15 |
| `auto_13` | `customer_churn.csv` | What is the churn rate in percent? | 50.0 |
| `auto_14` | `customer_churn.csv` | What is the average monthly charge? | 63.78 |
| `auto_15` | `customer_churn.csv` | What is the average tenure in months? | 25.23 |
| `auto_16` | `customer_churn.csv` | What is the total of total_charges across all customers? | 54826.17 |
| `auto_17` | `customer_churn.csv` | Which contract type has the most customers? | Month-to-month |
| `auto_18` | `customer_churn.csv` | Which contract type has the highest churn rate? | Month-to-month |
| `auto_19` | `customer_churn.csv` | What is the churn rate in percent for Month-to-month contracts? | 100.0 |
| `auto_20` | `customer_churn.csv` | What is the average monthly charge for customers who churned? | 49.47 |
| `auto_21` | `customer_churn.csv` | How many customers use Fiber optic internet? | 16 |
| `auto_22` | `customer_churn.csv` | What is the age of the oldest customer? | 58 |
| `auto_23` | `customer_churn.csv` | Which payment method is used most? | Electronic |
| `auto_24` | `house_prices.csv` | How many houses are in the dataset? | 25 |
| `auto_25` | `house_prices.csv` | What is the average house price? | 406400.0 |
| `auto_26` | `house_prices.csv` | What is the median house price? | 380000.0 |
| `auto_27` | `house_prices.csv` | What is the highest house price? | 720000 |
| `auto_28` | `house_prices.csv` | Which neighborhood has the highest average price? | Lakeside |
| `auto_29` | `house_prices.csv` | How many houses are waterfront? | 8 |
| `auto_30` | `house_prices.csv` | What is the average price of waterfront houses? | 607500.0 |
| `auto_31` | `house_prices.csv` | What is the average sqft_living? | 2062.0 |
| `auto_32` | `house_prices.csv` | What is the average price per sqft_living? | 193.03 |
| `auto_33` | `house_prices.csv` | How many houses have 4 or more bedrooms? | 11 |
| `auto_34` | `house_prices.csv` | What is the correlation between sqft_living and price? | 0.99 |
| `auto_35` | `customer_churn.csv` | How many customers have tenure under 12 months? | 6 |
| `auto_36` | `customer_churn.csv` | What is the median monthly charge? | 63.2 |
| `auto_37` | `customer_churn.csv` | How many customers pay by Electronic? | 8 |
| `auto_38` | `customer_churn.csv` | What is the average age of customers who churned? | 33.0 |
| `auto_39` | `customer_churn.csv` | Which internet service has the most customers? | Fiber optic |
| `auto_40` | `customer_churn.csv` | How many customers are on a Two year contract? | 7 |
| `auto_41` | `house_prices.csv` | What is the lowest house price? | 180000 |
| `auto_42` | `house_prices.csv` | What is the average number of bedrooms? | 3.4 |
| `auto_43` | `house_prices.csv` | How many houses have 2 or more bathrooms? | 20 |
| `auto_44` | `house_prices.csv` | Which neighborhood has the most houses? | Suburban |
| `auto_45` | `house_prices.csv` | What is the average lot size in sqft? | 6172.0 |
| `auto_46` | `house_prices.csv` | How many houses are under 10 years old? | 3 |

## Hugging Face Space

Docker Space deploy is wired in `.github/workflows/deploy-space.yml` (pushes `main` to the Space).

1. Create a free Docker Space named `agentic-data-analyst`.
2. Add Space secret `GROQ_API_KEY`.
3. In this GitHub repo, add secret `HF_TOKEN` (write) and variable `HF_SPACE` = `nishanttyagi28/agentic-data-analyst` (or your Space path).
4. Merge to `main` (or run the workflow manually).


## License

MIT. See [LICENSE](LICENSE).
