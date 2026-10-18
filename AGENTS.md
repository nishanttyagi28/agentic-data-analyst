# AGENTS.md

## Project

Agentic Data Analyst is a Streamlit app where you upload CSV files and ask questions in plain English. A router sends each question to a specialist (cleaning, read-only SQL, charts/stats, forecast, AutoML, or HTML report). Follow-ups can use results already computed in the session.

## Stack

- Python 3.11+, Streamlit, pandas, SQLAlchemy + SQLite, Plotly, scikit-learn, XGBoost, ChromaDB, Groq (`llama-3.3-70b-versatile`).
- Tests: pytest unit tests under `tests/` (no network; the LLM is faked), broader `self_test.py`.
- Deploy: Streamlit Community Cloud and Hugging Face Spaces (Docker). GitHub Actions for unit CI and Space deploy.

## Layout

- `app.py` — Streamlit entry
- `agents/` — orchestrator and specialists
- `db/`, `utils/`, `sample_data/`, `evals/`, `tests/`
- `Dockerfile`, `.github/workflows/`, `requirements.txt`, `.env.example`

## Commands

- `pip install -r requirements.txt` (plus `pip install pytest` for tests)
- `python -m streamlit run app.py`
- `python -m pytest -q tests/`
- `python self_test.py`
- `python evals/make_cases.py` and `python evals/make_manual_cases.py` (rebuild eval cases from the CSVs)
- `python evals/run_evals.py` (score the SQL agent on the eval cases; needs `GROQ_API_KEY`)
- `docker build -t agentic-data-analyst . && docker run -p 7860:7860 -e GROQ_API_KEY=... agentic-data-analyst`

## Always

- Keep each file under 300 lines when practical; split modules instead of suppressing linters.
- Only read-only SQL is executed; block write/DDL before run.
- Never log secrets or API keys. Do not commit `.env`.
- Plain lowercase commit messages, one logical change per commit.
- Git identity: Nishant Tyagi <253968642+nishanttyagi28@users.noreply.github.com>.
- Do not push unless asked. Stop after each milestone and wait for review.

## Ask first

- Before adding a dependency, Docker base image, or GitHub Action.
- Before changing the Space deploy workflow or README live-demo links.
- Before deleting or renaming public modules.

## Never

- Never force-push, amend published history, or rewrite pushed commits.
- Never weaken or delete tests to make them pass.
- Never add `noqa`, `type: ignore`, or formatter suppressions to hide errors.
- Never put AI/tool wording, generator notes, or `Co-authored-by` in files or commits.
- Never invent eval expected answers; compute them with pandas from the sample CSVs.

## Week 1 scope

- AGENTS.md, Dockerfile, HF Space deploy workflow, 36 scripted + 10 manual pandas eval cases, eval runner, README eval table.
- Do not merge until review. HF deploy runs after merge to `main`.
