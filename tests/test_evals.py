"""Eval cases and the SQL agent path, with a fake LLM (no network, no API key)."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import llm_client
from evals import run_evals

T = "user_data"
YES = "churn = 'Yes'"

# reference sql per question, used as the fake model's answer
REFERENCE_SQL = {
    "How many customers are in the dataset?": f"SELECT COUNT(*) FROM {T}",
    "How many customers churned?": f"SELECT COUNT(*) FROM {T} WHERE {YES}",
    "What is the churn rate in percent?": f"SELECT 100.0 * SUM({YES}) / COUNT(*) FROM {T}",
    "What is the average monthly charge?": f"SELECT AVG(monthly_charges) FROM {T}",
    "What is the average tenure in months?": f"SELECT AVG(tenure_months) FROM {T}",
    "What is the total of total_charges across all customers?":
        f"SELECT SUM(total_charges) FROM {T}",
    "Which contract type has the most customers?":
        f"SELECT contract_type FROM {T} GROUP BY contract_type ORDER BY COUNT(*) DESC LIMIT 1",
    "Which contract type has the highest churn rate?":
        f"SELECT contract_type FROM {T} GROUP BY contract_type "
        f"ORDER BY AVG({YES}) DESC LIMIT 1",
    "What is the churn rate in percent for Month-to-month contracts?":
        f"SELECT 100.0 * AVG({YES}) FROM {T} WHERE contract_type = 'Month-to-month'",
    "What is the average monthly charge for customers who churned?":
        f"SELECT AVG(monthly_charges) FROM {T} WHERE {YES}",
    "How many customers use Fiber optic internet?":
        f"SELECT COUNT(*) FROM {T} WHERE internet_service = 'Fiber optic'",
    "What is the age of the oldest customer?": f"SELECT MAX(age) FROM {T}",
    "Which payment method do most churned customers use?":
        f"SELECT payment_method FROM {T} WHERE {YES} GROUP BY payment_method "
        f"ORDER BY COUNT(*) DESC LIMIT 1",
    "How many customers have tenure under 12 months?":
        f"SELECT COUNT(*) FROM {T} WHERE tenure_months < 12",
    "What is the median monthly charge?":
        f"SELECT AVG(monthly_charges) FROM (SELECT monthly_charges FROM {T} "
        f"ORDER BY monthly_charges LIMIT 2 - (SELECT COUNT(*) FROM {T}) % 2 "
        f"OFFSET (SELECT (COUNT(*) - 1) / 2 FROM {T}))",
    "How many customers pay by Electronic?":
        f"SELECT COUNT(*) FROM {T} WHERE payment_method = 'Electronic'",
    "What is the average age of customers who churned?":
        f"SELECT AVG(age) FROM {T} WHERE {YES}",
    "Which internet service has the most customers?":
        f"SELECT internet_service FROM {T} GROUP BY internet_service "
        f"ORDER BY COUNT(*) DESC LIMIT 1",
    "How many customers are on a Two year contract?":
        f"SELECT COUNT(*) FROM {T} WHERE contract_type = 'Two year'",
    "How many houses are in the dataset?": f"SELECT COUNT(*) FROM {T}",
    "What is the average house price?": f"SELECT AVG(price) FROM {T}",
    "What is the median house price?":
        f"SELECT AVG(price) FROM (SELECT price FROM {T} ORDER BY price "
        f"LIMIT 2 - (SELECT COUNT(*) FROM {T}) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM {T}))",
    "What is the highest house price?": f"SELECT MAX(price) FROM {T}",
    "Which neighborhood has the highest average price?":
        f"SELECT neighborhood FROM {T} GROUP BY neighborhood ORDER BY AVG(price) DESC LIMIT 1",
    "How many houses are waterfront?": f"SELECT COUNT(*) FROM {T} WHERE waterfront = 1",
    "What is the average price of waterfront houses?":
        f"SELECT AVG(price) FROM {T} WHERE waterfront = 1",
    "What is the average sqft_living?": f"SELECT AVG(sqft_living) FROM {T}",
    "What is the average price per sqft_living?":
        f"SELECT AVG(1.0 * price / sqft_living) FROM {T}",
    "How many houses have 4 or more bedrooms?": f"SELECT COUNT(*) FROM {T} WHERE bedrooms >= 4",
    "What is the correlation between sqft_living and price?":
        f"SELECT (AVG(sqft_living * price) - AVG(sqft_living) * AVG(price)) / "
        f"(SQRT(AVG(sqft_living * sqft_living) - AVG(sqft_living) * AVG(sqft_living)) * "
        f"SQRT(AVG(1.0 * price * price) - AVG(price) * AVG(price))) FROM {T}",
    "What is the lowest house price?": f"SELECT MIN(price) FROM {T}",
    "What is the average number of bedrooms?": f"SELECT AVG(bedrooms) FROM {T}",
    "How many houses have 2 or more bathrooms?": f"SELECT COUNT(*) FROM {T} WHERE bathrooms >= 2",
    "Which neighborhood has the most houses?":
        f"SELECT neighborhood FROM {T} GROUP BY neighborhood ORDER BY COUNT(*) DESC LIMIT 1",
    "What is the average lot size in sqft?": f"SELECT AVG(sqft_lot) FROM {T}",
    "How many houses are under 10 years old?": f"SELECT COUNT(*) FROM {T} WHERE age < 10",
    "What percent of Fiber optic customers churned?":
        f"SELECT 100.0 * AVG({YES}) FROM {T} WHERE internet_service = 'Fiber optic'",
    "How many churned customers were on a Month-to-month contract and paid by Electronic?":
        f"SELECT COUNT(*) FROM {T} WHERE {YES} AND contract_type = 'Month-to-month' "
        f"AND payment_method = 'Electronic'",
    "What is the average tenure in months of customers who did not churn?":
        f"SELECT AVG(tenure_months) FROM {T} WHERE churn = 'No'",
    "Which payment method has the highest average monthly charge?":
        f"SELECT payment_method FROM {T} GROUP BY payment_method "
        f"ORDER BY AVG(monthly_charges) DESC LIMIT 1",
    "How many customers are older than 40 and pay more than 70 per month?":
        f"SELECT COUNT(*) FROM {T} WHERE age > 40 AND monthly_charges > 70",
    "What is the average price of 3-bedroom houses?":
        f"SELECT AVG(price) FROM {T} WHERE bedrooms = 3",
    "Which neighborhood has the lowest average price?":
        f"SELECT neighborhood FROM {T} GROUP BY neighborhood ORDER BY AVG(price) ASC LIMIT 1",
    "How many non-waterfront houses cost more than 400000?":
        f"SELECT COUNT(*) FROM {T} WHERE waterfront = 0 AND price > 400000",
    "What is the average sqft_living of houses with 2 floors?":
        f"SELECT AVG(sqft_living) FROM {T} WHERE floors = 2",
    "How much higher is the average waterfront price than the average non-waterfront price?":
        f"SELECT AVG(CASE WHEN waterfront = 1 THEN price END) - "
        f"AVG(CASE WHEN waterfront = 0 THEN price END) FROM {T}",
}

CASES = run_evals.load_cases()


class FakeCompletions:
    """Answers sql prompts from REFERENCE_SQL and anything else with a stub."""

    def __init__(self):
        self.calls = 0

    def create(self, messages, **_kwargs):
        self.calls += 1
        prompt = messages[-1]["content"]
        match = re.search(r"User question: (.+)", prompt)
        if match and "Database schema" in prompt:
            content = f"```sql\n{REFERENCE_SQL[match.group(1).strip()]}\n```"
        else:
            content = "Here is what the result shows."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


@pytest.fixture
def fake_llm(monkeypatch):
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_client, "get_groq_client", lambda: (client, None))
    return completions


def test_case_files_load_with_unique_ids():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert sum(i.startswith("auto_") for i in ids) >= 30
    assert sum(i.startswith("manual_") for i in ids) == 10
    for case in CASES:
        assert case["type"] in {"number", "text"}
        assert (Path(run_evals.ROOT) / "sample_data" / case["dataset"]).exists()


def test_loader_skips_blank_lines_and_rejects_bad_rows(tmp_path):
    row = {"id": "x", "dataset": "d.csv", "question": "q", "expected": 1, "type": "number"}
    (tmp_path / "cases.jsonl").write_text(json.dumps(row) + "\n\n")
    (tmp_path / "cases_manual.jsonl").write_text("")
    assert len(run_evals.load_cases(tmp_path)) == 1

    (tmp_path / "cases_manual.jsonl").write_text(json.dumps({"id": "y"}) + "\n")
    with pytest.raises(ValueError):
        run_evals.load_cases(tmp_path)


def test_score_number_and_text():
    num = {"expected": 63.78, "type": "number", "tolerance": 0.01}
    assert run_evals.score(num, 63.7833)
    assert not run_evals.score(num, 64)
    assert not run_evals.score(num, "n/a")
    assert not run_evals.score(num, None)
    text = {"expected": "Month-to-month", "type": "text", "tolerance": None}
    assert run_evals.score(text, " month-to-month ")
    assert not run_evals.score(text, "One year")


def test_every_case_has_reference_sql():
    assert {c["question"] for c in CASES} <= REFERENCE_SQL.keys()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_agent_path_matches_pandas_answer(case, fake_llm):
    result = run_evals.run_case(case)
    assert result["error"] is None
    assert result["passed"], result
    assert fake_llm.calls >= 2  # sql generation + explanation


def test_runner_reports_failure_for_wrong_sql(monkeypatch, fake_llm):
    case = next(c for c in CASES if c["id"] == "auto_11")
    monkeypatch.setitem(REFERENCE_SQL, case["question"], f"SELECT COUNT(*) + 1 FROM {T}")
    assert not run_evals.run_case(case)["passed"]


def test_runner_rejects_write_sql(monkeypatch, fake_llm):
    case = next(c for c in CASES if c["id"] == "auto_11")
    monkeypatch.setitem(REFERENCE_SQL, case["question"], f"DELETE FROM {T}")
    result = run_evals.run_case(case)
    assert not result["passed"]
    assert "safety" in result["error"]
