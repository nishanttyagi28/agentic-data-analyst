"""Run the eval cases through the SQL agent and score the answers.

Usage:
    python evals/run_evals.py              # all cases, needs GROQ_API_KEY
    python evals/run_evals.py --limit 5

Each case loads its CSV into an in-memory SQLite table, asks the SQL agent the
question, and compares the first cell of the result with the pandas answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
CASE_FILES = ("cases.jsonl", "cases_manual.jsonl")
REQUIRED_KEYS = {"id", "dataset", "question", "expected", "type"}


def load_cases(eval_dir: Path = EVAL_DIR) -> list[dict]:
    cases = []
    for name in CASE_FILES:
        path = eval_dir / name
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = REQUIRED_KEYS - case.keys()
            if missing:
                raise ValueError(f"{name}:{lineno} missing keys {sorted(missing)}")
            cases.append(case)
    return cases


def score(case: dict, actual) -> bool:
    if actual is None:
        return False
    expected = case["expected"]
    if case["type"] == "text":
        return str(actual).strip().lower() == str(expected).strip().lower()
    try:
        actual = float(actual)
    except (TypeError, ValueError):
        return False
    tol = case.get("tolerance") or 0.01
    return abs(actual - float(expected)) <= tol


def first_cell(df: pd.DataFrame | None):
    if df is None or df.empty:
        return None
    return df.iloc[0, 0]


def run_case(case: dict) -> dict:
    # imported here so the loader and scorer work without the agent stack
    from agents.sql_agent import run_sql_query
    from db.database import load_dataframe_to_table

    engine = create_engine("sqlite://")
    load_dataframe_to_table(pd.read_csv(ROOT / "sample_data" / case["dataset"]), engine)
    result = run_sql_query(case["question"], engine)
    actual = first_cell(result.get("result")) if result.get("success") else None
    return {
        "id": case["id"],
        "passed": score(case, actual),
        "actual": None if actual is None else str(actual),
        "expected": case["expected"],
        "sql": result.get("sql"),
        "error": result.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    cases = load_cases()[: args.limit]
    results = [run_case(c) for c in cases]
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark} {r['id']}: expected={r['expected']} actual={r['actual']}"
              + (f" error={r['error']}" if r["error"] else ""))
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
