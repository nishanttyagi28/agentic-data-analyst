"""Build evals/cases_manual.jsonl: hand-picked, harder questions.

Questions are written by hand; expected answers still come from pandas.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
churn = pd.read_csv(ROOT / "sample_data" / "customer_churn.csv")
house = pd.read_csv(ROOT / "sample_data" / "house_prices.csv")
churned = churn.churn == "Yes"

cases = []


def top(series, lowest=False):
    best = series.min() if lowest else series.max()
    winners = series[series == best].index.tolist()
    if len(winners) != 1:
        raise ValueError(f"tie: {winners}")
    return winners[0]


def add(dataset, question, expected, kind="number", tol=0.01):
    if isinstance(expected, float):
        expected = round(expected, 2)
    cases.append({
        "id": f"manual_{len(cases) + 1:02d}",
        "dataset": dataset,
        "question": question,
        "expected": expected,
        "type": kind,
        "tolerance": tol if kind == "number" else None,
    })


c = "customer_churn.csv"
fiber = churn.internet_service == "Fiber optic"
add(c, "What percent of Fiber optic customers churned?", float(churned[fiber].mean() * 100))
add(c, "How many churned customers were on a Month-to-month contract and paid by Electronic?",
    int((churned & (churn.contract_type == "Month-to-month")
         & (churn.payment_method == "Electronic")).sum()))
add(c, "What is the average tenure in months of customers who did not churn?",
    float(churn.loc[~churned, "tenure_months"].mean()))
add(c, "Which payment method has the highest average monthly charge?",
    top(churn.groupby("payment_method").monthly_charges.mean()), "text")
add(c, "How many customers are older than 40 and pay more than 70 per month?",
    int(((churn.age > 40) & (churn.monthly_charges > 70)).sum()))

h = "house_prices.csv"
add(h, "What is the average price of 3-bedroom houses?",
    float(house.loc[house.bedrooms == 3, "price"].mean()))
add(h, "Which neighborhood has the lowest average price?",
    top(house.groupby("neighborhood").price.mean(), lowest=True), "text")
add(h, "How many non-waterfront houses cost more than 400000?",
    int(((house.waterfront == 0) & (house.price > 400000)).sum()))
add(h, "What is the average sqft_living of houses with 2 floors?",
    float(house.loc[house.floors == 2, "sqft_living"].mean()))
add(h, "How much higher is the average waterfront price than the average non-waterfront price?",
    float(house.loc[house.waterfront == 1, "price"].mean()
          - house.loc[house.waterfront == 0, "price"].mean()))

out = Path(__file__).with_name("cases_manual.jsonl")
with out.open("w") as f:
    for row in cases:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(cases)} cases to {out}")
