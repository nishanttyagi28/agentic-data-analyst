"""Build evals/cases.jsonl from the sample csvs.

Expected answers come straight from pandas so they can be re-checked any time.
Cases 1-10 are left for hand-written questions (see cases_manual.jsonl).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
churn = pd.read_csv(ROOT / "sample_data" / "customer_churn.csv")
house = pd.read_csv(ROOT / "sample_data" / "house_prices.csv")

cases = []


def add(dataset, question, expected, kind="number", tol=0.01):
    if isinstance(expected, float):
        expected = round(expected, 2)
    cases.append({
        "id": f"auto_{len(cases) + 11:02d}",
        "dataset": dataset,
        "question": question,
        "expected": expected,
        "type": kind,
        "tolerance": tol if kind == "number" else None,
    })


c = "customer_churn.csv"
add(c, "How many customers are in the dataset?", int(len(churn)))
add(c, "How many customers churned?", int((churn.churn == "Yes").sum()))
add(c, "What is the churn rate in percent?", float((churn.churn == "Yes").mean() * 100))
add(c, "What is the average monthly charge?", float(churn.monthly_charges.mean()))
add(c, "What is the average tenure in months?", float(churn.tenure_months.mean()))
add(c, "What is the total of total_charges across all customers?", float(churn.total_charges.sum()))
add(c, "Which contract type has the most customers?", churn.contract_type.value_counts().idxmax(), "text")
rate = churn.assign(y=churn.churn.eq("Yes")).groupby("contract_type").y.mean()
add(c, "Which contract type has the highest churn rate?", rate.idxmax(), "text")
add(c, "What is the churn rate in percent for Month-to-month contracts?",
    float(rate.get("Month-to-month", 0) * 100))
add(c, "What is the average monthly charge for customers who churned?",
    float(churn.loc[churn.churn == "Yes", "monthly_charges"].mean()))
add(c, "How many customers use Fiber optic internet?", int((churn.internet_service == "Fiber optic").sum()))
add(c, "What is the age of the oldest customer?", int(churn.age.max()))
add(c, "Which payment method is used most?", churn.payment_method.value_counts().idxmax(), "text")

h = "house_prices.csv"
add(h, "How many houses are in the dataset?", int(len(house)))
add(h, "What is the average house price?", float(house.price.mean()))
add(h, "What is the median house price?", float(house.price.median()))
add(h, "What is the highest house price?", int(house.price.max()))
add(h, "Which neighborhood has the highest average price?",
    house.groupby("neighborhood").price.mean().idxmax(), "text")
add(h, "How many houses are waterfront?", int((house.waterfront == 1).sum()))
add(h, "What is the average price of waterfront houses?",
    float(house.loc[house.waterfront == 1, "price"].mean()))
add(h, "What is the average sqft_living?", float(house.sqft_living.mean()))
add(h, "What is the average price per sqft_living?", float((house.price / house.sqft_living).mean()))
add(h, "How many houses have 4 or more bedrooms?", int((house.bedrooms >= 4).sum()))
add(h, "What is the correlation between sqft_living and price?",
    float(house.sqft_living.corr(house.price)), tol=0.02)


add(c, "How many customers have tenure under 12 months?", int((churn.tenure_months < 12).sum()))
add(c, "What is the median monthly charge?", float(churn.monthly_charges.median()))
add(c, "How many customers pay by Electronic check?", int((churn.payment_method == "Electronic check").sum()))
add(c, "What is the average age of customers who churned?", float(churn.loc[churn.churn == "Yes", "age"].mean()))
add(c, "Which internet service has the most customers?", churn.internet_service.value_counts().idxmax(), "text")
add(c, "How many customers are on a Two year contract?", int((churn.contract_type == "Two year").sum()))

add(h, "What is the lowest house price?", int(house.price.min()))
add(h, "What is the average number of bedrooms?", float(house.bedrooms.mean()))
add(h, "How many houses have 2 or more bathrooms?", int((house.bathrooms >= 2).sum()))
add(h, "Which neighborhood has the most houses?", house.neighborhood.value_counts().idxmax(), "text")
add(h, "What is the average lot size in sqft?", float(house.sqft_lot.mean()))
add(h, "How many houses are under 10 years old?", int((house.age < 10).sum()))

out = Path(__file__).with_name("cases.jsonl")
with out.open("w") as f:
    for row in cases:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(cases)} cases to {out}")
