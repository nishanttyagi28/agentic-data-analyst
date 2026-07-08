"""Proactive Insight Agent — suggest concrete questions and flag anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _corr_pairs(df: pd.DataFrame, min_abs: float = 0.45) -> list[dict[str, Any]]:
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] < 2:
        return []
    corr = nums.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if abs(float(r)) >= min_abs:
                pairs.append({"a": a, "b": b, "r": float(r)})
    pairs.sort(key=lambda d: abs(d["r"]), reverse=True)
    return pairs[:5]


def _anomaly_flags(df: pd.DataFrame) -> list[dict[str, Any]]:
    flags = []
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in nums[:12]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        high = q3 + 2.5 * iqr  # stricter than standard IQR for "unusual spike"
        low = q1 - 2.5 * iqr
        hi_mask = s > high
        lo_mask = s < low
        if hi_mask.sum() > 0 and hi_mask.sum() <= max(3, len(s) * 0.05):
            flags.append({
                "type": "spike",
                "column": col,
                "detail": f"{int(hi_mask.sum())} unusually high value(s) in `{col}` (above ~{float(high):.3g})",
                "question": f"There's an unusual high spike in {col} — investigate those extreme values and what else differs on those rows",
            })
        if lo_mask.sum() > 0 and lo_mask.sum() <= max(3, len(s) * 0.05):
            flags.append({
                "type": "dip",
                "column": col,
                "detail": f"{int(lo_mask.sum())} unusually low value(s) in `{col}` (below ~{float(low):.3g})",
                "question": f"There's an unusual low outlier pattern in {col} — look into those rows",
            })

    cats = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cats[:8]:
        vc = df[col].astype(str).value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] >= 0.85:
            flags.append({
                "type": "imbalance",
                "column": col,
                "detail": f"`{col}` is heavily skewed toward '{vc.index[0]}' ({vc.iloc[0]*100:.0f}%)",
                "question": f"Is the heavy concentration of {col}={vc.index[0]} expected, or a data quality issue?",
            })
    return flags[:5]


def generate_insight_suggestions(
    df: pd.DataFrame,
    business_context: str = "",
    table_name: str = "user_data",
) -> dict[str, Any]:
    """
    Produce 3–5 concrete suggested questions from column profiles + anomalies.
    No LLM required (fast, free-tier friendly); optional context biases wording.
    """
    if df is None or df.empty:
        return {"success": False, "error": "No data loaded", "agent": "insight"}

    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    suggestions: list[dict[str, Any]] = []
    ctx = (business_context or "").strip()
    domain_hint = f" (in the context of: {ctx[:120]})" if ctx else ""

    # Correlation-driven
    for pair in _corr_pairs(df)[:2]:
        direction = "positively" if pair["r"] > 0 else "negatively"
        suggestions.append({
            "id": f"corr_{pair['a']}_{pair['b']}",
            "kind": "correlation",
            "label": f"`{pair['a']}` and `{pair['b']}` look {direction} correlated (r≈{pair['r']:.2f}) — quantify that?",
            "question": f"What's correlated with {pair['a']}? Focus on relationship with {pair['b']}",
            "detail": f"Pearson r={pair['r']:.3f}{domain_hint}",
        })

    # Category × metric
    if cats and nums:
        cat = min(cats, key=lambda c: df[c].nunique() if df[c].nunique() >= 2 else 999)
        metric = nums[0]
        for kw in ("goal", "price", "revenue", "charge", "amount", "sales", "churn"):
            for n in nums:
                if kw in n.lower():
                    metric = n
                    break
        if 2 <= df[cat].nunique() <= 20:
            suggestions.append({
                "id": f"group_{cat}_{metric}",
                "kind": "comparison",
                "label": f"Does average `{metric}` differ meaningfully across `{cat}`?",
                "question": f"Is there a significant difference in {metric} between {cat} groups?",
                "detail": f"Compare group means of {metric} by {cat}{domain_hint}",
            })

    # Target-like column
    for col in list(df.columns):
        if any(k in col.lower() for k in ("churn", "price", "goal", "target", "label", "outcome")):
            suggestions.append({
                "id": f"target_{col}",
                "kind": "modeling",
                "label": f"`{col}` looks like a key outcome — train a model to predict it?",
                "question": f"Train a model to predict {col} and explain the top drivers",
                "detail": f"AutoML on target `{col}`{domain_hint}",
            })
            break

    # Aggregate / SQL style
    if cats and nums:
        suggestions.append({
            "id": "top_n",
            "kind": "sql",
            "label": f"Top categories of `{cats[0]}` by average `{nums[0]}`",
            "question": f"Show the top 5 {cats[0]} values by average {nums[0]}",
            "detail": "Ranking / aggregate query",
        })

    # Anomalies as clickable suggestions
    anomalies = _anomaly_flags(df)
    for i, fl in enumerate(anomalies[:2]):
        suggestions.append({
            "id": f"anomaly_{i}_{fl['column']}",
            "kind": "anomaly",
            "label": fl["detail"] + " — investigate?",
            "question": fl["question"],
            "detail": fl.get("detail", ""),
        })

    # Missing-heavy column
    miss = df.isnull().mean()
    heavy = miss[miss > 0.1]
    if len(heavy):
        col = heavy.idxmax()
        suggestions.append({
            "id": f"missing_{col}",
            "kind": "quality",
            "label": f"`{col}` is {heavy[col]*100:.0f}% missing — is that a data issue or meaningful?",
            "question": f"Show data quality report focusing on missing values in {col}",
            "detail": "Data quality investigation",
        })

    # De-dupe by question text, keep 3–5
    seen = set()
    unique = []
    for s in suggestions:
        key = s["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    unique = unique[:5]
    if len(unique) < 3:
        unique.append({
            "id": "eda_default",
            "kind": "eda",
            "label": "Run exploratory analysis (distributions, correlations, summaries)",
            "question": "Run EDA and summarize the key patterns in this dataset",
            "detail": "Full EDA pass",
        })

    summary_lines = [f"Suggested explorations for `{table_name}` ({len(df)} rows × {len(df.columns)} cols):"]
    for i, s in enumerate(unique, 1):
        summary_lines.append(f"{i}. {s['label']}")

    return {
        "success": True,
        "agent": "insight",
        "suggestions": unique,
        "anomalies": anomalies,
        "business_context": ctx,
        "summary": "\n".join(summary_lines),
        "summary_for_rag": "Proactive insights:\n" + "\n".join(summary_lines),
    }
