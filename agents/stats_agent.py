"""Statistical Analysis Agent — hypothesis tests, correlations, outlier summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()


def _find_column(df: pd.DataFrame, query: str, prefer: str | None = None) -> str | None:
    q = query.lower()
    # Exact / substring column mention
    for col in df.columns:
        if col.lower() in q or col.lower().replace("_", " ") in q:
            return col
    if prefer == "numeric":
        nums = _numeric_cols(df)
        return nums[0] if nums else None
    if prefer == "cat":
        cats = _cat_cols(df)
        return cats[0] if cats else None
    return None


def _find_group_and_metric(df: pd.DataFrame, query: str) -> tuple[str | None, str | None]:
    """Best-effort: categorical group column + numeric metric from query or defaults."""
    q = query.lower()
    cats = _cat_cols(df)
    nums = [c for c in _numeric_cols(df) if df[c].nunique() > 2]

    group_col = None
    metric_col = None
    for col in cats:
        if col.lower() in q or col.lower().replace("_", " ") in q:
            group_col = col
            break
    for col in nums:
        if col.lower() in q or col.lower().replace("_", " ") in q:
            metric_col = col
            break

    if group_col is None and cats:
        # Prefer low-cardinality categoricals
        cats_sorted = sorted(cats, key=lambda c: df[c].nunique())
        group_col = cats_sorted[0]
    if metric_col is None and nums:
        # Prefer columns with target-like names
        for kw in ("revenue", "sales", "amount", "price", "charges", "income", "value"):
            for col in nums:
                if kw in col.lower():
                    metric_col = col
                    break
            if metric_col:
                break
        if metric_col is None:
            metric_col = nums[0]
    return group_col, metric_col


def run_group_comparison(df: pd.DataFrame, group_col: str, metric_col: str) -> dict[str, Any]:
    """t-test (2 groups) or one-way ANOVA (3+)."""
    work = df[[group_col, metric_col]].dropna()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna()
    groups = [g[metric_col].values for _, g in work.groupby(group_col) if len(g) >= 2]
    labels = [str(name) for name, g in work.groupby(group_col) if len(g) >= 2]
    n_groups = len(groups)

    if n_groups < 2:
        return {
            "success": False,
            "error": f"Need at least 2 groups with ≥2 observations in '{group_col}'",
            "agent": "stats",
        }

    group_means = {lab: float(np.mean(g)) for lab, g in zip(labels, groups)}
    n_total = sum(len(g) for g in groups)
    assumptions = [
        "Assumes approximately independent observations.",
        "Parametric tests assume roughly normal residuals within groups.",
        "Correlation/difference does not imply causation.",
    ]
    if n_total < 30:
        assumptions.append(f"Small sample (n={n_total}): interpret p-values cautiously.")

    if n_groups == 2:
        t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_name = "Welch's two-sample t-test"
        stat_val = float(t_stat)
    else:
        f_stat, p_val = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
        stat_val = float(f_stat)

    p_val = float(p_val)
    alpha = 0.05
    if p_val < alpha:
        interpretation = (
            f"There is a statistically significant difference in {metric_col} across "
            f"{group_col} groups (p={p_val:.4f} < {alpha}). "
            f"Group means: {', '.join(f'{k}={v:.3g}' for k, v in group_means.items())}."
        )
    else:
        interpretation = (
            f"No statistically significant difference in {metric_col} across {group_col} "
            f"at α={alpha} (p={p_val:.4f}). Observed means: "
            f"{', '.join(f'{k}={v:.3g}' for k, v in group_means.items())}. "
            f"A real effect may still exist but this sample does not show clear evidence."
        )

    return {
        "success": True,
        "agent": "stats",
        "test": test_name,
        "group_column": group_col,
        "metric_column": metric_col,
        "n_groups": n_groups,
        "group_means": group_means,
        "group_sizes": {lab: int(len(g)) for lab, g in zip(labels, groups)},
        "statistic": stat_val,
        "p_value": p_val,
        "alpha": alpha,
        "significant": p_val < alpha,
        "interpretation": interpretation,
        "assumptions": assumptions,
        "summary": interpretation + "\n\nCaveats: " + " ".join(assumptions),
        "summary_for_rag": (
            f"Stats test: {test_name} on {metric_col} by {group_col}\n"
            f"p={p_val:.4f}, significant={p_val < alpha}\n{interpretation}"
        ),
    }


def run_correlation_analysis(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    nums = _numeric_cols(df)
    if not nums:
        return {"success": False, "error": "No numeric columns for correlation analysis", "agent": "stats"}

    if target_col is None or target_col not in df.columns:
        # Prefer known targets
        for kw in ("churn", "price", "revenue", "sales", "target", "label"):
            for c in df.columns:
                if kw in c.lower():
                    target_col = c
                    break
            if target_col:
                break
        if target_col is None:
            target_col = nums[-1]

    work = df.copy()
    # Encode binary-ish categoricals for correlation
    if target_col not in nums:
        if work[target_col].nunique() <= 10:
            work["_target_enc"] = pd.factorize(work[target_col].astype(str))[0].astype(float)
            target_use = "_target_enc"
        else:
            return {
                "success": False,
                "error": f"Target '{target_col}' is non-numeric with high cardinality",
                "agent": "stats",
            }
    else:
        target_use = target_col

    features = [c for c in nums if c != target_col]
    if not features:
        return {"success": False, "error": "No feature columns to correlate with target", "agent": "stats"}

    rankings = []
    y = pd.to_numeric(work[target_use], errors="coerce")
    for col in features:
        x = pd.to_numeric(work[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        rankings.append({
            "column": col,
            "correlation": float(r),
            "abs_correlation": abs(float(r)),
            "p_value": float(p),
        })
    rankings.sort(key=lambda d: d["abs_correlation"], reverse=True)

    lines = [f"Correlations with target `{target_col}` (Pearson):"]
    for item in rankings[:10]:
        strength = (
            "strong" if item["abs_correlation"] >= 0.7
            else "moderate" if item["abs_correlation"] >= 0.4
            else "weak"
        )
        direction = "positive" if item["correlation"] > 0 else "negative"
        lines.append(
            f"- `{item['column']}`: r={item['correlation']:.3f} ({strength} {direction}, p={item['p_value']:.4f})"
        )
    lines.append(
        "\nNote: correlation measures linear association only and does not imply causation. "
        "Small samples can produce unstable estimates."
    )
    if len(df) < 30:
        lines.append(f"Sample size is small (n={len(df)}); treat rankings as exploratory.")

    interpretation = "\n".join(lines)
    return {
        "success": True,
        "agent": "stats",
        "task": "correlation",
        "target_column": target_col,
        "rankings": rankings,
        "interpretation": interpretation,
        "summary": interpretation,
        "summary_for_rag": f"Correlation analysis vs {target_col}\n{interpretation}",
    }


def run_outlier_summary(df: pd.DataFrame) -> dict[str, Any]:
    nums = _numeric_cols(df)
    if not nums:
        return {"success": False, "error": "No numeric columns for outlier detection", "agent": "stats"}

    details = {}
    total = 0
    for col in nums:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 4:
            continue
        q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        count = int(mask.fillna(False).sum())
        if count == 0:
            continue
        total += count
        sample_vals = s[mask.fillna(False)].head(5).tolist()
        details[col] = {
            "count": count,
            "pct": round(100.0 * count / len(df), 2),
            "low": low,
            "high": high,
            "examples": [float(v) for v in sample_vals if pd.notna(v)],
        }

    if not details:
        text = "No IQR-based outliers detected in numeric columns. This does not guarantee the data is free of anomalies."
    else:
        lines = [f"Outlier summary (IQR method) — {len(details)} column(s) affected:"]
        for col, info in details.items():
            lines.append(
                f"- `{col}`: {info['count']} ({info['pct']}%) outside [{info['low']:.3g}, {info['high']:.3g}]; "
                f"examples: {info['examples']}"
            )
        lines.append(
            "\nOutliers may be data errors or genuine rare events. Review before removing. "
            "IQR is a heuristic and can flag valid extreme values in skewed distributions."
        )
        text = "\n".join(lines)

    return {
        "success": True,
        "agent": "stats",
        "task": "outliers",
        "columns_affected": list(details.keys()),
        "details": details,
        "total_flagged_cells": total,
        "interpretation": text,
        "summary": text,
        "summary_for_rag": f"Outlier analysis\n{text}",
    }


def _detect_stats_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("outlier", "anomal", "unusual value", "extreme")):
        return "outliers"
    if any(w in q for w in ("correlat", "associated with", "related to", "drives", "feature importance", "what's linked")):
        return "correlation"
    if any(w in q for w in ("significant", "difference", "differ", "compare", "t-test", "anova", "vs", "versus", "between")):
        return "comparison"
    if any(w in q for w in ("stat", "hypothesis", "p-value", "p value")):
        return "comparison"
    return "correlation"


def run_stats_analysis(df: pd.DataFrame, query: str = "") -> dict[str, Any]:
    """Route natural-language stats questions to the right procedure."""
    if df is None or df.empty:
        return {"success": False, "error": "No data available for statistical analysis", "agent": "stats"}

    intent = _detect_stats_intent(query or "")

    try:
        if intent == "outliers":
            result = run_outlier_summary(df)
        elif intent == "correlation":
            target = _find_column(df, query) if query else None
            # Prefer target-like columns when query mentions them
            for col in df.columns:
                if col.lower() in (query or "").lower():
                    target = col
                    break
            result = run_correlation_analysis(df, target)
        else:
            group_col, metric_col = _find_group_and_metric(df, query or "")
            if not group_col or not metric_col:
                # Fall back to correlation if comparison not possible
                result = run_correlation_analysis(df)
            else:
                result = run_group_comparison(df, group_col, metric_col)

        if result.get("success") and not result.get("summary"):
            result["summary"] = result.get("interpretation", "Analysis complete.")
        return result
    except Exception as e:
        return {"success": False, "error": f"Statistical analysis failed: {e}", "agent": "stats"}
