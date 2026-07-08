"""Data Quality Agent — profile, report issues, suggest and apply safe cleans."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from db.database import TABLE_NAME, load_dataframe_to_table


def _is_numeric_looking(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    return float(parsed.notna().mean()) >= 0.8


def _iqr_outliers(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return {"count": 0, "pct": 0.0, "bounds": None, "indices": []}
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return {"count": 0, "pct": 0.0, "bounds": (q1, q3), "indices": []}
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < low) | (series > high)
    # Align to original index for numeric-convertible values only
    numeric = pd.to_numeric(series, errors="coerce")
    mask = (numeric < low) | (numeric > high)
    indices = series.index[mask.fillna(False)].tolist()[:50]
    count = int(mask.fillna(False).sum())
    return {
        "count": count,
        "pct": round(100.0 * count / max(len(series), 1), 2),
        "bounds": (low, high),
        "indices": indices,
    }


def _categorical_inconsistencies(series: pd.Series, threshold: float = 0.75) -> list[dict[str, Any]]:
    """Flag near-duplicate category labels for human review (no auto-merge)."""
    if pd.api.types.is_numeric_dtype(series):
        return []
    values = [str(v).strip() for v in series.dropna().unique() if str(v).strip()]
    if len(values) < 2 or len(values) > 80:
        return []

    groups: list[list[str]] = []
    used = set()
    for i, a in enumerate(values):
        if a in used:
            continue
        group = [a]
        for b in values[i + 1 :]:
            if b in used:
                continue
            # Normalize common variants lightly for comparison only
            a_norm = re.sub(r"[^a-z0-9]", "", a.lower())
            b_norm = re.sub(r"[^a-z0-9]", "", b.lower())
            if not a_norm or not b_norm:
                continue
            ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
            if ratio >= threshold or (a_norm in b_norm or b_norm in a_norm) and min(len(a_norm), len(b_norm)) >= 2:
                group.append(b)
                used.add(b)
        if len(group) > 1:
            used.add(a)
            groups.append(sorted(group))

    return [{"values": g, "note": "Possible synonyms — review before merging"} for g in groups[:15]]


def analyze_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Profile the dataframe and return a structured quality report."""
    if df is None or df.empty:
        return {
            "success": False,
            "error": "No data available for quality analysis",
            "agent": "quality",
        }

    n_rows, n_cols = len(df), len(df.columns)
    missing = {}
    for col in df.columns:
        cnt = int(df[col].isnull().sum())
        if cnt > 0:
            missing[col] = {"count": cnt, "pct": round(100.0 * cnt / n_rows, 2)}

    dup_mask = df.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())

    type_issues = []
    for col in df.columns:
        if _is_numeric_looking(df[col]):
            type_issues.append({
                "column": col,
                "issue": "numbers_as_text",
                "suggestion": f"Cast '{col}' to numeric",
            })

    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        info = _iqr_outliers(df[col])
        if info["count"] > 0:
            outliers[col] = info

    cat_issues = {}
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        groups = _categorical_inconsistencies(df[col])
        if groups:
            cat_issues[col] = groups

    suggestions = []
    if n_duplicates > 0:
        suggestions.append({
            "action": "drop_duplicates",
            "description": f"Remove {n_duplicates} exact duplicate row(s)",
            "auto_clean": True,
        })
    for col, info in missing.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            suggestions.append({
                "action": "impute_median",
                "column": col,
                "description": f"Impute missing in '{col}' with median ({info['pct']}% missing)",
                "auto_clean": True,
            })
        else:
            suggestions.append({
                "action": "impute_mode",
                "column": col,
                "description": f"Impute missing in '{col}' with mode ({info['pct']}% missing)",
                "auto_clean": True,
            })
    for issue in type_issues:
        suggestions.append({
            "action": "fix_dtype",
            "column": issue["column"],
            "description": issue["suggestion"],
            "auto_clean": True,
        })
    for col, groups in cat_issues.items():
        suggestions.append({
            "action": "review_categories",
            "column": col,
            "description": f"Review inconsistent categories in '{col}' ({len(groups)} group(s)) — not auto-merged",
            "auto_clean": False,
        })
    for col, info in outliers.items():
        suggestions.append({
            "action": "review_outliers",
            "column": col,
            "description": f"Review {info['count']} outlier(s) in '{col}' (IQR method) — not auto-removed",
            "auto_clean": False,
        })

    score = 100.0
    score -= min(30.0, sum(m["pct"] for m in missing.values()) / max(n_cols, 1))
    score -= min(20.0, 100.0 * n_duplicates / max(n_rows, 1))
    score -= min(15.0, 5.0 * len(type_issues))
    score -= min(15.0, sum(o["pct"] for o in outliers.values()) / max(len(outliers) or 1, 1) * 0.3)
    score = max(0.0, round(score, 1))

    summary_lines = [
        f"Data quality score: {score}/100",
        f"Rows: {n_rows}, Columns: {n_cols}",
        f"Missing columns: {len(missing)}",
        f"Duplicate rows: {n_duplicates}",
        f"Type issues: {len(type_issues)}",
        f"Columns with outliers: {len(outliers)}",
        f"Categorical inconsistencies flagged: {len(cat_issues)}",
    ]

    return {
        "success": True,
        "agent": "quality",
        "row_count": n_rows,
        "column_count": n_cols,
        "quality_score": score,
        "missing": missing,
        "duplicate_count": n_duplicates,
        "duplicate_indices": df.index[dup_mask].tolist()[:100],
        "type_issues": type_issues,
        "outliers": outliers,
        "categorical_issues": cat_issues,
        "suggestions": suggestions,
        "summary": "\n".join(summary_lines),
        "summary_for_rag": "Data Quality Report\n" + "\n".join(summary_lines),
    }


def apply_auto_clean(df: pd.DataFrame, engine=None) -> dict[str, Any]:
    """
    Apply safe defaults: drop exact duplicates, median/mode impute, fix numeric dtypes.
    Does NOT remove outliers or merge categories (those need human review).
    """
    if df is None or df.empty:
        return {"success": False, "error": "No data to clean", "agent": "quality"}

    cleaned = df.copy()
    log: list[str] = []
    before_rows = len(cleaned)

    n_dup = int(cleaned.duplicated().sum())
    if n_dup > 0:
        cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
        log.append(f"Dropped {n_dup} exact duplicate row(s) ({before_rows} → {len(cleaned)})")

    for col in list(cleaned.columns):
        if _is_numeric_looking(cleaned[col]):
            cleaned[col] = pd.to_numeric(
                cleaned[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            log.append(f"Cast '{col}' to numeric")

    for col in cleaned.columns:
        miss = int(cleaned[col].isnull().sum())
        if miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            med = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(med)
            log.append(f"Imputed {miss} missing value(s) in '{col}' with median ({med})")
        else:
            mode = cleaned[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill)
            log.append(f"Imputed {miss} missing value(s) in '{col}' with mode ({fill})")

    if engine is not None:
        load_dataframe_to_table(cleaned, engine, TABLE_NAME, if_exists="replace")

    report = analyze_data_quality(cleaned)
    return {
        "success": True,
        "agent": "quality",
        "dataframe": cleaned,
        "actions_log": log,
        "before_rows": before_rows,
        "after_rows": len(cleaned),
        "quality_report": report,
        "summary": (
            "Auto-clean applied:\n" + ("\n".join(f"- {a}" for a in log) if log else "- No changes needed")
            + f"\n\nUpdated quality score: {report.get('quality_score', 'n/a')}/100"
        ),
        "summary_for_rag": (
            "Auto-clean applied:\n" + "\n".join(log)
            + f"\nNew quality score: {report.get('quality_score')}"
        ),
    }


def format_quality_markdown(report: dict[str, Any]) -> str:
    """Human-readable quality report for chat / UI."""
    if not report.get("success"):
        return report.get("error", "Quality analysis failed")

    lines = [
        f"### Data Quality Report",
        f"**Score:** {report['quality_score']}/100  |  "
        f"**Rows:** {report['row_count']}  |  **Columns:** {report['column_count']}",
        "",
    ]

    if report.get("missing"):
        lines.append("**Missing values**")
        for col, info in report["missing"].items():
            lines.append(f"- `{col}`: {info['count']} ({info['pct']}%)")
    else:
        lines.append("**Missing values:** none detected")

    lines.append(f"\n**Exact duplicate rows:** {report.get('duplicate_count', 0)}")

    if report.get("type_issues"):
        lines.append("\n**Type inconsistencies**")
        for issue in report["type_issues"]:
            lines.append(f"- `{issue['column']}`: {issue['issue']} — {issue['suggestion']}")

    if report.get("outliers"):
        lines.append("\n**Outliers (IQR)**")
        for col, info in report["outliers"].items():
            lines.append(f"- `{col}`: {info['count']} values ({info['pct']}%) outside [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]")

    if report.get("categorical_issues"):
        lines.append("\n**Categorical values to review** (not auto-merged)")
        for col, groups in report["categorical_issues"].items():
            for g in groups:
                lines.append(f"- `{col}`: {', '.join(repr(v) for v in g['values'])}")

    if report.get("suggestions"):
        lines.append("\n**Suggested fixes**")
        for s in report["suggestions"]:
            flag = "✓ auto-clean" if s.get("auto_clean") else "👁 review only"
            lines.append(f"- [{flag}] {s['description']}")

    return "\n".join(lines)
