"""ML Agent — EDA, auto task detection, model training, summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from agents.llm_client import chat_completion
from utils.charts import (
    correlation_heatmap,
    distribution_chart,
    feature_importance_chart,
    groupby_bar_chart,
    missing_values_chart,
    pca_scatter,
    time_series_chart,
)

TARGET_KEYWORDS = {
    "target", "label", "class", "outcome", "churn", "survived",
    "price", "salary", "income", "revenue", "sales", "amount",
    "score", "rating", "default", "fraud", "diagnosis",
}


def detect_target_column(df: pd.DataFrame, user_query: str = "") -> str | None:
    query_lower = user_query.lower()
    for col in df.columns:
        if col.lower() in query_lower:
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in TARGET_KEYWORDS):
            return col
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) == 1:
        return cat_cols[0]
    low_card_numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].nunique() <= 10 and df[c].nunique() >= 2
    ]
    if len(low_card_numeric) == 1:
        return low_card_numeric[0]
    return None


def detect_task_type(df: pd.DataFrame, target_col: str | None, user_query: str = "") -> str:
    query_lower = user_query.lower()
    if any(w in query_lower for w in ["cluster", "segment", "group", "unsupervised"]):
        return "clustering"
    if target_col is None:
        return "clustering"
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
        return "regression"
    return "classification"


def _detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(k in str(col).lower() for k in ("date", "time", "month", "year", "day", "timestamp")):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if len(df) and parsed.notna().mean() >= 0.8:
                return col
    for col in df.select_dtypes(include=["object", "string"]).columns:
        # Skip high-cardinality free text / obvious non-dates
        if df[col].nunique() > min(50, max(3, len(df) // 2)):
            continue
        sample = df[col].dropna().astype(str).head(30)
        if sample.empty:
            continue
        # Heuristic: values should look date-like
        if not sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}", regex=True).mean() >= 0.5:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if len(df) and parsed.notna().mean() >= 0.8:
            return col
    return None


def _suggest_groupings(df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]) -> list[dict[str, str]]:
    """Suggest up to 3 useful group-by chart specs."""
    suggestions = []
    if not numeric_cols or not cat_cols:
        return suggestions
    # Prefer metrics that look like KPIs
    metrics = sorted(
        numeric_cols,
        key=lambda c: (
            0 if any(k in c.lower() for k in ("price", "revenue", "sales", "amount", "charge", "income")) else 1,
            -df[c].nunique() if c in df.columns else 0,
        ),
    )
    cats = sorted(cat_cols, key=lambda c: df[c].nunique())
    pairs = []
    for cat in cats:
        if df[cat].nunique() < 2 or df[cat].nunique() > 30:
            continue
        for metric in metrics:
            if metric == cat:
                continue
            pairs.append((cat, metric))
            if len(pairs) >= 3:
                break
        if len(pairs) >= 3:
            break
    for cat, metric in pairs[:3]:
        suggestions.append({"group_col": cat, "metric_col": metric, "agg": "mean"})
    return suggestions


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # Full descriptive statistics for numeric columns
    stats = {}
    describe_rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        col_stats = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "count": int(s.count()),
        }
        stats[col] = col_stats
        describe_rows.append({"column": col, **col_stats})

    describe_table = pd.DataFrame(describe_rows) if describe_rows else pd.DataFrame()

    cat_summary = {}
    for col in cat_cols:
        cat_summary[col] = {str(k): int(v) for k, v in df[col].value_counts().head(15).items()}

    # Correlation matrix values (for reports) + heatmap
    corr_matrix = None
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(3)

    charts: dict[str, Any] = {
        "missing": missing_values_chart(df),
        "correlation": correlation_heatmap(df),
        "distributions": {},
        "groupbys": {},
        "timeseries": {},
    }

    # Histograms for numeric, bar charts for categorical (top N)
    plot_cols = numeric_cols[:8] + cat_cols[:6]
    for col in plot_cols:
        chart = distribution_chart(df, col)
        if chart:
            charts["distributions"][col] = chart

    # Time-series if a date column exists
    date_col = _detect_date_column(df)
    if date_col and numeric_cols:
        metric_candidates = [
            c for c in numeric_cols
            if any(k in c.lower() for k in ("price", "revenue", "sales", "amount", "charge", "value", "total"))
        ] or numeric_cols[:2]
        for metric in metric_candidates[:2]:
            ts = time_series_chart(df, date_col, metric)
            if ts:
                charts["timeseries"][f"{metric}_over_time"] = ts

    # Auto group-by summaries (2–3 charts)
    groupings = _suggest_groupings(df, numeric_cols, cat_cols)
    for g in groupings:
        fig = groupby_bar_chart(df, g["group_col"], g["metric_col"], agg=g["agg"])
        if fig:
            key = f"{g['metric_col']}_by_{g['group_col']}"
            charts["groupbys"][key] = fig

    eda_text = _build_eda_summary(df, missing_report, stats, cat_summary, date_col, groupings)
    return {
        "missing_report": missing_report,
        "numeric_stats": stats,
        "describe_table": describe_table,
        "correlation_matrix": corr_matrix,
        "categorical_summary": cat_summary,
        "charts": charts,
        "summary_text": eda_text,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "date_column": date_col,
        "groupings": groupings,
    }


def _build_eda_summary(
    df: pd.DataFrame,
    missing: dict,
    stats: dict,
    cat_summary: dict,
    date_col: str | None = None,
    groupings: list | None = None,
) -> str:
    lines = [
        f"Dataset: {len(df)} rows, {len(df.columns)} columns",
        f"Numeric columns: {', '.join(stats.keys()) or 'none'}",
        f"Categorical columns: {', '.join(cat_summary.keys()) or 'none'}",
    ]
    if date_col:
        lines.append(f"Date/time column detected: {date_col}")
    if missing:
        lines.append("Missing values:")
        for col, info in missing.items():
            lines.append(f"  - {col}: {info['count']} ({info['pct']}%)")
    else:
        lines.append("No missing values detected.")
    if stats:
        lines.append("Numeric highlights (mean / median / std):")
        for col, s in list(stats.items())[:6]:
            lines.append(
                f"  - {col}: mean={s['mean']:.3g}, median={s['median']:.3g}, std={s.get('std', 0):.3g}, "
                f"range=[{s['min']:.3g}, {s['max']:.3g}]"
            )
    if groupings:
        lines.append("Suggested groupings: " + ", ".join(
            f"{g['metric_col']} by {g['group_col']}" for g in groupings
        ))
    return "\n".join(lines)


HIGH_CARDINALITY_THRESHOLD = 40


def _extract_date_parts(work: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Expand datetime columns into useful parts; drop original datetime."""
    notes: list[str] = []
    out = work.copy()
    for col in list(out.columns):
        series = out[col]
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        if not is_dt and series.dtype == object:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                series = parsed
                is_dt = True
        if not is_dt:
            continue
        out[f"{col}_month"] = pd.to_datetime(series, errors="coerce").dt.month
        out[f"{col}_dow"] = pd.to_datetime(series, errors="coerce").dt.dayofweek
        out[f"{col}_year"] = pd.to_datetime(series, errors="coerce").dt.year
        out = out.drop(columns=[col])
        notes.append(f"Expanded date column `{col}` → month/day-of-week/year features")
    return out, notes


def _prepare_features(
    df: pd.DataFrame,
    target_col: str | None,
    exclude_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str], list[str]]:
    """
    Returns X, y, feature_notes, high_card_flags.
    """
    exclude_cols = exclude_cols or set()
    work = df.copy()
    y = None
    notes: list[str] = []
    high_card: list[str] = []

    drop_ids = [c for c in work.columns if c != target_col and (
        c in exclude_cols
        or c.lower() in ("id", "index")
        or (c.lower().endswith("_id") and work[c].nunique() > max(20, int(0.5 * len(work))))
    )]
    if drop_ids:
        work = work.drop(columns=[c for c in drop_ids if c in work.columns])
        notes.append(f"Excluded likely ID columns from features: {', '.join(drop_ids)}")

    if target_col and target_col in work.columns:
        y = work.pop(target_col)

    work, date_notes = _extract_date_parts(work)
    notes.extend(date_notes)

    for col in list(work.select_dtypes(include=["object", "category", "string"]).columns):
        nuniq = work[col].nunique(dropna=True)
        if nuniq > HIGH_CARDINALITY_THRESHOLD:
            high_card.append(col)
            work = work.drop(columns=[col])
            notes.append(
                f"Flagged/dropped high-cardinality categorical `{col}` ({nuniq} levels) "
                "to reduce overfitting risk — not used as a raw feature."
            )

    for col in work.select_dtypes(include=["object", "category"]).columns:
        work[col] = work[col].astype(str).fillna("missing")
    work = work.fillna(work.median(numeric_only=True))
    for col in work.select_dtypes(exclude=[np.number]).columns:
        work[col] = work[col].fillna("missing")
    return work, y, notes, high_card


def _encode_for_ml(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    parts = []
    feature_names = []

    if numeric_cols:
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(X[numeric_cols].astype(float))
        parts.append(num_arr)
        feature_names.extend(numeric_cols)

    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
        cat_arr = ohe.fit_transform(X[cat_cols])
        parts.append(cat_arr)
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

    if not parts:
        return np.zeros((len(X), 1)), ["empty"]

    return np.hstack(parts), feature_names


def _importances_from_model(model, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in vals]))
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        if len(coef) == len(feature_names):
            return dict(zip(feature_names, [float(abs(v)) for v in coef]))
    return {f: 0.0 for f in feature_names}


def _plain_language_drivers(importances: dict[str, float], top_n: int = 5) -> str:
    if not importances:
        return "Feature importance was not available for the winning model."
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not ranked or ranked[0][1] == 0:
        return "No strong feature drivers were identified."
    parts = [f"**{name}**" for name, _ in ranked[:3]]
    if len(parts) == 1:
        return f"The model relied most on {parts[0]} to make predictions."
    if len(parts) == 2:
        return f"The model relied most on {parts[0]} and {parts[1]} to make predictions."
    return (
        f"The model relied most on {parts[0]}, {parts[1]}, and {parts[2]} to make predictions "
        f"(top signals among {len(importances)} features)."
    )


def _overfit_flags(n_rows: int, metrics: dict, task: str, n_features: int) -> list[str]:
    flags = []
    if n_rows < 50:
        flags.append(
            f"Small dataset (n={n_rows}): metrics can be unstable and may look better than real-world performance."
        )
    if task == "classification":
        acc = metrics.get("accuracy")
        if acc is not None and acc >= 0.98 and n_rows < 200:
            flags.append(
                f"Accuracy is very high ({acc:.3f}) on a modest sample — treat as a possible overfit / lucky split, not production-ready."
            )
        if acc is not None and acc >= 0.999:
            flags.append("Near-perfect accuracy often signals leakage or a trivial target — double-check features.")
    if task == "regression":
        r2 = metrics.get("r2")
        if r2 is not None and r2 >= 0.98 and n_rows < 200:
            flags.append(
                f"R² is very high ({r2:.3f}) on a modest sample — possible overfit; validate on fresh data."
            )
    if n_features > n_rows:
        flags.append(
            f"More features ({n_features}) than rows ({n_rows}) increases overfit risk."
        )
    return flags


def train_classification(X: np.ndarray, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    """AutoML: logistic regression, random forest, XGBoost with tiny grids; pick best F1."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    _, class_counts = np.unique(y_enc, return_counts=True)
    strat = y_enc if len(class_counts) > 1 and class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=strat
    )

    candidates = []
    # Logistic Regression
    for C in (0.5, 2.0):
        try:
            m = LogisticRegression(max_iter=400, C=C, random_state=42)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"LogisticRegression(C={C})",
                "model": m,
                "metrics": {
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                },
            })
        except Exception:
            pass
    # Random Forest — small grid
    for params in ParameterGrid({"n_estimators": [50, 100], "max_depth": [3, 6]}):
        try:
            m = RandomForestClassifier(random_state=42, **params)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"RandomForest(n={params['n_estimators']}, depth={params['max_depth']})",
                "model": m,
                "metrics": {
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                },
            })
        except Exception:
            pass
    # XGBoost — small grid
    for params in ParameterGrid({"n_estimators": [50, 100], "max_depth": [3, 4], "learning_rate": [0.1]}):
        try:
            m = XGBClassifier(
                eval_metric="logloss", random_state=42, **params,
            )
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"XGBoost(n={params['n_estimators']}, depth={params['max_depth']})",
                "model": m,
                "metrics": {
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                },
            })
        except Exception:
            pass

    if not candidates:
        raise RuntimeError("No classification candidate models trained successfully")

    best = max(candidates, key=lambda c: c["metrics"]["f1"])
    importances = _importances_from_model(best["model"], feature_names)
    leaderboard = [
        {"model": c["name"], **c["metrics"]}
        for c in sorted(candidates, key=lambda c: c["metrics"]["f1"], reverse=True)
    ]
    return {
        "model": best["model"],
        "model_name": best["name"],
        "metrics": best["metrics"],
        "feature_importances": importances,
        "leaderboard": leaderboard,
        "task": "classification",
        "why_best": (
            f"**{best['name']}** scored highest F1 ({best['metrics']['f1']:.3f}) "
            f"among {len(candidates)} candidates on a hold-out test split."
        ),
    }


def train_regression(X: np.ndarray, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    """AutoML: Ridge, RandomForest, XGBoost; pick best R² (then lowest RMSE)."""
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna()
    X, y_num = X[mask.values], y_num[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)

    candidates = []
    for alpha in (0.5, 2.0):
        try:
            m = Ridge(alpha=alpha)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"Ridge(alpha={alpha})",
                "model": m,
                "metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "r2": float(r2_score(y_test, pred)),
                },
            })
        except Exception:
            pass
    for params in ParameterGrid({"n_estimators": [50, 100], "max_depth": [3, 6]}):
        try:
            m = RandomForestRegressor(random_state=42, **params)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"RandomForest(n={params['n_estimators']}, depth={params['max_depth']})",
                "model": m,
                "metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "r2": float(r2_score(y_test, pred)),
                },
            })
        except Exception:
            pass
    for params in ParameterGrid({"n_estimators": [50, 100], "max_depth": [3, 4], "learning_rate": [0.1]}):
        try:
            m = XGBRegressor(random_state=42, **params)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            candidates.append({
                "name": f"XGBoost(n={params['n_estimators']}, depth={params['max_depth']})",
                "model": m,
                "metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "r2": float(r2_score(y_test, pred)),
                },
            })
        except Exception:
            pass

    if not candidates:
        raise RuntimeError("No regression candidate models trained successfully")

    best = max(candidates, key=lambda c: (c["metrics"]["r2"], -c["metrics"]["rmse"]))
    importances = _importances_from_model(best["model"], feature_names)
    leaderboard = [
        {"model": c["name"], **c["metrics"]}
        for c in sorted(candidates, key=lambda c: c["metrics"]["r2"], reverse=True)
    ]
    return {
        "model": best["model"],
        "model_name": best["name"],
        "metrics": best["metrics"],
        "feature_importances": importances,
        "leaderboard": leaderboard,
        "task": "regression",
        "why_best": (
            f"**{best['name']}** achieved the best hold-out R² ({best['metrics']['r2']:.3f}, "
            f"RMSE={best['metrics']['rmse']:.3g}) among {len(candidates)} candidates."
        ),
    }


def train_clustering(X: np.ndarray, n_clusters: int = 3) -> dict[str, Any]:
    n_clusters = min(n_clusters, max(2, len(X) // 5))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return {
        "model": model,
        "labels": labels,
        "pca_coords": coords,
        "metrics": {"silhouette_score": sil, "n_clusters": n_clusters},
        "pca_chart": pca_scatter(coords, labels),
        "task": "clustering",
    }


def generate_ml_summary(
    task: str,
    target_col: str | None,
    metrics: dict,
    eda_summary: str,
    user_query: str,
    model_name: str = "",
    why_best: str = "",
    drivers: str = "",
    risk_flags: list[str] | None = None,
    business_context: str = "",
    leaderboard: list | None = None,
) -> tuple[str, str | None]:
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    lb = ""
    if leaderboard:
        lb = "; ".join(
            f"{r.get('model')}: " + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.items() if k != "model"
            )
            for r in leaderboard[:4]
        )
    risks = "; ".join(risk_flags or []) or "none called out"
    ctx = business_context.strip()
    prompt = f"""Summarize these AutoML results in 4-6 sentences for a business user.
Be honest about limitations. Mention the winning model and why, top drivers, and any overfit risks.

Task: {task}
Target: {target_col or 'N/A'}
Winner: {model_name}
Why best: {why_best}
Metrics: {metrics_str}
Leaderboard: {lb}
Drivers: {drivers}
Risk flags: {risks}
Business context: {ctx or 'none'}
EDA: {eda_summary}
User request: {user_query}"""

    summary, err = chat_completion([
        {
            "role": "system",
            "content": (
                "You are a data science communicator. Clear, actionable, never oversell model quality. "
                "If risks are listed, state them plainly."
            ),
        },
        {"role": "user", "content": prompt},
    ], max_tokens=500)
    if summary:
        return summary, err
    # Intentional degraded-but-deterministic answer when summarization is unavailable;
    # preserve the provider error separately instead of leaking it into user-facing text.
    fallback = (
        f"Completed {task} with {model_name or 'baseline'}. Metrics: {metrics_str}. "
        f"{why_best} {drivers} Risks: {risks}."
    )
    return fallback, err


def run_ml_analysis(
    df: pd.DataFrame,
    user_query: str = "",
    business_context: str = "",
    exclude_feature_cols: set[str] | None = None,
) -> dict[str, Any]:
    """Full ML agent pipeline with AutoML comparison + risk flags."""
    try:
        if df is None or df.empty:
            return {"success": False, "error": "No data available for ML analysis", "agent": "ml"}

        eda = run_eda(df)
        target_col = detect_target_column(df, user_query)
        task = detect_task_type(df, target_col, user_query)

        X_df, y, feat_notes, high_card = _prepare_features(df, target_col, exclude_feature_cols)
        X, feature_names = _encode_for_ml(X_df)

        if X.shape[0] < 5:
            return {"success": False, "error": "Not enough rows for ML analysis (minimum 5)", "agent": "ml"}

        model_result: dict[str, Any] = {}
        charts = dict(eda["charts"])

        if task == "classification" and y is not None:
            model_result = train_classification(X, y, feature_names)
        elif task == "regression" and y is not None:
            model_result = train_regression(X, y, feature_names)
        else:
            task = "clustering"
            n_clusters = min(5, max(2, int(np.sqrt(len(df)))))
            model_result = train_clustering(X, n_clusters)
            model_result["model_name"] = "KMeans"
            model_result["why_best"] = "Unsupervised clustering (no labeled target detected)."
            model_result["leaderboard"] = [{"model": "KMeans", **model_result.get("metrics", {})}]
            if "pca_chart" in model_result:
                charts["pca"] = model_result["pca_chart"]

        importances = model_result.get("feature_importances") or {}
        if importances:
            charts["feature_importance"] = feature_importance_chart(importances)

        drivers = _plain_language_drivers(importances)
        risks = _overfit_flags(
            n_rows=X.shape[0],
            metrics=model_result.get("metrics", {}),
            task=task,
            n_features=X.shape[1],
        )
        if high_card:
            risks.append(
                "High-cardinality categoricals were dropped to limit overfit: " + ", ".join(high_card)
            )

        summary, sum_err = generate_ml_summary(
            task,
            target_col,
            model_result.get("metrics", {}),
            eda["summary_text"],
            user_query,
            model_name=model_result.get("model_name", ""),
            why_best=model_result.get("why_best", ""),
            drivers=drivers,
            risk_flags=risks,
            business_context=business_context,
            leaderboard=model_result.get("leaderboard"),
        )
        if drivers and drivers not in (summary or ""):
            summary = (summary or "") + f"\n\n**Drivers:** {drivers}"
        if risks:
            summary = (summary or "") + "\n\n**Caveats:**\n" + "\n".join(f"- {r}" for r in risks)
        if feat_notes:
            summary = (summary or "") + "\n\n**Feature prep:**\n" + "\n".join(f"- {n}" for n in feat_notes)

        return {
            "success": True,
            "agent": "ml",
            "task": task,
            "target_column": target_col,
            "eda": eda,
            "metrics": model_result.get("metrics", {}),
            "model_name": model_result.get("model_name"),
            "leaderboard": model_result.get("leaderboard"),
            "why_best": model_result.get("why_best"),
            "drivers": drivers,
            "risk_flags": risks,
            "feature_notes": feat_notes,
            "charts": charts,
            "summary": summary,
            "summary_error": sum_err,
            "summary_for_rag": (
                f"ML Task: {task}\n"
                f"Target: {target_col or 'N/A'}\n"
                f"Winner: {model_result.get('model_name')}\n"
                f"Metrics: {model_result.get('metrics', {})}\n"
                f"Drivers: {drivers}\n"
                f"Risks: {risks}\n"
                f"Summary: {summary}"
            ),
        }
    except Exception as e:
        return {"success": False, "error": f"ML analysis failed: {e}", "agent": "ml"}
