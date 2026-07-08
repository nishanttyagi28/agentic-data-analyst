"""ML Agent — EDA, auto task detection, model training, summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from agents.llm_client import chat_completion
from utils.charts import (
    correlation_heatmap,
    distribution_chart,
    feature_importance_chart,
    missing_values_chart,
    pca_scatter,
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


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()) if df[col].notna().any() else None,
            "std": float(df[col].std()) if df[col].notna().any() else None,
            "min": float(df[col].min()) if df[col].notna().any() else None,
            "max": float(df[col].max()) if df[col].notna().any() else None,
        }

    cat_summary = {}
    for col in cat_cols:
        cat_summary[col] = df[col].value_counts().head(10).to_dict()

    charts = {
        "missing": missing_values_chart(df),
        "correlation": correlation_heatmap(df),
        "distributions": {},
    }
    for col in (numeric_cols + cat_cols)[:6]:
        chart = distribution_chart(df, col)
        if chart:
            charts["distributions"][col] = chart

    eda_text = _build_eda_summary(df, missing_report, stats, cat_summary)
    return {
        "missing_report": missing_report,
        "numeric_stats": stats,
        "categorical_summary": cat_summary,
        "charts": charts,
        "summary_text": eda_text,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
    }


def _build_eda_summary(
    df: pd.DataFrame,
    missing: dict,
    stats: dict,
    cat_summary: dict,
) -> str:
    lines = [
        f"Dataset: {len(df)} rows, {len(df.columns)} columns",
        f"Numeric columns: {', '.join(stats.keys()) or 'none'}",
        f"Categorical columns: {', '.join(cat_summary.keys()) or 'none'}",
    ]
    if missing:
        lines.append("Missing values:")
        for col, info in missing.items():
            lines.append(f"  - {col}: {info['count']} ({info['pct']}%)")
    else:
        lines.append("No missing values detected.")
    return "\n".join(lines)


def _prepare_features(df: pd.DataFrame, target_col: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
    work = df.copy()
    y = None
    if target_col and target_col in work.columns:
        y = work.pop(target_col)
    for col in work.select_dtypes(include=["object", "category"]).columns:
        work[col] = work[col].astype(str).fillna("missing")
    work = work.fillna(work.median(numeric_only=True))
    for col in work.select_dtypes(exclude=[np.number]).columns:
        work[col] = work[col].fillna("missing")
    return work, y


def _encode_for_ml(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    parts = []
    feature_names = []

    if numeric_cols:
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(X[numeric_cols])
        parts.append(num_arr)
        feature_names.extend(numeric_cols)

    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_arr = ohe.fit_transform(X[cat_cols])
        parts.append(cat_arr)
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

    if not parts:
        return np.zeros((len(X), 1)), ["empty"]

    return np.hstack(parts), feature_names


def train_classification(X: np.ndarray, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    )
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return {"model": model, "metrics": metrics, "feature_importances": importances, "task": "classification"}


def train_regression(X: np.ndarray, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna()
    X, y_num = X[mask.values], y_num[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {"rmse": rmse, "r2": float(r2_score(y_test, y_pred))}
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return {"model": model, "metrics": metrics, "feature_importances": importances, "task": "regression"}


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
) -> tuple[str, str | None]:
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    prompt = f"""Summarize these ML analysis results in 3-5 sentences for a business user.

Task: {task}
Target column: {target_col or 'N/A (unsupervised)'}
Metrics: {metrics_str}
EDA overview: {eda_summary}
User request: {user_query}"""

    summary, err = chat_completion([
        {"role": "system", "content": "You are a data science communicator. Be clear and actionable."},
        {"role": "user", "content": prompt},
    ])
    if summary:
        return summary, err
    fallback = f"Completed {task} analysis. Metrics: {metrics_str}"
    return fallback, err


def run_ml_analysis(
    df: pd.DataFrame,
    user_query: str = "",
) -> dict[str, Any]:
    """Full ML agent pipeline."""
    try:
        if df is None or df.empty:
            return {"success": False, "error": "No data available for ML analysis", "agent": "ml"}

        eda = run_eda(df)
        target_col = detect_target_column(df, user_query)
        task = detect_task_type(df, target_col, user_query)

        X_df, y = _prepare_features(df, target_col)
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
            if "pca_chart" in model_result:
                charts["pca"] = model_result["pca_chart"]

        if model_result.get("feature_importances"):
            charts["feature_importance"] = feature_importance_chart(model_result["feature_importances"])

        summary, sum_err = generate_ml_summary(
            task, target_col, model_result.get("metrics", {}), eda["summary_text"], user_query
        )

        return {
            "success": True,
            "agent": "ml",
            "task": task,
            "target_column": target_col,
            "eda": eda,
            "metrics": model_result.get("metrics", {}),
            "charts": charts,
            "summary": summary,
            "summary_error": sum_err,
            "summary_for_rag": (
                f"ML Task: {task}\n"
                f"Target: {target_col or 'N/A'}\n"
                f"EDA: {eda['summary_text']}\n"
                f"Metrics: {model_result.get('metrics', {})}\n"
                f"Summary: {summary}"
            ),
        }
    except Exception as e:
        return {"success": False, "error": f"ML analysis failed: {e}", "agent": "ml"}