"""Forecasting Agent — lightweight trend forecasts with uncertainty bands."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def detect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(k in col.lower() for k in ("date", "time", "month", "year", "day", "period", "timestamp")):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                return col
    # Try object columns that look date-like (avoid parsing free text / categories)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique() > min(50, max(3, len(df) // 2)):
            continue
        sample = df[col].dropna().astype(str).head(30)
        if sample.empty:
            continue
        if not sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}", regex=True).mean() >= 0.5:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if len(df) > 0 and parsed.notna().mean() >= 0.8:
            return col
    return None


def detect_metric_column(df: pd.DataFrame, query: str = "", exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    q = (query or "").lower()
    nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not nums:
        return None
    for col in nums:
        if col.lower() in q or col.lower().replace("_", " ") in q:
            return col
    for kw in ("revenue", "sales", "amount", "price", "charges", "income", "value", "count", "total"):
        for col in nums:
            if kw in col.lower():
                return col
    return nums[0]


def _horizon_from_query(query: str, freq_days: float) -> int:
    q = (query or "").lower()
    if "year" in q:
        return max(1, int(round(365 / max(freq_days, 1))))
    if "quarter" in q:
        return max(1, int(round(90 / max(freq_days, 1))))
    if "week" in q:
        return max(1, int(round(7 / max(freq_days, 1))))
    if "month" in q:
        return max(1, int(round(30 / max(freq_days, 1))))
    if "day" in q:
        return 7
    # default: ~30% of history length, at least 3 points
    return 3


def run_forecast(df: pd.DataFrame, query: str = "", horizon: int | None = None) -> dict[str, Any]:
    """
    Linear-trend forecast on a datetime + numeric series.
    Returns point forecast + approximate 95% prediction band from residual std.
    """
    if df is None or df.empty:
        return {"success": False, "error": "No data available for forecasting", "agent": "forecast"}

    date_col = detect_datetime_column(df)
    if date_col is None:
        # Synthetic time index if user still asks for forecast
        metric = detect_metric_column(df, query)
        if metric is None:
            return {
                "success": False,
                "error": "No date/time column and no numeric metric found for forecasting",
                "agent": "forecast",
            }
        work = df[[metric]].copy().dropna()
        if len(work) < 3:
            return {"success": False, "error": "Need at least 3 data points to forecast", "agent": "forecast"}
        work = work.reset_index(drop=True)
        work["_t"] = np.arange(len(work))
        date_col = "_t"
        use_synthetic = True
        freq_days = 1.0
    else:
        use_synthetic = False
        metric = detect_metric_column(df, query, exclude={date_col})
        if metric is None:
            return {"success": False, "error": "No numeric metric found to forecast", "agent": "forecast"}
        work = df[[date_col, metric]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna().sort_values(date_col)
        if len(work) < 3:
            return {"success": False, "error": "Need at least 3 dated observations to forecast", "agent": "forecast"}
        # Aggregate duplicates on same date
        work = work.groupby(date_col, as_index=False)[metric].mean()
        deltas = work[date_col].diff().dt.total_seconds().dropna() / 86400.0
        freq_days = float(deltas.median()) if len(deltas) else 1.0
        if freq_days <= 0 or not np.isfinite(freq_days):
            freq_days = 1.0
        work["_t"] = (work[date_col] - work[date_col].min()).dt.total_seconds() / 86400.0

    y = work[metric].values.astype(float)
    X = work["_t"].values.reshape(-1, 1).astype(float)
    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat
    resid_std = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float(np.std(residuals))
    if not np.isfinite(resid_std) or resid_std == 0:
        resid_std = max(abs(float(np.mean(y))) * 0.05, 1e-6)

    n_ahead = horizon if horizon is not None else _horizon_from_query(query, freq_days)
    n_ahead = max(1, min(int(n_ahead), 36))

    last_t = float(X[-1, 0])
    future_t = np.array([last_t + freq_days * (i + 1) for i in range(n_ahead)]).reshape(-1, 1)
    pred = model.predict(future_t)
    # Simple expanding uncertainty with horizon (rule of thumb, not full prediction intervals)
    z = 1.96
    scales = 1.0 + 0.1 * np.arange(1, n_ahead + 1)
    lower = pred - z * resid_std * scales
    upper = pred + z * resid_std * scales

    if use_synthetic:
        hist_x = list(range(len(work)))
        fut_x = list(range(len(work), len(work) + n_ahead))
        x_title = "Observation index"
        future_labels = [f"t+{i+1}" for i in range(n_ahead)]
    else:
        hist_x = work[date_col].tolist()
        last_date = work[date_col].iloc[-1]
        fut_x = [last_date + pd.Timedelta(days=freq_days * (i + 1)) for i in range(n_ahead)]
        x_title = date_col
        future_labels = [str(d.date()) if hasattr(d, "date") else str(d) for d in fut_x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=y, mode="lines+markers", name="Historical", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=hist_x, y=y_hat, mode="lines", name="Fitted trend", line=dict(color="#AB63FA", dash="dot")))
    fig.add_trace(go.Scatter(
        x=list(fut_x) + list(fut_x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(0,204,150,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Approx. 95% band",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fut_x, y=pred, mode="lines+markers", name="Forecast",
        line=dict(color="#00CC96"),
    ))
    fig.update_layout(
        title=f"Forecast: {metric} (linear trend estimate)",
        xaxis_title=x_title,
        yaxis_title=metric,
        height=420,
    )

    r2 = float(model.score(X, y))
    forecast_rows = [
        {
            "period": future_labels[i],
            "forecast": float(pred[i]),
            "lower_95": float(lower[i]),
            "upper_95": float(upper[i]),
        }
        for i in range(n_ahead)
    ]

    caveats = [
        "This is a simple linear-trend estimate, not a full time-series model.",
        "The shaded band is an approximate residual-based range, not a formal statistical prediction interval.",
        "Limited history can produce unreliable forecasts; treat results as directional estimates only.",
    ]
    if len(work) < 12:
        caveats.append(f"Only {len(work)} historical points — confidence is low.")
    if use_synthetic:
        caveats.append("No datetime column detected; used row order as a time proxy.")

    summary = (
        f"Forecast for **{metric}** over the next {n_ahead} period(s) using a linear trend "
        f"(R² on history = {r2:.3f}).\n\n"
        f"Next point estimate: **{pred[0]:.3g}** "
        f"(approx. range {lower[0]:.3g} – {upper[0]:.3g}).\n\n"
        + "\n".join(f"- {c}" for c in caveats)
    )

    return {
        "success": True,
        "agent": "forecast",
        "metric_column": metric,
        "date_column": None if use_synthetic else date_col,
        "horizon": n_ahead,
        "r2_historical": r2,
        "forecast_table": forecast_rows,
        "charts": {"forecast": fig},
        "caveats": caveats,
        "summary": summary,
        "summary_for_rag": (
            f"Forecast of {metric}, horizon={n_ahead}, next={pred[0]:.4g}, "
            f"range=[{lower[0]:.4g}, {upper[0]:.4g}], R2={r2:.3f}. "
            + " ".join(caveats)
        ),
    }
