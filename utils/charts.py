"""Chart generation utilities for EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def missing_values_chart(df: pd.DataFrame) -> go.Figure:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Missing Values", height=300)
        return fig
    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        marker_color="#636EFA",
    ))
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Count",
        yaxis_title="Column",
        height=max(300, len(missing) * 40),
    )
    return fig


def distribution_chart(df: pd.DataFrame, column: str) -> go.Figure | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution: {column}")
    else:
        counts = series.value_counts().head(20)
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values))
        fig.update_layout(title=f"Value Counts: {column}", xaxis_title=column, yaxis_title="Count")
    fig.update_layout(height=350)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
    ))
    fig.update_layout(title="Correlation Heatmap", height=400)
    return fig


def pca_scatter(pca_coords: np.ndarray, labels: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    if labels is not None:
        for label in np.unique(labels):
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=pca_coords[mask, 0],
                y=pca_coords[mask, 1],
                mode="markers",
                name=str(label),
                marker=dict(size=8, opacity=0.7),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#636EFA"),
        ))
    fig.update_layout(
        title="PCA Visualization (2D)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=400,
    )
    return fig


def feature_importance_chart(importances: dict[str, float], top_n: int = 15) -> go.Figure:
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#00CC96"))
    fig.update_layout(title="Top Feature Importances", height=max(300, len(names) * 30))
    return fig