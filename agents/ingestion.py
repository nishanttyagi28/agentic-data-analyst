"""Ingestion Agent — CSV upload, cleaning, type inference, SQLite load."""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from db.database import TABLE_NAME, get_row_count, get_table_schema, load_dataframe_to_table


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def infer_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        series = result[col]
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() >= len(series) * 0.8:
                result[col] = numeric
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dt = pd.to_datetime(series, errors="coerce")
            if dt.notna().sum() >= len(series) * 0.8:
                result[col] = dt
    return result


def ingest_csv(
    file_bytes: bytes | None = None,
    file_path: str | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Load CSV into SQLite and return ingestion summary."""
    if engine is None:
        raise ValueError("Database engine is required")

    try:
        if file_bytes is not None:
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_bytes))
        elif file_path is not None:
            df = pd.read_csv(file_path)
        else:
            return {"success": False, "error": "No file provided"}

        if df.empty:
            return {"success": False, "error": "CSV file is empty"}

        df.columns = [clean_column_name(c) for c in df.columns]
        seen: dict[str, int] = {}
        unique_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                unique_cols.append(c)
        df.columns = unique_cols

        df = infer_and_cast_types(df)
        load_dataframe_to_table(df, engine, TABLE_NAME, if_exists="replace")

        schema = get_table_schema(engine, TABLE_NAME)
        row_count = get_row_count(engine, TABLE_NAME)

        dtype_summary = {}
        for col in df.columns:
            dtype_summary[col] = str(df[col].dtype)

        preview = df.head(10).to_dict(orient="records")

        return {
            "success": True,
            "table_name": TABLE_NAME,
            "row_count": row_count,
            "column_count": len(df.columns),
            "schema": schema,
            "dtypes": dtype_summary,
            "preview": preview,
            "columns": list(df.columns),
            "dataframe": df,
        }
    except pd.errors.EmptyDataError:
        return {"success": False, "error": "CSV file is empty or invalid"}
    except Exception as e:
        return {"success": False, "error": f"Failed to ingest CSV: {e}"}


def format_schema_for_display(schema: list[dict]) -> str:
    if not schema:
        return "No schema available"
    lines = ["| Column | Type | Nullable |", "|--------|------|----------|"]
    for col in schema:
        lines.append(f"| {col['name']} | {col['type']} | {col.get('nullable', True)} |")
    return "\n".join(lines)