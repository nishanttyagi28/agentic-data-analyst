"""Multi-table helpers: load registry, join-key detection, schema text for SQL."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from agents.ingestion import clean_column_name, infer_and_cast_types
from db.database import get_table_schema, load_dataframe_to_table


def sanitize_table_name(name: str) -> str:
    base = re.sub(r"[^\w]+", "_", str(name).strip().lower()).strip("_")
    if not base:
        base = "table"
    if base[0].isdigit():
        base = f"t_{base}"
    # reserved / default
    if base in ("user_data",):
        return base
    return base[:48]


def register_dataframe(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Clean columns lightly, store in SQLite + in-memory registry."""
    work = df.copy()
    work.columns = [clean_column_name(c) for c in work.columns]
    # uniquify columns
    seen: dict[str, int] = {}
    cols = []
    for c in work.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    work.columns = cols
    work = infer_and_cast_types(work)
    tname = sanitize_table_name(table_name)
    # avoid collision: only reuse an existing name if it holds the exact same
    # data (idempotent re-registration); any other content gets a new name.
    base = tname
    n = 2
    while tname in tables and not work.equals(tables.get(tname, pd.DataFrame())):
        tname = f"{base}_{n}"
        n += 1
        if n > 20:
            break

    load_dataframe_to_table(work, engine, tname, if_exists="replace")
    tables[tname] = work
    return {
        "success": True,
        "table_name": tname,
        "row_count": len(work),
        "column_count": len(work.columns),
        "columns": list(work.columns),
        "dataframe": work,
    }


def detect_join_keys(
    tables: dict[str, pd.DataFrame],
    max_pairs: int = 8,
) -> list[dict[str, Any]]:
    """
    Heuristic join suggestions: same column name, or high value-overlap for id-like cols.
    """
    names = list(tables.keys())
    suggestions: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dfa, dfb = tables[a], tables[b]
            cols_a, cols_b = set(dfa.columns), set(dfb.columns)
            shared = cols_a & cols_b
            for col in sorted(shared):
                # skip pure free-text high cardinality non-id
                na, nb = dfa[col].nunique(dropna=True), dfb[col].nunique(dropna=True)
                if na == 0 or nb == 0:
                    continue
                sa = set(dfa[col].dropna().astype(str).head(5000))
                sb = set(dfb[col].dropna().astype(str).head(5000))
                if not sa or not sb:
                    continue
                overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
                id_like = any(k in col.lower() for k in ("id", "key", "code", "sku", "uuid"))
                if overlap >= 0.1 or id_like:
                    suggestions.append({
                        "left_table": a,
                        "right_table": b,
                        "left_column": col,
                        "right_column": col,
                        "overlap": round(float(overlap), 3),
                        "id_like": id_like,
                        "message": (
                            f"Tables `{a}` and `{b}` share column `{col}` "
                            f"(value overlap ≈ {overlap*100:.0f}%) — I can JOIN on it."
                        ),
                    })
            # similar names (customer_id vs cust_id) — light check
            for ca in dfa.columns:
                for cb in dfb.columns:
                    if ca == cb:
                        continue
                    if ca.replace("_", "") == cb.replace("_", ""):
                        suggestions.append({
                            "left_table": a,
                            "right_table": b,
                            "left_column": ca,
                            "right_column": cb,
                            "overlap": None,
                            "id_like": True,
                            "message": (
                                f"Columns `{a}.{ca}` and `{b}.{cb}` look like the same key — possible JOIN."
                            ),
                        })
    # de-dupe
    seen = set()
    out = []
    for s in suggestions:
        key = (s["left_table"], s["right_table"], s["left_column"], s["right_column"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:max_pairs]


def build_multi_schema_context(engine: Engine, tables: dict[str, pd.DataFrame]) -> str:
    if not tables:
        return "No tables loaded."
    lines = ["Loaded tables (use exact names in SQL JOINs):"]
    for tname, df in tables.items():
        schema = get_table_schema(engine, tname)
        lines.append(f'\nTable: "{tname}" ({len(df)} rows)')
        if schema:
            for col in schema:
                lines.append(f"  - {col['name']} ({col['type']})")
        else:
            for c in df.columns:
                lines.append(f"  - {c} ({df[c].dtype})")
    joins = detect_join_keys(tables)
    if joins:
        lines.append("\nLikely join keys:")
        for j in joins[:5]:
            lines.append(f"  - {j['message']}")
    return "\n".join(lines)


def primary_dataframe(tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if not tables:
        return None
    if "user_data" in tables:
        return tables["user_data"]
    return next(iter(tables.values()))
