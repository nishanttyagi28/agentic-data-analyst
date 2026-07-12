"""Forecasting Agent — lightweight trend forecasts with uncertainty bands."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dateutil import parser as dateutil_parser
from sklearn.linear_model import LinearRegression

# If more than this share of non-null values fail to parse, warn the user.
DATE_PARSE_FAIL_WARN_THRESHOLD = 0.05
# CV of inter-event gaps above this → treat as irregular events (matches, etc.).
IRREGULAR_GAP_CV_THRESHOLD = 0.35

# ISO / sortable datetime prefix (must NOT be parsed with dayfirst=True — that mangles them).
_ISO_PREFIX = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
# Day/month/year slash or dot forms common in EU sports data.
_DMY_SLASH = re.compile(r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}")


def _parse_single_date(raw: Any) -> pd.Timestamp | pd.NaT:
    """Parse one value, choosing dayfirst based on string shape."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.NaT
    if isinstance(raw, pd.Timestamp):
        return raw
    if isinstance(raw, (np.datetime64,)):
        return pd.Timestamp(raw)

    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none", "nat", "null", ""):
        return pd.NaT

    # Already datetime-like objects
    if hasattr(raw, "year") and hasattr(raw, "month") and not isinstance(raw, str):
        try:
            return pd.Timestamp(raw)
        except (ValueError, TypeError, OverflowError):
            pass

    # 1) ISO / YYYY-MM-DD… — never use dayfirst (would turn 2024-07-03 into 2024-03-07)
    if _ISO_PREFIX.match(text):
        ts = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if pd.notna(ts):
            return pd.Timestamp(ts)
        try:
            return pd.Timestamp(dateutil_parser.parse(text, yearfirst=True, dayfirst=False))
        except (ValueError, OverflowError, TypeError):
            return pd.NaT

    # 2) dd/mm/yyyy or dd.mm.yyyy — prefer day-first
    if _DMY_SLASH.match(text):
        ts = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if pd.notna(ts):
            return pd.Timestamp(ts)
        try:
            return pd.Timestamp(dateutil_parser.parse(text, dayfirst=True, fuzzy=True))
        except (ValueError, OverflowError, TypeError):
            pass
        # US-style fallback mm/dd/yyyy
        ts = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if pd.notna(ts):
            return pd.Timestamp(ts)

    # 3) General fallback: try ISO-ish then day-first dateutil
    for kwargs in (
        {"yearfirst": True, "dayfirst": False},
        {"dayfirst": True},
        {"dayfirst": False},
    ):
        try:
            return pd.Timestamp(dateutil_parser.parse(text, fuzzy=True, **kwargs))
        except (ValueError, OverflowError, TypeError):
            continue
    return pd.NaT


def parse_mixed_dates(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """
    Best-effort parse of a column with mixed date formats
    (e.g. '13/03/2024' and '2024-07-03 00:00:00' in the same column).

    ISO strings are parsed year-first; slash dates prefer day-first (EU).
    Never applies dayfirst=True to the whole series (that corrupts ISO dates).

    Returns (parsed Series of datetime64[ns] with NaT for failures, quality meta).
    """
    n_total = int(len(series))
    as_str = series.astype(str)
    non_null_mask = (
        series.notna()
        & (as_str.str.strip() != "")
        & (~as_str.str.strip().str.lower().isin(["nan", "none", "nat", "null", "<na>"]))
    )
    n_non_null = int(non_null_mask.sum())
    if n_non_null == 0:
        empty = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        return empty, {
            "n_total": n_total,
            "n_non_null": 0,
            "n_parsed": 0,
            "n_failed": 0,
            "fail_rate": 1.0,
            "parse_ok": False,
            "warning": "No non-empty date values found.",
        }

    # Fast path for already-datetime columns
    if pd.api.types.is_datetime64_any_dtype(series):
        result = pd.to_datetime(series, errors="coerce")
        n_parsed = int(result.notna().sum())
        n_failed = n_non_null - n_parsed
        fail_rate = float(n_failed / n_non_null) if n_non_null else 1.0
        warning = None
        if fail_rate > DATE_PARSE_FAIL_WARN_THRESHOLD:
            warning = (
                f"{fail_rate * 100:.1f}% of dates could not be parsed reliably "
                f"({n_failed} of {n_non_null} non-empty values)."
            )
        return result, {
            "n_total": n_total,
            "n_non_null": n_non_null,
            "n_parsed": n_parsed,
            "n_failed": n_failed,
            "fail_rate": fail_rate,
            "parse_ok": n_parsed >= 3 and fail_rate <= 0.5,
            "warning": warning,
        }

    # Vectorized split: ISO vs slash vs other — avoid whole-series dayfirst=True
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    texts = series.where(non_null_mask).astype(str).str.strip()

    iso_mask = non_null_mask & texts.str.match(r"^\d{4}-\d{1,2}-\d{1,2}", na=False)
    dmy_mask = non_null_mask & texts.str.match(r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}", na=False) & ~iso_mask

    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(texts[iso_mask], errors="coerce", dayfirst=False)
    if dmy_mask.any():
        result.loc[dmy_mask] = pd.to_datetime(texts[dmy_mask], errors="coerce", dayfirst=True)

    # Row-by-row for remaining (or failed) non-null cells
    still_bad = non_null_mask & result.isna()
    if still_bad.any():
        for idx in series.index[still_bad]:
            result.loc[idx] = _parse_single_date(series.loc[idx])

    n_parsed = int((non_null_mask & result.notna()).sum())
    n_failed = n_non_null - n_parsed
    fail_rate = float(n_failed / n_non_null) if n_non_null else 1.0
    warning = None
    if fail_rate > DATE_PARSE_FAIL_WARN_THRESHOLD:
        warning = (
            f"{fail_rate * 100:.1f}% of dates could not be parsed reliably "
            f"({n_failed} of {n_non_null} non-empty values)."
        )

    return result, {
        "n_total": n_total,
        "n_non_null": n_non_null,
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "fail_rate": fail_rate,
        "parse_ok": n_parsed >= 3 and fail_rate <= 0.5,
        "warning": warning,
    }


def detect_datetime_column(df: pd.DataFrame) -> tuple[str | None, dict[str, Any] | None]:
    """Return (column_name, parse_meta) for the best date-like column, or (None, None)."""
    candidates: list[tuple[str, float, dict[str, Any]]] = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            meta = {
                "n_total": len(df),
                "n_non_null": int(df[col].notna().sum()),
                "n_parsed": int(df[col].notna().sum()),
                "n_failed": 0,
                "fail_rate": 0.0,
                "parse_ok": True,
                "warning": None,
            }
            return col, meta

    name_hint_cols = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ("date", "time", "month", "year", "day", "period", "timestamp", "match"))
    ]
    object_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    ordered = list(dict.fromkeys(name_hint_cols + object_cols))

    for col in ordered:
        sample = df[col].dropna().astype(str).head(40)
        if sample.empty:
            continue
        # Skip obvious non-dates (high cardinality free text without digits)
        if not sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}", regex=True).mean() >= 0.4:
            continue
        parsed, meta = parse_mixed_dates(df[col])
        if meta["n_parsed"] >= 3 and meta["fail_rate"] < 0.5:
            score = meta["n_parsed"] / max(meta["n_non_null"], 1)
            # Prefer columns with date-like names
            if any(k in str(col).lower() for k in ("date", "time", "match", "period")):
                score += 0.2
            candidates.append((col, score, meta))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, _, best_meta = candidates[0]
    return best_col, best_meta


# Measure-like tokens (target) vs dimension-like tokens (usually filters, not targets)
_METRIC_NAME_HINTS = (
    "weekly_sales", "sales", "revenue", "goal", "goals", "amount", "price",
    "charges", "income", "value", "count", "total", "score", "profit", "units",
    "demand", "volume", "qty", "quantity", "orders",
)
_DIMENSION_NAME_HINTS = (
    "store", "region", "country", "city", "state", "branch", "shop", "id",
    "flag", "code", "sku", "category", "segment", "dept", "department",
    "type", "class", "group", "zone", "index",
)
# Phrases that introduce a filter entity, not a metric
_FILTER_CONTEXT_RE = re.compile(
    r"\b(?:for|in|at|from|within|of)\s+(?:the\s+)?(?P<entity>[a-z_][\w\s]*?)\s+"
    r"(?P<val>\d+|['\"][^'\"]+['\"]|[A-Za-z][\w\-]*)",
    re.IGNORECASE,
)


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _col_tokens(col: str) -> set[str]:
    raw = re.split(r"[_\s]+", str(col).lower())
    return {t for t in raw if t} | {_norm_token(col)}


def _is_dimension_like(col: str) -> bool:
    tokens = _col_tokens(col)
    name = str(col).lower()
    if any(h in tokens or h in name for h in _DIMENSION_NAME_HINTS):
        # Exception: sales_target is a metric
        if any(m in name for m in ("sales", "revenue", "goal", "amount", "price")):
            return False
        return True
    return False


def _is_metric_like(col: str) -> bool:
    name = str(col).lower()
    tokens = _col_tokens(col)
    return any(h in name or h in tokens for h in _METRIC_NAME_HINTS)


def _query_mentions_column_as_metric(query: str, col: str) -> float:
    """
    Score how strongly the query names this column as the *thing to forecast*,
    not merely as a filter entity (e.g. 'for store 1').
    """
    q = (query or "").lower()
    q_norm = _norm_token(q)
    col_l = str(col).lower()
    col_space = col_l.replace("_", " ")
    col_norm = _norm_token(col)

    score = 0.0
    # Strong: full multi-word name in query ("weekly sales" -> Weekly_Sales)
    if len(col_space) >= 4 and col_space in q:
        score += 10.0
    if len(col_norm) >= 4 and col_norm in q_norm:
        score += 8.0

    # Metric keyword overlap with query
    for hint in _METRIC_NAME_HINTS:
        if hint in col_l or hint in _col_tokens(col):
            # prefer multi-word forms in query
            if hint.replace("_", " ") in q or hint in q:
                score += 6.0 if hint in ("sales", "revenue", "goals", "goal", "weekly_sales") else 4.0
            else:
                score += 1.0  # column is metric-like even if not in query

    # Penalize dimension columns mentioned only in filter context
    if _is_dimension_like(col):
        score -= 8.0
        # Extra penalty if "for <col> <value>" pattern
        for m in _FILTER_CONTEXT_RE.finditer(q):
            ent = m.group("entity").strip().lower()
            if col_l in ent or ent in col_l or col_space in ent or any(
                t in ent for t in _col_tokens(col) if len(t) > 2
            ):
                score -= 12.0

    # Short token match like "store" in query is weak and often a filter
    for tok in _col_tokens(col):
        if len(tok) <= 2:
            continue
        if re.search(rf"\b{re.escape(tok)}\b", q):
            if _is_dimension_like(col):
                score += 0.5  # almost ignore
            else:
                score += 3.0

    # Prefer higher-variance measure names when both match
    if _is_metric_like(col) and not _is_dimension_like(col):
        score += 2.0

    return score


def detect_metric_column(df: pd.DataFrame, query: str = "", exclude: set[str] | None = None) -> str | None:
    """
    Choose the numeric measure to forecast. Prefer metric-like columns
    (sales/revenue/goals) over dimension/ID columns (store/region) even when
    the query mentions the dimension as a filter ('for store 1').
    """
    exclude = exclude or set()
    nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not nums:
        # allow non-numeric that are metric-named after coercion
        nums = [c for c in df.columns if c not in exclude and _is_metric_like(c)]
    if not nums:
        return None

    scored = [(col, _query_mentions_column_as_metric(query, col)) for col in nums]
    scored.sort(key=lambda x: (x[1], _is_metric_like(x[0]), -int(_is_dimension_like(x[0]))), reverse=True)

    best_col, best_score = scored[0]
    # If best is still dimension-like and a metric-like column exists, prefer metric
    if _is_dimension_like(best_col):
        metric_candidates = [c for c in nums if _is_metric_like(c) and not _is_dimension_like(c)]
        if metric_candidates:
            # re-score metric-only
            metric_scored = [(c, _query_mentions_column_as_metric(query, c)) for c in metric_candidates]
            metric_scored.sort(key=lambda x: x[1], reverse=True)
            if metric_scored[0][1] >= best_score - 5 or best_score < 5:
                return metric_scored[0][0]

    # Default: first among metric-like if nothing scored well
    if best_score < 2.0:
        for c in nums:
            if _is_metric_like(c) and not _is_dimension_like(c):
                return c
        # fall back to highest variance numeric (not constant ID)
        variances = []
        for c in nums:
            s = pd.to_numeric(df[c], errors="coerce")
            variances.append((c, float(s.var()) if s.notna().sum() > 1 else 0.0))
        variances.sort(key=lambda x: x[1], reverse=True)
        if variances and variances[0][1] > 0:
            return variances[0][0]
    return best_col


def _cast_filter_value(series: pd.Series, val: Any) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return int(val) if str(val).isdigit() else float(val)
        except ValueError:
            return val
    return val


def parse_filters_from_query(df: pd.DataFrame, query: str) -> list[dict[str, Any]]:
    """
    Lightweight rule-based filters:
      - 'for store 1'           → Store = 1
      - 'for the north region'  → region = north
      - 'for region north'      → region = north
    Returns list of {column, op, value}.
    """
    q = query or ""
    filters: list[dict[str, Any]] = []
    q_lower = q.lower()

    dim_cols = [
        c for c in df.columns
        if _is_dimension_like(c) or (
            not _is_metric_like(c)
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and df[c].nunique(dropna=True) <= max(50, len(df) // 2 + 1)
        )
    ]

    for col in dim_cols:
        col_l = str(col).lower()
        col_space = col_l.replace("_", " ")
        # Pattern A: for <column> <value>   e.g. for store 1
        m_a = re.search(
            rf"\b(?:for|in|at)\s+(?:the\s+)?{re.escape(col_l)}\s+(\d+|['\"][^'\"]+['\"]|[A-Za-z][\w\-]*)",
            q_lower,
            re.IGNORECASE,
        )
        if not m_a and col_space != col_l:
            m_a = re.search(
                rf"\b(?:for|in|at)\s+(?:the\s+)?{re.escape(col_space)}\s+(\d+|['\"][^'\"]+['\"]|[A-Za-z][\w\-]*)",
                q_lower,
                re.IGNORECASE,
            )
        # Pattern B: for [the] <value> <column>  e.g. for the north region
        m_b = re.search(
            rf"\b(?:for|in|at)\s+(?:the\s+)?([A-Za-z][\w\-]*)\s+{re.escape(col_l)}\b",
            q_lower,
            re.IGNORECASE,
        )
        if not m_b and col_space != col_l:
            m_b = re.search(
                rf"\b(?:for|in|at)\s+(?:the\s+)?([A-Za-z][\w\-]*)\s+{re.escape(col_space)}\b",
                q_lower,
                re.IGNORECASE,
            )

        val = None
        if m_a:
            val = m_a.group(1).strip().strip("'\"")
            # skip if value is a time word (next month) not a real filter
            if val.lower() in ("next", "last", "this", "the", "a", "an"):
                val = None
        if val is None and m_b:
            val = m_b.group(1).strip().strip("'\"")
            if val.lower() in ("next", "last", "this", "the", "a", "an", "for"):
                val = None
        if val is None:
            continue
        # Skip if "value" is actually another column name
        if any(val.lower() == str(c).lower() for c in df.columns):
            continue
        cast_val = _cast_filter_value(df[col], val)
        filters.append({"column": col, "op": "=", "value": cast_val})

    # De-dupe by column (last wins)
    by_col = {f["column"]: f for f in filters}
    return list(by_col.values())


def resolve_forecast_intent(
    df: pd.DataFrame,
    query: str = "",
    exclude: set[str] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Resolve {target_column, filters} from NL + schema.
    Prefer LLM structured extraction when available; always fall back to rules.
    """
    exclude = exclude or set()
    schema_lines = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        nuniq = int(df[c].nunique(dropna=True))
        sample = df[c].dropna().astype(str).head(3).tolist()
        schema_lines.append(f"- {c} ({dtype}, nunique={nuniq}, sample={sample})")

    rule_metric = detect_metric_column(df, query, exclude=exclude)
    rule_filters = parse_filters_from_query(df, query)
    # Never filter on the target metric column
    rule_filters = [f for f in rule_filters if f["column"] != rule_metric]

    result = {
        "target_column": rule_metric,
        "filters": rule_filters,
        "source": "rules",
        "raw_llm": None,
    }

    if not use_llm or not (query or "").strip():
        return result

    try:
        from agents.llm_client import chat_completion
        import json

        prompt = f"""You extract forecast intent from a user question and a table schema.
Return JSON ONLY:
{{"target_column": "<exact column name to forecast>", "filters": [{{"column": "<col>", "op": "=", "value": "<value>"}}]}}

Rules:
- target_column must be the METRIC/MEASURE (e.g. Weekly_Sales, revenue, goals), NEVER a filter dimension like Store, Region, ID.
- Mentions like "for store 1" or "for the north region" are FILTERS, not targets.
- Use exact column names from the schema.
- filters may be empty.

Schema:
{chr(10).join(schema_lines)}

User question: {query}
"""
        text, err = chat_completion(
            [
                {"role": "system", "content": "You map NL forecast requests to target_column + filters. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        if err or not text:
            return result
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return result
        data = json.loads(match.group())
        target = data.get("target_column")
        filters = data.get("filters") or []
        valid_cols = set(df.columns)
        if target in valid_cols and target not in exclude:
            # Guard: reject dimension-only targets if a better metric exists
            if _is_dimension_like(target) and not _is_metric_like(target):
                metric_alt = detect_metric_column(df, query, exclude=exclude | {target})
                if metric_alt:
                    target = metric_alt
            result["target_column"] = target
            result["source"] = "llm+rules"
        cleaned = []
        for f in filters:
            col = f.get("column")
            if col not in valid_cols or col == result["target_column"]:
                continue
            cleaned.append({
                "column": col,
                "op": f.get("op") or "=",
                "value": f.get("value"),
            })
        # Merge rule filters if LLM omitted them
        if not cleaned and rule_filters:
            cleaned = rule_filters
        result["filters"] = cleaned
        result["raw_llm"] = data
    except Exception:
        return result
    return result


def apply_forecast_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply simple equality filters; return filtered df + human notes."""
    if df is None or df.empty or not filters:
        return df, []
    work = df.copy()
    notes = []
    for f in filters:
        col = f.get("column")
        if col not in work.columns:
            continue
        val = f.get("value")
        before = len(work)
        series = work[col]
        if pd.api.types.is_numeric_dtype(series):
            try:
                val_num = pd.to_numeric(val, errors="coerce")
                mask = series == val_num if pd.notna(val_num) else series.astype(str) == str(val)
            except Exception:
                mask = series.astype(str).str.lower() == str(val).lower()
        else:
            mask = series.astype(str).str.lower() == str(val).lower()
            # also try contains for region names
            if mask.sum() == 0:
                mask = series.astype(str).str.lower().str.contains(str(val).lower(), na=False)
        work = work.loc[mask]
        notes.append(f"Filter `{col}` = {val!r} ({before} → {len(work)} rows)")
    return work, notes


def low_variance_metric_warning(series: pd.Series, col: str) -> str | None:
    """
    Safety net: constant / near-constant targets (IDs, store numbers after filter)
    produce meaningless perfect R² forecasts.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return (
            f"The column `{col}` has too few numeric values after filtering to forecast reliably. "
            "Did you mean a different metric column?"
        )
    nuniq = int(s.nunique())
    std = float(s.std()) if len(s) > 1 else 0.0
    mean_abs = float(np.mean(np.abs(s))) if len(s) else 0.0
    if nuniq <= 1 or std == 0.0:
        return (
            f"The column `{col}` has very little variation in this filtered data "
            f"(unique values={nuniq}, std=0) — unusual for a forecast target "
            "(often an ID/category like Store after filtering). "
            "Did you mean to forecast a different column such as Weekly_Sales / revenue / goals?"
        )
    if nuniq <= 3 and mean_abs > 0 and std / mean_abs < 0.01:
        return (
            f"The column `{col}` is nearly constant after filtering "
            f"(nunique={nuniq}, CV≈{std / mean_abs:.4f}). "
            "This is unusual for a forecast target — did you mean a different metric column?"
        )
    return None


def _horizon_from_query(query: str, freq_days: float, irregular: bool) -> int:
    q = (query or "").lower()
    if irregular or any(w in q for w in ("season", "match", "game", "event", "fixture")):
        if "season" in q:
            return 10  # ~next season as next N events (directional)
        if "year" in q:
            return 12
        if "quarter" in q:
            return 6
        if "month" in q:
            return 5
        if "week" in q:
            return 3
        return 5
    if "year" in q or "season" in q:
        return max(1, int(round(365 / max(freq_days, 1))))
    if "quarter" in q:
        return max(1, int(round(90 / max(freq_days, 1))))
    if "week" in q:
        return max(1, int(round(7 / max(freq_days, 1))))
    if "month" in q:
        return max(1, int(round(30 / max(freq_days, 1))))
    if "day" in q:
        return 7
    return 3


def _is_nonnegative_count_like(y: np.ndarray) -> bool:
    """True if historical values are all non-negative (count/rate-like floor at 0)."""
    if len(y) == 0:
        return False
    return bool(np.all(y >= -1e-12))


def _is_irregular_spacing(deltas_days: np.ndarray) -> bool:
    if len(deltas_days) < 2:
        return False
    mean = float(np.mean(deltas_days))
    if mean <= 0 or not np.isfinite(mean):
        return True
    cv = float(np.std(deltas_days) / mean)
    # Also flag if max gap is much larger than median (sparse event calendar)
    med = float(np.median(deltas_days))
    mx = float(np.max(deltas_days))
    if med > 0 and mx / med >= 3.0:
        return True
    return cv >= IRREGULAR_GAP_CV_THRESHOLD


def run_forecast(
    df: pd.DataFrame,
    query: str = "",
    horizon: int | None = None,
    use_llm_intent: bool = True,
) -> dict[str, Any]:
    """
    Linear-trend forecast on a datetime + numeric series.
    Returns point forecast + approximate 95% prediction band from residual std.
    """
    if df is None or df.empty:
        return {"success": False, "error": "No data available for forecasting", "agent": "forecast"}

    warnings: list[str] = []
    date_col, parse_meta = detect_datetime_column(df)
    use_synthetic = False
    irregular = False
    event_mode = False

    # --- Resolve target metric + filters (not naive first-column-name-in-query) ---
    intent = resolve_forecast_intent(
        df,
        query,
        exclude={date_col} if date_col else set(),
        use_llm=use_llm_intent,
    )
    metric = intent.get("target_column")
    filters = intent.get("filters") or []
    filtered_df, filter_notes = apply_forecast_filters(df, filters)
    warnings.extend(filter_notes)

    if metric is None:
        return {
            "success": False,
            "error": "No numeric metric found to forecast",
            "agent": "forecast",
            "intent": intent,
        }

    if filtered_df is None or filtered_df.empty:
        return {
            "success": False,
            "error": (
                f"No rows left after applying filters {filters}. "
                "Check that the filter values exist in the data."
            ),
            "agent": "forecast",
            "intent": intent,
            "warnings": warnings,
        }

    # Safety net: wrong target often shows up as near-constant series after filter
    if metric in filtered_df.columns:
        lv = low_variance_metric_warning(filtered_df[metric], metric)
        if lv:
            # Try to auto-recover to a better metric once
            alt = detect_metric_column(
                filtered_df,
                query,
                exclude={metric, date_col} if date_col else {metric},
            )
            if alt and alt != metric:
                warnings.append(
                    f"Auto-switched forecast target from `{metric}` → `{alt}` because `{metric}` "
                    "had little/no variation (likely a filter/ID column, not a measure)."
                )
                metric = alt
                lv = low_variance_metric_warning(filtered_df[metric], metric)
            if lv:
                return {
                    "success": False,
                    "error": lv,
                    "agent": "forecast",
                    "metric_column": metric,
                    "intent": intent,
                    "warnings": warnings,
                    "summary": f"⚠️ {lv}",
                }

    if date_col is None:
        work = filtered_df[[metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna()
        if len(work) < 3:
            return {"success": False, "error": "Need at least 3 data points to forecast", "agent": "forecast"}
        work = work.reset_index(drop=True)
        work["_t"] = np.arange(len(work), dtype=float)
        work["_event_idx"] = np.arange(1, len(work) + 1)
        use_synthetic = True
        event_mode = True
        freq_days = 1.0
        warnings.append(
            "No datetime column could be parsed reliably — using observation/event order as the time axis "
            "(not calendar dates)."
        )
        parse_meta = None
    else:
        work = filtered_df[[date_col, metric]].copy()
        parsed_dates, col_meta = parse_mixed_dates(work[date_col])
        parse_meta = col_meta
        work[date_col] = parsed_dates
        work[metric] = pd.to_numeric(work[metric], errors="coerce")

        if col_meta.get("warning"):
            warnings.append(
                f"{col_meta['warning']} Column `{date_col}` — forecast timeline may be approximate."
            )

        n_valid_dates = int(work[date_col].notna().sum())
        if n_valid_dates < 3 or not col_meta.get("parse_ok", False):
            # Explicit fallback — never silently invent calendar dates
            work = work.drop(columns=[date_col], errors="ignore")
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
            work = work.dropna().reset_index(drop=True)
            if len(work) < 3:
                return {
                    "success": False,
                    "error": (
                        f"Could not parse enough dates in `{date_col}` "
                        f"({col_meta.get('n_parsed', 0)} ok, {col_meta.get('n_failed', 0)} failed) "
                        "and not enough numeric rows to fall back to event order."
                    ),
                    "agent": "forecast",
                    "date_parse": col_meta,
                    "warnings": warnings,
                }
            work["_t"] = np.arange(len(work), dtype=float)
            work["_event_idx"] = np.arange(1, len(work) + 1)
            use_synthetic = True
            event_mode = True
            freq_days = 1.0
            warnings.append(
                f"Date column `{date_col}` could not be parsed reliably "
                f"({col_meta.get('fail_rate', 1) * 100:.1f}% failed). "
                "Using event/observation order instead of calendar dates — "
                "do not treat forecast points as specific calendar days."
            )
            date_col = None
        else:
            work = work.dropna(subset=[date_col, metric]).sort_values(date_col)
            if len(work) < 3:
                return {
                    "success": False,
                    "error": "Need at least 3 dated observations to forecast",
                    "agent": "forecast",
                    "date_parse": col_meta,
                    "warnings": warnings,
                }
            # Aggregate duplicates on same date (mean of metric)
            work = work.groupby(date_col, as_index=False)[metric].mean()
            work = work.sort_values(date_col).reset_index(drop=True)
            if len(work) < 3:
                return {
                    "success": False,
                    "error": "Need at least 3 distinct dated observations to forecast",
                    "agent": "forecast",
                    "date_parse": col_meta,
                    "warnings": warnings,
                }
            deltas = work[date_col].diff().dt.total_seconds().dropna() / 86400.0
            deltas_arr = deltas.values.astype(float)
            freq_days = float(np.median(deltas_arr)) if len(deltas_arr) else 1.0
            if freq_days <= 0 or not np.isfinite(freq_days):
                freq_days = 1.0
            irregular = _is_irregular_spacing(deltas_arr)
            work["_t"] = (work[date_col] - work[date_col].min()).dt.total_seconds() / 86400.0
            work["_event_idx"] = np.arange(1, len(work) + 1)
            if irregular:
                event_mode = True
                warnings.append(
                    f"Timestamps in `{date_col}` are irregularly spaced (not a regular calendar series — "
                    "common for matches/events). Forecast steps are the next N events along the trend, "
                    "not precise calendar dates."
                )

    y = work[metric].values.astype(float)
    X = work["_t"].values.reshape(-1, 1).astype(float)
    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat
    resid_std = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float(np.std(residuals))
    if not np.isfinite(resid_std) or resid_std == 0:
        resid_std = max(abs(float(np.mean(y))) * 0.05, 1e-6)

    n_ahead = horizon if horizon is not None else _horizon_from_query(query, freq_days, event_mode or irregular)
    n_ahead = max(1, min(int(n_ahead), 36))

    last_t = float(X[-1, 0])
    # Step forward by typical gap for model input; labels may still be event-based
    future_t = np.array([last_t + freq_days * (i + 1) for i in range(n_ahead)]).reshape(-1, 1)
    pred = model.predict(future_t)
    z = 1.96
    scales = 1.0 + 0.1 * np.arange(1, n_ahead + 1)
    lower = pred - z * resid_std * scales
    upper = pred + z * resid_std * scales

    # Non-negativity for count-like / non-negative historical metrics
    nonneg = _is_nonnegative_count_like(y)
    if nonneg:
        pred = np.maximum(pred, 0.0)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        # Keep interval ordered
        upper = np.maximum(upper, lower)

    # --- Axis / labels ---
    if event_mode or use_synthetic:
        hist_x = work["_event_idx"].tolist()
        fut_x = list(range(int(work["_event_idx"].iloc[-1]) + 1, int(work["_event_idx"].iloc[-1]) + n_ahead + 1))
        x_title = "Event sequence (not calendar dates)"
        future_labels = [f"Event +{i + 1}" for i in range(n_ahead)]
        hist_hover = None
        if date_col and date_col in work.columns and not use_synthetic:
            hist_hover = [str(pd.Timestamp(d).date()) for d in work[date_col]]
    else:
        hist_x = work[date_col].tolist()
        last_date = work[date_col].iloc[-1]
        fut_x = [last_date + pd.Timedelta(days=freq_days * (i + 1)) for i in range(n_ahead)]
        x_title = str(date_col)
        future_labels = [str(pd.Timestamp(d).date()) for d in fut_x]
        hist_hover = None

    fig = go.Figure()
    hist_trace_kwargs: dict[str, Any] = dict(
        x=hist_x, y=y, mode="lines+markers", name="Historical", line=dict(color="#636EFA"),
    )
    if hist_hover is not None:
        hist_trace_kwargs["text"] = hist_hover
        hist_trace_kwargs["hovertemplate"] = "Event %{x}<br>Date %{text}<br>Value %{y}<extra></extra>"
    fig.add_trace(go.Scatter(**hist_trace_kwargs))
    fig.add_trace(go.Scatter(
        x=hist_x, y=y_hat, mode="lines", name="Fitted trend",
        line=dict(color="#AB63FA", dash="dot"),
    ))
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
        text=future_labels,
        hovertemplate="%{text}<br>Estimate %{y:.3g}<extra></extra>",
    ))
    title_suffix = "event trend estimate" if event_mode else "linear trend estimate"
    fig.update_layout(
        title=f"Forecast: {metric} ({title_suffix})",
        xaxis_title=x_title,
        yaxis_title=metric,
        height=420,
    )
    if nonneg:
        fig.update_yaxes(rangemode="tozero")

    r2 = float(model.score(X, y))
    # Final guard: perfect R² on near-constant y is almost always wrong target
    if r2 >= 0.999 and float(np.std(y)) < 1e-9:
        return {
            "success": False,
            "error": (
                f"The column `{metric}` is constant after filtering (R² would be trivially 1.0). "
                "This is almost never a valid forecast target — did you mean Weekly_Sales / revenue / goals?"
            ),
            "agent": "forecast",
            "metric_column": metric,
            "intent": intent,
            "warnings": warnings,
            "summary": (
                f"⚠️ Refused to forecast constant column `{metric}`. "
                "Pick a real measure (e.g. Weekly_Sales), not a filter/ID field."
            ),
        }

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
    if nonneg:
        caveats.append(
            f"Historical `{metric}` values are non-negative; forecast and interval lower bounds are clipped at 0 "
            "(count/amount-like data cannot go below zero)."
        )
    if event_mode:
        caveats.append(
            "Forecast periods are labeled as sequential events (Event +1, +2, …), not precise calendar dates."
        )
    if len(work) < 12:
        caveats.append(f"Only {len(work)} historical points — confidence is low.")
    caveats.extend(warnings)

    framing = (
        f"next {n_ahead} event(s) along the historical trend"
        if event_mode
        else f"next {n_ahead} period(s)"
    )
    filter_desc = ""
    if filters:
        filter_desc = " filtered by " + ", ".join(
            f"`{f['column']}`={f['value']!r}" for f in filters
        )
    warn_block = ""
    if warnings:
        warn_block = "\n\n**Notes:**\n" + "\n".join(f"- ⚠️ {w}" for w in warnings)

    summary = (
        f"Forecast for **{metric}**{filter_desc} over the {framing} using a linear trend "
        f"(R² on history = {r2:.3f}).\n\n"
        f"Next point estimate: **{pred[0]:.3g}** "
        f"(approx. range {lower[0]:.3g} – {upper[0]:.3g})."
        f"{warn_block}\n\n"
        + "\n".join(f"- {c}" for c in caveats)
    )

    return {
        "success": True,
        "agent": "forecast",
        "metric_column": metric,
        "filters": filters,
        "intent": intent,
        "date_column": None if use_synthetic else date_col,
        "horizon": n_ahead,
        "r2_historical": r2,
        "forecast_table": forecast_rows,
        "charts": {"forecast": fig},
        "caveats": caveats,
        "warnings": warnings,
        "date_parse": parse_meta,
        "event_mode": event_mode,
        "irregular_spacing": irregular,
        "nonnegative_clipped": nonneg,
        "summary": summary,
        "summary_for_rag": (
            f"Forecast of {metric}{filter_desc}, horizon={n_ahead}, next={pred[0]:.4g}, "
            f"range=[{lower[0]:.4g}, {upper[0]:.4g}], R2={r2:.3f}, event_mode={event_mode}. "
            + " ".join(caveats)
        ),
    }
