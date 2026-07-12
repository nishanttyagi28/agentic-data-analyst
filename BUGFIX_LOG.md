# Bugfix Log — Full Test & Bug-Fix Pass

Date: 2026-07-12

## Method

1. Recreated a clean Python virtualenv and installed `requirements.txt` (the
   repo's checked-in `.venv` was broken — no `pip`, no `activate` script).
2. Ran `self_test.py` as a baseline: **179 passed, 0 failed, 10 skipped**
   (skips are all LLM-dependent paths — no `GROQ_API_KEY` was available in
   this environment, so live Groq-backed SQL generation, RAG answering, and
   orchestrator `handle_query` e2e routing could not be exercised).
3. Dispatched three parallel code-review passes over every agent module
   (`forecast_agent`/`stats_agent`, `sql_agent`/`orchestrator`/`multitable`,
   `ml_agent`/`quality_agent`/`insight_agent`/`report_agent`/`rag_agent`/
   `ingestion`/`db`/`utils`) specifically hunting for bugs *not* already
   covered by `self_test.py`'s assertions — edge cases, degenerate data,
   silent failures, dead code. Every candidate bug below was reproduced by
   actually running the affected function against real input before being
   treated as confirmed.
4. Also manually drove RAG session isolation, ingestion edge cases (empty
   CSV, duplicate/reserved column names), and quality/EDA/ML/stats agents
   against empty, single-row, zero-variance, and tiny-n datasets.
5. Fixed each confirmed bug, re-verified the fix with a standalone repro
   script, then re-ran the full `self_test.py` suite after every fix to
   check for regressions. Final state: **179 passed, 0 failed, 10 skipped**
   (identical pass/skip set to the baseline — no test outcomes changed,
   only latent bugs were closed).

No public function signatures were changed. Two return-dict *value*
semantics changed (see items 3 and 6) — noted below since callers that
pattern-match on the old (wrong) values could theoretically be affected,
though no caller in this codebase does.

---

## Bugs found and fixed

### 1. `agents/multitable.py` — table registration silently overwrote unrelated data on name collision

**Symptom:** Uploading two different CSVs that happen to share the same
column names and row count (e.g. `orders_jan.csv` and `orders_feb.csv`,
same schema/length but different content) under the same auto-derived
table name caused the second upload to silently clobber the first — both
in the in-memory `tables` registry and the underlying SQLite table
(`if_exists="replace"`). No error, no rename, no warning.

**Root cause:** `register_dataframe`'s collision-avoidance loop used
"same column list + same row count" as a proxy for "this is the same
table being re-registered," and `break`-ed out of the disambiguation loop
without ever comparing actual values.

**Fix:** Removed the shape-based short-circuit. The loop now only treats a
name as safe to reuse when the content is byte-for-byte identical
(`work.equals(existing)`, already being computed in the loop condition);
any other collision gets suffixed (`_2`, `_3`, ...).

**Verified:** Registering two same-shaped-but-different dataframes under
name `"orders"` now yields `orders` and `orders_2` with correct,
non-clobbered content; idempotent re-registration of the identical
dataframe still reuses the original name.

---

### 2. `agents/forecast_agent.py` — minimum-row check ran before date deduplication

**Symptom:** A dated series with ≥3 rows but only 1–2 *unique* dates (e.g.
duplicate readings on the same day) passed the `len(work) < 3` guard,
then got collapsed to 1–2 points by the subsequent
`groupby(date_col).mean()`. The code proceeded to fit a trend line through
essentially 1–2 points, reporting a spuriously "perfect" forecast
(`r2_historical = 1.0`) with a zero-width confidence interval
(`lower_95 == upper_95 == forecast`) and no warning to the user.

**Root cause:** The `len(work) < 3` guard (line ~789) ran before the
`groupby` deduplication (line ~798) rather than after it.

**Fix:** Added a second `len(work) < 3` check immediately after the
`groupby` aggregation, returning the same "not enough dated observations"
error if deduplication drops the row count below 3.

**Verified:** A 4-row series with only 2 unique dates now correctly
returns `success=False` with a clear error instead of a fake
zero-uncertainty forecast.

---

### 3. `agents/stats_agent.py` — zero-variance data silently mislabeled as real findings

**Symptom (group comparison):** When a t-test/ANOVA group has zero
variance (all values identical), `scipy.stats.ttest_ind`/`f_oneway`
returns `nan` for the statistic and p-value. The code did `p_val < alpha`
directly on this `nan`, which evaluates to `False` in Python — so the
result confidently reported **"No statistically significant difference"**
(with a garbled `p=nan` embedded in the text) instead of flagging the
test as undefined.

**Symptom (correlation):** Same issue in `run_correlation_analysis` — a
constant feature or constant target produces `nan` from
`scipy.stats.pearsonr`, and `"positive" if r > 0 else "negative"` labeled
the undefined correlation as **"negative"** rather than flagging it as
undefined.

**Root cause:** Neither function checked for zero-variance
groups/columns before or after calling into scipy, so `nan` propagated
into boolean/comparison logic that silently resolves to a specific
(wrong) answer.

**Fix:**
- `run_group_comparison`: detect a non-finite p-value after the test runs
  and return an explicit "test statistic is undefined" interpretation with
  `significant: None` and `p_value: None`, instead of a false
  "not significant" claim.
- `run_correlation_analysis`: return an explicit error if the target
  column has no variance; skip (rather than mislabel) any feature column
  with no variance before computing Pearson r; return an explicit error if
  every feature ends up skipped.

**Verified:** A constant-value group comparison (`val=[5,5,5,5]` across
two groups) now reports `significant=None` with an "undefined" message
instead of a false negative. A constant target column now returns
`success=False` with a clear message instead of mislabeling every
feature's correlation direction. Normal (non-degenerate) cases produce
identical results to before (confirmed against the churn sample data).

**Note on return-value contract:** `significant` was previously always a
`bool`; it can now also be `None` (meaning "undefined, not computable").
`p_value`/`statistic` were previously always `float` (sometimes `nan`);
they can now also be `None`. No code in this repo pattern-matches on
these fields expecting only `bool`/`float`, so this is a behavior
improvement, not a break — but flagging per the task's signature-change
disclosure requirement since it's an observable value-type change.

---

### 4. `agents/ingestion.py` — automatic type inference was dead code under pinned pandas 3.0

**Symptom:** `infer_and_cast_types` was supposed to auto-cast text columns
that actually hold numbers or dates (e.g. an `order_date` column read as
plain text) into proper `int64`/`float64`/`datetime64` columns at
ingestion time. Under `pandas==3.0.3` (the version pinned in
`requirements.txt`), CSV text columns get dtype `"str"`, not `"object"`,
so the guard `if series.dtype == object:` never fired — the whole
function was silently a no-op for every text column.

**Impact in practice:** Narrower than it first looked — pandas' own CSV
parser already auto-infers purely-numeric columns as `int64`/`float64`
natively, so the numeric-cast branch was largely redundant anyway. The
real effect was date-like text columns (e.g. `"2024-01-01"` strings)
staying as plain string columns after ingestion instead of becoming
`datetime64`. Downstream, `forecast_agent.detect_datetime_column` does its
own robust re-parsing of string columns, so forecasting wasn't broken by
this — but any other code path expecting `ingest_csv` to hand back
already-typed dates would have silently gotten strings instead.

**Root cause:** `series.dtype == object` is not a version-portable way to
detect "this is text," and pandas 3.0 changed the default dtype backend
for CSV string columns.

**Fix:** Changed the guard to
`not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series)`,
which correctly identifies non-numeric, non-datetime columns regardless
of whether pandas represents them as `object` or `str` dtype.

**Verified:** A CSV with a `qty` (numeric-as-text) and `order_date`
(date-as-text) column now correctly ends up with `order_date` cast to
`datetime64[us]` after ingestion.

---

### 5. `agents/ingestion.py` — removed pandas kwarg would crash on any object-dtype column

**Symptom:** `pd.to_datetime(series, errors="coerce", infer_datetime_format=True)`
raises `TypeError: to_datetime() got an unexpected keyword argument
'infer_datetime_format'` under pandas ≥ 2.0 (the kwarg was deprecated in
1.x and removed in 2.0; pandas 3.0.3 is pinned here). This was masked by
bug #4 for ordinary CSV ingestion (since the branch never ran), but was a
live landmine that would fire as soon as #4 was fixed, or via any other
call path that reaches a genuine `object`-dtype column (e.g. one holding
mixed types).

**Fix:** Dropped the removed `infer_datetime_format=True` kwarg — modern
pandas infers the format automatically. Also wrapped the call in
`warnings.catch_warnings()` to suppress the resulting (harmless, but
noisy) `UserWarning` pandas now emits when it can't find one consistent
format across a column of arbitrary non-date text (which happens for
every ordinary categorical text column now that the date-cast branch is
live) — this is expected, functionally correct behavior; the warning was
just log noise.

**Verified:** Ingesting a mixed-type `object` column no longer raises;
full `self_test.py` run shows zero stray `UserWarning`s.

---

### 6. `agents/ml_agent.py` — stratified train/test split crashed on rare classes

**Symptom:** `run_ml_analysis` on a classification target where one class
has exactly 1 member (common in small/imbalanced datasets) crashed with a
raw `sklearn` error — `"The least populated classes in y have only 1
member, which is too few..."` — surfaced to the user as
`ML analysis failed: ...` instead of degrading gracefully.

**Root cause:** `strat = y_enc if len(np.unique(y_enc)) > 1 else None`
only disabled stratification when there was a single class total; it
didn't check that *every* class has at least the 2 members
`train_test_split(..., stratify=...)` requires (1 for train, 1 for test).

**Fix:** Compute per-class counts via `np.unique(..., return_counts=True)`
and only stratify when there are 2+ classes *and* the smallest class has
at least 2 members; otherwise fall back to an unstratified split (the
model still trains, just without the stratification guarantee for that
one rare class).

**Verified:** A 6-row dataset with a singleton class (`["a","a","a","a","a","b"]`)
now trains successfully instead of crashing.

---

## Minor cleanup (not a functional bug)

### `agents/quality_agent.py` — dead code in `_iqr_outliers`

`mask` was computed once against the raw (already-numeric-only, since the
caller pre-filters to `df.select_dtypes(include=[np.number])`) series and
immediately discarded/overwritten by an equivalent computation against a
`pd.to_numeric`-coerced copy two lines later. Harmless (never caused wrong
output — the caller only ever passes already-numeric columns, so both
computations were equivalent), but pure waste. Removed the dead
assignment.

---

## Investigated and confirmed NOT bugs (left alone)

- **`agents/sql_agent.py` comment-obfuscation in the safety check**
  (`DROP/**/TABLE` can evade the forbidden-keyword regex). Confirmed the
  regex can technically be evaded this way, but it's not exploitable in
  practice: `is_safe_select` separately requires the query to start with
  `SELECT`/`WITH` and blocks any multi-statement (`;`-separated) query, so
  there's no path to smuggle a standalone destructive statement through.
  Low-value defense-in-depth gap, not a functional bug — left as-is.
- **`agents/orchestrator.py` — `"predict" in q and not any(kw in
  forecast_keywords)`** is redundant (the forecast-keyword branch already
  returns earlier), but harmless dead logic, not a bug.
- **`agents/orchestrator.py` — "report" route has no `has_data()` guard**
  unlike other routes. Intentional: `generate_report`/`_collect_findings`
  already handle `dataframe=None` gracefully by design (a report can
  summarize prior findings even without an active dataframe).
- **`agents/sql_agent.py` — `is_safe_select` blocks `SELECT DROP FROM x`**
  (where `DROP` is a legitimate column name). Intentional conservative
  trade-off in a security-sensitive path — a false positive here is far
  cheaper than a false negative.
- **RAG session isolation, RAG retrieval relevance, report HTML
  escaping/XSS, chunking boundary logic, `db/database.py` identifier
  quoting for reserved words/spaces, `insight_agent.py` edge cases
  (no-numeric, no-categorical, all-null data), quality auto-clean/category-merge
  logic** — all specifically stress-tested by hand (see method above) and
  found correct; no changes made.
- **`pd.factorize`-based categorical encoding in
  `run_correlation_analysis`** (stats_agent.py) — for categorical targets
  with >2 levels, the encoding order is arbitrary, so the correlation
  sign/magnitude against such a target isn't strictly meaningful. This is
  a real analytical limitation worth knowing about, but it's a documented
  simplification of an inherently ill-posed problem (Pearson correlation
  against an unordered multi-class label), not a code defect — left alone
  per the task's guidance not to guess at "fixes" for intentional
  trade-offs.

---

## Not exercised (no `GROQ_API_KEY` available in this environment)

The following LLM-dependent paths could not be driven live and were only
covered by static code reading, not execution:
- `agents/sql_agent.py` LLM-driven SQL generation quality (the rule-based
  safety/self-check logic *was* fully exercised and passes)
- `agents/rag_agent.py` `.answer()` (LLM-backed answer synthesis; chunk
  indexing and similarity retrieval *were* exercised and verified correct,
  including cross-session isolation)
- `agents/orchestrator.py` `handle_query` end-to-end LLM routing
- `agents/llm_client.py` / `utils/env.py` error handling against a real
  Groq API failure/timeout (structurally reviewed — both return `(None,
  err)` tuples on failure rather than raising, which looks sound, but
  wasn't exercised against a live failure)

If a `GROQ_API_KEY` becomes available, re-running `self_test.py` would
additionally exercise these 10 currently-skipped checks.
