# Autonomous Decisions Log

## Architecture

1. **Custom Python router over LangGraph** — A rule-based classifier handles obvious cases (SQL/ML/RAG keywords); LLM classification is the fallback. This avoids LangGraph dependency and keeps orchestration simple.

2. **Session-scoped RAG collections** — Each dataset upload gets a new `session_id` and ChromaDB collection (`session_<id>`). Prevents cross-contamination between datasets.

3. **SQLite path `data/analyst.db`** — DB and Chroma persist under a `data/` directory created at runtime. Keeps the repo clean and matches zero-setup requirement.

4. **Embeddings loaded lazily** — `SentenceTransformer` is instantiated on first RAG use to speed up app startup when only SQL/ML is used.

## Ingestion

5. **Column name cleaning** — Lowercase, strip special chars, replace spaces with underscores, prefix numeric-leading names with `col_`. Handles messy real-world CSV headers.

6. **Type inference threshold 80%** — A column is cast to numeric/datetime only if ≥80% of values parse successfully. Balances auto-typing vs. corrupting categorical data.

## SQL Agent

7. **SELECT/CTE-only regex guard** — Blocks INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE and multi-statement queries before execution. Allows `WITH` (CTE) and window functions. Word-boundary matching avoids false positives on column names like `last_update`.

8. **Double-quoted identifiers** — Prompt instructs LLM to quote table/column names for SQLite compatibility with cleaned column names.

## ML Agent

9. **Target detection priority** — (1) column mentioned in user query, (2) known target keywords (`churn`, `price`, `label`, etc.), (3) sole categorical column, (4) sole low-cardinality numeric column.

10. **Task type defaults** — Numeric target with >10 unique values → regression; otherwise classification; no target or explicit cluster request → KMeans + PCA.

11. **XGBoost with `use_label_encoder=False`** — Avoids deprecation warnings in recent XGBoost versions.

12. **Minimum 5 rows for ML** — Prevents sklearn errors on tiny datasets during demo.

## RAG Agent

13. **Chunk size 500 / overlap 50** — Paragraph-aware splitting with light overlap for context continuity.

14. **Top-k=5 retrieval** — Balances context richness vs. LLM token limits.

15. **Auto-index after SQL/ML/stats/forecast/quality/report** — Orchestrator indexes successful agent results into Chroma so follow-up RAG works without manual step.

## UI

16. **Sidebar upload + main chat** — Upload/schema in sidebar; single chat input routes all agent types per spec.

17. **Sample datasets in sidebar** — Two CSVs (churn classification, house price regression) for immediate demo without upload.

## LLM

18. **Groq `llama-3.3-70b-versatile`** — Used for SQL generation, explanations, ML summaries, RAG answers, orchestration classification, and report executive summaries.

19. **Graceful API key fallback** — App runs without key; sidebar warning + agent-level errors instead of crashes.

---

## Full Analyst Upgrade (2026-07)

### Phase A — Data Quality Agent

20. **Quality as a soft gate, not a blocker** — After ingestion, auto-run `analyze_data_quality` and show a card with score + metrics. User can auto-clean or skip; analysis never blocked.

21. **Auto-clean defaults (safe only)** — Drop exact duplicates; median impute for numeric missing; mode impute for categorical missing; cast numbers-stored-as-text. **Never** auto-remove outliers or merge categories (flag for review only).

22. **Outliers via IQR (1.5×)** — Standard, dependency-free heuristic suitable for small demo datasets; z-score not used as primary (small-n unstable).

23. **Categorical near-duplicates** — `SequenceMatcher` + substring heuristics flag groups like USA/US/United States; never auto-merge.

### Phase B — Expanded EDA

24. **Full descriptive stats** — mean, median, std, min/max, q25/q75 per numeric column; `describe_table` exposed in UI.

25. **Group-by auto-suggestions** — Up to 3 low-cardinality categorical × KPI-like numeric pairs as bar charts.

26. **Time-series charts only when date-like** — Heuristic requires digit patterns before `to_datetime` to avoid parsing free-text categories (and noisy warnings).

### Phase C — Stats Agent

27. **Welch t-test (2 groups) / one-way ANOVA (3+)** — SciPy; always attach assumptions + small-sample caveats; never claim causation.

28. **Pearson correlation ranking** — Encode low-cardinality non-numeric targets via factorize for churn-style questions.

29. **Intent detection** — Keyword rules for outliers / correlation / comparison before falling back to correlation.

### Phase D — Forecasting

30. **LinearRegression trend + residual band** — Lightweight, no Prophet/statsmodels. Approx. 95% band = ±1.96 × residual_std × mild horizon scale. Explicitly labeled as estimates.

31. **No-date fallback** — Use observation index as time proxy so forecasts still work on churn/house samples; caveat in summary.

### Phase E — Report Export

32. **HTML over PDF** — HTML download works reliably in Streamlit Community Cloud without WeasyPrint/wkhtmltopdf. User can print-to-PDF from browser.

33. **Executive summary** — LLM when API available; deterministic fallback from findings so offline tests still produce a report.

### Phase F — Orchestrator

34. **Route priority** — report → quality → forecast → stats → ml → sql → rag. Specific multi-word phrases before broad keywords like "analysis" / "report".

35. **Report phrase variants** — Match `generate a report` (not only `generate report`) to avoid false ML/RAG routing.

### Phase G — UI

36. **Persistent sidebar "What I can help with"** — Lists cleaning, EDA, stats, forecast, ML, reports, SQL, RAG so non-technical users know the surface area.

37. **Generate Report button** — Always available when data is loaded; compiles whatever session findings exist so far.

---

## Testing log (upgrade)

### Bugs found and fixed during self-test loop

| # | Phase | Bug | Fix |
|---|-------|-----|-----|
| 1 | F | `"Generate a report of my analysis"` routed away from `report` because keyword was exact phrase `generate report` (no room for "a") and `"analysis"` matched ML | Added `generate a report`, `create a report`, `analysis report`, etc. |
| 2 | B/D | `pd.to_datetime` on categorical text columns emitted noisy parse warnings and risked false date detection | Date detection now requires digit/date-like patterns and caps high-cardinality text; uses `format="mixed"` |

### Regression results (final clean pass)

- `python self_test.py` → **125 passed, 0 failed, 0 skipped**
- Covered: SQL safety (SELECT/CTE/window + injection blocks), quality profile + auto-clean (outliers not dropped), expanded EDA, stats (t-test/ANOVA/corr/outliers), forecast bands, HTML report, orchestrator routes, full churn + houses flows (SQL, ML, RAG, quality, stats, forecast, report).

### Guardrails re-confirmed

- SQL safety validator unchanged in spirit (SELECT/WITH only, word-boundary forbidden keywords, no multi-statement).
- Auto-clean never drops outliers or merges categories without user action.
- Stats/forecast language always includes caveats (sample size, correlation ≠ causation, estimate bands).
- Works on small sample CSVs (~25–30 rows) and synthetic messy data.

---

## Maximum technical analyst upgrade (Phases H–L)

### Phase H — Proactive insights

38. **Rule-based suggestions (no LLM by default)** — Correlation pairs, group comparisons, target-like columns, anomaly spikes (strict IQR 2.5×), class imbalance, heavy missing. Keeps free-tier token use low; clickable labels map to normal orchestrator routes.

### Phase I — AutoML

39. **Multi-candidate models** — Classification: LogisticRegression, RandomForest, XGBoost; Regression: Ridge, RF, XGBoost. Tiny ParameterGrid only (session-friendly).
40. **Feature prep** — Date → month/dow/year; drop high-cardinality categoricals (>40 levels) with a flag; exclude likely IDs.
41. **Overfit flags** — Small n + very high accuracy/R² called out in summary (not left as silent “perfect score”).

### Phase J — Multi-table

42. **In-memory table registry + SQLite tables** — Multiple CSVs load as named tables; first/single file remains `user_data` for backward compatibility.
43. **Join-key heuristic** — Shared column names + value overlap; surfaced in sidebar and SQL schema prompt so the LLM can JOIN.

### Phase K — Business context

44. **Optional free-text context** — Stored on Orchestrator; passed into insight wording and ML summary prompts. Blank = previous behavior.

### Phase L — Ambiguous decisions

45. **Decision objects on quality report** — Category merge options with counts; ID-vs-feature options. Apply only on explicit user click (`apply_category_merge` / `exclude_ml_cols`). Never silent merge.

### Testing log (H–L)

| # | Phase | Bug | Fix |
|---|-------|-----|-----|
| 1 | I | `Ridge(random_state=...)` not valid on all sklearn builds | Removed `random_state` from Ridge |
| 2 | L | `build_decision_options` → `analyze_data_quality` recursion risk | Build cat issues from df/report without re-entering analyze |
| 3 | H–L | (preventive) multi-table SQL must see all tables | `run_sql_query(..., tables=)` + multi schema context |

Full regression re-run after H–L: see self_test output at ship time.
