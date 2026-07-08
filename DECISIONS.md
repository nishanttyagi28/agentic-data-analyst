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

7. **SELECT-only regex guard** — Blocks INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE and multi-statement queries before execution. Defense-in-depth beyond LLM prompting.

8. **Double-quoted identifiers** — Prompt instructs LLM to quote table/column names for SQLite compatibility with cleaned column names.

## ML Agent

9. **Target detection priority** — (1) column mentioned in user query, (2) known target keywords (`churn`, `price`, `label`, etc.), (3) sole categorical column, (4) sole low-cardinality numeric column.

10. **Task type defaults** — Numeric target with >10 unique values → regression; otherwise classification; no target or explicit cluster request → KMeans + PCA.

11. **XGBoost with `use_label_encoder=False`** — Avoids deprecation warnings in recent XGBoost versions.

12. **Minimum 5 rows for ML** — Prevents sklearn errors on tiny datasets during demo.

## RAG Agent

13. **Chunk size 500 / overlap 50** — Paragraph-aware splitting with light overlap for context continuity.

14. **Top-k=5 retrieval** — Balances context richness vs. LLM token limits.

15. **Auto-index after SQL/ML** — Orchestrator indexes successful agent results into Chroma so follow-up RAG works without manual step.

## UI

16. **Sidebar upload + main chat** — Upload/schema in sidebar; single chat input routes all agent types per spec.

17. **Sample datasets in sidebar** — Two CSVs (churn classification, house price regression) for immediate demo without upload.

## LLM

18. **Groq `llama-3.3-70b-versatile`** — Used for SQL generation, explanations, ML summaries, RAG answers, and orchestration classification.

19. **Graceful API key fallback** — App runs without key; sidebar warning + agent-level errors instead of crashes.