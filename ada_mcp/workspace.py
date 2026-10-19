"""Per-process state for the MCP server: one SQLite engine and the table registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

import agents
from db.database import get_engine

# sample_data/ sits next to agents/ both in the repo and in an installed wheel
SAMPLE_DIR = Path(agents.__file__).resolve().parent.parent / "sample_data"


def _memory_engine() -> Engine:
    # StaticPool keeps one connection, otherwise every connect() gets an empty db
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def rag_enabled() -> bool:
    return os.getenv("ADA_MCP_RAG", "").strip().lower() in ("1", "true", "yes")


@dataclass
class Workspace:
    engine: Engine
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    rag: Any = None

    @classmethod
    def in_memory(cls) -> "Workspace":
        return cls(engine=_memory_engine())

    @classmethod
    def from_env(cls) -> "Workspace":
        """ADA_MCP_DB=<path> keeps tables in a SQLite file between runs; default is in-memory."""
        db_path = os.getenv("ADA_MCP_DB", "").strip()
        if not db_path:
            return cls.in_memory()
        ws = cls(engine=get_engine(os.path.abspath(os.path.expanduser(db_path))))
        ws.reload_tables()
        return ws

    def reload_tables(self) -> None:
        """Rebuild the in-memory registry from tables already in the database."""
        for name in inspect(self.engine).get_table_names():
            self.tables[name] = pd.read_sql_table(name, self.engine)

    def get_rag(self):
        if not rag_enabled():
            return None
        if self.rag is None:
            # heavy imports (chromadb, sentence-transformers) only when asked for
            from agents.rag_agent import RAGAgent

            self.rag = RAGAgent(session_id="mcp")
        return self.rag


def list_samples() -> list[str]:
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(p.stem for p in SAMPLE_DIR.glob("*.csv"))
