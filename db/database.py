"""SQLAlchemy engine and session helpers for SQLite (Postgres-ready)."""

import os
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "analyst.db")
TABLE_NAME = "user_data"


def get_engine(db_path: str | None = None) -> Engine:
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_table_schema(engine: Engine, table_name: str = TABLE_NAME) -> list[dict[str, Any]]:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return []
    columns = []
    for col in inspector.get_columns(table_name):
        columns.append({
            "name": col["name"],
            "type": str(col["type"]),
            "nullable": col.get("nullable", True),
        })
    return columns


def get_row_count(engine: Engine, table_name: str = TABLE_NAME) -> int:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return 0
    with engine.connect() as conn:
        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        return result.scalar() or 0


def load_dataframe_to_table(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str = TABLE_NAME,
    if_exists: str = "replace",
) -> None:
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)


def execute_select_query(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})