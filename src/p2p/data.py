"""Data access helpers. Every notebook and the app read the warehouse through these functions."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from p2p.config import APP_DATA_DIR, DUCKDB_PATH


def db_path() -> Path:
    """The full dbt warehouse when it exists (laboratory), otherwise the compact copy shipped with the app."""
    if os.getenv("P2P_DB"):
        return Path(os.environ["P2P_DB"])
    return DUCKDB_PATH if DUCKDB_PATH.exists() else APP_DATA_DIR / "p2p_app.duckdb"


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Connect to the DuckDB warehouse."""
    return duckdb.connect(str(db_path()), read_only=read_only)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with connect() as con:
        return con.execute(sql, params or []).df()


@lru_cache(maxsize=None)
def weekly_section() -> pd.DataFrame:
    """Forecasting base table: one row per section and week."""
    df = query("select * from marts.fct_weekly_section order by section_id, week_start")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def panel(target: str = "pr_lines") -> pd.DataFrame:
    """Long panel in Nixtla format (unique_id, ds, y) at section level."""
    df = weekly_section()
    return df.rename(columns={"section_id": "unique_id", "week_start": "ds", target: "y"})[["unique_id", "ds", "y"]].copy()


def company_series(target: str) -> pd.DataFrame:
    """Company-level weekly series for a target column of fct_weekly_section."""
    df = weekly_section().groupby("week_start", as_index=False)[target].sum()
    return pd.DataFrame({"unique_id": "Company", "ds": df.week_start, "y": df[target].astype(float)})


def export_app_tables(tables: dict[str, str]) -> None:
    """Export warehouse tables to parquet for the lightweight deployed app."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        for name, sql in tables.items():
            con.execute(f"copy ({sql}) to '{APP_DATA_DIR / (name + '.parquet')}' (format parquet)")
