"""Run-or-load cache for backtests, so notebooks stay fast and results are reproducible.

Every backtest result is stored as parquet under reports/experiments/<arena>/<split>/ with its
runtime. Delete a file (or pass force=True) to recompute it.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from p2p.config import REPORTS_DIR
from p2p.forecasting import backtest as bt

EXP_DIR = REPORTS_DIR / "experiments"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_or_load(name: str, fn, df: pd.DataFrame, origins, arena: str, split: str, force: bool = False, h: int = bt.H) -> pd.DataFrame:
    d = EXP_DIR / arena / split
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{_slug(name)}.parquet"
    meta = d / f"{_slug(name)}.json"
    if f.exists() and not force:
        out = pd.read_parquet(f)
        out.attrs["seconds"] = json.loads(meta.read_text())["seconds"]
        return out
    out = bt.rolling_origin(fn, df, origins, name, h=h)
    out.to_parquet(f, index=False)
    meta.write_text(json.dumps({"model": name, "seconds": out.attrs["seconds"], "origins": [str(o.date()) for o in origins]}))
    return out


def load_arena(arena: str, split: str) -> tuple[pd.DataFrame, dict]:
    d = EXP_DIR / arena / split
    frames, secs = [], {}
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f)
        frames.append(df)
        secs[df.model.iloc[0]] = json.loads(f.with_suffix(".json").read_text())["seconds"]
    return pd.concat(frames, ignore_index=True), secs
