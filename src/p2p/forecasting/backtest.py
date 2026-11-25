"""Rolling-origin backtesting shared by every model family.

A forecaster is any function with the signature

    fn(train: DataFrame[unique_id, ds, y], h: int, cutoff: Timestamp) -> DataFrame

that returns columns unique_id, ds, yhat and, when available, the quantile columns
q0.025, q0.1, q0.5, q0.9, q0.975. The engine handles the expanding window, the
alignment with actuals and the horizon index, so every model is scored the same way.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from p2p.config import TEST_END, TEST_START, TRAIN_END
from p2p.forecasting.metrics import QUANTILES

Forecaster = Callable[[pd.DataFrame, int, pd.Timestamp], pd.DataFrame]
H = 12
STEP = 4


def cutoffs(start: str, end: str, h: int = H, step: int = STEP) -> list[pd.Timestamp]:
    """Forecast origins every `step` weeks such that origin + h stays within `end`."""
    last = pd.Timestamp(end) - pd.Timedelta(weeks=h)
    return list(pd.date_range(pd.Timestamp(start), last, freq=f"{step}W-MON"))


def validation_cutoffs() -> list[pd.Timestamp]:
    """Origins inside 2024, used to choose and tune models (never touches 2025)."""
    return cutoffs("2024-01-01", TRAIN_END)


def test_cutoffs() -> list[pd.Timestamp]:
    """Origins for the 2025 holdout. The first origin is the last training week."""
    return cutoffs(TRAIN_END, TEST_END)


def _finish(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    for q in QUANTILES:
        c = f"q{q}"
        if c not in fc:
            fc[c] = np.nan
    if fc["q0.5"].isna().all():
        fc["q0.5"] = fc["yhat"]
    num = ["yhat"] + [f"q{q}" for q in QUANTILES]
    fc[num] = fc[num].clip(lower=0)
    # enforce monotone quantiles
    qs = fc[[f"q{q}" for q in QUANTILES]].to_numpy()
    if not np.isnan(qs).all():
        fc[[f"q{q}" for q in QUANTILES]] = np.sort(qs, axis=1)
    return fc


def rolling_origin(fn: Forecaster, df: pd.DataFrame, origins: list[pd.Timestamp], name: str,
                   h: int = H, verbose: bool = False) -> pd.DataFrame:
    """Run `fn` at every origin and return forecasts joined with actuals."""
    out = []
    t0 = time.time()
    for c in origins:
        train = df[df.ds <= c]
        fc = fn(train, h, c)
        fc = _finish(fc)
        fc["cutoff"] = c
        fc["h"] = ((fc.ds - c).dt.days // 7).astype(int)
        out.append(fc[fc.h.between(1, h)])
        if verbose:
            print(f"{name}: origin {c.date()} done")
    res = pd.concat(out, ignore_index=True)
    res = res.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")
    res["model"] = name
    res.attrs["seconds"] = time.time() - t0
    return res


def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df.ds <= TRAIN_END].copy(), df[(df.ds >= TEST_START) & (df.ds <= TEST_END)].copy()
