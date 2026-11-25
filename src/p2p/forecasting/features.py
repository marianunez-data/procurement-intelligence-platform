"""Known-future covariates.

Only information a planner genuinely has in advance is used here: the calendar (Fourier
terms, year-end shutdown, budget push), the bi-weekly MRP run and the planned-maintenance
schedule. Corrective work orders, utilisation and FX are NOT known in advance and are only
used as lagged features inside the ML feature store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2p.data import weekly_section

FOURIER_K = 3


def calendar_features(ds: pd.Series) -> pd.DataFrame:
    ds = pd.to_datetime(pd.Series(ds)).reset_index(drop=True)
    woy = ds.dt.isocalendar().week.astype(int).to_numpy()
    out = {}
    for k in range(1, FOURIER_K + 1):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    out["shutdown"] = (((ds.dt.month == 12) & (ds.dt.day >= 15)) | ((ds.dt.month == 1) & (ds.dt.day <= 6))).astype(int).to_numpy()
    out["budget_push"] = ds.dt.month.isin([10, 11]).astype(int).to_numpy()
    ref = pd.Timestamp("2022-01-03")
    out["mrp_week"] = ((((ds - ref).dt.days // 7) % 2) == 0).astype(int).to_numpy()
    return pd.DataFrame(out)


def known_future_exog(level: str = "section") -> pd.DataFrame:
    """Exogenous frame (unique_id, ds, features) for every section-week in the data."""
    ws = weekly_section()
    if level == "section":
        base = ws[["section_id", "week_start", "pm_planned_next_3w"]].rename(columns={"section_id": "unique_id", "week_start": "ds"})
    else:
        base = ws.groupby("week_start", as_index=False).pm_planned_next_3w.sum().rename(columns={"week_start": "ds"})
        base.insert(0, "unique_id", "Company")
    cal = calendar_features(base.ds)
    return pd.concat([base.reset_index(drop=True), cal], axis=1)
