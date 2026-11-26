"""Hierarchical forecasting: make section, business-unit and company forecasts add up.

Planners use the company total, managers use their business unit and buyers use sections.
Independent forecasts at each level do not add up; reconciliation fixes that and often
improves accuracy at the top levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta

from p2p.data import weekly_section

SPEC = [["Company"], ["Company", "business_unit"], ["Company", "business_unit", "section_id"]]


def hierarchical_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ws = weekly_section()
    df = ws.rename(columns={"week_start": "ds", "pr_lines": "y"})[["business_unit", "section_id", "ds", "y"]].copy()
    df["Company"] = "Company"
    df["y"] = df.y.astype(float)
    Y, S, tags = aggregate(df, SPEC)
    return Y, S, tags


def reconcile_backtest(origins: list[pd.Timestamp], h: int = 12) -> pd.DataFrame:
    """Base forecasts (ETS and Theta) at every level, then BottomUp and MinT reconciliation."""
    Y, S, tags = hierarchical_frame()
    rows = []
    for c in origins:
        train = Y[Y.ds <= c]
        sf = StatsForecast(models=[AutoETS(season_length=1, alias="ETS"), AutoTheta(season_length=52, alias="Theta")],
                           freq="W-MON", n_jobs=1)
        base = sf.forecast(df=train, h=h, fitted=True)
        fitted = sf.forecast_fitted_values()
        hrec = HierarchicalReconciliation(reconcilers=[BottomUp(), MinTrace(method="mint_shrink")])
        rec = hrec.reconcile(Y_hat_df=base, Y_df=fitted, S_df=S, tags=tags)
        rec["cutoff"] = c
        rows.append(rec)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(Y, on=["unique_id", "ds"], how="inner")
    out["h"] = ((out.ds - out.cutoff).dt.days // 7).astype(int)
    out["level"] = np.where(out.unique_id == "Company", "Company",
                            np.where(out.unique_id.str.count("/") == 1, "Business unit", "Section"))
    return out


def level_accuracy(rec: pd.DataFrame) -> pd.DataFrame:
    """WAPE by hierarchy level for each base and reconciled forecast."""
    methods = [c for c in rec.columns if c not in ("unique_id", "ds", "y", "cutoff", "h", "level")]
    res = []
    for lvl, g in rec.groupby("level"):
        for m in methods:
            res.append({"level": lvl, "method": m, "WAPE": float((g.y - g[m]).abs().sum() / g.y.abs().sum())})
    return pd.DataFrame(res).pivot(index="method", columns="level", values="WAPE")[["Company", "Business unit", "Section"]]
