"""Baselines and statistical forecasters (statsforecast and Prophet)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    CrostonOptimized,
    Naive,
    SeasonalNaive,
    TSB,
    WindowAverage,
)
from statsforecast.utils import ConformalIntervals

from p2p.forecasting.features import known_future_exog

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

LEVELS = [80, 95]


def _to_quantiles(fc: pd.DataFrame, name: str) -> pd.DataFrame:
    out = pd.DataFrame({"unique_id": fc.unique_id, "ds": fc.ds, "yhat": fc[name]})
    if f"{name}-lo-80" in fc:
        out["q0.1"], out["q0.9"] = fc[f"{name}-lo-80"], fc[f"{name}-hi-80"]
        out["q0.025"], out["q0.975"] = fc[f"{name}-lo-95"], fc[f"{name}-hi-95"]
    return out


def statsforecast_model(model, name: str, exog: bool = False):
    """Wrap a statsforecast model into the backtest signature."""
    X = known_future_exog() if exog else None

    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        m = model
        sf = StatsForecast(models=[m], freq="W-MON", n_jobs=1)
        if exog:
            tr = train.merge(X, on=["unique_id", "ds"], how="left")
            fut = X[(X.ds > cutoff) & (X.ds <= cutoff + pd.Timedelta(weeks=h))]
            fut = fut[fut.unique_id.isin(train.unique_id.unique())]
            fc = sf.forecast(df=tr, h=h, X_df=fut, level=LEVELS)
        else:
            fc = sf.forecast(df=train, h=h, level=LEVELS)
        return _to_quantiles(fc.reset_index() if "unique_id" not in fc else fc, m.alias)
    fn.__name__ = name
    return fn


def baseline_models() -> dict:
    return {
        "Naive": statsforecast_model(Naive(alias="Naive"), "Naive"),
        "Seasonal Naive": statsforecast_model(SeasonalNaive(season_length=52, alias="Seasonal Naive"), "Seasonal Naive"),
        "Moving Average (13w)": statsforecast_model(
            WindowAverage(window_size=13, alias="Moving Average (13w)", prediction_intervals=ConformalIntervals(h=12, n_windows=4)),
            "Moving Average (13w)"),
    }


def statistical_models() -> dict:
    ci = ConformalIntervals(h=12, n_windows=4)
    return {
        "ETS": statsforecast_model(AutoETS(season_length=1, alias="ETS"), "ETS"),
        "Theta": statsforecast_model(AutoTheta(season_length=52, alias="Theta"), "Theta"),
        "SARIMAX (Fourier + PM plan)": statsforecast_model(
            AutoARIMA(season_length=1, alias="SARIMAX (Fourier + PM plan)"), "SARIMAX (Fourier + PM plan)", exog=True),
        "Croston": statsforecast_model(CrostonOptimized(alias="Croston", prediction_intervals=ci), "Croston"),
        "TSB": statsforecast_model(TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB", prediction_intervals=ci), "TSB"),
    }


def prophet_model(name: str = "Prophet"):
    from prophet import Prophet

    X = known_future_exog()

    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for uid, g in train.groupby("unique_id"):
            g = g.merge(X[X.unique_id == uid], on=["unique_id", "ds"], how="left")
            m = Prophet(yearly_seasonality=8, weekly_seasonality=False, daily_seasonality=False,
                        interval_width=0.8, uncertainty_samples=300, changepoint_prior_scale=0.05)
            for reg in ["pm_planned_next_3w", "shutdown", "budget_push", "mrp_week"]:
                m.add_regressor(reg)
            m.fit(g[["ds", "y", "pm_planned_next_3w", "shutdown", "budget_push", "mrp_week"]])
            fut = X[(X.unique_id == uid) & (X.ds > cutoff) & (X.ds <= cutoff + pd.Timedelta(weeks=h))]
            p = m.predict(fut[["ds", "pm_planned_next_3w", "shutdown", "budget_push", "mrp_week"]])
            samples = m.predictive_samples(fut[["ds", "pm_planned_next_3w", "shutdown", "budget_push", "mrp_week"]])["yhat"]
            qs = np.quantile(samples, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
            rows.append(pd.DataFrame({"unique_id": uid, "ds": fut.ds.values, "yhat": p.yhat.values,
                                      "q0.025": qs[0], "q0.1": qs[1], "q0.5": qs[2], "q0.9": qs[3], "q0.975": qs[4]}))
        return pd.concat(rows, ignore_index=True)
    fn.__name__ = name
    return fn
