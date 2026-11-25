"""Forecast evaluation metrics: point accuracy, bias and probabilistic quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)


def mase(y, yhat, scale) -> float:
    """Mean absolute scaled error. `scale` is the in-sample MAE of the seasonal naive."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))) / scale)


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: total absolute error over total actuals."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)))


def bias(y, yhat) -> float:
    """Signed bias as a share of actuals. Positive means over-forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sum(yhat - y) / np.sum(y))


def pinball(y, q_pred, q: float) -> float:
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def crps_from_quantiles(y, qpreds: dict[float, np.ndarray]) -> float:
    """Approximate CRPS as twice the average pinball loss over the available quantiles."""
    return float(2 * np.mean([pinball(y, qpreds[q], q) for q in qpreds]))


def seasonal_naive_scale(train: pd.DataFrame, season: int = 52) -> pd.Series:
    """Per-series MASE denominator from the training data."""
    def _s(g):
        y = g.sort_values("ds")["y"].to_numpy()
        if len(y) <= season:
            return np.mean(np.abs(np.diff(y)))
        return np.mean(np.abs(y[season:] - y[:-season]))
    return train.groupby("unique_id").apply(_s, include_groups=False)


def evaluate(fc: pd.DataFrame, scale: pd.Series, model_col: str = "model") -> pd.DataFrame:
    """Score a forecast frame with columns unique_id, ds, cutoff, h, y, model, yhat and quantile columns q0.1 ..."""
    rows = []
    for (model,), g in fc.groupby([model_col]):
        sc = g.unique_id.map(scale).to_numpy()
        ae = np.abs(g.y - g.yhat).to_numpy()
        r = dict(
            model=model,
            MASE=float(np.mean(ae / sc)),
            WAPE=wape(g.y, g.yhat),
            Bias=bias(g.y, g.yhat),
        )
        qcols = {q: f"q{q}" for q in QUANTILES if f"q{q}" in g}
        if len(qcols) == len(QUANTILES):
            r["Pinball"] = float(np.mean([pinball(g.y, g[c], q) for q, c in qcols.items()]))
            r["CRPS"] = crps_from_quantiles(g.y, {q: g[c].to_numpy() for q, c in qcols.items()})
            r["Coverage80"] = coverage(g.y, g["q0.1"], g["q0.9"])
            r["Coverage95"] = coverage(g.y, g["q0.025"], g["q0.975"])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
