"""Model selection: statistical significance and a weighted decision scorecard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from p2p.forecasting.metrics import evaluate

DEFAULT_WEIGHTS = {"Accuracy": 0.40, "Uncertainty": 0.20, "Robustness": 0.15, "Explainability": 0.10,
                   "Cost": 0.10, "Maintainability": 0.05}

# Expert judgement (1 to 5) recorded once and justified in notebook 11.
EXPLAINABILITY = {
    "Naive": 5, "Seasonal Naive": 5, "Moving Average (13w)": 5, "ETS": 4, "Theta": 4, "SARIMAX (Fourier + PM plan)": 4,
    "Croston": 4, "TSB": 4, "Prophet": 4, "LightGBM": 3, "XGBoost": 3, "Random Forest": 3, "N-HiTS": 2, "N-BEATS": 1,
    "Chronos-Bolt (zero-shot)": 1, "Monte Carlo (structural demand)": 5, "Ensemble (top 3)": 2,
}
MAINTAINABILITY = {
    "Naive": 5, "Seasonal Naive": 5, "Moving Average (13w)": 5, "ETS": 5, "Theta": 5, "SARIMAX (Fourier + PM plan)": 4,
    "Croston": 5, "TSB": 5, "Prophet": 4, "LightGBM": 3, "XGBoost": 3, "Random Forest": 3, "N-HiTS": 2, "N-BEATS": 2,
    "Chronos-Bolt (zero-shot)": 3, "Monte Carlo (structural demand)": 3, "Ensemble (top 3)": 2,
}


def diebold_mariano(fc: pd.DataFrame, model_a: str, model_b: str, max_lag: int = 11) -> tuple[float, float]:
    """Diebold-Mariano test on absolute errors (averaged across series per forecast week).

    Negative statistic means model_a has lower loss. Newey-West variance handles the overlap
    of multi-step forecasts. Returns (statistic, two-sided p-value).
    """
    a = fc[fc.model == model_a].assign(ae=lambda d: (d.y - d.yhat).abs()).groupby(["cutoff", "h"]).ae.mean()
    b = fc[fc.model == model_b].assign(ae=lambda d: (d.y - d.yhat).abs()).groupby(["cutoff", "h"]).ae.mean()
    d = (a - b).dropna().to_numpy()
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(max_lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (max_lag + 1)) * gamma[k] for k in range(1, max_lag + 1))
    stat = dbar / np.sqrt(max(var, 1e-12) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def robustness(fc: pd.DataFrame, scale: pd.Series) -> pd.Series:
    """Worst quarter MASE in the test period (lower is better): how badly does a model break?"""
    f = fc.assign(q=fc.ds.dt.to_period("Q"), ae=(fc.y - fc.yhat).abs() / fc.unique_id.map(scale))
    return f.groupby(["model", "q"]).ae.mean().groupby("model").max()


def scorecard(fc: pd.DataFrame, scale: pd.Series, seconds: dict, weights: dict | None = None) -> pd.DataFrame:
    """Score every model 0-100 on each criterion and combine with the given weights."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    ev = evaluate(fc, scale).set_index("model")
    rb = robustness(fc, scale)
    raw = pd.DataFrame({
        "MASE": ev.MASE,
        "CRPS": ev.get("CRPS"),
        "Coverage80": ev.get("Coverage80"),
        "WorstQuarterMASE": rb,
        "Seconds": pd.Series(seconds),
        "ExplainabilityScore": pd.Series(EXPLAINABILITY),
        "MaintainabilityScore": pd.Series(MAINTAINABILITY),
    }).loc[ev.index]

    def lower_better(x):
        x = x.astype(float)
        return 100 * (x.max() - x) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(100.0, index=x.index)

    unc = 0.5 * lower_better(raw.CRPS) + 0.5 * lower_better((raw.Coverage80 - 0.8).abs())
    out = pd.DataFrame({
        "Accuracy": lower_better(raw.MASE),
        "Uncertainty": unc,
        "Robustness": lower_better(raw.WorstQuarterMASE),
        "Explainability": (raw.ExplainabilityScore - 1) / 4 * 100,
        "Cost": lower_better(np.log1p(raw.Seconds)),
        "Maintainability": (raw.MaintainabilityScore - 1) / 4 * 100,
    })
    out["Weighted score"] = sum(out[k] * v for k, v in w.items()) / sum(w.values())
    out = pd.concat([raw[["MASE", "CRPS", "Coverage80", "WorstQuarterMASE", "Seconds"]], out], axis=1)
    return out.sort_values("Weighted score", ascending=False)


def ensemble(fc: pd.DataFrame, members: list[str], name: str = "Ensemble (top 3)") -> pd.DataFrame:
    """Equal-weight average of point forecasts and quantiles of the member models."""
    cols = ["yhat", "q0.025", "q0.1", "q0.5", "q0.9", "q0.975"]
    e = fc[fc.model.isin(members)].groupby(["unique_id", "ds", "cutoff", "h", "y"], as_index=False)[cols].mean()
    e["model"] = name
    return e
