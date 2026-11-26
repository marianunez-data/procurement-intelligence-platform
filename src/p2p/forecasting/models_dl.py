"""Deep-learning forecasters (N-HiTS, N-BEATS) and a zero-shot foundation model (Chronos-Bolt)."""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

from p2p.forecasting.features import known_future_exog

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for name in ["pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric", "neuralforecast"]:
    logging.getLogger(name).setLevel(logging.ERROR)

FUTR = ["pm_planned_next_3w", "sin1", "cos1", "sin2", "cos2", "shutdown", "budget_push", "mrp_week"]
Z_RATIO = 1.959964 / 1.281552  # extrapolate 95% bounds from 80% bounds (normal approximation)


def _mq_to_frame(fc: pd.DataFrame, alias: str) -> pd.DataFrame:
    med = f"{alias}-median" if f"{alias}-median" in fc else alias
    out = pd.DataFrame({"unique_id": fc.unique_id, "ds": fc.ds, "yhat": fc[med]})
    out["q0.5"] = fc[med]
    out["q0.1"], out["q0.9"] = fc[f"{alias}-lo-80"], fc[f"{alias}-hi-80"]
    out["q0.025"], out["q0.975"] = fc[f"{alias}-lo-95"], fc[f"{alias}-hi-95"]
    return out


def neural_model(kind: str, max_steps: int = 500, seed: int = 1):
    """Global neural network trained on all sections at once.

    A negative-binomial output layer suits weekly counts, and early stopping on the last
    12 training weeks plus dropout stop the network from memorising the short history
    (without them the prediction intervals collapse, see notebook 09).
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import NBEATS, NHITS

    X = known_future_exog()

    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        common = dict(h=h, input_size=26, loss=DistributionLoss(distribution="NegativeBinomial", level=[80, 95]),
                      max_steps=max_steps, random_seed=seed, enable_progress_bar=False, enable_model_summary=False,
                      logger=False, accelerator="cpu", devices=1, batch_size=12, windows_batch_size=128, learning_rate=1e-3,
                      early_stop_patience_steps=5, val_check_steps=20)
        if kind == "N-HiTS":
            model = NHITS(alias="N-HiTS", futr_exog_list=FUTR, dropout_prob_theta=0.3, **common)
            tr = train.merge(X[["unique_id", "ds"] + FUTR], on=["unique_id", "ds"], how="left")
            fut = X[(X.ds > cutoff) & (X.ds <= cutoff + pd.Timedelta(weeks=h))][["unique_id", "ds"] + FUTR]
        else:
            model = NBEATS(alias="N-BEATS", **common)
            tr, fut = train, None
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=tr, val_size=12)
        fc = nf.predict(futr_df=fut) if fut is not None else nf.predict()
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        return _mq_to_frame(fc, model.alias)
    fn.__name__ = kind
    return fn


_CHRONOS = None


def chronos_model(model_id: str = "amazon/chronos-bolt-small", name: str = "Chronos-Bolt (zero-shot)"):
    """Zero-shot forecasts from a pretrained time-series foundation model. No training on our data."""
    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        global _CHRONOS
        import torch
        from chronos import BaseChronosPipeline
        if _CHRONOS is None:
            _CHRONOS = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        uids, ctx = [], []
        for uid, g in train.groupby("unique_id"):
            uids.append(uid)
            ctx.append(torch.tensor(g.sort_values("ds").y.to_numpy()[-208:], dtype=torch.float32))
        q, mean = _CHRONOS.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
        q = q.numpy()
        rows = []
        dates = pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=h, freq="W-MON")
        for i, uid in enumerate(uids):
            q10, q50, q90 = q[i, :, 0], q[i, :, 1], q[i, :, 2]
            rows.append(pd.DataFrame({"unique_id": uid, "ds": dates, "yhat": q50, "q0.1": q10, "q0.5": q50, "q0.9": q90,
                                      "q0.025": q50 - Z_RATIO * (q50 - q10), "q0.975": q50 + Z_RATIO * (q90 - q50)}))
        return pd.concat(rows, ignore_index=True)
    fn.__name__ = name
    return fn
