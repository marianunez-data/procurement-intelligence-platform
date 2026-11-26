"""Machine-learning forecasters with a direct multi-horizon strategy.

For each horizon h (1..12) a separate model learns  y(t+h) = f(features at t, calendar at t+h).
Features come from the SQL feature store `marts.ml_features_weekly`, so the warehouse is the
single source of feature logic. Prediction intervals use split-conformal calibration on the
most recent 26 weeks of the training window, with residuals scaled by sqrt(level) because
count data is heteroscedastic.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from p2p.data import query
from p2p.forecasting.features import calendar_features, known_future_exog

warnings.filterwarnings("ignore", category=UserWarning)

SQL_FEATURES = [
    "y_t", "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_13", "lag_26", "lag_51",
    "roll_mean_4", "roll_mean_13", "roll_mean_52", "roll_std_13", "roll_max_13",
    "cm_roll_mean_4", "project_roll_mean_13", "cm_wo_roll_mean_4", "fleet_utilisation",
    "fleet_utilisation_chg_4", "fx_chg_13w",
]
TARGET_FEATURES = ["pm_planned_next_3w", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "shutdown", "budget_push", "mrp_week"]
CATEGORICAL = ["section_code", "bu_code"]
FEATURES = SQL_FEATURES + [f"target_{c}" for c in TARGET_FEATURES] + CATEGORICAL
QS = (0.025, 0.1, 0.5, 0.9, 0.975)
CALIBRATION_WEEKS = 26

_STORE: pd.DataFrame | None = None


def feature_store() -> pd.DataFrame:
    """Load the SQL feature store and attach known-future covariates of the target week."""
    global _STORE
    if _STORE is None:
        fs = query("select * from marts.ml_features_weekly order by section_id, origin_week")
        fs["origin_week"] = pd.to_datetime(fs.origin_week)
        fs["section_code"] = fs.section_id.str[3:].astype(int)
        fs["bu_code"] = fs.business_unit.astype("category").cat.codes
        _STORE = fs
    return _STORE


_DM: dict[int, pd.DataFrame] = {}


def design_matrix(h: int) -> pd.DataFrame:
    """Rows indexed by origin week with features and the target y(t+h)."""
    if h in _DM:
        return _DM[h]
    fs = feature_store().copy()
    fs["target_week"] = fs.origin_week + pd.Timedelta(weeks=h)
    X = known_future_exog().rename(columns={"unique_id": "section_id", "ds": "target_week"})
    X = X.rename(columns={c: f"target_{c}" for c in TARGET_FEATURES})
    fs = fs.merge(X, on=["section_id", "target_week"], how="left")
    # calendar of target weeks beyond the data range (still known in advance)
    miss = fs["target_sin1"].isna()
    if miss.any():
        cal = calendar_features(fs.loc[miss, "target_week"])
        for c in TARGET_FEATURES:
            if c in cal:
                fs.loc[miss, f"target_{c}"] = cal[c].to_numpy()
    fs["y"] = fs[f"y_lead_{h}"]
    _DM[h] = fs
    return fs


def _make(kind: str, params: dict | None = None):
    params = params or {}
    if kind == "LightGBM":
        base = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    objective="poisson", verbose=-1, random_state=42, n_jobs=1)
        return lgb.LGBMRegressor(**{**base, **params})
    if kind == "XGBoost":
        base = dict(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                    objective="count:poisson", random_state=42, n_jobs=1)
        return XGBRegressor(**{**base, **params})
    if kind == "Random Forest":
        base = dict(n_estimators=150, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=2)
        return RandomForestRegressor(**{**base, **params})
    raise ValueError(kind)


def fit_predict_horizon(kind: str, h: int, cutoff: pd.Timestamp, params: dict | None = None):
    """Fit the horizon-h model with data known at `cutoff` and forecast the week cutoff + h."""
    dm = design_matrix(h)
    known = dm[(dm.origin_week + pd.Timedelta(weeks=h) <= cutoff) & dm.y.notna() & dm.lag_13.notna()]
    calib_start = cutoff - pd.Timedelta(weeks=CALIBRATION_WEEKS)
    fit_part = known[known.origin_week + pd.Timedelta(weeks=h) <= calib_start]
    cal_part = known[known.origin_week + pd.Timedelta(weeks=h) > calib_start]
    # 1) calibration fit -> scaled residual quantiles
    m_cal = _make(kind, params).fit(fit_part[FEATURES], fit_part.y)
    p_cal = np.clip(m_cal.predict(cal_part[FEATURES]), 0, None)
    scaled = (cal_part.y.to_numpy() - p_cal) / np.sqrt(p_cal + 1)
    res_q = np.quantile(scaled, QS)
    # 2) final fit on everything known
    model = _make(kind, params).fit(known[FEATURES], known.y)
    origin_rows = dm[dm.origin_week == cutoff]
    p = np.clip(model.predict(origin_rows[FEATURES]), 0, None)
    out = pd.DataFrame({"unique_id": origin_rows.section_id.values,
                        "ds": (origin_rows.origin_week + pd.Timedelta(weeks=h)).values, "yhat": p})
    for q, r in zip(QS, res_q):
        out[f"q{q}"] = p + r * np.sqrt(p + 1)
    return out, model


def ml_model(kind: str, params: dict | None = None, name: str | None = None):
    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        return pd.concat([fit_predict_horizon(kind, k, cutoff, params)[0] for k in range(1, h + 1)], ignore_index=True)
    fn.__name__ = name or kind
    return fn


def ml_models() -> dict:
    return {k: ml_model(k) for k in ["LightGBM", "XGBoost", "Random Forest"]}
