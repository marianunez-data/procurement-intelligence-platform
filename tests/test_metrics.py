import pytest
import numpy as np
import pandas as pd

from p2p.forecasting.metrics import bias, coverage, pinball, seasonal_naive_scale, wape


def test_wape_and_bias():
    y, yhat = np.array([10, 20, 30]), np.array([12, 18, 30])
    assert wape(y, yhat) == 4 / 60
    assert bias(y, yhat) == 0.0


def test_pinball_is_asymmetric():
    y = np.array([10.0])
    assert pinball(y, np.array([8.0]), 0.9) == pytest.approx(0.9 * 2)  # under-forecast penalised by q
    assert pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.1 * 2)  # over-forecast penalised by 1 - q


def test_coverage():
    assert coverage([1, 5, 9], [0, 0, 0], [6, 6, 6]) == 2 / 3


def test_seasonal_naive_scale_uses_season():
    ds = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    df = pd.DataFrame({"unique_id": "A", "ds": ds, "y": np.arange(60, dtype=float)})
    assert seasonal_naive_scale(df, season=52)["A"] == 52.0
