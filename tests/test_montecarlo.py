import numpy as np
import pandas as pd

from conftest import requires_warehouse
from p2p.forecasting.montecarlo import ProcessSimulator, Scenario, _sample, build_state


def test_conditional_sampling_exceeds_elapsed():
    rng = np.random.default_rng(0)
    d = np.sort(rng.exponential(30, 2000))
    elapsed = np.full(500, 45.0)
    s = _sample(rng, d, 500, elapsed)
    assert (s > 45.0).all()


@requires_warehouse
def test_faster_process_reduces_backlog():
    state = build_state(pd.Timestamp("2025-06-29"))
    arrivals = np.full((12, 300), 150)
    base = ProcessSimulator(state, n_sims=300, seed=1).run(12, arrivals, Scenario())
    fast = ProcessSimulator(state, n_sims=300, seed=1).run(12, arrivals, Scenario(approval_time=0.5, downpayment_time=0.5, buyer_speed=0.7))
    assert base["open_lines"].shape == (12, 300)
    assert np.median(fast["open_lines"][-1]) < np.median(base["open_lines"][-1])
