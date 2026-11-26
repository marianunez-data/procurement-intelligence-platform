"""Monte Carlo simulation of Quayline's procurement system.

Two simulators share this module:

1. DemandSimulator, a structural model of requisition inflow. Each demand driver is
   simulated separately (corrective maintenance, planned maintenance, MRP restock,
   projects) and the components are summed, so the forecast reflects how demand is created.

2. ProcessSimulator, a stage-by-stage simulation of every open requisition line and of
   every future line. For each line it samples the remaining stage durations from the
   empirical distributions of the last 12 months (conditioned on the time already spent in
   the current stage), so it forecasts backlog, receipts and cash out, and it answers
   what-if questions by scaling the stages a decision would change.

Both simulators only use information available at the origin date, so they are backtested
exactly like the other models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from p2p.data import query, weekly_section
from p2p.forecasting.features import calendar_features

STAGES = ["release", "rfq_wait", "quotation", "approval", "downpayment", "supplier", "gr_lag"]
EVENTS = ["pr_created_on", "pr_released_on", "rfq_created_on", "po_created_on", "po_approved_on", "prod_start", "delivered_on", "gr_posted_on"]


# ---------------------------------------------------------------------------
# 1. Demand
# ---------------------------------------------------------------------------
class DemandSimulator:
    """Component-based simulation of weekly requisition lines per section."""

    def __init__(self, history: pd.DataFrame, lookback_weeks: int = 26, level_sigma: float = 0.08):
        self.hist = history.sort_values(["section_id", "week_start"]).copy()
        self.lookback = lookback_weeks
        self.level_sigma = level_sigma  # uncertainty about the current demand level, shared within a simulation

    def _params(self, g: pd.DataFrame) -> dict:
        g = g.copy()
        cal = calendar_features(g.week_start)
        g["shutdown"], g["budget_push"], g["mrp_week"] = cal.shutdown.values, cal.budget_push.values, cal.mrp_week.values
        g["month"] = g.week_start.dt.month
        normal = g[g.shutdown == 0]
        # monthly seasonal index of corrective demand (multi-year average)
        cm_month = normal.groupby("month").cm_lines.mean()
        cm_index = (cm_month / normal.cm_lines.mean()).reindex(range(1, 13)).fillna(1.0)
        recent = g.tail(self.lookback)
        recent_n = recent[recent.shutdown == 0]
        deseason = (recent_n.cm_lines / recent_n.month.map(cm_index)).mean()
        cm_var = recent_n.cm_lines.var()
        shut_factor = (g[g.shutdown == 1].cm_lines.mean() / normal.cm_lines.mean()) if (g.shutdown == 1).any() else 0.5
        rs_shut = (g[g.shutdown == 1].restock_lines.mean() / max(normal.restock_lines.mean(), 1e-6)) if (g.shutdown == 1).any() else 0.5
        pm_ratio = g.pm_lines.sum() / max(g.pm_planned_next_3w.sum(), 1)
        rs_mrp = recent_n[recent_n.mrp_week == 1].restock_lines.mean()
        rs_off = recent_n[recent_n.mrp_week == 0].restock_lines.mean()
        push = normal[normal.budget_push == 1].restock_lines.mean() / max(normal[normal.budget_push == 0].restock_lines.mean(), 1e-6)
        proj = g.project_lines.to_numpy()
        p_burst = float(np.mean(proj > 0))
        sizes = proj[proj > 0] if (proj > 0).any() else np.array([0])
        return dict(cm_index=cm_index, cm_level=deseason, cm_var=cm_var, shut=np.nan_to_num(shut_factor, nan=0.5),
                    rs_shut=np.nan_to_num(rs_shut, nan=0.5), pm_ratio=pm_ratio, rs_mrp=np.nan_to_num(rs_mrp),
                    rs_off=np.nan_to_num(rs_off), push=np.nan_to_num(push, nan=1.0), p_burst=p_burst, sizes=sizes)

    def simulate(self, future: pd.DataFrame, n_sims: int = 2000, seed: int = 0, demand_change: float = 0.0) -> dict:
        """future: rows (section_id, week_start, pm_planned_next_3w). Returns {section: array (weeks, sims)}."""
        rng = np.random.default_rng(seed)
        level = rng.lognormal(0, self.level_sigma, n_sims)  # company-wide demand regime per simulation
        out = {}
        for sec, f in future.groupby("section_id"):
            p = self._params(self.hist[self.hist.section_id == sec])
            f = f.sort_values("week_start")
            cal = calendar_features(f.week_start)
            months = f.week_start.dt.month.to_numpy()
            cm_mu = p["cm_level"] * p["cm_index"].reindex(months).to_numpy() * np.where(cal.shutdown == 1, p["shut"], 1.0)
            cm_mu = np.maximum(cm_mu * (1 + demand_change), 0.05)
            # negative binomial with dispersion matched to recent variance
            k = max(cm_mu.mean() ** 2 / max(p["cm_var"] - cm_mu.mean(), 1e-3), 1.0)
            cm_mu2 = cm_mu[:, None] * level[None, :]
            cm = rng.negative_binomial(k, k / (k + cm_mu2))
            pm = rng.poisson((p["pm_ratio"] * f.pm_planned_next_3w.to_numpy() * (1 + demand_change))[:, None], size=(len(f), n_sims))
            rs_mu = np.where(cal.mrp_week == 1, p["rs_mrp"], p["rs_off"]) * np.where(cal.budget_push == 1, p["push"], 1.0)
            rs_mu = rs_mu * np.where(cal.shutdown == 1, p["rs_shut"], 1.0) * (1 + demand_change)
            rs = rng.poisson(np.maximum(rs_mu, 0)[:, None], size=(len(f), n_sims))
            burst = rng.random((len(f), n_sims)) < p["p_burst"]
            proj = np.where(burst, rng.choice(p["sizes"], size=(len(f), n_sims)), 0)
            out[sec] = cm + pm + rs + proj
        return out


def demand_mc_model(n_sims: int = 2000, name: str = "Monte Carlo (structural demand)"):
    """Backtest wrapper: forecasts requisition lines with the demand simulator."""
    ws = weekly_section()

    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        hist = ws[ws.week_start <= cutoff]
        fut = ws[(ws.week_start > cutoff) & (ws.week_start <= cutoff + pd.Timedelta(weeks=h))][["section_id", "week_start", "pm_planned_next_3w"]]
        sims = DemandSimulator(hist).simulate(fut, n_sims=n_sims, seed=int(cutoff.strftime("%Y%m%d")) % 2**31)
        rows = []
        for sec, arr in sims.items():
            dates = fut[fut.section_id == sec].sort_values("week_start").week_start.to_numpy()
            q = np.quantile(arr, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
            rows.append(pd.DataFrame({"unique_id": sec, "ds": dates, "yhat": arr.mean(axis=1), "q0.025": q[0],
                                      "q0.1": q[1], "q0.5": q[2], "q0.9": q[3], "q0.975": q[4]}))
        return pd.concat(rows, ignore_index=True)
    fn.__name__ = name
    return fn


# ---------------------------------------------------------------------------
# 2. Process
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """What-if levers. Multipliers scale stage durations; shares are absolute changes."""

    name: str = "Baseline"
    buyer_speed: float = 1.0          # RFQ wait and quotation time
    approval_time: float = 1.0        # PO approval time
    downpayment_time: float = 1.0     # downpayment processing time
    foreign_lead_time: float = 1.0    # supplier lead time, foreign
    local_lead_time: float = 1.0      # supplier lead time, local
    local_share_change: float = 0.0   # e.g. +0.10 moves 10 points of new demand to local suppliers
    demand_change: float = 0.0        # e.g. +0.15 = 15% more requisition lines
    extra_downpayment_days_foreign: float = 0.0  # e.g. an FX allocation delay

    def multipliers(self) -> dict:
        return {"rfq_wait": self.buyer_speed, "quotation": self.buyer_speed, "approval": self.approval_time,
                "downpayment": self.downpayment_time}


@dataclass
class ProcessState:
    """Everything the process simulator needs, reconstructed as of an origin date."""

    asof: pd.Timestamp
    open_lines: pd.DataFrame
    payables: pd.DataFrame
    durations: dict = field(default_factory=dict)
    p_local: float = 0.5
    p_dp: dict = field(default_factory=dict)
    dp_pct: dict = field(default_factory=dict)
    p_cancel: float = 0.03
    values: dict = field(default_factory=dict)
    pay_lag: dict = field(default_factory=dict)


_LINES: pd.DataFrame | None = None


def _all_lines() -> pd.DataFrame:
    global _LINES
    if _LINES is None:
        df = query("""
            select l.pr_line_key, l.section_id, l.supplier_origin, l.line_value_usd, l.is_critical, l.reason_code,
                   l.pr_created_on, l.pr_released_on, l.rfq_created_on, l.po_created_on, l.po_approved_on,
                   l.dp_required, l.dp_paid_on, l.delivered_on, l.gr_posted_on, l.final_paid_on, l.cancelled_on,
                   coalesce(s.downpayment_pct, 0) as dp_pct, l.po_number, l.supplier_id, l.purchasing_group_id
            from marts.fct_requisition_line l
            left join marts.dim_supplier s using (supplier_id)
        """)
        for c in [c for c in df.columns if c.endswith("_on")]:
            df[c] = pd.to_datetime(df[c])
        df["prod_start"] = df.dp_paid_on.where(df.dp_required.fillna(False), df.po_approved_on)
        _LINES = df
    return _LINES


def build_state(asof: pd.Timestamp, window_days: int = 365) -> ProcessState:
    """Reconstruct the pipeline as it looked at `asof` (no information after it is used)."""
    L = _all_lines().copy()
    L = L[L.pr_created_on <= asof]
    date_cols = EVENTS + ["dp_paid_on", "final_paid_on", "cancelled_on"]
    for c in date_cols:
        L.loc[L[c] > asof, c] = pd.NaT
    # supplier/origin is only known once the PO exists
    L.loc[L.po_created_on.isna(), "supplier_origin"] = "Not Sourced"
    L["closed_on"] = L.gr_posted_on.fillna(L.cancelled_on)
    open_l = L[L.closed_on.isna()].copy()

    # current stage = first missing event after creation
    stage = np.full(len(open_l), 0)
    last = open_l.pr_created_on.copy()
    for k, ev in enumerate(EVENTS[1:]):
        has = open_l[ev].notna().to_numpy()
        stage = np.where(has, k + 1, stage)
        last = open_l[ev].where(open_l[ev].notna(), last)
    open_l["stage"] = stage  # index into STAGES of the stage in progress
    open_l["elapsed"] = (asof - last).dt.days.to_numpy()
    # a non-downpayment line that is approved is already in production
    nodp = (open_l.stage == 4) & ~open_l.dp_required.fillna(False)
    open_l.loc[nodp, "stage"] = 5

    # empirical stage durations from events completed in the trailing window
    lo = asof - pd.Timedelta(days=window_days)
    done = L.copy()
    pairs = {"release": ("pr_created_on", "pr_released_on"), "rfq_wait": ("pr_released_on", "rfq_created_on"),
             "quotation": ("rfq_created_on", "po_created_on"), "approval": ("po_created_on", "po_approved_on"),
             "downpayment": ("po_approved_on", "dp_paid_on"), "supplier": ("prod_start", "delivered_on"),
             "gr_lag": ("delivered_on", "gr_posted_on")}
    durations = {}
    for st, (a, b) in pairs.items():
        m = done[a].notna() & done[b].notna() & (done[b] > lo)
        for origin in ["Local", "Foreign"]:
            sel = m & (done.supplier_origin == origin) if st not in ("release", "rfq_wait") else m
            d = (done.loc[sel, b] - done.loc[sel, a]).dt.days.to_numpy().astype(float)
            durations[(st, origin)] = np.sort(d[d >= 0]) if len(d) else np.array([1.0])

    recent_po = L[L.po_created_on.notna() & (L.po_created_on > lo) & L.supplier_origin.isin(["Local", "Foreign"])]
    p_local = float((recent_po.supplier_origin == "Local").mean())
    p_dp = {o: float(recent_po[recent_po.supplier_origin == o].dp_required.mean()) for o in ["Local", "Foreign"]}
    dp_pct = {o: float(recent_po[(recent_po.supplier_origin == o) & recent_po.dp_required].dp_pct.mean() or 0.4) for o in ["Local", "Foreign"]}
    created = L[L.pr_created_on > lo]
    p_cancel = float(created.cancelled_on.notna().mean())
    values = {o: recent_po[recent_po.supplier_origin == o].line_value_usd.to_numpy() for o in ["Local", "Foreign"]}
    # payment lag after goods receipt
    paid = L[L.gr_posted_on.notna() & L.final_paid_on.notna() & (L.final_paid_on > lo)]
    pay_lag = {o: np.sort((paid[paid.supplier_origin == o].final_paid_on - paid[paid.supplier_origin == o].gr_posted_on).dt.days.to_numpy().astype(float))
               for o in ["Local", "Foreign"]}
    payables = L[L.gr_posted_on.notna() & L.final_paid_on.isna() & L.cancelled_on.isna()].copy()
    payables["elapsed"] = (asof - payables.gr_posted_on).dt.days
    return ProcessState(asof=asof, open_lines=open_l, payables=payables, durations=durations, p_local=p_local,
                        p_dp=p_dp, dp_pct=dp_pct, p_cancel=p_cancel, values=values, pay_lag=pay_lag)


def _sample(rng, sorted_d: np.ndarray, n: int, elapsed: np.ndarray | None = None) -> np.ndarray:
    """Sample durations; with `elapsed`, sample from the distribution conditioned on d > elapsed."""
    m = len(sorted_d)
    if elapsed is None:
        return sorted_d[rng.integers(0, m, n)]
    idx = np.searchsorted(sorted_d, elapsed, side="right")
    u = rng.random(n)
    k = (idx + np.floor(u * (m - idx))).astype(int)
    tail = idx >= m  # already longer than anything observed: add an exponential tail
    out = sorted_d[np.minimum(k, m - 1)]
    out = np.where(tail, elapsed + rng.exponential(7.0, n), out)
    return out


class ProcessSimulator:
    """`regime_sigma` adds parameter uncertainty: every simulation draws one shared multiplier for
    internal (buyer, approval, finance) times and one for supplier times. Without it the lines are
    independent, their errors average out and the intervals are far too narrow (see notebook 10)."""

    def __init__(self, state: ProcessState, n_sims: int = 1000, seed: int = 0, regime_sigma: float = 0.20):
        self.s = state
        self.n = n_sims
        self.rng = np.random.default_rng(seed)
        self.f_internal = self.rng.lognormal(0, regime_sigma, n_sims)
        self.f_supplier = self.rng.lognormal(0, regime_sigma, n_sims)

    def _journey(self, origin: np.ndarray, start_stage: np.ndarray, elapsed: np.ndarray, dp_req: np.ndarray,
                 sc: Scenario, cancel_ok: np.ndarray, sim_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Days from asof (or from creation for new lines) to production start, delivery and goods receipt."""
        rng, s = self.rng, self.s
        n = len(origin)
        t = np.zeros(n)
        prod = np.full(n, np.nan)
        deliv = np.full(n, np.nan)
        mult = sc.multipliers()
        for k, st in enumerate(STAGES):
            active = start_stage <= k
            if st == "downpayment":
                active &= dp_req
            if not active.any():
                continue
            for o in ["Local", "Foreign"]:
                sel = active & (origin == o)
                if not sel.any():
                    continue
                cond = sel & (start_stage == k) & (elapsed > 0)
                d = np.empty(sel.sum())
                sub_cond = cond[sel]
                if sub_cond.any():
                    d[sub_cond] = _sample(rng, s.durations[(st, o)], sub_cond.sum(), elapsed[sel][sub_cond]) - elapsed[sel][sub_cond]
                if (~sub_cond).any():
                    d[~sub_cond] = _sample(rng, s.durations[(st, o)], (~sub_cond).sum())
                f = mult.get(st, 1.0)
                if st == "supplier":
                    f = sc.foreign_lead_time if o == "Foreign" else sc.local_lead_time
                regime = (self.f_supplier if st == "supplier" else self.f_internal)[sim_idx[sel]]
                d = np.maximum(d * f * regime, 0)
                if st == "downpayment" and o == "Foreign":
                    d = d + sc.extra_downpayment_days_foreign
                t[sel] += d
            if k == 3:  # after approval: production starts unless a downpayment is due
                prod = np.where(dp_req, prod, t)
            if k == 4:
                prod = np.where(dp_req, t, prod)
            if k == 5:
                deliv = t.copy()
        prod = np.where(start_stage >= 5, np.nan, prod)
        cancelled = cancel_ok & (rng.random(n) < self.s.p_cancel)
        return prod, deliv, t, cancelled

    def run(self, horizon_weeks: int = 12, new_lines_per_week: np.ndarray | None = None, scenario: Scenario | None = None) -> dict:
        """Simulate. `new_lines_per_week` has shape (horizon_weeks, n_sims) of future arrivals (company total)."""
        sc = scenario or Scenario()
        s, rng, N = self.s, self.rng, self.n
        ol = s.open_lines
        nl = len(ol)
        # ---- existing open lines, replicated across simulations
        rep = np.repeat(np.arange(nl), N)
        sim_e = np.tile(np.arange(N), nl)
        origin = ol.supplier_origin.to_numpy()[rep].astype(object)
        unsourced = origin == "Not Sourced"
        p_loc = np.clip(s.p_local + sc.local_share_change, 0, 1)
        origin[unsourced] = np.where(rng.random(unsourced.sum()) < p_loc, "Local", "Foreign")
        origin[(origin != "Local") & (origin != "Foreign")] = "Foreign"
        dp_known = ol.po_created_on.notna().to_numpy()[rep]
        dp_req = np.where(dp_known, ol.dp_required.fillna(False).to_numpy()[rep],
                          rng.random(len(rep)) < np.where(origin == "Local", s.p_dp["Local"], s.p_dp["Foreign"]))
        stage = ol.stage.to_numpy()[rep]
        elapsed = ol.elapsed.to_numpy()[rep].astype(float)
        prod, deliv, gr, canc = self._journey(origin, stage, elapsed, dp_req.astype(bool), sc, cancel_ok=stage <= 2, sim_idx=sim_e)
        value = ol.line_value_usd.to_numpy()[rep]
        dp_pct = np.where(ol.dp_pct.to_numpy()[rep] > 0, ol.dp_pct.to_numpy()[rep],
                          np.where(origin == "Local", s.dp_pct["Local"], s.dp_pct["Foreign"]))
        # ---- new lines
        if new_lines_per_week is not None:
            counts = new_lines_per_week.astype(int)
            w_idx, s_idx = np.nonzero(counts)
            reps = counts[w_idx, s_idx]
            nw = np.repeat(w_idx, reps)
            ns = np.repeat(s_idx, reps)
            m = len(nw)
            created = 7 * nw + rng.integers(0, 7, m) + 1.0  # days after asof
            o_new = np.where(rng.random(m) < p_loc, "Local", "Foreign").astype(object)
            dp_new = rng.random(m) < np.where(o_new == "Local", s.p_dp["Local"], s.p_dp["Foreign"])
            p2, d2, g2, c2 = self._journey(o_new, np.zeros(m, int), np.zeros(m), dp_new, sc, cancel_ok=np.ones(m, bool), sim_idx=ns)
            v_new = np.where(o_new == "Local", rng.choice(s.values["Local"], m), rng.choice(s.values["Foreign"], m))
            dpp_new = np.where(o_new == "Local", s.dp_pct["Local"], s.dp_pct["Foreign"])
        else:
            m = 0
            created = p2 = d2 = g2 = v_new = dpp_new = np.array([])
            c2 = dp_new = np.array([], bool)
            ns = np.array([], int)
            o_new = np.array([], object)

        H = horizon_weeks
        def week_of(day):  # day offset from asof -> forecast week index 0..H-1 (or -1 if outside)
            day = np.nan_to_num(day, nan=-1.0)
            w = (np.ceil(day / 7) - 1).astype(int)
            return np.where((day > 0) & (w < H), w, -1)

        open_count = np.zeros((H, N))
        receipts = np.zeros((H, N))
        cash = np.zeros((H, N))
        # backlog at each week end: existing lines close at gr (or cancellation ~ quotation end)
        close_e = np.where(canc, np.minimum(gr, 30.0), gr)
        close_n = created + np.where(c2, np.minimum(g2, 30.0), g2) if m else np.array([])
        for w in range(H):
            end = 7 * (w + 1)
            open_count[w] = np.bincount(sim_e[close_e > end], minlength=N)
            if m:
                alive = (created <= end) & (close_n > end)
                open_count[w] += np.bincount(ns[alive], minlength=N)
        # receipts
        we = week_of(gr)
        ok = (we >= 0) & ~canc
        np.add.at(receipts, (we[ok], sim_e[ok]), 1)
        if m:
            wn = week_of(created + g2)
            okn = (wn >= 0) & ~c2
            np.add.at(receipts, (wn[okn], ns[okn]), 1)
        # cash: downpayments at production start, final payment after receipt
        def pay(origin_arr, day_gr, n):
            out = np.empty(n)
            for o in ["Local", "Foreign"]:
                sel = origin_arr == o
                if sel.any():
                    out[sel] = _sample(rng, s.pay_lag[o], sel.sum())
            return day_gr + out
        wdp = week_of(prod)
        okdp = (wdp >= 0) & dp_req & ~canc
        np.add.at(cash, (wdp[okdp], sim_e[okdp]), (value * dp_pct)[okdp])
        wf = week_of(pay(origin, gr, len(gr)))
        okf = (wf >= 0) & ~canc
        np.add.at(cash, (wf[okf], sim_e[okf]), (value * (1 - dp_pct * dp_req))[okf])
        if m:
            wdp2 = week_of(created + p2)
            ok2 = (wdp2 >= 0) & dp_new & ~c2
            np.add.at(cash, (wdp2[ok2], ns[ok2]), (v_new * dpp_new)[ok2])
            wf2 = week_of(pay(o_new, created + g2, m))
            okf2 = (wf2 >= 0) & ~c2
            np.add.at(cash, (wf2[okf2], ns[okf2]), (v_new * (1 - dpp_new * dp_new))[okf2])
        # received but unpaid lines at asof
        pb = s.payables
        if len(pb):
            repb = np.repeat(np.arange(len(pb)), N)
            simb = np.tile(np.arange(N), len(pb))
            ob = pb.supplier_origin.to_numpy()[repb]
            ob = np.where(np.isin(ob, ["Local", "Foreign"]), ob, "Foreign")
            el = pb.elapsed.to_numpy()[repb].astype(float)
            lag = np.empty(len(repb))
            for o in ["Local", "Foreign"]:
                sel = ob == o
                if sel.any():
                    lag[sel] = _sample(rng, s.pay_lag[o], sel.sum(), el[sel]) - el[sel]
            wb = week_of(lag)
            vb = (pb.line_value_usd.to_numpy() * (1 - pb.dp_pct.to_numpy() * pb.dp_required.fillna(False).to_numpy()))[repb]
            okb = wb >= 0
            np.add.at(cash, (wb[okb], simb[okb]), vb[okb])
        # line-level arrival distribution for the open lines (days from asof to goods receipt)
        gr_matrix = np.where(canc, np.nan, gr).reshape(nl, N)
        return {"open_lines": open_count, "receipts": receipts, "cash_out": cash, "line_receipt_days": gr_matrix}


def process_mc_model(target: str, n_sims: int = 500, name: str | None = None, regime_sigma: float = 0.20):
    """Backtest wrapper for company-level backlog ('open_lines'), receipts or cash ('cash_out')."""
    ws = weekly_section()

    def fn(train: pd.DataFrame, h: int, cutoff: pd.Timestamp) -> pd.DataFrame:
        asof = cutoff + pd.Timedelta(days=6)
        state = build_state(asof)
        hist = ws[ws.week_start <= cutoff]
        fut = ws[(ws.week_start > cutoff) & (ws.week_start <= cutoff + pd.Timedelta(weeks=h))][["section_id", "week_start", "pm_planned_next_3w"]]
        dsim = DemandSimulator(hist).simulate(fut, n_sims=n_sims, seed=1)
        arrivals = sum(dsim.values())  # (h, n_sims) company total
        res = ProcessSimulator(state, n_sims=n_sims, seed=2, regime_sigma=regime_sigma).run(h, arrivals)[target]
        dates = pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=h, freq="W-MON")
        q = np.quantile(res, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
        return pd.DataFrame({"unique_id": "Company", "ds": dates, "yhat": res.mean(axis=1), "q0.025": q[0],
                             "q0.1": q[1], "q0.5": q[2], "q0.9": q[3], "q0.975": q[4]})
    fn.__name__ = name or f"Monte Carlo (process, {target})"
    return fn
