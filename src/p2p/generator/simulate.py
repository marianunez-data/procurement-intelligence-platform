"""Event-level simulation of Quayline's procure-to-pay process (2022 to 2025).

The simulation is built in layers so every pattern in the data has a known cause:

1. Leading indicators: fleet utilisation, FX rate, corrective and planned work orders.
2. Demand: purchase-requisition (PR) lines from four drivers (CM, PM, Restock, Project).
3. Lifecycle: every line moves through release, RFQ, PO, approval, downpayment,
   supplier production, delivery, goods receipt and payment, with stage durations that
   react to workload, policy changes and shocks.
4. Snapshot: anything that happens after the extraction date is unknown (open lines).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2p.config import CFG, GeneratorConfig
from p2p.generator import master_data as md


def _ln(rng: np.random.Generator, median, sigma, size=None):
    """Lognormal sample parameterised by its median (more readable than mu)."""
    return np.asarray(median) * rng.lognormal(0.0, sigma, size)


def _weeks(cfg: GeneratorConfig) -> pd.DatetimeIndex:
    return pd.date_range(cfg.start, cfg.snapshot, freq="W-MON")


def _in(dates: pd.Series | pd.DatetimeIndex, start: str, end: str):
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def rainy_season(dates) -> np.ndarray:
    """0..1 bump that peaks in mid-July (West African rainy season)."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    return np.exp(-(((doy - 198) / 50.0) ** 2))


def shutdown_factor(dates) -> np.ndarray:
    """Year-end shutdown: activity drops between 20 Dec and 2 Jan."""
    d = pd.DatetimeIndex(dates)
    f = np.ones(len(d))
    late_dec = (d.month == 12) & (d.day >= 18)
    early_jan = (d.month == 1) & (d.day <= 6)
    f[late_dec] = 0.45
    f[early_jan] = 0.75
    return f


# ---------------------------------------------------------------------------
# 1. Leading indicators
# ---------------------------------------------------------------------------
def simulate_fx(rng, cfg: GeneratorConfig) -> pd.DataFrame:
    days = pd.date_range(cfg.start, cfg.snapshot, freq="D")
    crisis = cfg.shock("FX liquidity crisis")
    c0 = pd.Timestamp(crisis.start)
    drift = np.full(len(days), 0.0004)
    vol = np.full(len(days), 0.0035)
    in_crisis = _in(days, crisis.start, crisis.end)
    devaluation_days = (days >= c0) & (days < c0 + pd.Timedelta(days=84))
    drift[devaluation_days] = np.log(1.55) / 84  # 55% devaluation over 12 weeks
    vol[in_crisis] = 0.009
    steps = drift + vol * rng.standard_normal(len(days))
    rate = 760 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"rate_date": days, "lcu_per_usd": np.round(rate, 2)})


def simulate_operations(rng, cfg: GeneratorConfig) -> pd.DataFrame:
    weeks = _weeks(cfg)
    rows = []
    phase = {"Port Operations": 0.4, "Fleet and Workshop": 0.0, "Power Generation": 1.2, "Marine Services": 2.0}
    for bu, ph in phase.items():
        noise = np.zeros(len(weeks))
        for i in range(1, len(weeks)):
            noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 0.018)
        doy = weeks.dayofyear.to_numpy()
        util = 0.72 + 0.05 * np.sin(2 * np.pi * doy / 365.25 + ph) + noise
        if bu == "Marine Services":
            util = util + 0.07 * (weeks >= pd.Timestamp(cfg.shock("New marine contract").start))
        util = util * shutdown_factor(weeks) ** 0.3
        rows.append(pd.DataFrame({"week_start": weeks, "business_unit": bu, "fleet_utilisation": np.clip(util, 0.3, 0.98).round(4)}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Demand: work orders and PR lines
# ---------------------------------------------------------------------------
def simulate_demand(rng, cfg, materials, equipment, operations):
    weeks = _weeks(cfg)
    t_years = ((weeks - cfg.start).days / 365.25).to_numpy()
    rain = rainy_season(weeks)
    shut = shutdown_factor(weeks)
    q4 = np.where(weeks.month.isin([10, 11]), 1.30, 1.0)
    marine_step = np.where(weeks >= pd.Timestamp(cfg.shock("New marine contract").start), 1.25, 1.0)
    util = operations.pivot(index="week_start", columns="business_unit", values="fleet_utilisation").reindex(weeks)

    mats_by_sec = {s: g for s, g in materials.groupby("section_id")}
    eq_by_sec = {s: g for s, g in equipment.groupby("section_id")}

    wo_rows, line_rows = [], []
    wo_seq, pr_seq = 0, 0
    week_idx = np.arange(len(weeks))

    for _, s in md.SECTIONS.iterrows():
        trend = (1 + cfg.annual_growth + s.growth) ** t_years
        bu_mult = marine_step if s.business_unit == "Marine Services" else 1.0
        u = (util[s.business_unit].to_numpy() / 0.72) ** 1.5
        mats = mats_by_sec[s.section_id]
        p_mat = (mats.popularity / mats.popularity.sum()).to_numpy()
        eqs = eq_by_sec[s.section_id]

        # Corrective maintenance: random breakdowns, more in the rainy season and at high utilisation
        cm_mean = s.base_lines * 0.45 / 1.4 * trend * (1 + 1.6 * s.rain_amp * rain) * u * shut * bu_mult
        cm_wo = rng.negative_binomial(20, 20 / (20 + cm_mean))
        # Planned maintenance: a fixed schedule (monthly services plus quarterly majors) known in advance
        pm_pattern = 1 + 0.6 * (week_idx % 4 == 0) + 1.0 * (week_idx % 13 == 0)
        pm_mean = s.base_lines * 0.25 / 1.25 / 1.33 * trend * pm_pattern * bu_mult
        pm_wo = np.maximum(0, np.round(pm_mean + rng.normal(0, 0.3, len(weeks)))).astype(int)
        # Restock: MRP run every second week, bigger in the Q4 budget push
        mrp_week = (week_idx % 2 == 0)
        rs_mean = s.base_lines * 0.22 * trend * q4 * shut * np.where(mrp_week, 1.6, 0.4) * bu_mult
        rs_lines = rng.poisson(rs_mean)
        # Projects: lumpy bursts
        proj_burst = rng.random(len(weeks)) < 0.03 * np.where(weeks.month.isin([10, 11]), 1.6, 1.0)

        for w, wk in enumerate(weeks):
            # corrective work orders -> PR lines raised 0-10 days after the breakdown
            for _ in range(cm_wo[w]):
                eq = eqs.iloc[rng.integers(len(eqs))]
                created = wk + pd.Timedelta(days=int(rng.integers(0, 6)))
                wo_seq += 1
                wo_id = f"WO{4000000 + wo_seq}"
                wo_rows.append(dict(work_order_id=wo_id, section_id=s.section_id, equipment_id=eq.equipment_id,
                                    wo_type="CM", created_on=created, planned_date=created, criticality=eq.criticality))
                n = 1 + rng.poisson(0.4)
                pr_seq += 1
                pr_date = created + pd.Timedelta(days=int(min(rng.exponential(1.5), 10)))
                crit = eq.criticality == "A"
                for k in range(n):
                    line_rows.append((pr_seq, k + 1, pr_date, s.section_id, "CM", wo_id, eq.equipment_id, crit,
                                      rng.choice(len(mats), p=p_mat)))
            # planned maintenance: work order planned for this week, parts requested 14-21 days earlier
            for _ in range(pm_wo[w]):
                eq = eqs.iloc[rng.integers(len(eqs))]
                planned = wk + pd.Timedelta(days=int(rng.integers(0, 5)))
                created = planned - pd.Timedelta(days=int(rng.integers(21, 29)))
                wo_seq += 1
                wo_id = f"WO{4000000 + wo_seq}"
                wo_rows.append(dict(work_order_id=wo_id, section_id=s.section_id, equipment_id=eq.equipment_id,
                                    wo_type="PM", created_on=created, planned_date=planned, criticality=eq.criticality))
                n = rng.poisson(1.25)
                if n == 0:
                    continue
                pr_seq += 1
                pr_date = planned - pd.Timedelta(days=int(rng.integers(14, 22)))
                for k in range(n):
                    line_rows.append((pr_seq, k + 1, pr_date, s.section_id, "PM", wo_id, eq.equipment_id, False,
                                      rng.choice(len(mats), p=p_mat)))
            # restock PRs: 3-8 lines each, raised on the MRP day (Tuesday) or ad hoc
            remaining = rs_lines[w]
            while remaining > 0:
                n = int(min(remaining, rng.integers(3, 9)))
                remaining -= n
                pr_seq += 1
                day = 1 if mrp_week[w] else int(rng.integers(0, 5))
                pr_date = wk + pd.Timedelta(days=day)
                for k in range(n):
                    line_rows.append((pr_seq, k + 1, pr_date, s.section_id, "RESTOCK", None, None, False,
                                      rng.choice(len(mats), p=p_mat)))
            if proj_burst[w]:
                n = 12 + rng.poisson(22)
                pr_seq += 1
                pr_date = wk + pd.Timedelta(days=int(rng.integers(0, 5)))
                for k in range(n):
                    line_rows.append((pr_seq, k + 1, pr_date, s.section_id, "PROJECT", None, None, False,
                                      rng.integers(len(mats))))

    lines = pd.DataFrame(line_rows, columns=["pr_seq", "pr_item", "pr_created_on", "section_id", "reason_code",
                                             "work_order_id", "equipment_id", "is_critical", "mat_pos"])
    # resolve material ids
    lines["material_id"] = [mats_by_sec[s].iloc[p].material_id for s, p in zip(lines.section_id, lines.mat_pos)]
    lines = lines.drop(columns="mat_pos")
    lines = lines[(lines.pr_created_on >= cfg.start) & (lines.pr_created_on <= cfg.snapshot)].copy()
    wos = pd.DataFrame(wo_rows)
    wos = wos[wos.created_on >= cfg.start].reset_index(drop=True)
    return wos, lines


# ---------------------------------------------------------------------------
# 3. Lifecycle
# ---------------------------------------------------------------------------
def assign_preferred_suppliers(rng, materials, suppliers):
    sec = md.SECTIONS.set_index("section_id")
    focus_for_bu = {"Marine Services": "Marine", "Power Generation": "Power"}
    pref = []
    for _, m in materials.iterrows():
        bu = sec.loc[m.section_id, "business_unit"]
        p_foreign = sec.loc[m.section_id, "foreign_share"]
        if m.category in ("Consumables", "Safety and PPE"):
            p_foreign *= 0.35
        origin = "Foreign" if rng.random() < p_foreign else "Local"
        pool = suppliers[suppliers.supplier_origin == origin]
        focus = focus_for_bu.get(bu)
        if focus and rng.random() < 0.7:
            pool = pool[pool.focus.isin([focus, "General"])]
        else:
            pool = pool[pool.focus == "General"] if (pool.focus == "General").any() else pool
        weights = np.linspace(2.0, 0.5, len(pool))  # some suppliers are more popular
        pref.append(pool.supplier_id.to_numpy()[rng.choice(len(pool), p=weights / weights.sum())])
    return materials.assign(preferred_supplier_id=pref)


def simulate_lifecycle(rng, cfg, lines, materials, suppliers, fx):
    sup = suppliers.set_index("supplier_id")
    mat = materials.set_index("material_id")
    lines = lines.merge(materials[["material_id", "category", "standard_price_usd", "qty_low", "qty_high",
                                   "lead_time_factor", "preferred_supplier_id", "uom", "short_text"]],
                        on="material_id", how="left")
    n = len(lines)
    snapshot = cfg.snapshot
    policy = pd.Timestamp(cfg.policy_change_date)
    fx_shock = cfg.shock("FX liquidity crisis")
    congestion = cfg.shock("Port congestion")
    suspension = cfg.shock("Key supplier suspension")

    # --- supplier choice: preferred, occasional alternative, local substitution after the FX crisis
    origin_pref = sup.loc[lines.preferred_supplier_id, "supplier_origin"].to_numpy()
    supplier_id = lines.preferred_supplier_id.to_numpy().copy()
    alt = rng.random(n) < 0.12
    local_ids = sup.index[sup.supplier_origin == "Local"].to_numpy()
    foreign_ids = sup.index[sup.supplier_origin == "Foreign"].to_numpy()
    for i in np.where(alt)[0]:
        supplier_id[i] = rng.choice(foreign_ids if origin_pref[i] == "Foreign" else local_ids)
    substitution = (origin_pref == "Foreign") & (lines.pr_created_on >= pd.Timestamp("2024-01-01")).to_numpy() & (rng.random(n) < 0.10)
    supplier_id[substitution] = rng.choice(local_ids, substitution.sum())
    lines["supplier_id"] = supplier_id
    lines["supplier_origin"] = sup.loc[supplier_id, "supplier_origin"].to_numpy()
    foreign = (lines.supplier_origin == "Foreign").to_numpy()

    # --- quantities and prices
    qty = rng.integers(lines.qty_low, lines.qty_high + 1)
    qty = np.where(lines.reason_code == "RESTOCK", qty * 2, qty)
    t_years = ((lines.pr_created_on - cfg.start).dt.days / 365.25).to_numpy()
    inflation = 1.04 ** t_years
    fx_tbl = fx.set_index("rate_date").lcu_per_usd
    fx_at = fx_tbl.reindex(lines.pr_created_on).to_numpy()
    local_fx_effect = np.where(foreign, 1.0, (fx_tbl.iloc[0] / fx_at) ** 0.35)
    unit_price = lines.standard_price_usd.to_numpy() * inflation * local_fx_effect * rng.lognormal(0, 0.08, n)
    lines["quantity"] = qty
    lines["estimated_price_usd"] = np.round(lines.standard_price_usd.to_numpy() * inflation, 2)
    lines["po_unit_price_usd"] = np.round(unit_price, 2)
    lines["line_value_usd"] = np.round(unit_price * qty, 2)

    # --- PR numbering in date order
    pr_order = lines.groupby("pr_seq").pr_created_on.min().sort_values()
    pr_number = pd.Series(np.arange(len(pr_order)) + 1000100001, index=pr_order.index)
    lines["pr_number"] = lines.pr_seq.map(pr_number)

    # --- workload index: busy weeks slow the buyers down (queueing effect)
    wk = lines.pr_created_on.dt.to_period("W-SUN").dt.start_time
    weekly = wk.value_counts().sort_index()
    expected = weekly.rolling(26, min_periods=4).median().shift(1).bfill()
    load = (weekly.rolling(4, min_periods=1).mean() / expected).clip(0.6, 2.2)
    load_line = wk.map(load).to_numpy()

    after_policy = (lines.pr_created_on >= policy).to_numpy()
    crit = lines.is_critical.to_numpy()
    project = (lines.reason_code == "PROJECT").to_numpy()

    def add_days(base, days):
        return base + pd.to_timedelta(np.round(days).astype(int), unit="D")

    # PR release (HOD)
    # release is a PR-level decision: every line of a PR is released together
    pr_codes, pr_idx = np.unique(lines.pr_number.to_numpy(), return_inverse=True)
    rel_pr = _ln(rng, 1.0, 0.6, len(pr_codes))
    rel = rel_pr[pr_idx] * np.where(after_policy, 1.0, 2.0)
    pr_released = add_days(lines.pr_created_on, rel)
    # Year-end shutdown pushes internal steps into January
    def apply_shutdown(dates: pd.Series) -> pd.Series:
        d = dates.copy()
        m = ((d.dt.month == 12) & (d.dt.day >= 20)) | ((d.dt.month == 1) & (d.dt.day <= 2))
        if m.any():
            nxt_year = np.where(d[m].dt.month == 12, d[m].dt.year + 1, d[m].dt.year)
            jan3 = pd.to_datetime(pd.DataFrame({"year": nxt_year, "month": 1, "day": 3}))
            d[m] = jan3.values  # work resumes on 3 January
        return d

    pr_released = apply_shutdown(pd.Series(pr_released))

    # PR-supplier groups become one RFQ / PO
    lines["group_key"] = lines.pr_number.astype(str) + "-" + lines.supplier_id
    g = lines.groupby("group_key")
    gkeys = g.ngroup().to_numpy()
    n_g = gkeys.max() + 1
    g_first = lines.assign(pr_released=pr_released.values).groupby(gkeys).agg(
        pr_released=("pr_released", "min"), foreign=("supplier_origin", lambda x: x.iloc[0] == "Foreign"),
        crit=("is_critical", "max"), project=("reason_code", lambda x: (x == "PROJECT").any()),
        value=("line_value_usd", "sum"), load=("pr_created_on", "size"))
    g_first["load"] = pd.Series(load_line).groupby(gkeys).mean().to_numpy()
    gf = g_first["foreign"].to_numpy()
    gc = g_first["crit"].to_numpy()
    gp = g_first["project"].to_numpy()
    g_after = (g_first.pr_released >= policy).to_numpy()

    # Not Edited -> RFQ (buyer queue)
    q = _ln(rng, np.where(gf, 6.0, 4.0) * g_first["load"].to_numpy() ** 0.9 * np.where(gc, 0.5, 1.0) * np.where(gp, 1.5, 1.0), 0.7, n_g)
    rfq_on = apply_shutdown(pd.Series(add_days(g_first.pr_released, q)))
    rfq_on = pd.Series(np.maximum(rfq_on.values, g_first.pr_released.values))
    # RFQ -> PO (quotation)
    qt = _ln(rng, np.where(gf, 12.0, 7.0) * np.where(gc, 0.6, 1.0), 0.5, n_g)
    po_on = apply_shutdown(pd.Series(add_days(rfq_on, qt)))
    po_on = pd.Series(np.maximum(po_on.values, rfq_on.values))
    cancelled_rfq = rng.random(n_g) < 0.03

    # Approval: MD above the threshold (threshold raised by the new policy)
    threshold = np.where(g_after, cfg.md_threshold_after, cfg.md_threshold_before)
    needs_md = g_first["value"].to_numpy() > threshold
    appr_med = np.where(needs_md, np.where(g_after, 2.5, 6.0), 1.0)
    appr = _ln(rng, appr_med, np.where(needs_md, 0.7, 0.5), n_g)
    submitted = po_on.copy()
    approved = pd.Series(add_days(submitted, appr))
    reject_p = np.where(needs_md, np.where(g_after, 0.03, 0.06), 0.01)
    rejected = rng.random(n_g) < reject_p
    cancelled_after_reject = rejected & (rng.random(n_g) < 0.3)
    resubmitted = pd.Series(add_days(approved, _ln(rng, 8.0, 0.5, n_g)))
    approved2 = pd.Series(add_days(resubmitted, _ln(rng, appr_med, 0.6, n_g)))
    final_approved = approved.where(~rejected, approved2)
    final_approved = apply_shutdown(final_approved)
    final_approved = pd.Series(np.maximum(final_approved.values, po_on.values))

    # Downpayment (foreign downpayments wait for FX allocation during the crisis)
    g_sup = lines.groupby(gkeys).supplier_id.first().to_numpy()
    dp_req = sup.loc[g_sup, "downpayment_required"].to_numpy()
    in_fx = _in(final_approved, fx_shock.start, fx_shock.end).to_numpy()
    dp_days = _ln(rng, 8.0, 0.6, n_g) + np.where(in_fx & gf, _ln(rng, 25.0, 0.5, n_g), 0) + np.where(in_fx & ~gf, _ln(rng, 4.0, 0.5, n_g), 0)
    dp_paid = pd.Series(add_days(final_approved, dp_days)).where(dp_req)
    prod_start = dp_paid.where(dp_req, final_approved)

    groups = pd.DataFrame({
        "gkey": np.arange(n_g), "rfq_created_on": rfq_on.values, "po_created_on": po_on.values,
        "approval_submitted_on": submitted.values, "first_decision_on": approved.values,
        "rejected": rejected, "resubmitted_on": resubmitted.where(rejected).values,
        "po_approved_on": final_approved.values, "needs_md": needs_md, "dp_required": dp_req,
        "dp_paid_on": dp_paid.values, "prod_start": prod_start.values,
        "cancelled_rfq": cancelled_rfq, "cancelled_after_reject": cancelled_after_reject,
        "po_value_usd": g_first["value"].to_numpy(), "supplier_id": g_sup,
    })

    lines["gkey"] = gkeys
    lines["pr_released_on"] = pr_released.values
    lines = lines.merge(groups.drop(columns="supplier_id"), on="gkey", how="left")

    # Supplier production and delivery
    sup_lt = sup.loc[lines.supplier_id, "quoted_lead_time_days"].to_numpy() * lines.lead_time_factor.to_numpy()
    quoted = np.round(sup_lt).astype(int)
    rel_s = sup.loc[lines.supplier_id, "reliability"].to_numpy()
    mu = (1 - rel_s) * 0.6 - 0.05
    actual_lt = quoted * np.exp(mu + rng.normal(0, 0.25, len(lines)))
    lines["quoted_lead_time_days"] = quoted
    lines["promised_date"] = lines.po_approved_on + pd.to_timedelta(quoted, "D")
    delivery = lines.prod_start + pd.to_timedelta(np.round(actual_lt).astype(int), "D")

    # Port congestion: foreign deliveries landing in the window are held until it clears
    fgn = (lines.supplier_origin == "Foreign").to_numpy()
    held = fgn & _in(delivery, congestion.start, congestion.end).to_numpy()
    release = pd.Timestamp(congestion.end) + pd.to_timedelta(np.round(rng.exponential(8, held.sum())).astype(int), "D")
    delivery = delivery.copy()
    delivery[held] = release.values

    # Supplier suspension: the largest foreign marine supplier stops shipping for eight weeks
    marine_secs = md.SECTIONS[md.SECTIONS.business_unit == "Marine Services"].section_id
    marine_sup = (lines[lines.section_id.isin(marine_secs) & fgn].supplier_id.value_counts().index[0])
    s0, s1 = pd.Timestamp(suspension.start), pd.Timestamp(suspension.end)
    hit = (lines.supplier_id == marine_sup).to_numpy() & (delivery >= s0).to_numpy() & (lines.prod_start <= s1).to_numpy()
    delivery[hit] = delivery[hit] + pd.Timedelta(days=56)

    # Inspection rejection and redelivery
    rej_p = np.where(fgn, 0.03, 0.02)
    insp_rej = rng.random(len(lines)) < rej_p
    redelivery = delivery + pd.to_timedelta(np.round(_ln(rng, 30, 0.4, len(lines))).astype(int), "D")
    # Partial deliveries
    partial = rng.random(len(lines)) < 0.06
    second = delivery + pd.to_timedelta(np.round(_ln(rng, 21, 0.5, len(lines))).astype(int), "D")
    final_delivery = delivery.where(~insp_rej, redelivery)
    final_delivery = final_delivery.where(~partial, np.maximum(final_delivery, second))
    # Goods receipt posting (3% posted weeks late: "received but not posted")
    gr_lag = _ln(rng, 1.5, 0.5, len(lines))
    late_post = rng.random(len(lines)) < 0.03
    gr_lag = gr_lag + np.where(late_post, rng.uniform(20, 60, len(lines)), 0)
    gr_posted = final_delivery + pd.to_timedelta(np.round(gr_lag).astype(int), "D")

    # Final payment after goods receipt
    terms = sup.loc[lines.supplier_id, "payment_terms_days"].to_numpy()
    pay_fx = _in(gr_posted, fx_shock.start, fx_shock.end).to_numpy() & fgn
    pay_delay = _ln(rng, 5, 0.8, len(lines)) + np.where(pay_fx, _ln(rng, 15, 0.5, len(lines)), 0)
    final_paid = gr_posted + pd.to_timedelta(np.round(terms + pay_delay).astype(int), "D")

    lines["first_delivery_on"] = delivery
    lines["inspection_rejected"] = insp_rej
    lines["partial_delivery"] = partial
    lines["delivered_on"] = final_delivery
    lines["late_gr_posting"] = late_post
    lines["gr_posted_on"] = gr_posted
    lines["final_paid_on"] = final_paid
    lines["suspended_supplier"] = lines.supplier_id == marine_sup

    # Cancellations stop the lifecycle
    cancel_rfq = lines.cancelled_rfq.to_numpy()
    cancel_rej = lines.cancelled_after_reject.to_numpy()
    for col in ["po_created_on", "approval_submitted_on", "first_decision_on", "resubmitted_on", "po_approved_on",
                "dp_paid_on", "prod_start", "promised_date", "first_delivery_on", "delivered_on", "gr_posted_on", "final_paid_on"]:
        lines.loc[cancel_rfq, col] = pd.NaT
    for col in ["resubmitted_on", "po_approved_on", "dp_paid_on", "prod_start", "promised_date",
                "first_delivery_on", "delivered_on", "gr_posted_on", "final_paid_on"]:
        lines.loc[cancel_rej, col] = pd.NaT
    lines["cancelled_on"] = pd.NaT
    lines.loc[cancel_rfq, "cancelled_on"] = lines.loc[cancel_rfq, "rfq_created_on"] + pd.Timedelta(days=14)
    lines.loc[cancel_rej, "cancelled_on"] = lines.loc[cancel_rej, "first_decision_on"] + pd.Timedelta(days=3)

    # 4. Snapshot: the future is unknown
    date_cols = ["pr_released_on", "rfq_created_on", "po_created_on", "approval_submitted_on", "first_decision_on",
                 "resubmitted_on", "po_approved_on", "dp_paid_on", "prod_start", "first_delivery_on", "delivered_on",
                 "gr_posted_on", "final_paid_on", "cancelled_on"]
    truth = lines.copy()  # the full, uncensored lifecycle (ground truth, never shown to models)
    for col in date_cols:
        lines.loc[lines[col] > snapshot, col] = pd.NaT
    # an event cannot be known if the previous one is unknown
    chain = ["pr_released_on", "rfq_created_on", "po_created_on", "po_approved_on", "delivered_on", "gr_posted_on", "final_paid_on"]
    for a, b in zip(chain, chain[1:]):
        lines.loc[lines[a].isna(), b] = pd.NaT
    lines.loc[lines.po_created_on.isna(), ["approval_submitted_on", "first_decision_on", "resubmitted_on", "dp_paid_on", "promised_date"]] = pd.NaT
    lines.loc[lines.po_approved_on.isna(), ["dp_paid_on"]] = pd.NaT
    lines.loc[lines.first_decision_on.isna(), "resubmitted_on"] = pd.NaT
    lines.loc[lines.cancelled_on > snapshot, "cancelled_on"] = pd.NaT
    lines.loc[lines.delivered_on.isna(), ["inspection_rejected", "partial_delivery"]] = False
    return lines, truth, marine_sup
