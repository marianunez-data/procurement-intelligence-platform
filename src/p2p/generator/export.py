"""Turn the simulated lifecycle into SAP-like raw extracts, inject documented
data-quality defects and write everything to data/raw/.

The defect counts are written to ground_truth.json so the data-audit notebook can
prove it found every issue that was planted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from p2p.config import CFG, DATA_DIR, RAW_DIR, WAREHOUSE_DIR, GeneratorConfig
from p2p.generator import master_data as md
from p2p.generator.simulate import (
    _ln,
    assign_preferred_suppliers,
    simulate_demand,
    simulate_fx,
    simulate_lifecycle,
    simulate_operations,
)

GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


def _fmt(d: pd.Series) -> pd.Series:
    return d.dt.strftime("%Y-%m-%d")


def _purchasing_group(lines: pd.DataFrame) -> pd.Series:
    sec = md.SECTIONS.set_index("section_id").business_unit
    bu = lines.section_id.map(sec)
    pg = np.where(lines.supplier_origin == "Local", "PG03",
                  np.where(lines.supplier_id.str[-1].astype(int) % 2 == 0, "PG01", "PG02"))
    pg = np.where((bu == "Power Generation") & (lines.pr_number % 3 == 0), "PG04", pg)
    pg = np.where(bu == "Marine Services", "PG05", pg)
    pg = np.where(lines.reason_code == "PROJECT", "PG06", pg)
    return pd.Series(pg, index=lines.index)


def build(cfg: GeneratorConfig = CFG) -> dict:
    rng = np.random.default_rng(cfg.seed)
    suppliers = md.build_suppliers(rng)
    materials = md.build_materials(rng)
    materials = assign_preferred_suppliers(rng, materials, suppliers)
    equipment = md.build_equipment(rng)
    fx = simulate_fx(rng, cfg)
    operations = simulate_operations(rng, cfg)
    work_orders, lines = simulate_demand(rng, cfg, materials, equipment, operations)
    lines, truth, suspended_supplier = simulate_lifecycle(rng, cfg, lines, materials, suppliers, fx)
    lines["purchasing_group_id"] = _purchasing_group(lines)
    return dict(suppliers=suppliers, materials=materials, equipment=equipment, fx=fx,
                operations=operations, work_orders=work_orders, lines=lines, truth=truth,
                suspended_supplier=suspended_supplier, rng=rng)


def to_raw_tables(b: dict, cfg: GeneratorConfig = CFG) -> tuple[dict[str, pd.DataFrame], dict]:
    rng = b["rng"]
    lines = b["lines"].sort_values(["pr_created_on", "pr_number", "pr_item"]).reset_index(drop=True)
    snap = cfg.snapshot
    defects: dict[str, int] = {}

    # ---------------- purchase requisitions ----------------
    status = np.select(
        [lines.cancelled_on.notna(), lines.gr_posted_on.notna(), lines.po_created_on.notna(), lines.rfq_created_on.notna()],
        ["Cancelled", "Closed", "PO Created", "RFQ Created"], default="Not Edited")
    sec_name = lines.section_id.map(md.SECTIONS.set_index("section_id").section_name)
    pr = pd.DataFrame({
        "pr_number": lines.pr_number, "pr_item": lines.pr_item * 10,
        "created_on": _fmt(lines.pr_created_on), "released_on": _fmt(lines.pr_released_on),
        "section": sec_name, "material_id": lines.material_id, "short_text": lines.short_text,
        "quantity": lines.quantity, "uom": lines.uom, "estimated_price_usd": lines.estimated_price_usd,
        "reason_code": lines.reason_code, "work_order_id": lines.work_order_id, "equipment_id": lines.equipment_id,
        "priority": np.where(lines.is_critical, "1-Critical", "3-Normal"), "processing_status": status,
        "status_changed_on": _fmt(lines.cancelled_on),
    })
    # defect: messy manual section names
    messy = rng.random(len(pr)) < 0.15
    pr.loc[messy, "section"] = [rng.choice(md.SECTION_ALIASES[s]) for s in pr.loc[messy, "section"]]
    defects["pr_section_messy_text"] = int(messy.sum())
    # defect: status inconsistent with the documents (status says Not Edited but a PO exists)
    has_po = (pr.processing_status == "PO Created").to_numpy()
    wrong = has_po & (rng.random(len(pr)) < 0.006)
    pr.loc[wrong, "processing_status"] = "Not Edited"
    defects["pr_status_inconsistent"] = int(wrong.sum())

    # ---------------- RFQs ----------------
    rfq_lines = lines[lines.rfq_created_on.notna()]
    rfq_keys = rfq_lines.drop_duplicates("group_key").sort_values("rfq_created_on")
    rfq_no = pd.Series(np.arange(len(rfq_keys)) + 6000100001, index=rfq_keys.group_key.values)
    rfq = pd.DataFrame({
        "rfq_number": rfq_lines.group_key.map(rfq_no).values, "pr_number": rfq_lines.pr_number.values,
        "pr_item": rfq_lines.pr_item.values * 10, "purchasing_group_id": rfq_lines.purchasing_group_id.values,
        "supplier_invited_id": rfq_lines.supplier_id.values, "created_on": _fmt(rfq_lines.rfq_created_on).values,
    })

    # ---------------- purchase orders ----------------
    po_lines = lines[lines.po_created_on.notna()].copy()
    po_keys = po_lines.drop_duplicates("group_key").sort_values("po_created_on")
    po_no = pd.Series(np.arange(len(po_keys)) + 4500100001, index=po_keys.group_key.values)
    po_lines["po_number"] = po_lines.group_key.map(po_no)
    po_lines["po_item"] = (po_lines.groupby("po_number").cumcount() + 1) * 10
    lines = lines.merge(po_lines[["pr_number", "pr_item", "po_number", "po_item"]], on=["pr_number", "pr_item"], how="left")
    po_status = np.select([po_lines.cancelled_on.notna(), po_lines.gr_posted_on.notna(), po_lines.po_approved_on.notna()],
                          ["Cancelled", "Closed", "Released"], default="Awaiting Approval")
    po = pd.DataFrame({
        "po_number": po_lines.po_number, "po_item": po_lines.po_item, "pr_number": po_lines.pr_number,
        "pr_item": po_lines.pr_item * 10, "supplier_id": po_lines.supplier_id, "purchasing_group_id": po_lines.purchasing_group_id,
        "created_on": _fmt(po_lines.po_created_on), "material_id": po_lines.material_id, "quantity": po_lines.quantity,
        "net_price_usd": po_lines.po_unit_price_usd, "net_value_usd": po_lines.line_value_usd,
        "promised_date": _fmt(po_lines.promised_date), "quoted_lead_time_days": po_lines.quoted_lead_time_days,
        "po_status": po_status,
    }).reset_index(drop=True)
    # defects
    miss = rng.random(len(po)) < 0.005
    po.loc[miss, "supplier_id"] = None
    defects["po_missing_supplier"] = int(miss.sum())
    zero = rng.random(len(po)) < 0.002
    po.loc[zero, "net_price_usd"] = 0.0
    po.loc[zero, "net_value_usd"] = 0.0
    defects["po_zero_price"] = int(zero.sum())
    typo = rng.random(len(po)) < 0.002
    po.loc[typo, "created_on"] = (pd.to_datetime(po.loc[typo, "created_on"]) - pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    defects["po_date_before_pr"] = int(typo.sum())
    orphan = rng.random(len(po)) < 0.0015
    po.loc[orphan, "pr_number"] = 1999900000 + np.arange(orphan.sum())
    defects["po_orphan_without_pr"] = int(orphan.sum())

    # ---------------- approvals ----------------
    first = po_lines.drop_duplicates("po_number")
    appr_rows = []
    for r in first.itertuples():
        level = "MD" if r.needs_md else "HOD"
        if pd.notna(r.first_decision_on):
            appr_rows.append((r.po_number, level, 1, r.approval_submitted_on, r.first_decision_on,
                              "REJECTED" if r.rejected else "APPROVED"))
        elif pd.notna(r.approval_submitted_on):
            appr_rows.append((r.po_number, level, 1, r.approval_submitted_on, pd.NaT, "PENDING"))
        if r.rejected and pd.notna(r.resubmitted_on) and pd.isna(r.cancelled_on):
            decided = r.po_approved_on
            appr_rows.append((r.po_number, level, 2, r.resubmitted_on, decided, "APPROVED" if pd.notna(decided) else "PENDING"))
    approvals = pd.DataFrame(appr_rows, columns=["po_number", "approval_level", "approval_cycle", "submitted_on", "decided_on", "decision"])
    approvals["submitted_on"] = _fmt(approvals.submitted_on)
    approvals["decided_on"] = _fmt(approvals.decided_on)

    # ---------------- payments ----------------
    sup = b["suppliers"].set_index("supplier_id")
    dp = first[first.dp_paid_on.notna()]
    po_val = po_lines.groupby("po_number").line_value_usd.sum()
    pay_dp = pd.DataFrame({"po_number": dp.po_number, "po_item": None, "payment_type": "DOWNPAYMENT",
                           "amount_usd": (po_val.reindex(dp.po_number).values * sup.loc[dp.supplier_id, "downpayment_pct"].values).round(2),
                           "paid_on": _fmt(dp.dp_paid_on)})
    fin = po_lines[po_lines.final_paid_on.notna()]
    pct = sup.loc[fin.supplier_id, "downpayment_pct"].values * fin.dp_required.values
    pay_fin = pd.DataFrame({"po_number": fin.po_number, "po_item": fin.po_item, "payment_type": "FINAL",
                            "amount_usd": (fin.line_value_usd.values * (1 - pct)).round(2), "paid_on": _fmt(fin.final_paid_on)})
    payments = pd.concat([pay_dp, pay_fin], ignore_index=True).sort_values("paid_on").reset_index(drop=True)
    payments.insert(0, "payment_id", [f"PAY{700000 + i}" for i in range(len(payments))])

    # ---------------- goods receipts (SAP date format dd.mm.yyyy) ----------------
    gr_rows = []
    rec = po_lines[po_lines.first_delivery_on.notna() & (po_lines.first_delivery_on <= snap)]
    for r in rec.itertuples():
        q = r.quantity
        if r.inspection_rejected:
            post1 = r.first_delivery_on + pd.Timedelta(days=1)
            if post1 <= snap:
                gr_rows.append((r.po_number, r.po_item, "101", post1, r.first_delivery_on, q))
                if post1 + pd.Timedelta(days=5) <= snap:
                    gr_rows.append((r.po_number, r.po_item, "122", post1 + pd.Timedelta(days=5), r.first_delivery_on, -q))
        if pd.isna(r.gr_posted_on):
            continue
        if r.partial_delivery and q >= 2:
            q1 = max(1, int(round(q * 0.6)))
            p1 = r.first_delivery_on + pd.Timedelta(days=2) if not r.inspection_rejected else r.gr_posted_on - pd.Timedelta(days=10)
            gr_rows.append((r.po_number, r.po_item, "101", min(p1, r.gr_posted_on), r.first_delivery_on, q1))
            gr_rows.append((r.po_number, r.po_item, "101", r.gr_posted_on, r.delivered_on, q - q1))
        else:
            gr_rows.append((r.po_number, r.po_item, "101", r.gr_posted_on, r.delivered_on, q))
    gr = pd.DataFrame(gr_rows, columns=["po_number", "po_item", "movement_type", "posting_date", "delivery_date", "quantity"])
    gr = gr.sort_values("posting_date").reset_index(drop=True)
    gr.insert(0, "material_document", [f"5000{100000 + i}" for i in range(len(gr))])
    over = (gr.movement_type == "101").to_numpy() & (rng.random(len(gr)) < 0.003)
    gr.loc[over, "quantity"] = gr.loc[over, "quantity"] + 1
    defects["gr_over_delivery"] = int(over.sum())
    gr["posting_date"] = gr.posting_date.dt.strftime("%d.%m.%Y")
    gr["delivery_date"] = gr.delivery_date.dt.strftime("%d.%m.%Y")
    fut = rng.choice(len(gr), 6, replace=False)
    gr.loc[fut, "posting_date"] = gr.loc[fut, "posting_date"].str[:-4] + "2052"
    defects["gr_future_posting_date"] = 6
    dup_gr = gr.sample(frac=0.004, random_state=11)
    gr = pd.concat([gr, dup_gr]).sort_index(kind="stable").reset_index(drop=True)
    defects["gr_exact_duplicates"] = len(dup_gr)
    dup_pr = pr.sample(frac=0.003, random_state=12)
    pr = pd.concat([pr, dup_pr]).sort_index(kind="stable").reset_index(drop=True)
    defects["pr_exact_duplicates"] = len(dup_pr)

    # ---------------- work orders ----------------
    wo = b["work_orders"].copy()
    wo_lines = lines[lines.work_order_id.notna()].groupby("work_order_id").agg(
        parts_received=("gr_posted_on", "max"), n_lines=("pr_item", "size"), open_lines=("gr_posted_on", lambda x: x.isna().sum()))
    wo = wo.merge(wo_lines, on="work_order_id", how="left")
    repair = pd.to_timedelta(np.round(_ln(rng, 2, 0.6, len(wo))).astype(int), "D")
    base_close = np.where(wo.wo_type == "PM", wo.planned_date, wo.created_on)
    no_parts_close = pd.Series(pd.to_datetime(base_close)) + repair
    waiting = wo.open_lines.fillna(0) > 0
    with_parts_close = np.maximum(wo.parts_received.fillna(pd.Timestamp("1900-01-01")), pd.Series(pd.to_datetime(base_close))) + repair
    closed = no_parts_close.where(wo.n_lines.isna(), with_parts_close)
    closed = closed.where(~waiting, pd.NaT)
    closed = closed.where(closed <= snap, pd.NaT)
    wo = wo[wo.created_on <= snap].copy()
    wo["closed_on"] = closed
    work_orders = pd.DataFrame({
        "work_order_id": wo.work_order_id, "equipment_id": wo.equipment_id, "wo_type": wo.wo_type,
        "created_on": _fmt(wo.created_on), "planned_date": _fmt(wo.planned_date), "closed_on": _fmt(wo.closed_on),
    }).sort_values("created_on").reset_index(drop=True)

    sections = md.SECTIONS[["section_id", "section_name", "business_unit"]]
    suppliers = b["suppliers"].drop(columns=["reliability", "focus"])  # reliability is hidden ground truth
    materials = b["materials"][["material_id", "short_text", "category", "section_id", "standard_price_usd", "uom", "preferred_supplier_id"]]
    tables = {
        "purchase_requisitions": pr, "rfqs": rfq, "purchase_orders": po, "po_approvals": approvals,
        "payments": payments, "goods_receipts": gr, "work_orders": work_orders, "suppliers": suppliers,
        "materials": materials, "equipment": b["equipment"], "sections": sections,
        "purchasing_groups": md.PURCHASING_GROUPS, "fx_rates": b["fx"].assign(rate_date=_fmt(b["fx"].rate_date)),
        "operations_weekly": b["operations"].assign(week_start=_fmt(b["operations"].week_start)),
    }
    return tables, defects


def write(cfg: GeneratorConfig = CFG) -> dict:
    b = build(cfg)
    tables, defects = to_raw_tables(b, cfg)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)  # dbt writes the DuckDB file here
    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
    truth = b["truth"].drop(columns=["group_key", "gkey"], errors="ignore")
    truth.to_parquet(GROUND_TRUTH_DIR / "lifecycle_uncensored.parquet", index=False)
    reliability = b["suppliers"][["supplier_id", "reliability", "quoted_lead_time_days", "supplier_origin"]]
    reliability.to_csv(GROUND_TRUTH_DIR / "supplier_reliability.csv", index=False)
    gt = {
        "company": "Quayline Offshore Logistics (fictional)",
        "seed": cfg.seed, "start_date": cfg.start_date, "snapshot_date": cfg.snapshot_date,
        "annual_growth": cfg.annual_growth,
        "md_threshold_usd": {"before_policy": cfg.md_threshold_before, "after_policy": cfg.md_threshold_after},
        "shocks": [s.__dict__ for s in cfg.shocks],
        "suspended_supplier_id": b["suspended_supplier"],
        "demand_drivers": {
            "CM": "Corrective maintenance: negative binomial, rises with rainy season (peak mid-July) and fleet utilisation",
            "PM": "Planned maintenance: fixed 4-week and 13-week schedule, parts requested 14-21 days ahead (known future covariate)",
            "RESTOCK": "MRP run every second Tuesday, +30% in the October-November budget push",
            "PROJECT": "Lumpy bursts of 12-40 lines in about 3% of weeks",
        },
        "seasonality": {"rainy_season_peak_doy": 198, "shutdown": "18 Dec to 6 Jan at 45-75% activity", "q4_restock_push": 1.30},
        "planted_defects": defects,
        "row_counts": {k: int(len(v)) for k, v in tables.items()},
    }
    (GROUND_TRUTH_DIR / "ground_truth.json").write_text(json.dumps(gt, indent=2, default=str))
    return gt


if __name__ == "__main__":
    import time
    t = time.time()
    out = write()
    print(json.dumps(out["row_counts"], indent=2))
    print(json.dumps(out["planted_defects"], indent=2))
    print(f"done in {time.time() - t:.1f}s")
