"""KPI engine for interactive use (dashboard and agent).

The governed monthly KPI table is built by dbt (`kpis.kpi_monthly_company`). The dashboard needs the
same KPIs for any period and any filter (business unit, section, supplier origin), so this module
recomputes them with parameterised SQL. A contract test (tests/test_kpis.py) checks that, without
filters, the engine returns exactly the dbt values, so the two can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from p2p.data import query


@dataclass
class Filters:
    business_units: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)

    def where(self, alias: str = "l") -> tuple[str, list]:
        clauses, params = [], []
        if self.business_units:
            clauses.append(f"{alias}.business_unit in ({','.join('?' * len(self.business_units))})")
            params += self.business_units
        if self.sections:
            clauses.append(f"{alias}.section_name in ({','.join('?' * len(self.sections))})")
            params += self.sections
        if self.origins:
            clauses.append(f"{alias}.supplier_origin in ({','.join('?' * len(self.origins))})")
            params += self.origins
        return (" and " + " and ".join(clauses)) if clauses else "", params


def catalog() -> pd.DataFrame:
    return query("select * from reference.kpi_catalog order by kpi_id")


def compute(start: str, end: str, filters: Filters | None = None) -> pd.DataFrame:
    """KPIs for the period [start, end] (dates as 'YYYY-MM-DD'). Stock KPIs are measured at `end`."""
    f = filters or Filters()
    w, p = f.where("l")
    flows = query(f"""
        select
            median(l.pr_to_receipt_days) filter (where l.gr_posted_on between ?::date and ?::date)                       as K01,
            avg(case when l.pr_to_receipt_days <= 45 then 1.0 else 0.0 end)
                filter (where l.gr_posted_on between ?::date and ?::date and l.is_critical and l.reason_code = 'CM')      as K02,
            median(l.rfq_wait_days) filter (where l.gr_posted_on between ?::date and ?::date)                            as K08,
            median(l.po_to_delivery_days) filter (where l.gr_posted_on between ?::date and ?::date)                      as K10,
            avg(case when l.is_otif then 1.0 else 0.0 end) filter (where l.gr_posted_on between ?::date and ?::date)      as K11,
            stddev_samp(l.po_to_delivery_days) filter (where l.gr_posted_on between ?::date and ?::date)
              / nullif(avg(l.po_to_delivery_days) filter (where l.gr_posted_on between ?::date and ?::date), 0)          as K12,
            avg(case when l.had_inspection_return then 1.0 else 0.0 end)
                filter (where l.gr_posted_on between ?::date and ?::date)                                                 as K13,
            avg(case when l.supplier_origin = 'Local' then 1.0 else 0.0 end)
                filter (where l.po_created_on between ?::date and ?::date and l.supplier_origin in ('Local', 'Foreign'))  as K14,
            avg(case when l.reason_code = 'CM' then 1.0 else 0.0 end) filter (where l.pr_created_on between ?::date and ?::date) as K16,
            sum(l.po_value_usd) filter (where l.po_created_on between ?::date and ?::date and l.supplier_origin in ('Local', 'Foreign'))
              / nullif(sum(l.estimated_value_usd) filter (where l.po_created_on between ?::date and ?::date
                                                           and l.supplier_origin in ('Local', 'Foreign')), 0) - 1          as K19,
            count(*) filter (where l.closed_on between ?::date and ?::date) * 1.0
              / nullif(count(*) filter (where l.pr_created_on between ?::date and ?::date), 0)                           as K07,
            count(*) filter (where l.pr_created_on <= ?::date and (l.closed_on is null or l.closed_on > ?::date))         as K03,
            sum(l.line_value_usd) filter (where l.pr_created_on <= ?::date and (l.closed_on is null or l.closed_on > ?::date)) as K04,
            avg(case when date_diff('day', l.pr_created_on, ?::date) > 90 then 1.0 else 0.0 end)
                filter (where l.pr_created_on <= ?::date and (l.closed_on is null or l.closed_on > ?::date))              as K05,
            avg(case when date_diff('day', l.pr_created_on, ?::date) > 150 then 1.0 else 0.0 end)
                filter (where l.pr_created_on <= ?::date and (l.closed_on is null or l.closed_on > ?::date))              as K06,
            count(distinct l.po_number) filter (where l.dp_required and l.po_approved_on <= ?::date
                and (l.dp_paid_on is null or l.dp_paid_on > ?::date) and (l.cancelled_on is null or l.cancelled_on > ?::date)) as K17,
            1 - avg(case when l.dq_missing_supplier or l.dq_zero_price or l.dq_po_date_corrected or l.dq_over_delivery
                              or l.dq_future_posting_date then 1.0 else 0.0 end)
                filter (where l.po_created_on between ?::date and ?::date and l.supplier_origin <> 'Not Sourced')        as K22
        from marts.fct_requisition_line l
        where true {w}
    """, [start, end] * 14 + [end] * 13 + [start, end] + p)
    cash = query(f"""
        select sum(l.amount_usd) as K18 from marts.fct_payment l
        where l.paid_on between ?::date and ?::date {w}
    """, [start, end] + p)
    wf, wp = Filters(f.business_units, f.sections).where("w")  # supplier origin does not apply to work orders
    wo = query(f"""
        with base as (
            select w.*, s.section_name from marts.fct_work_order w left join marts.dim_section s using (section_id)
        )
        select
            count(distinct w.work_order_id) filter (where (w.closed_on is null or w.closed_on > ?::date) and exists (
                select 1 from marts.fct_requisition_line l
                where l.work_order_id = w.work_order_id and l.pr_created_on <= ?::date and (l.closed_on is null or l.closed_on > ?::date))) as K20,
            count(distinct w.work_order_id) filter (where w.wo_type = 'CM' and w.criticality = 'A' and w.pr_lines > 0
                and (w.last_part_received_on is null or w.last_part_received_on > ?::date))                               as K21
        from base w
        where w.created_on <= ?::date {wf}
    """, [end, end, end, end, end] + wp)
    appr = query("""
        select median(date_diff('day', a.approval_submitted_on, a.po_approved_on)) filter (where a.approval_level = 'MD') as K09,
               avg(case when a.first_decision = 'REJECTED' then 1.0 else 0.0 end)                                       as K15
        from intermediate.int_po_approvals a where a.first_decision_on between ?::date and ?::date
    """, [start, end]) if not (f.business_units or f.sections or f.origins) else pd.DataFrame([{"K09": None, "K15": None}])
    vals = pd.concat([flows, cash, wo, appr], axis=1).T[0].rename("value")
    cat = catalog().set_index("kpi_id")
    out = cat.join(vals, how="left")
    out["status"] = [_status(r.direction, r.value, r.target, r.amber_threshold) for r in out.itertuples()]
    return out.reset_index()


def _status(direction, value, target, amber) -> str:
    if value is None or pd.isna(value):
        return "No data"
    if target is None or pd.isna(target):
        return "Monitor"
    if direction == "lower_is_better":
        return "On target" if value <= target else ("At risk" if value <= amber else "Off target")
    if direction == "higher_is_better":
        return "On target" if value >= target else ("At risk" if value >= amber else "Off target")
    return "Monitor"


def monthly_history(kpi_id: str) -> pd.DataFrame:
    return query("select month_start, kpi_value, target, status from kpis.kpi_monthly_company where kpi_id = ? order by month_start", [kpi_id])
