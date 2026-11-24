-- Governed monthly KPI table (company level), one row per month and KPI.
-- Definitions, targets and owners live in the kpi_catalog seed.
with months as (
    select distinct month_start, last_day(month_start) as month_end
    from {{ ref('dim_date') }}
    where month_start between cast('2022-01-01' as date) and cast('2025-12-01' as date)
),
l as (select * from {{ ref('fct_requisition_line') }}),
received as (
    select m.month_start,
        median(l.pr_to_receipt_days)                                                       as k01,
        avg(case when l.pr_to_receipt_days <= 45 then 1.0 else 0.0 end) filter (where l.is_critical and l.reason_code = 'CM') as k02,
        median(l.rfq_wait_days)                                                            as k08,
        median(l.po_to_delivery_days)                                                      as k10,
        avg(case when l.is_otif then 1.0 else 0.0 end)                                     as k11,
        stddev_samp(l.po_to_delivery_days) / nullif(avg(l.po_to_delivery_days), 0)         as k12,
        avg(case when l.had_inspection_return then 1.0 else 0.0 end)                       as k13
    from months m join l on l.gr_posted_on between m.month_start and m.month_end
    group by m.month_start
),
open_eom as (
    select m.month_start,
        count(*)                                                                            as k03,
        sum(l.line_value_usd)                                                               as k04,
        avg(case when date_diff('day', l.pr_created_on, m.month_end) > 90 then 1.0 else 0.0 end)  as k05,
        avg(case when date_diff('day', l.pr_created_on, m.month_end) > 150 then 1.0 else 0.0 end) as k06
    from months m join l on l.pr_created_on <= m.month_end and (l.closed_on is null or l.closed_on > m.month_end)
    group by m.month_start
),
flow as (
    select m.month_start,
        (select count(*) from l where l.closed_on between m.month_start and m.month_end) * 1.0
          / nullif((select count(*) from l where l.pr_created_on between m.month_start and m.month_end), 0) as k07,
        (select avg(case when l.reason_code = 'CM' then 1.0 else 0.0 end) from l where l.pr_created_on between m.month_start and m.month_end) as k16
    from months m
),
approvals as (
    select m.month_start,
        median(date_diff('day', a.approval_submitted_on, a.po_approved_on)) filter (where a.approval_level = 'MD') as k09,
        avg(case when a.first_decision = 'REJECTED' then 1.0 else 0.0 end)                as k15
    from months m join {{ ref('int_po_approvals') }} a on a.first_decision_on between m.month_start and m.month_end
    group by m.month_start
),
po_lines as (
    select m.month_start,
        avg(case when l.supplier_origin = 'Local' then 1.0 else 0.0 end) filter (where l.supplier_origin <> 'Unknown') as k14,
        sum(l.po_value_usd) filter (where l.supplier_origin <> 'Unknown')
          / nullif(sum(l.estimated_value_usd) filter (where l.supplier_origin <> 'Unknown'), 0) - 1 as k19,
        1 - avg(case when l.dq_missing_supplier or l.dq_zero_price or l.dq_po_date_corrected
                          or l.dq_over_delivery or l.dq_future_posting_date then 1.0 else 0.0 end) as k22
    from months m join l on l.po_created_on between m.month_start and m.month_end
    where l.supplier_origin in ('Local', 'Foreign', 'Unknown')
    group by m.month_start
),
dp_pending as (
    select m.month_start, count(distinct l.po_number) as k17
    from months m join l
      on l.dp_required and l.po_approved_on <= m.month_end
     and (l.dp_paid_on is null or l.dp_paid_on > m.month_end)
     and (l.cancelled_on is null or l.cancelled_on > m.month_end)
    group by m.month_start
),
cash as (
    select m.month_start, sum(p.amount_usd) as k18
    from months m join {{ ref('fct_payment') }} p on p.paid_on between m.month_start and m.month_end
    group by m.month_start
),
wo_wait as (
    select m.month_start, count(distinct w.work_order_id) as k20
    from months m
    join {{ ref('fct_work_order') }} w on w.created_on <= m.month_end and (w.closed_on is null or w.closed_on > m.month_end)
    join l on l.work_order_id = w.work_order_id
         and l.pr_created_on <= m.month_end and (l.closed_on is null or l.closed_on > m.month_end)
    group by m.month_start
),
downtime as (
    select m.month_start, count(distinct w.work_order_id) as k21
    from months m
    join {{ ref('fct_work_order') }} w
      on w.wo_type = 'CM' and w.criticality = 'A' and w.pr_lines > 0
     and w.created_on <= m.month_end
     and (w.last_part_received_on is null or w.last_part_received_on > m.month_end)
    group by m.month_start
),
wide as (
    select m.month_start, r.k01, r.k02, o.k03, o.k04, o.k05, o.k06, f.k07, r.k08, a.k09, r.k10, r.k11, r.k12, r.k13,
           p.k14, a.k15, f.k16, d.k17, c.k18, p.k19, ww.k20, dt.k21, p.k22
    from months m
    left join received r using (month_start)
    left join open_eom o using (month_start)
    left join flow f using (month_start)
    left join approvals a using (month_start)
    left join po_lines p using (month_start)
    left join dp_pending d using (month_start)
    left join cash c using (month_start)
    left join wo_wait ww using (month_start)
    left join downtime dt using (month_start)
),
long as (
    unpivot wide on columns(* exclude (month_start)) into name kpi_key value kpi_value
)
select
    l.month_start,
    upper(l.kpi_key)                    as kpi_id,
    k.kpi_name, k.area, k.unit, k.direction, k.target, k.amber_threshold,
    l.kpi_value,
    case
        when k.target is null then 'Monitor'
        when k.direction = 'lower_is_better' and l.kpi_value <= k.target then 'On target'
        when k.direction = 'lower_is_better' and l.kpi_value <= k.amber_threshold then 'At risk'
        when k.direction = 'higher_is_better' and l.kpi_value >= k.target then 'On target'
        when k.direction = 'higher_is_better' and l.kpi_value >= k.amber_threshold then 'At risk'
        else 'Off target'
    end                                 as status
from long l
join {{ ref('kpi_catalog') }} k on k.kpi_id = upper(l.kpi_key)
