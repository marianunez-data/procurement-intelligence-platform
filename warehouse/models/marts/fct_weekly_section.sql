-- The forecasting base table: one row per section and week with the targets
-- (requisition inflow, cash out) and the leading indicators available to the planner.
with weeks as (
    select distinct week_start from {{ ref('dim_date') }}
    where week_start between cast('{{ var("series_start") }}' as date) and cast('{{ var("series_end") }}' as date)
),
grid as (
    select w.week_start, s.section_id, s.section_name, s.business_unit
    from weeks w cross join {{ ref('dim_section') }} s
),
inflow as (
    select pr_week_start as week_start, section_id,
           count(*)                                         as pr_lines,
           sum(line_value_usd)                              as pr_value_usd,
           count(*) filter (where reason_code = 'CM')       as cm_lines,
           count(*) filter (where reason_code = 'PM')       as pm_lines,
           count(*) filter (where reason_code = 'RESTOCK')  as restock_lines,
           count(*) filter (where reason_code = 'PROJECT')  as project_lines,
           count(*) filter (where is_critical)              as critical_lines
    from {{ ref('fct_requisition_line') }}
    group by all
),
cash as (
    select paid_week_start as week_start, section_id, sum(amount_usd) as cash_out_usd
    from {{ ref('fct_payment') }} group by all
),
backlog as (
    select week_start, section_id, sum(open_lines) as open_lines, sum(open_value_usd) as open_value_usd,
           sum(received_lines) as received_lines
    from {{ ref('fct_backlog_weekly') }} group by all
),
wo as (
    select {{ week_start('created_on') }} as week_start, section_id,
           count(*) filter (where wo_type = 'CM') as cm_work_orders
    from {{ ref('fct_work_order') }} group by all
),
pm_plan as (
    select {{ week_start('planned_date') }} as week_start, section_id, count(*) as pm_work_orders_planned
    from {{ ref('fct_work_order') }} where wo_type = 'PM' group by all
),
ops as (select * from {{ ref('stg_operations_weekly') }}),
fx as (
    select {{ week_start('rate_date') }} as week_start, avg(lcu_per_usd) as fx_lcu_per_usd
    from {{ ref('stg_fx_rates') }} group by all
)
select
    g.week_start, g.section_id, g.section_name, g.business_unit,
    coalesce(i.pr_lines, 0)               as pr_lines,
    coalesce(i.pr_value_usd, 0)           as pr_value_usd,
    coalesce(i.cm_lines, 0)               as cm_lines,
    coalesce(i.pm_lines, 0)               as pm_lines,
    coalesce(i.restock_lines, 0)          as restock_lines,
    coalesce(i.project_lines, 0)          as project_lines,
    coalesce(i.critical_lines, 0)         as critical_lines,
    coalesce(c.cash_out_usd, 0)           as cash_out_usd,
    coalesce(b.open_lines, 0)             as open_lines,
    coalesce(b.open_value_usd, 0)         as open_value_usd,
    coalesce(b.received_lines, 0)         as received_lines,
    coalesce(w.cm_work_orders, 0)         as cm_work_orders,
    coalesce(p.pm_work_orders_planned, 0) as pm_work_orders_planned,
    -- planned maintenance is scheduled weeks ahead, so these leads are legitimately known at forecast time
    coalesce(lead(p.pm_work_orders_planned, 1) over w_sec, 0) + coalesce(lead(p.pm_work_orders_planned, 2) over w_sec, 0)
        + coalesce(lead(p.pm_work_orders_planned, 3) over w_sec, 0) as pm_planned_next_3w,
    o.fleet_utilisation,
    f.fx_lcu_per_usd
from grid g
left join inflow i using (week_start, section_id)
left join cash c using (week_start, section_id)
left join backlog b using (week_start, section_id)
left join wo w using (week_start, section_id)
left join pm_plan p using (week_start, section_id)
left join ops o on o.week_start = g.week_start and o.business_unit = g.business_unit
left join fx f on f.week_start = g.week_start
window w_sec as (partition by g.section_id order by g.week_start)
