-- Feature store for the machine-learning forecasters (direct multi-horizon strategy).
-- Every feature on row t uses information available at the end of week t only.
-- Targets y_lead_h are the requisition lines h weeks ahead (h = 1..12).
with base as (select * from {{ ref('fct_weekly_section') }})
select
    section_id,
    business_unit,
    week_start                                                                   as origin_week,
    pr_lines                                                                     as y_t,
    lag(pr_lines, 1)  over w as lag_1,
    lag(pr_lines, 2)  over w as lag_2,
    lag(pr_lines, 3)  over w as lag_3,
    lag(pr_lines, 4)  over w as lag_4,
    lag(pr_lines, 8)  over w as lag_8,
    lag(pr_lines, 13) over w as lag_13,
    lag(pr_lines, 26) over w as lag_26,
    lag(pr_lines, 51) over w as lag_51,
    avg(pr_lines)        over (w rows between 3 preceding and current row)  as roll_mean_4,
    avg(pr_lines)        over (w rows between 12 preceding and current row) as roll_mean_13,
    avg(pr_lines)        over (w rows between 51 preceding and current row) as roll_mean_52,
    stddev_samp(pr_lines) over (w rows between 12 preceding and current row) as roll_std_13,
    max(pr_lines)        over (w rows between 12 preceding and current row) as roll_max_13,
    avg(cm_lines)        over (w rows between 3 preceding and current row)  as cm_roll_mean_4,
    avg(project_lines)   over (w rows between 12 preceding and current row) as project_roll_mean_13,
    avg(cm_work_orders)  over (w rows between 3 preceding and current row)  as cm_wo_roll_mean_4,
    pm_planned_next_3w,
    fleet_utilisation,
    fleet_utilisation - lag(fleet_utilisation, 4) over w                    as fleet_utilisation_chg_4,
    fx_lcu_per_usd / nullif(lag(fx_lcu_per_usd, 13) over w, 0) - 1         as fx_chg_13w,
    {% for h in range(1, 13) %}
    lead(pr_lines, {{ h }}) over w as y_lead_{{ h }},
    {% endfor %}
    row_number() over w                                                      as week_index
from base
window w as (partition by section_id order by week_start)
