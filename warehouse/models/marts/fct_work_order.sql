-- Work orders with their parts situation. Downtime is counted for corrective work orders on
-- criticality-A equipment while at least one requisition line is still open.
with parts as (
    select work_order_id,
           count(*)                                    as pr_lines,
           count(*) filter (where is_open)             as open_pr_lines,
           max(gr_posted_on)                           as last_part_received_on
    from {{ ref('fct_requisition_line') }}
    where work_order_id is not null and lifecycle_status <> 'Cancelled'
    group by work_order_id
)
select
    w.*,
    e.criticality,
    e.section_id,
    e.business_unit,
    coalesce(p.pr_lines, 0)                                         as pr_lines,
    coalesce(p.open_pr_lines, 0)                                    as open_pr_lines,
    p.last_part_received_on,
    w.closed_on is null                                             as is_open,
    coalesce(p.open_pr_lines, 0) > 0                                as is_waiting_for_parts,
    case when w.wo_type = 'CM' and e.criticality = 'A' and p.pr_lines > 0 then
        24 * date_diff('day', w.created_on, coalesce(p.last_part_received_on, cast('{{ var("snapshot_date") }}' as date)))
    end                                                              as downtime_hours_waiting_parts
from {{ ref('stg_work_orders') }} w
left join {{ ref('dim_equipment') }} e using (equipment_id)
left join parts p using (work_order_id)
