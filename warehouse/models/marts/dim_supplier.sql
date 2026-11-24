-- Supplier master enriched with delivery performance over the full history.
with perf as (
    select
        supplier_id,
        count(*)                                              as po_lines,
        count(delivered_on)                                   as delivered_lines,
        median(po_to_delivery_days)                           as lead_time_p50_days,
        quantile_cont(po_to_delivery_days, 0.9)               as lead_time_p90_days,
        avg(case when is_otif then 1.0 when is_otif is not null then 0.0 end) as otif_rate,
        sum(po_value_usd)                                     as total_po_value_usd,
        max(po_created_on)                                    as last_po_on
    from {{ ref('int_requisition_lifecycle') }}
    where po_number is not null
    group by supplier_id
)
select s.*, p.* exclude (supplier_id),
       rank() over (order by p.total_po_value_usd desc)       as spend_rank
from {{ ref('stg_suppliers') }} s
left join perf p using (supplier_id)
