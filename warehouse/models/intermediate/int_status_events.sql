-- Long format: one row per (line, event). Used for process mining and gaps-and-islands analysis.
with wide as (
    select
        pr_line_key,
        pr_created_on   as "PR Created",
        pr_released_on  as "PR Released",
        rfq_created_on  as "RFQ Created",
        po_created_on   as "PO Created",
        po_approved_on  as "PO Approved",
        dp_paid_on      as "Downpayment Paid",
        delivered_on    as "Delivered",
        gr_posted_on    as "Goods Receipt Posted",
        final_paid_on   as "Final Payment",
        cancelled_on    as "Cancelled"
    from {{ ref('int_requisition_lifecycle') }}
),
long as (
    unpivot wide on columns(* exclude (pr_line_key)) into name event_name value event_date
)
select
    pr_line_key,
    event_name,
    case event_name
        when 'PR Created' then 1 when 'PR Released' then 2 when 'RFQ Created' then 3
        when 'PO Created' then 4 when 'PO Approved' then 5 when 'Downpayment Paid' then 6
        when 'Delivered' then 7 when 'Goods Receipt Posted' then 8 when 'Final Payment' then 9
        else 10 end as event_order,
    event_date
from long
