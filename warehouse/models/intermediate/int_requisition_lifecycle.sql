-- The core table: one row per requisition line with every stage of its journey,
-- derived from the documents themselves (never from the unreliable SAP status field).
with pr as (select * from {{ ref('stg_purchase_requisitions') }}),
rfq as (
    select pr_line_key, min(rfq_created_on) as rfq_created_on, any_value(purchasing_group_id) as purchasing_group_id
    from {{ ref('stg_rfqs') }} group by pr_line_key
),
po as (select * from {{ ref('stg_purchase_orders') }}),
appr as (select * from {{ ref('int_po_approvals') }}),
rec as (select * from {{ ref('int_po_line_receipts') }}),
pay as (select * from {{ ref('int_po_payments') }}),
sup as (select * from {{ ref('stg_suppliers') }}),
mat as (select * from {{ ref('stg_materials') }}),

joined as (
    select
        pr.pr_line_key, pr.pr_number, pr.pr_item, pr.section_id, pr.material_id, mat.category, pr.short_text,
        pr.reason_code, pr.is_critical, pr.work_order_id, pr.equipment_id,
        coalesce(po.purchasing_group_id, rfq.purchasing_group_id) as purchasing_group_id,
        po.po_number, po.po_item, po.po_line_key, po.supplier_id,
        case when po.po_number is null then 'Not Sourced' else coalesce(sup.supplier_origin, 'Unknown') end as supplier_origin,
        pr.quantity, pr.estimated_price_usd, pr.estimated_value_usd,
        case when po.dq_zero_price then pr.estimated_value_usd else po.net_value_usd end as po_value_usd,
        pr.pr_created_on, pr.pr_released_on, rfq.rfq_created_on,
        -- Business rule: a PO dated before its requisition is a keying error; the approval
        -- submission date is the reliable PO creation date in that case.
        case when po.po_created_on_raw < pr.pr_created_on then appr.approval_submitted_on
             else po.po_created_on_raw end                      as po_created_on,
        appr.approval_level, appr.approval_submitted_on, appr.first_decision_on, appr.first_decision,
        appr.was_rejected, appr.approval_cycles, appr.po_approved_on,
        sup.downpayment_required                                as dp_required,
        pay.dp_paid_on, po.promised_date, po.quoted_lead_time_days,
        rec.first_delivery_on,
        case when rec.is_fully_received then rec.last_delivery_on end      as delivered_on,
        case when rec.is_fully_received then rec.last_receipt_posted_on end as gr_posted_on,
        rec.net_received_qty, rec.n_receipts, coalesce(rec.had_inspection_return, false) as had_inspection_return,
        pay.final_paid_on,
        pr.cancelled_on,
        pr.sap_processing_status,
        -- data quality flags
        pr.dq_unmapped_section,
        coalesce(po.dq_missing_supplier, false)                 as dq_missing_supplier,
        coalesce(po.dq_zero_price, false)                       as dq_zero_price,
        coalesce(po.po_created_on_raw < pr.pr_created_on, false) as dq_po_date_corrected,
        coalesce(rec.dq_over_delivery, false)                   as dq_over_delivery,
        coalesce(rec.dq_future_posting_date, false)             as dq_future_posting_date
    from pr
    left join rfq using (pr_line_key)
    left join po on po.pr_line_key = pr.pr_line_key
    left join appr on appr.po_number = po.po_number
    left join rec on rec.po_line_key = po.po_line_key
    left join pay on pay.po_line_key = po.po_line_key
    left join sup on sup.supplier_id = po.supplier_id
    left join mat on mat.material_id = pr.material_id
),

statused as (
    select
        *,
        case
            when cancelled_on is not null                          then 'Cancelled'
            when gr_posted_on is not null                          then 'Received'
            when po_approved_on is not null and dp_required and dp_paid_on is null then 'Awaiting Downpayment'
            when po_approved_on is not null                        then 'Awaiting Delivery'
            when po_created_on is not null                         then 'Awaiting Approval'
            when rfq_created_on is not null                        then 'RFQ Created'
            else 'Not Edited'
        end as lifecycle_status
    from joined
)

select
    *,
    lifecycle_status not in ('Cancelled', 'Received')                       as is_open,
    coalesce(gr_posted_on, cancelled_on)                                    as closed_on,
    date_diff('day', pr_created_on, coalesce(gr_posted_on, cancelled_on, cast('{{ var("snapshot_date") }}' as date))) as age_days,
    coalesce(po_value_usd, estimated_value_usd)                             as line_value_usd,
    date_diff('day', pr_created_on, pr_released_on)                         as release_days,
    date_diff('day', pr_released_on, rfq_created_on)                        as rfq_wait_days,
    date_diff('day', rfq_created_on, po_created_on)                         as quotation_days,
    date_diff('day', approval_submitted_on, po_approved_on)                 as approval_days,
    date_diff('day', po_approved_on, dp_paid_on)                            as downpayment_days,
    date_diff('day', po_approved_on, delivered_on)                          as po_to_delivery_days,
    date_diff('day', delivered_on, gr_posted_on)                            as gr_posting_lag_days,
    date_diff('day', pr_created_on, gr_posted_on)                           as pr_to_receipt_days,
    case when delivered_on is not null then delivered_on <= promised_date end                       as is_on_time,
    case when delivered_on is not null then delivered_on <= promised_date and n_receipts = 1 and not had_inspection_return end as is_otif,
    -- the SAP status field disagrees with the documents (audit check)
    (sap_processing_status = 'Not Edited' and po_number is not null)       as dq_status_mismatch
from statused
