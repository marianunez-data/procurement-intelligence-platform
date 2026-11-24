-- Cash out. Downpayments are made at PO level and attributed to the section of the PO's first line.
with po_owner as (
    select po_number, any_value(section_id) as section_id, any_value(supplier_origin) as supplier_origin
    from {{ ref('fct_requisition_line') }}
    where po_number is not null
    group by po_number
)
select
    p.payment_id, p.po_number, p.po_item, p.payment_type, p.amount_usd, p.paid_on,
    {{ week_start('p.paid_on') }} as paid_week_start,
    o.section_id, s.business_unit, o.supplier_origin
from {{ ref('stg_payments') }} p
left join po_owner o using (po_number)
left join {{ ref('stg_sections') }} s using (section_id)
