-- Downpayments are made per PO, final payments per PO line.
with dp as (
    select po_number, min(paid_on) as dp_paid_on, sum(amount_usd) as dp_amount_usd
    from {{ ref('stg_payments') }}
    where payment_type = 'DOWNPAYMENT'
    group by po_number
),
fin as (
    select po_number || '-' || po_item as po_line_key, min(paid_on) as final_paid_on, sum(amount_usd) as final_amount_usd
    from {{ ref('stg_payments') }}
    where payment_type = 'FINAL'
    group by 1
)
select po.po_line_key, po.po_number, dp.dp_paid_on, dp.dp_amount_usd, fin.final_paid_on, fin.final_amount_usd
from {{ ref('stg_purchase_orders') }} po
left join dp using (po_number)
left join fin using (po_line_key)
