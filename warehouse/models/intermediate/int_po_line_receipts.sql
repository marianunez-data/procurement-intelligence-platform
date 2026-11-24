-- Collapse material documents into one row per PO line.
-- A line counts as received when net receipts (101 minus 122 returns) cover the ordered quantity;
-- the receipt date is the last 101 posting, so redeliveries after a rejection and second
-- shipments of partial deliveries are handled correctly.
with movements as (
    select * from {{ ref('stg_goods_receipts') }}
),
po as (
    select po_line_key, quantity as ordered_qty from {{ ref('stg_purchase_orders') }}
)
select
    po.po_line_key,
    min(case when m.movement_type = '101' then m.delivery_date end)  as first_delivery_on,
    max(case when m.movement_type = '101' then m.delivery_date end)  as last_delivery_on,
    max(case when m.movement_type = '101' then m.posting_date end)   as last_receipt_posted_on,
    sum(m.quantity)                                                  as net_received_qty,
    po.ordered_qty,
    sum(m.quantity) >= po.ordered_qty                                as is_fully_received,
    count(case when m.movement_type = '101' then 1 end)              as n_receipts,
    bool_or(m.movement_type = '122')                                 as had_inspection_return,
    sum(m.quantity) > po.ordered_qty                                 as dq_over_delivery,
    bool_or(m.dq_future_posting_date)                                as dq_future_posting_date
from po
join movements m using (po_line_key)
group by po.po_line_key, po.ordered_qty
