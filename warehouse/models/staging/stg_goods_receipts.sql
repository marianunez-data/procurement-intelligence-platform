-- SAP exports dates as dd.mm.yyyy. Exact duplicate documents and impossible future dates are removed.
with parsed as (
    select
        material_document,
        cast(po_number as bigint)                       as po_number,
        cast(po_item as integer)                        as po_item,
        cast(po_number as varchar) || '-' || po_item    as po_line_key,
        movement_type,
        cast(strptime(posting_date, '%d.%m.%Y') as date)  as posting_date_raw,
        cast(strptime(delivery_date, '%d.%m.%Y') as date) as delivery_date,
        cast(quantity as integer)                       as quantity
    from {{ source('sap_raw', 'goods_receipts') }}
    qualify row_number() over (partition by material_document order by posting_date_raw) = 1
)
-- Business rule: a posting date after the extraction date is a keying error; fall back to the delivery date.
select
    material_document, po_number, po_item, po_line_key, movement_type,
    case when posting_date_raw > cast('{{ var("snapshot_date") }}' as date) then delivery_date else posting_date_raw end as posting_date,
    delivery_date,
    quantity,
    posting_date_raw > cast('{{ var("snapshot_date") }}' as date) as dq_future_posting_date
from parsed
