select
    payment_id,
    cast(po_number as bigint)                           as po_number,
    cast(nullif(po_item, '') as integer)                as po_item,
    payment_type,
    cast(amount_usd as double)                          as amount_usd,
    cast(paid_on as date)                               as paid_on
from {{ source('sap_raw', 'payments') }}
