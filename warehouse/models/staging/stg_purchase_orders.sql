-- PO lines with data-quality flags. Defects are flagged here and resolved in the intermediate layer.
select
    cast(po_number as bigint)                           as po_number,
    cast(po_item as integer)                            as po_item,
    cast(po_number as varchar) || '-' || po_item        as po_line_key,
    cast(pr_number as bigint) || '-' || pr_item         as pr_line_key,
    coalesce(nullif(supplier_id, ''), 'UNKNOWN')        as supplier_id,
    purchasing_group_id,
    cast(created_on as date)                            as po_created_on_raw,
    material_id,
    cast(quantity as integer)                           as quantity,
    cast(net_price_usd as double)                       as net_price_usd,
    cast(net_value_usd as double)                       as net_value_usd,
    cast(nullif(promised_date, '') as date)             as promised_date,
    cast(quoted_lead_time_days as integer)              as quoted_lead_time_days,
    po_status,
    supplier_id is null or supplier_id = ''             as dq_missing_supplier,
    cast(net_price_usd as double) <= 0                  as dq_zero_price
from {{ source('sap_raw', 'purchase_orders') }}
