select
    cast(rfq_number as bigint)                          as rfq_number,
    cast(pr_number as bigint) || '-' || pr_item         as pr_line_key,
    purchasing_group_id,
    supplier_invited_id,
    cast(created_on as date)                            as rfq_created_on
from {{ source('sap_raw', 'rfqs') }}
