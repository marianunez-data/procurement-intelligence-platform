select
    supplier_id,
    supplier_name,
    supplier_origin,
    country,
    cast(quoted_lead_time_days as integer)              as quoted_lead_time_days,
    downpayment_required = 'True'                       as downpayment_required,
    cast(downpayment_pct as double)                     as downpayment_pct,
    cast(payment_terms_days as integer)                 as payment_terms_days
from {{ source('sap_raw', 'suppliers') }}
