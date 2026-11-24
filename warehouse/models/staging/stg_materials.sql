select
    material_id,
    short_text,
    category,
    section_id,
    cast(standard_price_usd as double)                  as standard_price_usd,
    uom,
    preferred_supplier_id
from {{ source('sap_raw', 'materials') }}
