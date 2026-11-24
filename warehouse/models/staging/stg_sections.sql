select section_id, section_name, business_unit
from {{ source('sap_raw', 'sections') }}
