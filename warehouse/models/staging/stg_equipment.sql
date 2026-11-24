select equipment_id, section_id, criticality, description
from {{ source('sap_raw', 'equipment') }}
