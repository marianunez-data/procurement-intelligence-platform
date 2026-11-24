select purchasing_group_id, purchasing_group_name, scope
from {{ source('sap_raw', 'purchasing_groups') }}
