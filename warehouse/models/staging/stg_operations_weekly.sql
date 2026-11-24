select cast(week_start as date) as week_start, business_unit, cast(fleet_utilisation as double) as fleet_utilisation
from {{ source('sap_raw', 'operations_weekly') }}
