select cast(rate_date as date) as rate_date, cast(lcu_per_usd as double) as lcu_per_usd
from {{ source('sap_raw', 'fx_rates') }}
