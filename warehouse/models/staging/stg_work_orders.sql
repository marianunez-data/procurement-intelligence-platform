select
    work_order_id,
    equipment_id,
    wo_type,
    cast(created_on as date)                            as created_on,
    cast(planned_date as date)                          as planned_date,
    cast(nullif(closed_on, '') as date)                 as closed_on
from {{ source('sap_raw', 'work_orders') }}
