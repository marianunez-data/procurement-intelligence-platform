select
    cast(po_number as bigint)                           as po_number,
    approval_level,
    cast(approval_cycle as integer)                     as approval_cycle,
    cast(submitted_on as date)                          as submitted_on,
    cast(nullif(decided_on, '') as date)                as decided_on,
    decision
from {{ source('sap_raw', 'po_approvals') }}
