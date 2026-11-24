-- One clean row per PR line: exact duplicates removed, section text mapped to a section id.
with source as (
    select * from {{ source('sap_raw', 'purchase_requisitions') }}
),
deduplicated as (
    select *
    from source
    qualify row_number() over (partition by pr_number, pr_item order by created_on) = 1
)
select
    cast(d.pr_number as bigint)                         as pr_number,
    cast(d.pr_item as integer)                          as pr_item,
    cast(d.pr_number as varchar) || '-' || d.pr_item    as pr_line_key,
    cast(d.created_on as date)                          as pr_created_on,
    cast(nullif(d.released_on, '') as date)             as pr_released_on,
    a.section_id,
    d.section                                           as section_raw_text,
    d.material_id,
    d.short_text,
    cast(d.quantity as integer)                         as quantity,
    d.uom,
    cast(d.estimated_price_usd as double)               as estimated_price_usd,
    cast(d.quantity as integer) * cast(d.estimated_price_usd as double) as estimated_value_usd,
    d.reason_code,
    nullif(d.work_order_id, '')                         as work_order_id,
    nullif(d.equipment_id, '')                          as equipment_id,
    d.priority = '1-Critical'                           as is_critical,
    d.processing_status                                 as sap_processing_status,
    cast(nullif(d.status_changed_on, '') as date)       as cancelled_on,
    a.section_id is null                                as dq_unmapped_section
from deduplicated d
left join {{ ref('section_aliases') }} a
    on a.raw_section_text = lower(trim(d.section))
