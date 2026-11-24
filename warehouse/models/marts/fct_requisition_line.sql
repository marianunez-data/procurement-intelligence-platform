-- Main fact table: one row per requisition line (grain: pr_line_key).
select
    l.*,
    s.section_name,
    s.business_unit,
    {{ week_start('l.pr_created_on') }}                 as pr_week_start,
    {{ week_start('l.gr_posted_on') }}                  as receipt_week_start
from {{ ref('int_requisition_lifecycle') }} l
left join {{ ref('stg_sections') }} s using (section_id)
