select e.*, s.section_name, s.business_unit
from {{ ref('stg_equipment') }} e
left join {{ ref('stg_sections') }} s using (section_id)
