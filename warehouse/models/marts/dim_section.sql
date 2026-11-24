select section_id, section_name, business_unit from {{ ref('stg_sections') }}
