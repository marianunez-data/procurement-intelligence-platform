select e.*, f.section_id, f.business_unit, f.supplier_origin
from {{ ref('int_status_events') }} e
join {{ ref('fct_requisition_line') }} f using (pr_line_key)
