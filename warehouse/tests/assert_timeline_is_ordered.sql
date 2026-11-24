-- Every stage must happen on or after the stage before it (after cleaning rules are applied).
select pr_line_key
from {{ ref('fct_requisition_line') }}
where pr_released_on < pr_created_on
   or rfq_created_on < pr_released_on
   or po_approved_on < po_created_on
   or gr_posted_on < delivered_on
   or delivered_on < po_approved_on
