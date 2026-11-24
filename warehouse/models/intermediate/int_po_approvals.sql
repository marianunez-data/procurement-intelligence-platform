-- One row per PO summarising its approval journey (including rejection and resubmission).
select
    po_number,
    any_value(approval_level)                                              as approval_level,
    min(case when approval_cycle = 1 then submitted_on end)                as approval_submitted_on,
    min(case when approval_cycle = 1 then decided_on end)                  as first_decision_on,
    any_value(case when approval_cycle = 1 then decision end)              as first_decision,
    max(case when decision = 'APPROVED' then decided_on end)               as po_approved_on,
    bool_or(decision = 'REJECTED')                                         as was_rejected,
    max(approval_cycle)                                                    as approval_cycles
from {{ ref('stg_po_approvals') }}
group by po_number
