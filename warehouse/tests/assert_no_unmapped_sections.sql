-- Every free-text section entry must map to a governed section (extend the seed if this fails).
select * from {{ ref('stg_purchase_requisitions') }} where dq_unmapped_section
