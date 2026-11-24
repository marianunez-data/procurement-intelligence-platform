-- Weekly backlog movement by section and supplier origin.
-- A line is open at the end of a week if it was created on or before the week end and was
-- neither received nor cancelled by then. Built with a range join on a week spine.
with weeks as (
    select distinct week_start, week_start + 6 as week_end
    from {{ ref('dim_date') }}
    where week_start between cast('{{ var("series_start") }}' as date) and cast('{{ var("series_end") }}' as date)
),
lines as (
    select pr_line_key, section_id, supplier_origin, pr_created_on, closed_on, gr_posted_on, cancelled_on, line_value_usd
    from {{ ref('fct_requisition_line') }}
),
grid as (
    select w.week_start, w.week_end, s.section_id, o.supplier_origin
    from weeks w
    cross join (select distinct section_id from lines where section_id is not null) s
    cross join (select distinct supplier_origin from lines) o
),
open_eow as (
    select
        w.week_start, l.section_id, l.supplier_origin,
        count(*)                                                                      as open_lines,
        sum(l.line_value_usd)                                                         as open_value_usd,
        count(*) filter (where date_diff('day', l.pr_created_on, w.week_end) > {{ var('aged_warning_days') }})  as open_aged_90,
        count(*) filter (where date_diff('day', l.pr_created_on, w.week_end) > {{ var('aged_critical_days') }}) as open_aged_150
    from weeks w
    join lines l
      on l.pr_created_on <= w.week_end
     and (l.closed_on is null or l.closed_on > w.week_end)
    group by all
),
created as (
    select {{ week_start('pr_created_on') }} as week_start, section_id, supplier_origin,
           count(*) as created_lines, sum(line_value_usd) as created_value_usd
    from lines group by all
),
received as (
    select {{ week_start('gr_posted_on') }} as week_start, section_id, supplier_origin, count(*) as received_lines
    from lines where gr_posted_on is not null group by all
),
cancelled as (
    select {{ week_start('cancelled_on') }} as week_start, section_id, supplier_origin, count(*) as cancelled_lines
    from lines where cancelled_on is not null group by all
)
select
    g.week_start,
    g.section_id,
    s.section_name,
    s.business_unit,
    g.supplier_origin,
    coalesce(c.created_lines, 0)        as created_lines,
    coalesce(c.created_value_usd, 0)    as created_value_usd,
    coalesce(r.received_lines, 0)       as received_lines,
    coalesce(x.cancelled_lines, 0)      as cancelled_lines,
    coalesce(o.open_lines, 0)           as open_lines,
    coalesce(o.open_value_usd, 0)       as open_value_usd,
    coalesce(o.open_aged_90, 0)         as open_aged_90,
    coalesce(o.open_aged_150, 0)        as open_aged_150
from grid g
left join open_eow o using (week_start, section_id, supplier_origin)
left join created c using (week_start, section_id, supplier_origin)
left join received r using (week_start, section_id, supplier_origin)
left join cancelled x using (week_start, section_id, supplier_origin)
left join {{ ref('dim_section') }} s on s.section_id = g.section_id
