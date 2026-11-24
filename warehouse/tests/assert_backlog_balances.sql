-- Stock-and-flow check: open(t) = open(t-1) + created(t) - received(t) - cancelled(t)
with w as (
    select week_start,
           sum(open_lines) as open_lines, sum(created_lines) as created,
           sum(received_lines) as received, sum(cancelled_lines) as cancelled
    from {{ ref('fct_backlog_weekly') }}
    group by week_start
),
chk as (
    select *, lag(open_lines) over (order by week_start) as prev_open from w
)
select * from chk
where prev_open is not null and open_lines <> prev_open + created - received - cancelled
