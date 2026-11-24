with spine as (
    select cast(d as date) as date_day
    from range(date '2021-12-01', date '2027-01-01', interval 1 day) t(d)
)
select
    date_day,
    {{ week_start('date_day') }}                         as week_start,
    cast(date_trunc('month', date_day) as date)          as month_start,
    cast(date_trunc('quarter', date_day) as date)        as quarter_start,
    year(date_day)                                       as year,
    quarter(date_day)                                    as quarter,
    month(date_day)                                      as month,
    weekofyear(date_day)                                 as iso_week,
    dayofweek(date_day)                                  as day_of_week,
    strftime(date_day, '%b %Y')                          as month_label,
    (month(date_day) = 12 and day(date_day) >= 18) or (month(date_day) = 1 and day(date_day) <= 6) as is_shutdown_period,
    month(date_day) between 5 and 9                      as is_rainy_season,
    month(date_day) in (10, 11)                          as is_budget_push
from spine
