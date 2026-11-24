-- Share-type KPIs must stay within 0 and 1. Purchase price variance (K19) is a signed % and is excluded.
select * from {{ ref('kpi_monthly_company') }}
where unit = '%' and kpi_id <> 'K19' and kpi_value is not null and (kpi_value < 0 or kpi_value > 1)
