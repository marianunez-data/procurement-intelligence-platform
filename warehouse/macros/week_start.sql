{# Monday that starts the ISO week of a date #}
{% macro week_start(col) -%}
    cast(date_trunc('week', {{ col }}) as date)
{%- endmacro %}
