# Architecture

## Data flow

```mermaid
flowchart TB
    subgraph Generation
        G1[config.py<br/>dates, shocks, policies] --> G2[master_data.py<br/>sections, suppliers, materials, equipment]
        G2 --> G3[simulate.py<br/>indicators, demand, lifecycle]
        G3 --> G4[export.py<br/>SAP-like CSV extracts + planted defects]
        G3 --> G5[ground_truth/<br/>uncensored lifecycle, defect counts]
    end
    subgraph Warehouse[dbt + DuckDB]
        W1[sources: read_csv, all text] --> W2[staging: types, de-duplication, aliases, flags]
        W2 --> W3[intermediate: receipts, approvals, payments, lifecycle, status events]
        W3 --> W4[marts: fct_requisition_line, fct_backlog_weekly, fct_weekly_section, fct_payment, fct_work_order, dims]
        W4 --> W5[kpis: kpi_monthly_company + kpi_catalog seed]
        W4 --> W6[ml_features_weekly: SQL feature store]
    end
    subgraph Science[src/p2p/forecasting + survival]
        S1[backtest.py: rolling origin] --> S2[models_stats / models_ml / models_dl / montecarlo]
        S2 --> S3[metrics.py, selection.py, hierarchy.py]
        S4[leadtime.py: Kaplan-Meier, Cox, AFT]
    end
    subgraph Serving
        P[production.py: forecasts, ETAs, SHAP, app database] --> A[Streamlit app]
        P --> AG[Agent tools] --> A
        AG --> M[MCP server]
    end
    G4 --> W1
    W4 --> S1
    W6 --> S2
    W4 --> S4
    S3 --> P
    S4 --> P
    G5 -. validation only .-> S4
```

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Engine | DuckDB | Columnar analytics in one file: runs in CI, notebooks and the app with no server |
| Transformations | dbt | Versioned SQL, tests, documentation and lineage; portable to BigQuery or Snowflake |
| Status logic | Derived from documents | The SAP status field was proven unreliable in the audit |
| Feature store | SQL window functions | One source of feature logic; leakage audited empirically |
| Forecast strategy | Direct multi-horizon for ML | No error accumulation across the 12 weeks |
| Uncertainty | Native intervals, conformal for ML, simulation quantiles for Monte Carlo | Every model is scored on calibration, not only accuracy |
| Model choice | Weighted scorecard | Makes trade-offs explicit and adjustable by stakeholders |
| Lead time | Survival analysis | Open orders are censored observations; ignoring them biases estimates |
| App data | Compact DuckDB copy + parquet artifacts | The deployed app needs no warehouse rebuild and no heavy ML stack |
| Agent | OpenAI-compatible tool calling + offline router | Provider-agnostic, free-tier friendly, never breaks in a demo |

## Layer contracts

* Raw extracts are never modified; every repair happens in dbt and leaves a `dq_` flag.
* Marts are the only schemas exposed to the app, the agent and the MCP server (`marts`, `kpis`, `reference`).
* The KPI engine used by the app is contract-tested against the dbt KPI table (`tests/test_warehouse.py`).
