# Procurement Intelligence Platform

End-to-end procurement and supply chain intelligence for a fictional offshore logistics company:
data generation, analytics warehouse, forecasting, simulation, survival analysis, dashboard and AI agent.

> Work in progress. Quayline Offshore Logistics, its suppliers and all figures are fictional.

## Business problem
Imported spare parts take months to arrive and nobody can say when a specific order will land.
This project builds one trusted view of the procure-to-pay pipeline, forecasts demand, backlog and cash
with honest uncertainty, predicts arrival dates for open orders and simulates management decisions.

## Roadmap
- [x] Project structure and configuration
- [ ] Fictional data generator and data audit (notebooks 00 to 02)
- [ ] dbt + DuckDB warehouse, KPI layer and SQL lab (notebooks 03 to 04)
- [ ] EDA, time-based split and statistical models (notebooks 05 to 07)
- [ ] Machine learning, deep learning, Monte Carlo and model arena (notebooks 08 to 11)
- [ ] Survival analysis, conclusions, dashboard and AI agent (notebooks 12 to 13, app, agent)

## Tech stack
Python, SQL, dbt, DuckDB, statsforecast, LightGBM, PyTorch, lifelines, SHAP, Streamlit, LLM agents, MCP.
