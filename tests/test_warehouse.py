import pandas as pd

from conftest import requires_warehouse
from p2p.data import query


@requires_warehouse
def test_backlog_stock_and_flow_identity():
    w = query("""select week_start, sum(open_lines) o, sum(created_lines) c, sum(received_lines) r, sum(cancelled_lines) x
                 from marts.fct_backlog_weekly group by 1 order by 1""")
    lhs = w.o.iloc[1:].to_numpy()
    rhs = (w.o.shift(1) + w.c - w.r - w.x).iloc[1:].to_numpy()
    assert (lhs == rhs).all()


@requires_warehouse
def test_kpi_engine_matches_dbt():
    from p2p.kpis.engine import compute
    for month in ["2023-06-01", "2024-11-01", "2025-12-01"]:
        m = pd.Timestamp(month)
        eng = compute(str(m.date()), str((m + pd.offsets.MonthEnd(0)).date())).set_index("kpi_id").value.astype(float)
        dbt = query("select kpi_id, kpi_value from kpis.kpi_monthly_company where month_start = ?", [month]).set_index("kpi_id").kpi_value
        both = pd.concat([eng, dbt], axis=1, keys=["engine", "dbt"]).dropna()
        assert len(both) >= 20
        assert (both.engine - both.dbt).abs().max() < 1e-6, both[(both.engine - both.dbt).abs() > 1e-6]


@requires_warehouse
def test_feature_store_has_no_leakage():
    from p2p.config import DBT_DIR
    from p2p.data import connect
    compiled = (DBT_DIR / "target" / "compiled" / "p2p" / "models" / "marts" / "ml_features_weekly.sql").read_text()
    cutoff = "2024-06-03"
    sql = compiled.replace('"p2p"."marts"."fct_weekly_section"', f"(select * from marts.fct_weekly_section where week_start <= date '{cutoff}')")
    with connect() as con:
        full = con.execute("select * from marts.ml_features_weekly").df()
        trunc = con.execute(sql).df()
    cols = [c for c in full.columns if c.startswith(("lag_", "roll_")) or c == "y_t"]
    a = full[full.origin_week <= pd.Timestamp(cutoff)].set_index(["section_id", "origin_week"])[cols].sort_index()
    b = trunc.set_index(["section_id", "origin_week"])[cols].sort_index()
    assert ((a - b).abs().fillna(0) < 1e-9).all().all()
