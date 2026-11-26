"""Tools available to the procurement agent (and exposed through the MCP server).

Each tool is a plain Python function with a JSON schema. Tools return compact, JSON-serialisable
results so the language model only sees what it needs, and every number it quotes can be traced.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from p2p.agent.guardrails import check_sql
from p2p.config import APP_DATA_DIR, KNOWLEDGE_BASE_DIR
from p2p.data import query

ART = APP_DATA_DIR


def _art(name: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"{name}.parquet")


def _records(df: pd.DataFrame, n: int = 50) -> list[dict]:
    df = df.head(n).copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(3)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _section_id(name: str | None) -> str | None:
    if not name or name.lower() in ("company", "all", "total"):
        return None
    secs = query("select section_id, section_name from marts.dim_section")
    for r in secs.itertuples():
        if name.lower() in (r.section_id.lower(), r.section_name.lower()) or name.lower() in r.section_name.lower():
            return r.section_id
    return None


# ---------------------------------------------------------------- data access
def query_warehouse(sql: str) -> dict:
    """Run a read-only SQL query on the warehouse (schemas marts, kpis, reference)."""
    safe = check_sql(sql)
    df = query(safe)
    return {"sql": safe, "row_count": len(df), "rows": _records(df, 50)}


def describe_schema() -> dict:
    """List the analytical tables and their columns."""
    cols = query("""
        select table_schema || '.' || table_name as table_name, string_agg(column_name, ', ' order by ordinal_position) as columns
        from information_schema.columns where table_schema in ('marts', 'kpis', 'reference')
        group by 1 order by 1
    """)
    return {"tables": _records(cols, 40)}


# ---------------------------------------------------------------- KPIs
def get_kpis(period_start: str = "2025-12-01", period_end: str = "2025-12-31", business_unit: str | None = None,
             supplier_origin: str | None = None) -> dict:
    """Governed KPIs for a period, optionally filtered by business unit or supplier origin."""
    from p2p.kpis.engine import Filters, compute
    f = Filters(business_units=[business_unit] if business_unit else [], origins=[supplier_origin] if supplier_origin else [])
    k = compute(period_start, period_end, f)[["kpi_id", "kpi_name", "unit", "value", "target", "status"]]
    return {"period": [period_start, period_end], "filters": {"business_unit": business_unit, "supplier_origin": supplier_origin},
            "kpis": _records(k)}


def explain_kpi_change(kpi_id: str, month: str = "2025-12-01") -> dict:
    """Compare a KPI with the previous month and break the change down by business unit and supplier origin."""
    from p2p.kpis.engine import Filters, compute
    m = pd.Timestamp(month).to_period("M")
    cur = (str(m.start_time.date()), str(m.end_time.date()))
    prev = (str((m - 1).start_time.date()), str((m - 1).end_time.date()))
    total = {lbl: float(compute(*p).set_index("kpi_id").loc[kpi_id, "value"] or np.nan) for lbl, p in [("previous", prev), ("current", cur)]}
    rows = []
    for dim, values in [("business_unit", ["Port Operations", "Fleet and Workshop", "Power Generation", "Marine Services"]),
                        ("supplier_origin", ["Local", "Foreign"])]:
        for v in values:
            f = Filters(business_units=[v]) if dim == "business_unit" else Filters(origins=[v])
            a = compute(*prev, f).set_index("kpi_id").loc[kpi_id, "value"]
            b = compute(*cur, f).set_index("kpi_id").loc[kpi_id, "value"]
            rows.append({"dimension": dim, "segment": v, "previous": a, "current": b,
                         "change": (b - a) if a is not None and b is not None and pd.notna(a) and pd.notna(b) else None})
    name = query("select kpi_name, unit, direction from reference.kpi_catalog where kpi_id = ?", [kpi_id]).iloc[0].to_dict()
    return {"kpi": kpi_id, **name, "month": cur[0], "company": total, "breakdown": _records(pd.DataFrame(rows))}


# ---------------------------------------------------------------- forecasts
def forecast_demand(section: str | None = None) -> dict:
    """12-week forecast of requisition lines (champion model) for a section or the company."""
    fc = _art("demand_forecast")
    sid = _section_id(section)
    uid = sid or "Company"
    f = fc[fc.unique_id == uid].sort_values("ds")
    return {"series": uid, "model": f.model.iloc[0], "origin": str(f.ds.min().date() - pd.Timedelta(days=7)),
            "next_4_weeks_total": round(float(f.yhat.head(4).sum()), 1), "next_12_weeks_total": round(float(f.yhat.sum()), 1),
            "weekly": _records(f[["ds", "yhat", "q0.1", "q0.9"]], 12)}


def forecast_backlog_and_cash() -> dict:
    """Monte Carlo forecast of open backlog lines and weekly cash out for the next 12 weeks (baseline scenario)."""
    fc = _art("process_forecast")
    out = {}
    for metric, g in fc.groupby("metric"):
        g = g.sort_values("ds")
        out[metric] = _records(g[["ds", "yhat", "q0.1", "q0.9"]], 12)
    return {"model": "Monte Carlo (process)", "forecast": out}


def explain_forecast(section: str) -> dict:
    """Top drivers (SHAP) behind the 4-weeks-ahead LightGBM forecast for a section."""
    sh = _art("shap_drivers")
    sid = _section_id(section)
    s = sh[sh.section_id == sid].sort_values("abs_effect", ascending=False).head(6)
    return {"section": section, "baseline_lines": round(float(s.baseline.iloc[0]), 1), "forecast_lines": round(float(s.prediction.iloc[0]), 1),
            "drivers": _records(s[["feature", "value", "effect_pct"]])}


def recommend_model(use_case: str) -> dict:
    """Which forecasting approach to trust for a planning use case, with the evidence from the backtests."""
    cards = json.loads((ART / "selection_cards.json").read_text())
    text = use_case.lower()
    best = max(cards, key=lambda c: sum(w in text for w in c["keywords"]))
    return best


# ---------------------------------------------------------------- lead time
def lead_time_for(po_number: str | None = None, pr_line_key: str | None = None) -> dict:
    """Predicted arrival (P50 and P90 dates) for an open PO or requisition line, from the survival model."""
    eta = _art("open_line_eta")
    if pr_line_key:
        r = eta[eta.pr_line_key == pr_line_key]
    elif po_number:
        r = eta[eta.po_number.astype(str) == str(po_number).split(".")[0]]
    else:
        return {"error": "Provide a PO number or a requisition line key."}
    if r.empty:
        return {"error": "No open line found with that reference. It may be received, cancelled or not yet ordered."}
    return {"lines": _records(r[["pr_line_key", "po_number", "section_name", "supplier_name", "supplier_origin", "lifecycle_status",
                                 "age_days", "eta_p50", "eta_p90", "risk_over_150_days", "is_critical"]], 20)}


def at_risk_lines(section: str | None = None, top: int = 10, critical_only: bool = False) -> dict:
    """Open lines with the highest risk of exceeding 150 days, optionally for one section."""
    eta = _art("open_line_eta")
    if section:
        sid = _section_id(section)
        eta = eta[eta.section_id == sid]
    if critical_only:
        eta = eta[eta.is_critical]
    r = eta.sort_values(["risk_over_150_days", "age_days"], ascending=False).head(int(top))
    return {"count_open": int(len(eta)), "lines": _records(r[["pr_line_key", "po_number", "section_name", "supplier_name", "supplier_origin",
                                                             "lifecycle_status", "age_days", "eta_p50", "eta_p90", "risk_over_150_days", "is_critical"]], top)}


# ---------------------------------------------------------------- scenarios
@lru_cache(maxsize=1)
def _baseline_state():
    from p2p.forecasting.montecarlo import build_state
    from p2p.data import weekly_section
    asof = pd.Timestamp(weekly_section().week_start.max()) + pd.Timedelta(days=6)
    return build_state(asof), asof


def run_scenario(approval_time: float = 1.0, downpayment_time: float = 1.0, buyer_speed: float = 1.0,
                 foreign_lead_time: float = 1.0, local_share_change: float = 0.0, demand_change: float = 0.0,
                 extra_downpayment_days_foreign: float = 0.0, n_sims: int = 400) -> dict:
    """Monte Carlo what-if: compare a scenario with the baseline over 12 weeks (backlog, receipts, cash out)."""
    from p2p.forecasting.montecarlo import DemandSimulator, ProcessSimulator, Scenario
    from p2p.data import weekly_section
    state, asof = _baseline_state()
    ws = weekly_section()
    fut = future_exog_frame(ws)
    arrivals = {}
    for lbl, dc in [("baseline", 0.0), ("scenario", demand_change)]:
        arrivals[lbl] = sum(DemandSimulator(ws).simulate(fut, n_sims=n_sims, seed=7, demand_change=dc).values())
    sc = Scenario("Scenario", buyer_speed=buyer_speed, approval_time=approval_time, downpayment_time=downpayment_time,
                  foreign_lead_time=foreign_lead_time, local_share_change=local_share_change, demand_change=demand_change,
                  extra_downpayment_days_foreign=extra_downpayment_days_foreign)
    res = {}
    for lbl, scen in [("baseline", Scenario()), ("scenario", sc)]:
        r = ProcessSimulator(state, n_sims=n_sims, seed=11).run(12, arrivals[lbl], scen)
        res[lbl] = {"open_lines_week12_p50": float(np.median(r["open_lines"][-1])),
                    "open_lines_week12_p10": float(np.quantile(r["open_lines"][-1], 0.1)),
                    "open_lines_week12_p90": float(np.quantile(r["open_lines"][-1], 0.9)),
                    "receipts_12w_mean": float(r["receipts"].sum(axis=0).mean()),
                    "cash_out_12w_mean_usd": float(r["cash_out"].sum(axis=0).mean())}
    delta = {k: res["scenario"][k] - res["baseline"][k] for k in res["baseline"]}
    return {"as_of": str(asof.date()), "levers": sc.__dict__, "baseline": res["baseline"], "scenario": res["scenario"], "difference": delta}


def future_exog_frame(ws: pd.DataFrame, h: int = 12) -> pd.DataFrame:
    """Future weeks with the planned-maintenance load. In production this comes from the maintenance plan in the CMMS;
    here the average of 8 recent complete weeks per section is used beyond the extraction date."""
    last = ws.week_start.max()
    weeks = pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq="W-MON")
    # the last 3 weeks are excluded: their 3-week look-ahead runs past the extraction date and is truncated
    window = (ws.week_start > last - pd.Timedelta(weeks=11)) & (ws.week_start <= last - pd.Timedelta(weeks=3))
    recent = ws[window].groupby("section_id").pm_planned_next_3w.mean()
    return pd.DataFrame([{"section_id": s, "week_start": w, "pm_planned_next_3w": v} for s, v in recent.items() for w in weeks])


# ---------------------------------------------------------------- policies (RAG)
@lru_cache(maxsize=1)
def _kb():
    from sklearn.feature_extraction.text import TfidfVectorizer
    chunks = []
    for p in sorted(Path(KNOWLEDGE_BASE_DIR).glob("*.md")):
        text = p.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        for part in re.split(r"\n(?=## )", text):
            body = [ln for ln in part.strip().splitlines() if ln.strip() and not ln.startswith("#")]
            if sum(len(ln) for ln in body) > 80:  # skip chunks that are only a heading or a version line
                chunks.append({"source": p.name, "title": title, "text": part.strip()})
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform([c["title"] + " " + c["text"] for c in chunks])
    return chunks, vec, mat


def search_policies(question: str, top_k: int = 3) -> dict:
    """Search Quayline's procurement policies and procedures. Returns the most relevant passages with their source."""
    chunks, vec, mat = _kb()
    scores = (mat @ vec.transform([question]).T).toarray().ravel()
    idx = np.argsort(-scores)[:top_k]
    return {"passages": [{"source": chunks[i]["source"], "document": chunks[i]["title"], "score": round(float(scores[i]), 3),
                          "text": chunks[i]["text"][:1200]} for i in idx if scores[i] > 0]}


# ---------------------------------------------------------------- actions (draft only)
def draft_email(recipient_role: str, subject: str, key_points: list[str]) -> dict:
    """Draft an email for a person to review and send. The agent never sends messages."""
    body = "\n\n".join(["Hello,", *[p.strip().rstrip(".") + "." for p in key_points],
                        "Please confirm the updated date or let me know if you need anything from our side.",
                        "Thanks and I remain available.\n\nPlanning Department\nQuayline Offshore Logistics"])
    return {"status": "draft (not sent)", "to": recipient_role, "subject": subject, "body": body}


# ---------------------------------------------------------------- registry
TOOLS = {
    "query_warehouse": (query_warehouse, {"sql": {"type": "string", "description": "A single read-only SELECT on schemas marts, kpis or reference."}}, ["sql"]),
    "describe_schema": (describe_schema, {}, []),
    "get_kpis": (get_kpis, {
        "period_start": {"type": "string", "description": "YYYY-MM-DD"}, "period_end": {"type": "string", "description": "YYYY-MM-DD"},
        "business_unit": {"type": "string", "enum": ["Port Operations", "Fleet and Workshop", "Power Generation", "Marine Services"]},
        "supplier_origin": {"type": "string", "enum": ["Local", "Foreign"]}}, []),
    "explain_kpi_change": (explain_kpi_change, {"kpi_id": {"type": "string", "description": "K01..K22"},
                                                "month": {"type": "string", "description": "First day of the month, YYYY-MM-01"}}, ["kpi_id"]),
    "forecast_demand": (forecast_demand, {"section": {"type": "string", "description": "Section name, or 'Company'"}}, []),
    "forecast_backlog_and_cash": (forecast_backlog_and_cash, {}, []),
    "explain_forecast": (explain_forecast, {"section": {"type": "string"}}, ["section"]),
    "recommend_model": (recommend_model, {"use_case": {"type": "string"}}, ["use_case"]),
    "lead_time_for": (lead_time_for, {"po_number": {"type": "string"}, "pr_line_key": {"type": "string"}}, []),
    "at_risk_lines": (at_risk_lines, {"section": {"type": "string"}, "top": {"type": "integer"}, "critical_only": {"type": "boolean"}}, []),
    "run_scenario": (run_scenario, {
        "approval_time": {"type": "number", "description": "Multiplier on PO approval time, e.g. 0.5 = twice as fast"},
        "downpayment_time": {"type": "number", "description": "Multiplier on downpayment processing time"},
        "buyer_speed": {"type": "number", "description": "Multiplier on RFQ and quotation time"},
        "foreign_lead_time": {"type": "number", "description": "Multiplier on foreign supplier lead time"},
        "local_share_change": {"type": "number", "description": "Change in local sourcing share, e.g. 0.1 = +10 points"},
        "demand_change": {"type": "number", "description": "Change in requisition volume, e.g. 0.15 = +15%"},
        "extra_downpayment_days_foreign": {"type": "number", "description": "Extra days on foreign downpayments (FX stress)"}}, []),
    "search_policies": (search_policies, {"question": {"type": "string"}}, ["question"]),
    "draft_email": (draft_email, {"recipient_role": {"type": "string"}, "subject": {"type": "string"},
                                  "key_points": {"type": "array", "items": {"type": "string"}}}, ["recipient_role", "subject", "key_points"]),
}


def openai_tool_specs() -> list[dict]:
    specs = []
    for name, (fn, props, required) in TOOLS.items():
        specs.append({"type": "function", "function": {
            "name": name, "description": (fn.__doc__ or "").strip().split("\n")[0],
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return specs


def call_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool {name}")
    return TOOLS[name][0](**(args or {}))
