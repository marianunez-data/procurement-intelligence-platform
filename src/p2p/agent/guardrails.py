"""Guardrails for the procurement agent.

The agent may only read data. SQL is checked before execution, results are capped, and requests
outside the procurement domain or attempts to override instructions are declined.
"""
from __future__ import annotations

import re

ALLOWED_SCHEMAS = ("marts", "kpis", "reference")
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|copy|export|import|install|load|pragma|set|call|"
    r"read_csv|read_parquet|read_json|read_text|read_blob|glob|httpfs|system)\b",
    re.IGNORECASE,
)
MAX_ROWS = 200

INJECTION_PATTERNS = re.compile(
    r"(ignore (all |any )?(the )?(previous |prior |your |above )?(instructions|rules|guidelines)|you are now|system prompt|"
    r"developer mode|jailbreak|disregard (all |the |your )?(previous )?(rules|instructions)|act as (an?|the) (admin|root))",
    re.IGNORECASE,
)
DOMAIN_TERMS = re.compile(
    r"(procure|purchas|requisition|\bpr\b|\bpo\b|order|supplier|vendor|backlog|lead time|lead-time|deliver|receipt|"
    r"kpi|otif|cash|payment|downpayment|forecast|demand|section|business unit|marine|fleet|port|power|spare|part|"
    r"stock|approval|rfq|quotation|scenario|monte carlo|model|expedit|critical|work order|sql|table|policy|"
    r"cycle time|aged|ageing|aging|spend|price|fx|currency|local|foreign|p2p|planning|data quality|shap|driver|"
    r"line|risk|open|crane|stacker|infrastructure|vehicle|truck|equipment|generator|voltage|switchgear|tug|engine|deck|safety|"
    r"maintenance|warehouse|inventory|eta|arriv|late|expedite|budget|threshold|approve|finance)",
    re.IGNORECASE,
)


class GuardrailError(ValueError):
    """Raised when a request or a tool call violates a guardrail."""


def check_sql(sql: str) -> str:
    """Validate a read-only analytical query and return it with a row limit."""
    s = sql.strip().rstrip(";")
    if ";" in s:
        raise GuardrailError("Only one SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", s, re.IGNORECASE):
        raise GuardrailError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise GuardrailError("The query uses a forbidden keyword or function. The agent has read-only access.")
    tables = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)", s, re.IGNORECASE)
    for schema, _ in tables:
        if schema.lower() not in ALLOWED_SCHEMAS:
            raise GuardrailError(f"Schema '{schema}' is not available to the agent. Use marts, kpis or reference.")
    if not re.search(r"\blimit\s+\d+\s*$", s, re.IGNORECASE):
        s = f"select * from ({s}) as q limit {MAX_ROWS}"
    return s


DESTRUCTIVE = re.compile(r"\b(delete|drop|truncate|remove|update|insert|alter|wipe|overwrite)\b.*\b(table|data|record|records|rows?|database|po|order)s?\b",
                         re.IGNORECASE)


def check_request(text: str) -> str | None:
    """Return a refusal message when the request should not be processed, else None."""
    if DESTRUCTIVE.search(text):
        return "I have read-only access to Quayline's data and cannot change or delete anything."
    if INJECTION_PATTERNS.search(text):
        return ("I cannot change my instructions. I can answer questions about Quayline's procurement data, forecasts, "
                "KPIs and policies.")
    if not DOMAIN_TERMS.search(text):
        return ("I am the Quayline procurement assistant. I can help with backlog, suppliers, lead times, KPIs, forecasts, "
                "cash out, scenarios and procurement policies. That request is outside my scope.")
    return None
