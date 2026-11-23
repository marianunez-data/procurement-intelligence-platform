"""Master data for the fictional company: business units, sections, suppliers,
materials, equipment and purchasing groups. All names are invented."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Organisation
# ---------------------------------------------------------------------------
# base_lines: average PR lines per week in 2022; foreign: share of foreign sourcing;
# rain_amp: amplitude of the rainy-season effect on corrective maintenance;
# growth: section-specific annual growth on top of the company trend.
SECTIONS = pd.DataFrame(
    [
        ("SEC01", "Cranes and Lifting", "Port Operations", 14, 0.75, 0.30, 0.00),
        ("SEC02", "Reach Stackers", "Port Operations", 10, 0.70, 0.25, 0.01),
        ("SEC03", "Port Infrastructure", "Port Operations", 6, 0.30, 0.35, 0.00),
        ("SEC04", "Light Vehicles", "Fleet and Workshop", 16, 0.35, 0.40, 0.04),
        ("SEC05", "Heavy Duty Trucks", "Fleet and Workshop", 12, 0.55, 0.35, 0.02),
        ("SEC06", "Minor Equipment", "Fleet and Workshop", 8, 0.40, 0.20, 0.00),
        ("SEC07", "Generators", "Power Generation", 10, 0.80, 0.30, 0.01),
        ("SEC08", "Low Voltage Electrical", "Power Generation", 12, 0.45, 0.45, 0.00),
        ("SEC09", "HV Switchgear", "Power Generation", 4, 0.85, 0.25, -0.03),
        ("SEC10", "Tugboats", "Marine Services", 9, 0.80, 0.10, 0.00),
        ("SEC11", "Engine Room", "Marine Services", 7, 0.85, 0.10, 0.00),
        ("SEC12", "Deck and Safety", "Marine Services", 6, 0.50, 0.15, 0.00),
    ],
    columns=["section_id", "section_name", "business_unit", "base_lines", "foreign_share", "rain_amp", "growth"],
)

# How the same section name appears in messy manual entries (used for data-quality defects)
SECTION_ALIASES = {
    "Cranes and Lifting": ["CRANES & LIFTING", "Cranes & Lifting", "cranes and lifting "],
    "Reach Stackers": ["REACH STACKERS", "Reach Stacker"],
    "Port Infrastructure": ["PORT INFRA", "Port infrastructure"],
    "Light Vehicles": ["LIGHT VEHICLE", "Light Vehicle", " light vehicles"],
    "Heavy Duty Trucks": ["HEAVY DUTY", "Heavy Duty"],
    "Minor Equipment": ["MINOR EQUIPMENT", "Minor Equip."],
    "Generators": ["GENERATORS", "Generator"],
    "Low Voltage Electrical": ["LOW VOLTAGE", "Low Voltage", "LV Electrical"],
    "HV Switchgear": ["HV SWITCHGEAR", "H.V. Switchgear"],
    "Tugboats": ["TUGBOATS", "Tug Boats"],
    "Engine Room": ["ENGINE ROOM", "Engine room"],
    "Deck and Safety": ["DECK & SAFETY", "Deck/Safety"],
}

PURCHASING_GROUPS = pd.DataFrame(
    [
        ("PG01", "Spares Foreign A", "Foreign"),
        ("PG02", "Spares Foreign B", "Foreign"),
        ("PG03", "Spares Local", "Local"),
        ("PG04", "Electrical and Power", "Both"),
        ("PG05", "Marine Procurement", "Both"),
        ("PG06", "Projects and Capex", "Both"),
    ],
    columns=["purchasing_group_id", "purchasing_group_name", "scope"],
)

# category: (median unit price USD, price sigma, qty low, qty high, lead-time factor)
CATEGORIES = {
    "Filters": (45, 0.6, 2, 24, 0.9),
    "Bearings and Seals": (120, 0.8, 1, 10, 1.0),
    "Hydraulics": (650, 0.9, 1, 4, 1.15),
    "Electrical Components": (220, 1.0, 1, 12, 1.0),
    "Engine Parts": (900, 1.0, 1, 3, 1.2),
    "Tyres and Tracks": (480, 0.5, 2, 8, 0.9),
    "Safety and PPE": (35, 0.7, 5, 60, 0.7),
    "Consumables": (18, 0.8, 10, 120, 0.6),
    "Tools": (160, 0.8, 1, 6, 0.8),
    "Electronics and Controls": (1400, 1.0, 1, 2, 1.3),
}

# Category mix per business unit (probabilities, same order as CATEGORIES)
CATEGORY_MIX = {
    "Port Operations": [0.12, 0.14, 0.20, 0.10, 0.10, 0.06, 0.08, 0.10, 0.05, 0.05],
    "Fleet and Workshop": [0.20, 0.12, 0.08, 0.08, 0.14, 0.16, 0.06, 0.10, 0.04, 0.02],
    "Power Generation": [0.12, 0.08, 0.04, 0.30, 0.14, 0.00, 0.08, 0.10, 0.06, 0.08],
    "Marine Services": [0.12, 0.12, 0.10, 0.10, 0.18, 0.00, 0.14, 0.12, 0.06, 0.06],
}

FOREIGN_COUNTRIES = ["Germany", "Italy", "Netherlands", "United Arab Emirates", "China",
                     "United States", "United Kingdom", "Singapore", "South Africa", "Norway"]

_SUPPLIER_STEMS = [
    "Nordhavn", "Castellane", "Veltrix", "Halvorsund", "Ostrava Line", "Sundara", "Marlowe Gate",
    "Redfenn", "Altamira Bay", "Kestrelton", "Harrowmere", "Tidewell", "Ironvale", "Callowind",
    "Solvane", "Brennholt", "Quayside Atlas", "Vantoria", "Lumora", "Grevik", "Asterline",
    "Corvane", "Pellucid", "Darnholm", "Zephyra", "Okavo", "Benwick", "Torvald", "Meroway",
    "Ashcombe", "Valdera", "Kintora", "Selbourne", "Rivenna", "Hollin", "Mabrey", "Pentravis",
    "Orinoco Ridge", "Wexmoor", "Lyndhurst Vale", "Esperanza Coast", "Tamsin", "Calderon Works",
    "Brightfold", "Norrland", "Avonlea", "Sorrento Field", "Yarrowby", "Fenwright", "Ondura",
    "Graniteholm", "Westerly", "Belvista", "Coralmont", "Dunmere", "Ivorybridge", "Kalhari",
    "Pinewater", "Rosslare Point", "Talvik",
]
_SUPPLIER_SUFFIX = {
    "Marine": ["Marine Systems", "Marine Supply", "Ship Spares"],
    "Power": ["Power Solutions", "Energy Systems", "Electrical"],
    "General": ["Industrial Supply", "Engineering", "Trading", "Spares and Services", "Hydraulics"],
}


def build_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    """60 fictional suppliers: 36 foreign and 24 local, each with a lead-time profile."""
    rows = []
    focus_cycle = ["Marine", "Power", "General", "General"]
    for i, stem in enumerate(_SUPPLIER_STEMS):
        foreign = i < 36
        focus = focus_cycle[i % 4]
        suffix = rng.choice(_SUPPLIER_SUFFIX[focus])
        name = f"{stem} {suffix}" + (" GmbH" if foreign and i % 7 == 0 else "") + (" Ltd" if not foreign else "")
        if foreign:
            country = FOREIGN_COUNTRIES[i % len(FOREIGN_COUNTRIES)]
            base_lt = float(rng.uniform(45, 110))
            reliability = float(np.clip(rng.normal(0.72, 0.12), 0.35, 0.95))
            dp_required = bool(rng.random() < 0.65)
        else:
            country = "Domestic"
            base_lt = float(rng.uniform(8, 28))
            reliability = float(np.clip(rng.normal(0.80, 0.10), 0.45, 0.97))
            dp_required = bool(rng.random() < 0.15)
        rows.append(
            dict(
                supplier_id=f"V{10001 + i}",
                supplier_name=name,
                supplier_origin="Foreign" if foreign else "Local",
                country=country,
                focus=focus,
                quoted_lead_time_days=round(base_lt),
                reliability=round(reliability, 3),
                downpayment_required=dp_required,
                downpayment_pct=float(rng.choice([0.3, 0.4, 0.5])) if dp_required else 0.0,
                payment_terms_days=int(rng.choice([30, 45, 60])),
            )
        )
    return pd.DataFrame(rows)


def build_materials(rng: np.random.Generator, n: int = 1500) -> pd.DataFrame:
    """Spare-part catalogue. Each material belongs to one section and has a demand profile."""
    cats = list(CATEGORIES)
    sec = SECTIONS.sample(n=n, replace=True, weights=SECTIONS["base_lines"], random_state=7).reset_index(drop=True)
    rows = []
    for i, s in sec.iterrows():
        cat = rng.choice(cats, p=CATEGORY_MIX[s.business_unit])
        med, sig, qlo, qhi, ltf = CATEGORIES[cat]
        price = float(np.round(med * rng.lognormal(0, sig), 2))
        rows.append(
            dict(
                material_id=f"M{300000 + i}",
                short_text=f"{cat.split(' ')[0].upper()} {s.section_name.split(' ')[0].upper()}-{i:04d}",
                category=cat,
                section_id=s.section_id,
                standard_price_usd=max(price, 1.5),
                qty_low=qlo,
                qty_high=qhi,
                lead_time_factor=ltf,
                uom=rng.choice(["EA", "EA", "EA", "SET", "PC", "L"] if cat == "Consumables" else ["EA", "EA", "SET", "PC"]),
                popularity=float(rng.pareto(1.3) + 0.05),  # long tail: few materials drive most demand
            )
        )
    return pd.DataFrame(rows)


def build_equipment(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    counts = {"SEC01": 14, "SEC02": 18, "SEC03": 10, "SEC04": 60, "SEC05": 28, "SEC06": 24,
              "SEC07": 16, "SEC08": 12, "SEC09": 6, "SEC10": 6, "SEC11": 6, "SEC12": 6}
    k = 0
    for sec_id, c in counts.items():
        for j in range(c):
            crit = rng.choice(["A", "B", "C"], p=[0.25, 0.45, 0.30])
            rows.append(dict(equipment_id=f"EQ{5000 + k}", section_id=sec_id, criticality=crit,
                             description=f"{sec_id}-UNIT-{j + 1:03d}"))
            k += 1
    return pd.DataFrame(rows)
