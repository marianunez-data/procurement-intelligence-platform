"""Central configuration for the Procurement Intelligence Platform project.

Every date, shock and policy parameter used by the data generator lives here, so the
"ground truth" that the models are later tested against is explicit and versioned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
APP_DATA_DIR = DATA_DIR / "app"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DUCKDB_PATH = WAREHOUSE_DIR / "p2p.duckdb"
DBT_DIR = ROOT / "warehouse"
KNOWLEDGE_BASE_DIR = ROOT / "docs" / "knowledge_base"

COMPANY_NAME = "Quayline Offshore Logistics"
CURRENCY = "USD"


@dataclass(frozen=True)
class Shock:
    """A dated event the generator injects into the data."""

    name: str
    start: str
    end: str
    description: str


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20251231
    start_date: str = "2021-06-07"  # simulation warm-up; analysis starts 2022-01-03
    analysis_start: str = "2022-01-03"
    snapshot_date: str = "2025-12-31"  # extraction date: everything after is unknown
    annual_growth: float = 0.08
    md_threshold_before: float = 5_000.0
    md_threshold_after: float = 25_000.0
    policy_change_date: str = "2024-04-01"
    shocks: tuple[Shock, ...] = field(
        default_factory=lambda: (
            Shock(
                "FX liquidity crisis",
                "2023-06-05",
                "2024-03-31",
                "Local currency devalues about 55%; foreign downpayments wait for FX allocation.",
            ),
            Shock(
                "Approval policy change",
                "2024-04-01",
                "2025-12-31",
                "New delegation of authority: MD threshold raised from USD 5k to 25k and e-approvals.",
            ),
            Shock(
                "Port congestion",
                "2024-09-02",
                "2024-10-13",
                "Six weeks of congestion at the import port; foreign deliveries are held.",
            ),
            Shock(
                "New marine contract",
                "2025-03-03",
                "2025-12-31",
                "Two additional tug vessels under contract; Marine demand steps up about 25%.",
            ),
            Shock(
                "Key supplier suspension",
                "2025-06-02",
                "2025-07-27",
                "A top foreign marine supplier suspends shipments for eight weeks.",
            ),
        )
    )

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date)

    @property
    def snapshot(self) -> pd.Timestamp:
        return pd.Timestamp(self.snapshot_date)

    def shock(self, name: str) -> Shock:
        return next(s for s in self.shocks if s.name == name)


# Forecasting split used in every notebook (weeks start on Monday)
TRAIN_END = "2024-12-30"  # last training week start
TEST_START = "2025-01-06"
TEST_END = "2025-12-22"  # last complete week before the extraction date
HORIZONS = (4, 8, 12)
SEASON_LENGTH = 52

CFG = GeneratorConfig()
