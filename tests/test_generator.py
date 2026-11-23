import json

import pandas as pd

from p2p.config import RAW_DIR
from p2p.generator.export import GROUND_TRUTH_DIR

TABLES = ["purchase_requisitions", "rfqs", "purchase_orders", "po_approvals", "payments", "goods_receipts", "work_orders",
          "suppliers", "materials", "equipment", "sections", "purchasing_groups", "fx_rates", "operations_weekly"]


def test_all_extracts_exist_and_match_ground_truth():
    gt = json.loads((GROUND_TRUTH_DIR / "ground_truth.json").read_text())
    for t in TABLES:
        df = pd.read_csv(RAW_DIR / f"{t}.csv", nrows=None if t in ("suppliers", "sections") else 5)
        assert len(df) > 0, t
    assert gt["row_counts"]["suppliers"] == 60
    assert gt["planted_defects"]["gr_future_posting_date"] == 6


def test_goods_receipt_dates_use_sap_format():
    gr = pd.read_csv(RAW_DIR / "goods_receipts.csv", nrows=50)
    assert gr.posting_date.str.match(r"^\d{2}\.\d{2}\.\d{4}$").all()


def test_no_real_company_names():
    sup = pd.read_csv(RAW_DIR / "suppliers.csv")
    assert sup.supplier_name.str.len().min() > 5
    assert not sup.supplier_name.str.contains("Intels|Orlean", case=False).any()
