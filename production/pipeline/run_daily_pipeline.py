from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import overwrite_sheet
from production.pipeline.build_xgb_snapshot import build_xgb_snapshot
from production.pipeline.common import utc_now_iso
from production.pipeline.refresh_sentiment_data import refresh_sentiment_data
from production.sheet_contract import SPREADSHEET_NAME, TAB_PIPELINE_STATUS


def main() -> None:
    generated_at = utc_now_iso()
    xgb_payload = build_xgb_snapshot()
    sentiment_status = refresh_sentiment_data()

    status_row = pd.DataFrame(
        [
            {
                "generated_at_utc": generated_at,
                "spreadsheet_name": SPREADSHEET_NAME,
                "xgb_status": "success",
                "xgb_forecast_date": xgb_payload["forecast_date"],
                "xgb_signal": xgb_payload["signal"],
                "xgb_current_price": xgb_payload["current_price"],
                "xgb_pred_price_t1": xgb_payload["pred_price_final_t1"],
                "sentiment_status": sentiment_status.get("status", "unknown"),
                "sentiment_error": sentiment_status.get("error", ""),
                "sentiment_candidate_rows": sentiment_status.get("candidate_rows", 0),
                "sentiment_newly_scored_rows": sentiment_status.get("newly_scored_rows", 0),
                "sentiment_article_rows": sentiment_status.get("article_rows", 0),
                "sentiment_daily_rows": sentiment_status.get("daily_rows", 0),
                "sentiment_fetch_failed_windows": sentiment_status.get("fetch_failed_windows", 0),
            }
        ]
    )
    overwrite_sheet(status_row, TAB_PIPELINE_STATUS)

    print("production spreadsheet state updated")
    print(f"- spreadsheet        : {SPREADSHEET_NAME}")
    print(f"- xgb forecast date  : {xgb_payload['forecast_date']}")
    print(f"- xgb signal         : {xgb_payload['signal']}")
    print(f"- sentiment status   : {sentiment_status.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
