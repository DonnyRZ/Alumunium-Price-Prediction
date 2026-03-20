from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import overwrite_sheet, read_sheet
from production.pipeline.build_ai_brief import build_ai_brief
from production.pipeline.build_signal_snapshot import build_signal_snapshot
from production.pipeline.common import utc_now_iso
from production.pipeline.refresh_sentiment_data import refresh_sentiment_data
from production.sheet_contract import (
    SPREADSHEET_NAME,
    TAB_ACCOUNT_PRIORITY_QUEUE,
    TAB_QUOTE_DECISION_LOG,
    TAB_RUN_STATUS,
)


def _seed_if_empty(tab_name: str, columns: list[str]) -> None:
    existing = read_sheet(tab_name)
    if not existing.empty:
        return
    overwrite_sheet(pd.DataFrame(columns=columns), tab_name)


def main() -> None:
    generated_at = utc_now_iso()
    print("[1/5] build signal snapshot...")
    signal_payload = build_signal_snapshot(write_sheets=True)
    print("[2/5] refresh sentiment context...")
    sentiment_status = refresh_sentiment_data()
    print("[3/5] build AI brief...")
    ai_payload = build_ai_brief(write_sheets=True)

    print("[4/5] seed optional operational sheets...")
    _seed_if_empty(
        TAB_ACCOUNT_PRIORITY_QUEUE,
        [
            "account_name",
            "account_tier",
            "open_inquiry_flag",
            "active_quote_flag",
            "signal_latest",
            "recommended_priority",
            "last_contact_date",
            "action_note",
        ],
    )
    _seed_if_empty(
        TAB_QUOTE_DECISION_LOG,
        [
            "date",
            "account_name",
            "signal_level",
            "probability_action",
            "decision_taken",
            "quote_validity",
            "follow_up_speed",
            "outcome_note",
        ],
    )

    print("[5/5] write run status...")
    status_row = pd.DataFrame(
        [
            {
                "generated_at_utc": generated_at,
                "spreadsheet_name": SPREADSHEET_NAME,
                "signal_status": "success",
                "signal_model_name": signal_payload["model_name"],
                "signal_model_version": signal_payload["model_version"],
                "signal_base_date": signal_payload["latest_base_date"],
                "signal_forecast_window_end": signal_payload["forecast_window_end"],
                "signal_probability_action": signal_payload["latest_probability_action"],
                "signal_action_level": signal_payload["latest_action_level"],
                "signal_history_rows": signal_payload["history_rows"],
                "signal_metrics_rows": signal_payload["metrics_rows"],
                "sentiment_status": sentiment_status.get("status", "unknown"),
                "sentiment_error": sentiment_status.get("error", ""),
                "sentiment_candidate_rows": sentiment_status.get("candidate_rows", 0),
                "sentiment_newly_scored_rows": sentiment_status.get("newly_scored_rows", 0),
                "sentiment_article_rows": sentiment_status.get("article_rows", 0),
                "sentiment_daily_rows": sentiment_status.get("daily_rows", 0),
                "sentiment_fetch_failed_windows": sentiment_status.get("fetch_failed_windows", 0),
                "ai_brief_status": ai_payload.get("ai_status", "unknown"),
                "ai_brief_status_note": ai_payload.get("ai_status_note", ""),
                "ai_brief_date": ai_payload.get("brief_date", ""),
                "ai_forecast_window_end": ai_payload.get("forecast_window_end", ""),
            }
        ]
    )
    overwrite_sheet(status_row, TAB_RUN_STATUS)

    print("marketing production spreadsheet state updated")
    print(f"- spreadsheet           : {SPREADSHEET_NAME}")
    print(f"- signal base date      : {signal_payload['latest_base_date']}")
    print(f"- signal action         : {signal_payload['latest_label_id']}")
    print(f"- signal probability    : {signal_payload['latest_probability_action']:.4f}")
    print(f"- sentiment status      : {sentiment_status.get('status', 'unknown')}")
    print(f"- ai brief status       : {ai_payload.get('ai_status', 'unknown')}")


if __name__ == "__main__":
    main()
