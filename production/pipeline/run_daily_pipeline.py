from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.pipeline.build_dashboard_payload import build_dashboard_payload
from production.pipeline.build_sentiment_snapshot import build_sentiment_snapshot
from production.pipeline.refresh_sentiment_data import refresh_sentiment_data
from production.pipeline.build_xgb_snapshot import build_xgb_snapshot


def main() -> None:
    model_path = build_xgb_snapshot()
    sentiment_refresh = refresh_sentiment_data()
    sentiment_path = build_sentiment_snapshot()
    dashboard_path = build_dashboard_payload()

    print("production snapshots updated")
    print(f"- model     : {model_path}")
    print(f"- sentiment refresh: {sentiment_refresh}")
    print(f"- sentiment : {sentiment_path}")
    print(f"- dashboard : {dashboard_path}")


if __name__ == "__main__":
    main()
