from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_processing import build_flags, clean_model_ready_v3, detect_outliers, flag_suspect_outliers
from src.download_raw import download_ohlcv


RAW_PATH = ROOT / "data" / "raw data" / "ali_f_raw.csv"
PROCESSED_PATH = ROOT / "data" / "processed data" / "ali_f_event_model_ready_v3.csv"
SYMBOL = "ALI=F"
START_DATE = "2000-01-01"
INTERVAL = "1d"


def refresh_market_data() -> dict:
    raw_df = download_ohlcv(symbol=SYMBOL, start=START_DATE, interval=INTERVAL)
    if raw_df.empty:
        raise ValueError("Yahoo Finance mengembalikan data kosong.")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_PATH)

    flagged = build_flags(raw_df)
    flagged = detect_outliers(flagged)
    flagged = flag_suspect_outliers(flagged)
    _, model_df, diagnostics = clean_model_ready_v3(
        flagged,
        max_gap_days=7,
        drop_residual_suspect=True,
        max_suspect_passes=5,
        max_suspect_drop_pct=5.0,
    )
    model_df.to_csv(PROCESSED_PATH)

    latest_date = model_df.index.max()
    return {
        "raw_rows": int(len(raw_df)),
        "processed_rows": int(len(model_df)),
        "latest_date": latest_date.date().isoformat() if latest_date is not None else None,
        "diagnostics": diagnostics,
        "raw_path": str(RAW_PATH.relative_to(ROOT)),
        "processed_path": str(PROCESSED_PATH.relative_to(ROOT)),
    }


if __name__ == "__main__":
    result = refresh_market_data()
    for key, value in result.items():
        print(f"{key}: {value}")
