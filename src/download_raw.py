#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def download_ohlcv(symbol: str, start: str, interval: str) -> pd.DataFrame:
    os.makedirs("/tmp/yf-cache", exist_ok=True)
    try:
        yf.set_tz_cache_location("/tmp/yf-cache")
    except Exception:
        pass

    df = yf.download(
        symbol,
        start=start,
        interval=interval,
        auto_adjust=False,
        repair=True,
        keepna=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Download raw OHLCV data with yfinance.")
    parser.add_argument("--symbol", default="ALI=F", help="Ticker symbol (default: ALI=F)")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Interval (default: 1d)")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw data", "ali_f_raw.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = download_ohlcv(args.symbol, args.start, args.interval)
    if df.empty:
        raise SystemExit("Download failed or returned empty data.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)

    print("Saved:", args.out)
    print("Rows:", len(df))
    print("Range:", df.index.min().date(), "to", df.index.max().date())
    print("Downloaded at:", datetime.utcnow().isoformat() + "Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
