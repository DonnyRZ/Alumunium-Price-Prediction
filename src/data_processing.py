#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def load_raw(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["is_missing"] = close.isna()
    df["is_stale"] = close.notna() & close.eq(close.shift(1))
    df["price_changed"] = close.notna() & (~df["is_stale"])

    # Streak group based on stale flag only (missing does not count as stale)
    df["streak_group"] = (df["is_stale"] != df["is_stale"].shift(1)).cumsum()
    streak_sizes = df[df["is_stale"]].groupby("streak_group").size()
    df["streak_len"] = df["streak_group"].map(streak_sizes).fillna(0).astype(int)

    if "Volume" in df.columns:
        df["volume_zero"] = df["Volume"] == 0
    else:
        df["volume_zero"] = False

    if set(["Open", "High", "Low", "Close"]).issubset(df.columns):
        valid = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        df["flat_candle"] = (
            (df["Open"] == df["High"])
            & (df["High"] == df["Low"])
            & (df["Low"] == df["Close"])
            & valid
        )
    else:
        df["flat_candle"] = False

    return df


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change(fill_method=None)
    ret = df["Return"]

    # IQR outliers (global)
    q1 = ret.quantile(0.25)
    q3 = ret.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["is_outlier_iqr"] = (ret < lower) | (ret > upper)

    # Rolling MAD outliers (robust)
    window = 90
    min_periods = 30
    median = ret.rolling(window, min_periods=min_periods).median()
    mad = (ret - median).abs().rolling(window, min_periods=min_periods).median()
    mad = mad.replace(0, np.nan).clip(lower=1e-6)
    robust_z = 0.6745 * (ret - median) / mad
    df["is_outlier_mad"] = robust_z.abs() > 5

    return df


def flag_suspect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    streak_sizes = df[df["is_stale"]].groupby("streak_group").size()
    long_streak_groups = set(streak_sizes[streak_sizes >= 5].index)
    last_day_mask = df["streak_group"].isin(long_streak_groups) & df["is_stale"] & (
        df["streak_group"] != df["streak_group"].shift(-1)
    )
    next_day_idx = df.index.to_series().shift(-1)[last_day_mask].dropna()
    df["next_day_after_long_streak"] = df.index.isin(next_day_idx.values)

    df["is_suspect_outlier"] = (df["is_outlier_iqr"] | df["is_outlier_mad"]) & (
        df["volume_zero"] | df["next_day_after_long_streak"]
    )
    return df


def clean_event_based(df: pd.DataFrame) -> pd.DataFrame:
    # Best-practice rules:
    # 1) Drop missing Close
    # 2) Keep only event-based days (price changes)
    # 3) Drop suspect outliers
    df_clean = df.copy()
    df_clean = df_clean[df_clean["Close"].notna()]
    df_clean = df_clean[~df_clean["is_stale"]]
    df_clean = df_clean[~df_clean["is_suspect_outlier"]]

    # Recompute return after cleaning
    df_clean["Return"] = df_clean["Close"].pct_change(fill_method=None)
    return df_clean


def build_report(raw_df: pd.DataFrame, flagged_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    report = {}

    report["raw_rows"] = len(raw_df)
    report["raw_missing_close"] = int(raw_df["Close"].isna().sum())
    report["raw_volume_zero"] = int((raw_df["Volume"] == 0).sum()) if "Volume" in raw_df.columns else 0
    report["raw_stale_days"] = int(flagged_df["is_stale"].sum())
    report["raw_long_streak_days"] = int((flagged_df["streak_len"] >= 5).sum())
    report["raw_suspect_outliers"] = int(flagged_df["is_suspect_outlier"].sum())

    report["clean_rows"] = len(clean_df)
    report["dropped_missing"] = int(raw_df["Close"].isna().sum())
    report["dropped_stale"] = int(flagged_df["is_stale"].sum())
    report["dropped_suspect_outliers"] = int(flagged_df["is_suspect_outlier"].sum())

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Process raw ALI=F data with best-practice cleaning.")
    parser.add_argument(
        "--raw",
        default=os.path.join("data", "raw data", "ali_f_raw.csv"),
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "processed data", "ali_f_event_clean.csv"),
        help="Output path for cleaned event-based CSV",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path}")

    raw_df = load_raw(raw_path)
    flagged = build_flags(raw_df)
    flagged = detect_outliers(flagged)
    flagged = flag_suspect_outliers(flagged)

    clean_df = clean_event_based(flagged)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path)

    report = build_report(raw_df, flagged, clean_df)
    print("PROCESSING REPORT")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k}: {v}")
    print("Saved:", out_path)
    print("Processed at:", datetime.utcnow().isoformat() + "Z")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
