#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.news.config import build_settings, ensure_parent_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate scored aluminium news into daily sentiment features.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input scored CSV path. Default comes from NEWS_SENTIMENT_SCORED_OUTPUT_PATH.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output daily feature CSV path. Default comes from NEWS_SENTIMENT_DAILY_OUTPUT_PATH.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings()

    input_path = Path(args.input) if args.input else settings.scored_output_path
    output_path = Path(args.output) if args.output else settings.daily_output_path

    if not input_path.exists():
        raise SystemExit(f"Input scored file tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit("Input scored file kosong.")

    df["news_date"] = pd.to_datetime(df["news_date"]).dt.date
    df["market_impact_score"] = pd.to_numeric(df["market_impact_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    confidence_cutoff = settings.confidence_threshold
    df["is_bullish"] = df["market_impact_score"] > 0.20
    df["is_bearish"] = df["market_impact_score"] < -0.20
    df["is_high_confidence"] = df["confidence"] >= confidence_cutoff

    daily = (
        df.groupby("news_date")
        .agg(
            news_count_model=("article_id", "nunique"),
            market_sentiment_mean=("market_impact_score", "mean"),
            market_sentiment_sum=("market_impact_score", "sum"),
            bullish_ratio=("is_bullish", "mean"),
            bearish_ratio=("is_bearish", "mean"),
            high_confidence_ratio=("is_high_confidence", "mean"),
        )
        .reset_index()
        .sort_values("news_date")
    )

    for channel in ["price", "supply", "policy", "logistics", "inventory", "demand", "macro", "unclear"]:
        channel_count = (
            df.assign(channel_match=df["impact_channel"].eq(channel))
            .groupby("news_date")["channel_match"]
            .sum()
            .rename(f"channel_{channel}_count")
            .reset_index()
        )
        daily = daily.merge(channel_count, on="news_date", how="left")

    daily = daily.fillna(0.0)
    ensure_parent_dir(output_path)
    daily.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows :", len(daily))
    print("Date :", daily["news_date"].min(), "->", daily["news_date"].max())
    print("Columns:", ", ".join(daily.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
