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


CHANNEL_PRIORITY = [
    "supply",
    "logistics",
    "policy",
    "price",
    "inventory",
    "demand",
    "macro",
    "unclear",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build conservative daily news overlay signals from aggregated sentiment features.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input daily sentiment CSV path. Default comes from NEWS_SENTIMENT_DAILY_OUTPUT_PATH.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output overlay signal CSV path. Default comes from NEWS_SENTIMENT_OVERLAY_OUTPUT_PATH.",
    )
    return parser


def infer_dominant_channel(row: pd.Series) -> tuple[str, int]:
    best_channel = "unclear"
    best_count = -1
    for channel in CHANNEL_PRIORITY:
        count = int(row.get(f"channel_{channel}_count", 0) or 0)
        if count > best_count:
            best_channel = channel
            best_count = count
    return best_channel, max(best_count, 0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings()

    input_path = Path(args.input) if args.input else settings.daily_output_path
    output_path = Path(args.output) if args.output else settings.overlay_output_path

    if not input_path.exists():
        raise SystemExit(f"Input daily sentiment file tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit("Input daily sentiment file kosong.")

    df["news_date"] = pd.to_datetime(df["news_date"]).dt.date
    numeric_columns = [
        "news_count_model",
        "market_sentiment_mean",
        "market_sentiment_sum",
        "bullish_ratio",
        "bearish_ratio",
        "high_confidence_ratio",
    ] + [column for column in df.columns if column.startswith("channel_") and column.endswith("_count")]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    dominant = df.apply(infer_dominant_channel, axis=1, result_type="expand")
    df["dominant_channel"] = dominant[0]
    df["dominant_channel_count"] = dominant[1].astype(int)
    df["dominant_channel_share"] = np.where(
        df["news_count_model"] > 0,
        df["dominant_channel_count"] / df["news_count_model"],
        0.0,
    )

    min_news_count = settings.overlay_news_min_count
    min_high_conf_ratio = settings.overlay_min_high_conf_ratio
    medium_threshold = settings.overlay_medium_sentiment_threshold
    strong_threshold = settings.overlay_strong_sentiment_threshold

    directional_ready = (df["news_count_model"] >= min_news_count) & (df["high_confidence_ratio"] >= min_high_conf_ratio)

    conditions = [
        directional_ready & (df["market_sentiment_mean"] >= medium_threshold),
        directional_ready & (df["market_sentiment_mean"] <= -medium_threshold),
        (df["news_count_model"] >= min_news_count) & (df["market_sentiment_mean"].abs() < medium_threshold),
    ]
    choices = ["bullish_overlay", "bearish_overlay", "weak_news"]
    df["overlay_state"] = np.select(conditions, choices, default="watchlist")

    df["overlay_bias"] = np.select(
        [df["overlay_state"].eq("bullish_overlay"), df["overlay_state"].eq("bearish_overlay")],
        ["bullish", "bearish"],
        default="neutral_or_hold",
    )
    df["overlay_strength"] = np.select(
        [df["market_sentiment_mean"].abs() >= strong_threshold, df["market_sentiment_mean"].abs() >= medium_threshold],
        ["strong", "medium"],
        default="weak",
    )

    df["overlay_action_if_model_bullish"] = "ignore_news"
    df["overlay_action_if_model_bearish"] = "ignore_news"
    df["overlay_role"] = "ignore"

    strengthen_mask = (
        df["overlay_state"].eq("bullish_overlay")
        & df["overlay_strength"].eq("strong")
        & df["dominant_channel"].isin(["supply", "logistics"])
    )
    caution_mask = df["overlay_state"].eq("bearish_overlay") & df["dominant_channel"].eq("policy")
    bullish_watch_mask = df["overlay_state"].eq("bullish_overlay") & ~strengthen_mask
    bearish_watch_mask = df["overlay_state"].eq("bearish_overlay") & ~caution_mask
    watchlist_mask = df["overlay_state"].eq("watchlist")

    df.loc[strengthen_mask, "overlay_action_if_model_bullish"] = "size_up"
    df.loc[strengthen_mask, "overlay_action_if_model_bearish"] = "hold_or_review"
    df.loc[strengthen_mask, "overlay_role"] = "bullish_conviction"

    df.loc[caution_mask, "overlay_action_if_model_bullish"] = "size_down"
    df.loc[caution_mask, "overlay_action_if_model_bearish"] = "keep_bearish_but_not_news_only"
    df.loc[caution_mask, "overlay_role"] = "bearish_caution"

    df.loc[bullish_watch_mask, "overlay_action_if_model_bullish"] = "keep_with_normal_size"
    df.loc[bullish_watch_mask, "overlay_action_if_model_bearish"] = "review_conflict"
    df.loc[bullish_watch_mask, "overlay_role"] = "bullish_watch"

    df.loc[bearish_watch_mask, "overlay_action_if_model_bullish"] = "review_conflict"
    df.loc[bearish_watch_mask, "overlay_action_if_model_bearish"] = "keep_with_normal_size"
    df.loc[bearish_watch_mask, "overlay_role"] = "bearish_watch"

    df.loc[watchlist_mask, "overlay_action_if_model_bullish"] = "watch_only"
    df.loc[watchlist_mask, "overlay_action_if_model_bearish"] = "watch_only"
    df.loc[watchlist_mask, "overlay_role"] = "watchlist"

    df["overlay_conviction_adjustment"] = np.select(
        [strengthen_mask, caution_mask],
        [0.25, -0.25],
        default=0.0,
    )

    df["overlay_note"] = np.select(
        [
            strengthen_mask,
            caution_mask,
            bullish_watch_mask,
            bearish_watch_mask,
            df["overlay_state"].eq("weak_news"),
            watchlist_mask,
        ],
        [
            "News supply/logistics bullish yang kuat. Layak memperkuat conviction bullish.",
            "News policy bearish. Pakai sebagai caution, bukan hard veto.",
            "News bullish ada, tetapi belum cukup kuat untuk size up otomatis.",
            "News bearish ada, tetapi belum cukup kuat untuk dipakai sebagai veto.",
            "Ada berita, tetapi sinyal terlalu lemah untuk mengubah keputusan model.",
            "Ada konteks yang patut dipantau, tetapi belum cukup kuat untuk tindakan.",
        ],
        default="Tidak ada aksi khusus dari news overlay.",
    )

    keep_columns = [
        "news_date",
        "news_count_model",
        "market_sentiment_mean",
        "market_sentiment_sum",
        "bullish_ratio",
        "bearish_ratio",
        "high_confidence_ratio",
        "dominant_channel",
        "dominant_channel_count",
        "dominant_channel_share",
        "overlay_state",
        "overlay_bias",
        "overlay_strength",
        "overlay_role",
        "overlay_conviction_adjustment",
        "overlay_action_if_model_bullish",
        "overlay_action_if_model_bearish",
        "overlay_note",
    ]
    overlay_df = df[keep_columns].sort_values("news_date").reset_index(drop=True)

    ensure_parent_dir(output_path)
    overlay_df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows :", len(overlay_df))
    print("Date :", overlay_df["news_date"].min(), "->", overlay_df["news_date"].max())
    print("Overlay state counts:")
    print(overlay_df["overlay_state"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
