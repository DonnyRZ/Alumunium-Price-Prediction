from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.pipeline.common import SENTIMENT_SNAPSHOT_PATH, utc_now_iso, write_json
from production.pipeline.common import SENTIMENT_ARTICLES_PATH, SENTIMENT_DAILY_PATH

CHANNEL_MAP = {
    "price": "Harga pasar",
    "supply": "Pasokan",
    "policy": "Kebijakan",
    "logistics": "Logistik",
    "inventory": "Persediaan",
    "demand": "Permintaan",
    "macro": "Makro",
    "unclear": "Belum jelas",
}


def _dominant_channel(row: pd.Series) -> str:
    mapping = {
        "channel_price_count": "Harga pasar",
        "channel_supply_count": "Pasokan",
        "channel_policy_count": "Kebijakan",
        "channel_logistics_count": "Logistik",
        "channel_inventory_count": "Persediaan",
        "channel_demand_count": "Permintaan",
        "channel_macro_count": "Makro",
        "channel_unclear_count": "Belum jelas",
    }
    cols = list(mapping.keys())
    if row[cols].sum() <= 0:
        return "Belum jelas"
    return mapping[row[cols].idxmax()]


def _tone_label(score: float) -> str:
    if score >= 0.20:
        return "Positif"
    if score <= -0.20:
        return "Negatif"
    return "Netral"


def build_sentiment_snapshot() -> Path:
    scored = pd.read_csv(SENTIMENT_ARTICLES_PATH, parse_dates=["news_date", "news_datetime"])
    daily = pd.read_csv(SENTIMENT_DAILY_PATH, parse_dates=["news_date"])

    if scored.empty or daily.empty:
        payload = {
            "generated_at_utc": utc_now_iso(),
            "data_sources": {
                "scored_articles": str(SENTIMENT_ARTICLES_PATH.relative_to(ROOT)),
                "daily_sentiment": str(SENTIMENT_DAILY_PATH.relative_to(ROOT)),
            },
            "article_count": 0,
            "day_count": 0,
            "date_min": None,
            "date_max": None,
            "average_score": None,
            "average_confidence": None,
            "latest_news_date": None,
            "latest_daily": None,
            "label_distribution": {},
            "channel_distribution": {},
            "recent_daily": [],
            "top_articles": [],
            "production_note": (
                "Belum ada news relevan yang siap dipakai pada refresh terbaru. "
                "Dashboard tetap jalan tanpa konteks sentiment."
            ),
        }
        return write_json(payload, SENTIMENT_SNAPSHOT_PATH)

    daily = daily.sort_values("news_date").copy()
    daily["dominant_channel"] = daily.apply(_dominant_channel, axis=1)
    daily["tone_label"] = daily["market_sentiment_mean"].apply(_tone_label)

    latest_day = daily.sort_values("news_date").tail(1).iloc[0]
    latest_news_date = pd.Timestamp(latest_day["news_date"])

    recent_articles = (
        scored.sort_values("news_datetime", ascending=False)
        .head(8)
        .assign(
            impact_channel_readable=lambda x: x["impact_channel"].map(CHANNEL_MAP).fillna(x["impact_channel"]),
            impact_label_readable=lambda x: x["impact_label"].map(
                {"bullish": "Cenderung naik", "bearish": "Cenderung turun", "neutral": "Netral"}
            ),
        )
    )

    payload = {
        "generated_at_utc": utc_now_iso(),
        "data_sources": {
            "scored_articles": str(SENTIMENT_ARTICLES_PATH.relative_to(ROOT)),
            "daily_sentiment": str(SENTIMENT_DAILY_PATH.relative_to(ROOT)),
        },
        "article_count": int(len(scored)),
        "day_count": int(daily["news_date"].nunique()),
        "date_min": pd.Timestamp(scored["news_date"].min()).date().isoformat(),
        "date_max": pd.Timestamp(scored["news_date"].max()).date().isoformat(),
        "average_score": float(scored["market_impact_score"].mean()),
        "average_confidence": float(scored["confidence"].mean()),
        "latest_news_date": latest_news_date.date().isoformat(),
        "latest_daily": {
            "news_date": latest_news_date.date().isoformat(),
            "news_count_model": int(latest_day["news_count_model"]),
            "market_sentiment_mean": float(latest_day["market_sentiment_mean"]),
            "bullish_ratio": float(latest_day["bullish_ratio"]),
            "bearish_ratio": float(latest_day["bearish_ratio"]),
            "high_confidence_ratio": float(latest_day["high_confidence_ratio"]),
            "dominant_channel": str(latest_day["dominant_channel"]),
            "tone_label": str(latest_day["tone_label"]),
        },
        "label_distribution": scored["impact_label"].value_counts().to_dict(),
        "channel_distribution": scored["impact_channel"].value_counts().to_dict(),
        "recent_daily": [
            {
                "news_date": pd.Timestamp(row["news_date"]).date().isoformat(),
                "news_count_model": int(row["news_count_model"]),
                "market_sentiment_mean": float(row["market_sentiment_mean"]),
                "bullish_ratio": float(row["bullish_ratio"]),
                "bearish_ratio": float(row["bearish_ratio"]),
                "dominant_channel": str(row["dominant_channel"]),
                "tone_label": str(row["tone_label"]),
            }
            for _, row in daily.tail(60).iterrows()
        ],
        "top_articles": [
            {
                "news_date": pd.Timestamp(row["news_date"]).date().isoformat(),
                "title": str(row["title"]),
                "domain": str(row["domain"]),
                "impact_label": str(row["impact_label_readable"]),
                "impact_channel": str(row["impact_channel_readable"]),
                "market_impact_score": float(row["market_impact_score"]),
                "confidence": float(row["confidence"]),
                "reason_short": str(row["reason_short"]),
                "url": str(row["url"]),
            }
            for _, row in recent_articles.iterrows()
        ],
        "production_note": (
            "Sentiment dipakai sebagai konteks pasar harian. "
            "Output ini tidak mengubah prediksi model secara otomatis."
        ),
    }
    return write_json(payload, SENTIMENT_SNAPSHOT_PATH)


if __name__ == "__main__":
    out = build_sentiment_snapshot()
    print(f"saved: {out}")
