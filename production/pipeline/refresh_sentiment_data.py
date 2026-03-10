from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.pipeline.common import SENTIMENT_ARTICLES_PATH, SENTIMENT_DAILY_PATH, ensure_parent
from src.news.backfill_gdelt_history import (
    QUERY_MAP,
    USER_AGENT,
    fetch_gdelt_window,
    normalize_news,
    split_windows,
)
from src.news.config import build_settings, has_real_api_key
from src.news.io import ensure_article_ids, load_existing_scores, upsert_scores
from src.news.score_sentiment import score_one_article


@dataclass(frozen=True)
class ProductionSentimentSettings:
    lookback_days: int
    languages: list[str]
    window_days: int
    maxrecords: int
    retries: int
    pause_seconds: float


def build_production_settings() -> ProductionSentimentSettings:
    return ProductionSentimentSettings(
        lookback_days=int(os.getenv("PRODUCTION_SENTIMENT_LOOKBACK_DAYS", "30")),
        languages=[
            item.strip().lower()
            for item in os.getenv("PRODUCTION_SENTIMENT_LANGUAGES", "english").split(",")
            if item.strip()
        ],
        window_days=int(os.getenv("PRODUCTION_SENTIMENT_WINDOW_DAYS", "7")),
        maxrecords=int(os.getenv("PRODUCTION_SENTIMENT_MAXRECORDS", "25")),
        retries=int(os.getenv("PRODUCTION_SENTIMENT_RETRIES", "3")),
        pause_seconds=float(os.getenv("PRODUCTION_SENTIMENT_PAUSE_SECONDS", "3.0")),
    )


def _is_fatal_network_error(text: str) -> bool:
    lowered = str(text).lower()
    patterns = [
        "failed to resolve",
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
    ]
    return any(pattern in lowered for pattern in patterns)


def fetch_recent_candidate_news(settings: ProductionSentimentSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=settings.lookback_days)
    windows = split_windows(start_date, end_date, settings.window_days)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    frames: list[pd.DataFrame] = []
    fetch_logs: list[dict] = []

    for query_group in QUERY_MAP:
        for language in settings.languages:
            for window_start, window_end in windows:
                df_part, log = fetch_gdelt_window(
                    session=session,
                    query_group=query_group,
                    language=language,
                    window_start=window_start,
                    window_end=window_end,
                    maxrecords=settings.maxrecords,
                    retries=settings.retries,
                    pause_seconds=settings.pause_seconds,
                )
                frames.append(df_part)
                fetch_logs.append(log)
                if log.get("status") == "failed" and _is_fatal_network_error(log.get("error", "")):
                    return pd.DataFrame(), pd.DataFrame(fetch_logs)
                time.sleep(settings.pause_seconds)

    raw = (
        pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in frames)
        else pd.DataFrame()
    )
    logs = pd.DataFrame(fetch_logs)
    if raw.empty:
        return pd.DataFrame(), logs

    news = normalize_news(raw)
    candidate = news[news["usage_bucket"].eq("candidate_model")].copy()
    if candidate.empty:
        return candidate, logs

    candidate = ensure_article_ids(candidate)
    candidate["news_date"] = pd.to_datetime(candidate["news_date"]).dt.date
    candidate = candidate.sort_values(["news_date", "news_datetime", "title"]).reset_index(drop=True)
    return candidate, logs


def aggregate_daily_features(scored_df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame(
            columns=[
                "news_date",
                "news_count_model",
                "market_sentiment_mean",
                "market_sentiment_sum",
                "bullish_ratio",
                "bearish_ratio",
                "high_confidence_ratio",
            ]
        )

    df = scored_df.copy()
    df["news_date"] = pd.to_datetime(df["news_date"]).dt.date
    df["market_impact_score"] = pd.to_numeric(df["market_impact_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["is_bullish"] = df["market_impact_score"] > 0.20
    df["is_bearish"] = df["market_impact_score"] < -0.20
    df["is_high_confidence"] = df["confidence"] >= confidence_threshold

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

    return daily.fillna(0.0)


def _empty_scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "article_id",
            "news_date",
            "news_datetime",
            "title",
            "snippet",
            "url",
            "language",
            "domain",
            "domain_quality",
            "query_group",
            "relevance",
            "usage_bucket",
            "prompt_version",
            "scored_model",
            "scored_at_utc",
            "market_impact_score",
            "impact_label",
            "impact_channel",
            "confidence",
            "reason_short",
            "raw_model_output",
        ]
    )


def refresh_sentiment_data() -> dict:
    production_settings = build_production_settings()
    news_settings = build_settings()

    candidate_df, fetch_logs = fetch_recent_candidate_news(production_settings)
    if candidate_df.empty:
        if SENTIMENT_ARTICLES_PATH.exists() and SENTIMENT_DAILY_PATH.exists():
            return {
                "status": "used_existing",
                "candidate_rows": 0,
                "fetch_failed_windows": int(fetch_logs["status"].eq("failed").sum()) if not fetch_logs.empty else 0,
                "articles_path": str(SENTIMENT_ARTICLES_PATH.relative_to(ROOT)),
                "daily_path": str(SENTIMENT_DAILY_PATH.relative_to(ROOT)),
            }
        empty_scored = _empty_scored_df()
        empty_daily = aggregate_daily_features(empty_scored, news_settings.confidence_threshold)
        ensure_parent(SENTIMENT_ARTICLES_PATH)
        ensure_parent(SENTIMENT_DAILY_PATH)
        empty_scored.to_csv(SENTIMENT_ARTICLES_PATH, index=False)
        empty_daily.to_csv(SENTIMENT_DAILY_PATH, index=False)
        return {
            "status": "empty_refresh",
            "candidate_rows": 0,
            "newly_scored_rows": 0,
            "article_rows": 0,
            "daily_rows": 0,
            "fetch_failed_windows": int(fetch_logs["status"].eq("failed").sum()) if not fetch_logs.empty else 0,
            "articles_path": str(SENTIMENT_ARTICLES_PATH.relative_to(ROOT)),
            "daily_path": str(SENTIMENT_DAILY_PATH.relative_to(ROOT)),
        }

    existing_df = load_existing_scores(SENTIMENT_ARTICLES_PATH)
    if not existing_df.empty:
        existing_df = existing_df[existing_df["article_id"].astype(str).isin(candidate_df["article_id"].astype(str))]

    done_ids = set(existing_df["article_id"].astype(str)) if not existing_df.empty else set()
    pending_df = candidate_df[~candidate_df["article_id"].astype(str).isin(done_ids)].copy()

    if not has_real_api_key(news_settings.gemini_api_key):
        raise RuntimeError("GEMINI_API_KEY belum diisi. Refresh sentiment production tidak bisa berjalan.")

    scored_rows: list[dict] = []
    for row in pending_df.itertuples(index=False):
        title = str(getattr(row, "title", "")).strip()
        snippet = str(getattr(row, "snippet", "") or "").strip()[: news_settings.max_snippet_chars]
        parsed, raw_output = score_one_article(
            api_key=news_settings.gemini_api_key,
            model=news_settings.gemini_model,
            title=title,
            snippet=snippet,
            max_reason_chars=news_settings.max_reason_chars,
        )
        scored_rows.append(
            {
                "article_id": getattr(row, "article_id"),
                "news_date": getattr(row, "news_date"),
                "news_datetime": getattr(row, "news_datetime", ""),
                "title": title,
                "snippet": snippet,
                "url": getattr(row, "url", ""),
                "language": getattr(row, "language", ""),
                "domain": getattr(row, "domain", ""),
                "domain_quality": getattr(row, "domain_quality", ""),
                "query_group": getattr(row, "query_group", ""),
                "relevance": getattr(row, "relevance", ""),
                "usage_bucket": getattr(row, "usage_bucket", ""),
                "prompt_version": news_settings.prompt_version,
                "scored_model": news_settings.gemini_model,
                "scored_at_utc": datetime.now(timezone.utc).isoformat(),
                "market_impact_score": parsed["market_impact_score"],
                "impact_label": parsed["impact_label"],
                "impact_channel": parsed["impact_channel"],
                "confidence": parsed["confidence"],
                "reason_short": parsed["reason_short"][: news_settings.max_reason_chars],
                "raw_model_output": raw_output,
            }
        )
        time.sleep(news_settings.sleep_seconds)

    combined_df = upsert_scores(existing_df, scored_rows)
    combined_df = combined_df[combined_df["article_id"].astype(str).isin(candidate_df["article_id"].astype(str))].copy()
    combined_df["news_date"] = pd.to_datetime(combined_df["news_date"]).dt.date
    combined_df = combined_df.sort_values(["news_date", "news_datetime", "title"]).reset_index(drop=True)

    daily_df = aggregate_daily_features(combined_df, news_settings.confidence_threshold)

    ensure_parent(SENTIMENT_ARTICLES_PATH)
    ensure_parent(SENTIMENT_DAILY_PATH)
    combined_df.to_csv(SENTIMENT_ARTICLES_PATH, index=False)
    daily_df.to_csv(SENTIMENT_DAILY_PATH, index=False)

    return {
        "status": "refreshed",
        "candidate_rows": int(len(candidate_df)),
        "newly_scored_rows": int(len(scored_rows)),
        "article_rows": int(len(combined_df)),
        "daily_rows": int(len(daily_df)),
        "fetch_failed_windows": int(fetch_logs["status"].eq("failed").sum()) if not fetch_logs.empty else 0,
        "articles_path": str(SENTIMENT_ARTICLES_PATH.relative_to(ROOT)),
        "daily_path": str(SENTIMENT_DAILY_PATH.relative_to(ROOT)),
    }


if __name__ == "__main__":
    result = refresh_sentiment_data()
    for key, value in result.items():
        print(f"{key}: {value}")
