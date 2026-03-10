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

from production.gsheet_manager import overwrite_sheet, read_sheet
from production.sheet_contract import TAB_SENTIMENT_ARTICLES, TAB_SENTIMENT_DAILY
from src.news.backfill_gdelt_history import (
    QUERY_MAP,
    USER_AGENT,
    fetch_gdelt_window,
    normalize_news,
    split_windows,
)
from src.news.config import build_settings, has_real_api_key
from src.news.io import ensure_article_ids, upsert_scores
from src.news.score_sentiment import score_one_article


@dataclass(frozen=True)
class ProductionSentimentSettings:
    lookback_days: int
    languages: list[str]
    window_days: int
    maxrecords: int
    retries: int
    pause_seconds: float
    max_new_articles_per_run: int


SCORED_COLUMNS = [
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

DAILY_COLUMNS = [
    "news_date",
    "news_count_model",
    "market_sentiment_mean",
    "market_sentiment_sum",
    "bullish_ratio",
    "bearish_ratio",
    "high_confidence_ratio",
    "channel_price_count",
    "channel_supply_count",
    "channel_policy_count",
    "channel_logistics_count",
    "channel_inventory_count",
    "channel_demand_count",
    "channel_macro_count",
    "channel_unclear_count",
]

LOCAL_SCORED_SEED_PATHS = [
    ROOT / "data/news/scored/gdelt_backfill_180d_english_v2_candidate_model_scored.csv",
    ROOT / "data/news/scored/gdelt_candidate_model_scored_v1.csv",
]


def build_production_settings() -> ProductionSentimentSettings:
    return ProductionSentimentSettings(
        lookback_days=int(os.getenv("PRODUCTION_SENTIMENT_LOOKBACK_DAYS", "7")),
        languages=[
            item.strip().lower()
            for item in os.getenv("PRODUCTION_SENTIMENT_LANGUAGES", "english").split(",")
            if item.strip()
        ],
        window_days=int(os.getenv("PRODUCTION_SENTIMENT_WINDOW_DAYS", "3")),
        maxrecords=int(os.getenv("PRODUCTION_SENTIMENT_MAXRECORDS", "15")),
        retries=int(os.getenv("PRODUCTION_SENTIMENT_RETRIES", "2")),
        pause_seconds=float(os.getenv("PRODUCTION_SENTIMENT_PAUSE_SECONDS", "4.0")),
        max_new_articles_per_run=int(os.getenv("PRODUCTION_SENTIMENT_MAX_NEW_ARTICLES_PER_RUN", "5")),
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


def _empty_scored_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCORED_COLUMNS)


def _empty_daily_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_COLUMNS)


def _load_local_scored_seed() -> pd.DataFrame:
    for path in LOCAL_SCORED_SEED_PATHS:
        if path.exists():
            try:
                seed = pd.read_csv(path)
            except Exception:
                continue
            if not seed.empty:
                return _coerce_scored_sheet(seed)
    return _empty_scored_df()


def _coerce_scored_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_scored_df()
    out = df.copy()
    for col in ["market_impact_score", "confidence"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in SCORED_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = ensure_article_ids(out[SCORED_COLUMNS].copy())
    return out


def _coerce_daily_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_daily_df()
    out = df.copy()
    numeric_cols = [column for column in DAILY_COLUMNS if column != "news_date"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in DAILY_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col != "news_date" else ""
    return out[DAILY_COLUMNS].copy()


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
        return _empty_daily_df()

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

    daily = daily.fillna(0.0)
    daily["news_date"] = pd.to_datetime(daily["news_date"]).dt.date.astype(str)
    return daily[DAILY_COLUMNS].copy()


def _write_sentiment_state(scored_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    overwrite_sheet(scored_df, TAB_SENTIMENT_ARTICLES)
    overwrite_sheet(daily_df, TAB_SENTIMENT_DAILY)


def refresh_sentiment_data() -> dict:
    production_settings = build_production_settings()
    news_settings = build_settings()

    existing_scored = _coerce_scored_sheet(read_sheet(TAB_SENTIMENT_ARTICLES))
    existing_daily = _coerce_daily_sheet(read_sheet(TAB_SENTIMENT_DAILY))

    if existing_scored.empty:
        seed_scored = _load_local_scored_seed()
        if not seed_scored.empty:
            seed_daily = aggregate_daily_features(seed_scored, news_settings.confidence_threshold)
            _write_sentiment_state(seed_scored, seed_daily)
            existing_scored = seed_scored
            existing_daily = seed_daily

    try:
        candidate_df, fetch_logs = fetch_recent_candidate_news(production_settings)
    except Exception as exc:
        return {
            "status": "used_existing_after_fetch_error",
            "error": str(exc),
            "candidate_rows": 0,
            "newly_scored_rows": 0,
            "article_rows": int(len(existing_scored)),
            "daily_rows": int(len(existing_daily)),
        }

    fetch_failed_windows = int(fetch_logs["status"].eq("failed").sum()) if not fetch_logs.empty else 0
    if candidate_df.empty:
        if existing_scored.empty and existing_daily.empty:
            _write_sentiment_state(_empty_scored_df(), _empty_daily_df())
            return {
                "status": "empty_refresh",
                "candidate_rows": 0,
                "newly_scored_rows": 0,
                "article_rows": 0,
                "daily_rows": 0,
                "fetch_failed_windows": fetch_failed_windows,
            }
        return {
            "status": "used_existing_no_candidates",
            "candidate_rows": 0,
            "newly_scored_rows": 0,
            "article_rows": int(len(existing_scored)),
            "daily_rows": int(len(existing_daily)),
            "fetch_failed_windows": fetch_failed_windows,
        }

    candidate_ids = set(candidate_df["article_id"].astype(str))
    done_ids = set(existing_scored["article_id"].astype(str)) if not existing_scored.empty else set()
    pending_df = candidate_df[~candidate_df["article_id"].astype(str).isin(done_ids)].copy()
    pending_df = pending_df.sort_values("news_datetime", ascending=False).head(
        production_settings.max_new_articles_per_run
    )

    if pending_df.empty:
        combined_df = existing_scored.copy()
        daily_df = aggregate_daily_features(combined_df, news_settings.confidence_threshold)
        _write_sentiment_state(combined_df, daily_df)
        return {
            "status": "up_to_date",
            "candidate_rows": int(len(candidate_df)),
            "newly_scored_rows": 0,
            "article_rows": int(len(combined_df)),
            "daily_rows": int(len(daily_df)),
            "fetch_failed_windows": fetch_failed_windows,
        }

    if not has_real_api_key(news_settings.gemini_api_key):
        daily_df = aggregate_daily_features(existing_scored, news_settings.confidence_threshold)
        _write_sentiment_state(existing_scored, daily_df)
        return {
            "status": "used_existing_no_api_key",
            "candidate_rows": int(len(candidate_df)),
            "newly_scored_rows": 0,
            "article_rows": int(len(existing_scored)),
            "daily_rows": int(len(daily_df)),
            "fetch_failed_windows": fetch_failed_windows,
        }

    scored_rows: list[dict] = []
    scoring_error = None

    for row in pending_df.itertuples(index=False):
        title = str(getattr(row, "title", "")).strip()
        snippet = str(getattr(row, "snippet", "") or "").strip()[: news_settings.max_snippet_chars]
        try:
            parsed, raw_output = score_one_article(
                api_key=news_settings.gemini_api_key,
                model=news_settings.gemini_model,
                title=title,
                snippet=snippet,
                max_reason_chars=news_settings.max_reason_chars,
            )
        except Exception as exc:
            scoring_error = str(exc)
            break

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

    combined_df = upsert_scores(existing_scored, scored_rows)
    combined_df = ensure_article_ids(combined_df)
    combined_df = combined_df[combined_df["article_id"].astype(str).isin(set(combined_df["article_id"].astype(str)) | candidate_ids)]
    combined_df["news_date"] = pd.to_datetime(combined_df["news_date"], errors="coerce").dt.date.astype(str)
    combined_df = combined_df.sort_values(["news_date", "news_datetime", "title"]).reset_index(drop=True)

    daily_df = aggregate_daily_features(combined_df, news_settings.confidence_threshold)
    _write_sentiment_state(combined_df, daily_df)

    status = "refreshed" if scoring_error is None else "partial_refresh_after_scoring_error"
    return {
        "status": status,
        "error": scoring_error,
        "candidate_rows": int(len(candidate_df)),
        "newly_scored_rows": int(len(scored_rows)),
        "article_rows": int(len(combined_df)),
        "daily_rows": int(len(daily_df)),
        "fetch_failed_windows": fetch_failed_windows,
    }


if __name__ == "__main__":
    result = refresh_sentiment_data()
    for key, value in result.items():
        print(f"{key}: {value}")
