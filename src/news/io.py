from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_INPUT_COLUMNS = [
    "title",
    "url",
    "news_date",
]


def build_article_id(row: pd.Series) -> str:
    raw = "|".join(
        [
            str(row.get("news_date", "")),
            str(row.get("url", "")),
            str(row.get("title", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ensure_article_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "article_id" not in df.columns:
        df["article_id"] = df.apply(build_article_id, axis=1)
    return df


def load_candidate_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input candidate news missing columns: {missing}")

    df = ensure_article_ids(df)
    df["news_date"] = pd.to_datetime(df["news_date"]).dt.date
    return df


def load_existing_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df
    return ensure_article_ids(df)


def upsert_scores(existing_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        if new_df.empty:
            return pd.DataFrame()
        return ensure_article_ids(new_df)

    if new_df.empty:
        return ensure_article_ids(existing_df)

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = ensure_article_ids(merged)
    merged = merged.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
    return merged
