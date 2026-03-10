#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.news.config import ensure_parent_dir


BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "INALUM-News-Backfill/1.0"

COMMODITY_ANCHORS = [
    "aluminium", "aluminum", "alumina", "bauxite", "lme",
]

BUSINESS_DIRECT_TERMS = [
    "aluminium price", "aluminum price", "alumina price", "bauxite price",
    "aluminium premium", "aluminum premium",
    "aluminium market", "aluminum market", "alumina market",
    "lme aluminium", "lme aluminum",
    "aluminium inventory", "aluminum inventory",
    "aluminium tariff", "aluminum tariff",
    "aluminium sanctions", "aluminum sanctions",
    "aluminium export ban", "aluminum export ban",
    "harga aluminium", "harga alumina", "tarif aluminium", "sanksi aluminium",
]

MODEL_SIGNAL_TERMS = [
    "price", "prices", "premium", "market",
    "supply", "demand", "inventory", "stockpile", "warehouse",
    "tariff", "tariffs", "sanctions", "export ban",
    "shipping", "logistics", "freight",
    "force majeure", "production", "output",
    "harga", "pasokan", "permintaan", "produksi", "tarif", "sanksi",
]

INDUSTRY_STRUCTURE_TERMS = [
    "smelter", "refinery", "alumina refinery", "bauxite mine",
    "mining", "capacity", "shutdown", "reopening", "closure",
    "acquire", "acquires", "acquisition", "deal", "agreement",
]

OTHER_METAL_TERMS = [
    "copper", "zinc", "nickel", "steel", "iron ore",
]

PRODUCT_NOISE_TERMS = [
    "ceraluminum",
    "laptop", "notebook", "smartphone", "phone", "tablet",
    "review", "oled", "camera", "android",
    "iphone", "samsung", "asus", "pixel", "xiaomi", "huawei", "oppo", "vivo",
    "watch", "headphone", "earbuds", "bike", "bikes", "bicycle",
    "lighting", "interior", "design", "furniture",
    "health", "recipe", "restaurant", "travel", "magazine", "fashion",
    "movie", "film", "concert", "award", "beverage",
]

EVENT_NOISE_TERMS = [
    "trade fair", "expo", "conference", "represented at",
    "leadership", "leadership pipeline",
    "appointment", "appointed", "executive", "chief",
    "briefing", "dealmaking",
]

FINANCE_NOISE_TERMS = [
    "analysts explain", "price target", "buy rating", "sell rating",
    "stock surges", "stock target", "stock targets", "metal stocks",
    "shares", "stake", "nasdaq", "nyse", "cao sells",
]

NOISE_KEYWORDS = [
    "celebrity", "actor", "actress", "singer", "dating", "selingkuh",
    "instagram", "tiktok", "youtube", "wedding", "divorce", "affair",
] + PRODUCT_NOISE_TERMS + EVENT_NOISE_TERMS + FINANCE_NOISE_TERMS

HIGH_SIGNAL_DOMAINS = {
    "mining.com",
    "hellenicshippingnews.com",
    "news.metal.com",
    "fastmarkets.com",
    "business-standard.com",
}

LOW_SIGNAL_DOMAINS = {
    "finance.yahoo.com",
    "economictimes.indiatimes.com",
    "businesstoday.in",
    "moneycontrol.com",
    "livemint.com",
    "tickerreport.com",
    "bicycleretailer.com",
    "gadget.viva.co.id",
    "eturbonews.com",
    "tradearabia.com",
    "interest.co.nz",
    "themarketsdaily.com",
    "dailypolitical.com",
    "fool.com",
    "insidermonkey.com",
}

# Precision-first retrieval. Weak groups from EDA are intentionally dropped.
QUERY_MAP = {
    "price_lme": '"aluminium price" OR "aluminum price" OR "LME aluminium" OR "LME aluminum"',
    "supply_assets": '"aluminium smelter" OR "aluminum smelter" OR "alumina refinery" OR "aluminium refinery"',
    "inventory_warehouse": '"aluminium inventory" OR "aluminum inventory" OR "LME warehouse" OR "aluminium stockpile"',
    "trade_policy_strict": '"aluminium tariff" OR "aluminum tariff" OR "aluminium sanctions" OR "aluminum sanctions" OR "aluminium export ban" OR "aluminum export ban"',
}

LANGUAGE_QUERY_OVERRIDES = {
    "price_lme": {
        "indonesian": '"harga aluminium" OR "harga alumina"',
    },
    "supply_assets": {
        "indonesian": '"smelter aluminium" OR "kilang alumina"',
    },
    "inventory_warehouse": {
        "indonesian": '"stok aluminium" OR "inventori aluminium"',
    },
    "trade_policy_strict": {
        "indonesian": '"tarif aluminium" OR "sanksi aluminium" OR "larangan ekspor bauksit"',
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill historical GDELT news with the same precision filters used in EDA-NEWS.",
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--languages",
        default="english",
        help="Comma-separated language list. Default: english",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Window size in days per GDELT request. Default: 7",
    )
    parser.add_argument(
        "--maxrecords",
        type=int,
        default=25,
        help="Max records per request. Default: 25",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output prefix. Default derived from date range.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=3.0,
        help="Pause between successful requests. Default: 3",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count per request. Default: 3",
    )
    return parser


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_domain(url: str) -> str:
    url = safe_text(url)
    if not url:
        return ""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return ""
    domain = domain.replace("www.", "")
    return domain


def parse_news_date(value) -> pd.Timestamp | pd.NaT:
    text = safe_text(value)
    if not text:
        return pd.NaT
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                return pd.Timestamp(dt, tz="UTC")
            return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            continue
    try:
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        return parsed
    except Exception:
        return pd.NaT


def extract_hits(text: str, terms: list[str]) -> list[str]:
    text_norm = normalize_text(text)
    return [term for term in terms if term in text_norm]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def resolve_query_for_language(query_group: str, language: str) -> str:
    return LANGUAGE_QUERY_OVERRIDES.get(query_group, {}).get(language, QUERY_MAP[query_group])


def build_query_string(query: str, language: str) -> str:
    query = safe_text(query).strip()
    if " OR " in query and not (query.startswith("(") and query.endswith(")")):
        query = f"({query})"
    return f"{query} sourcelang:{language}"


def format_gdelt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def split_windows(start_date: datetime, end_date: datetime, window_days: int) -> list[tuple[datetime, datetime]]:
    windows = []
    cursor = start_date
    while cursor < end_date:
        window_end = min(cursor + timedelta(days=window_days), end_date)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


def is_permanent_query_error(text: str) -> bool:
    msg = safe_text(text).lower()
    patterns = [
        "queries containing or'd terms must be surrounded by ()",
        "parentheses may only be used around or'd statements",
        "your query was too short or too long",
        "must be within the last 3 months",
    ]
    return any(pattern in msg for pattern in patterns)


def fetch_gdelt_window(
    session: requests.Session,
    query_group: str,
    language: str,
    window_start: datetime,
    window_end: datetime,
    maxrecords: int,
    retries: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    query = resolve_query_for_language(query_group, language)
    full_query = build_query_string(query, language)
    params = {
        "query": full_query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": maxrecords,
        "startdatetime": format_gdelt_dt(window_start),
        "enddatetime": format_gdelt_dt(window_end),
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(BASE_URL, params=params, timeout=60)
            if response.status_code == 429:
                wait_time = pause_seconds * (attempt + 1)
                print(
                    f"[429] {query_group}/{language} {window_start.date()}->{window_end.date()} -> sleep {wait_time:.0f}s"
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            content_type = safe_text(response.headers.get("content-type", "")).lower()
            if "json" not in content_type:
                body_preview = response.text[:240]
                if is_permanent_query_error(body_preview):
                    raise ValueError(f"Permanent query error: {body_preview}")
                raise ValueError(
                    f"Non-JSON response | status={response.status_code} | content_type={content_type} | body={body_preview!r}"
                )

            payload = response.json()
            articles = payload.get("articles", [])
            df = pd.DataFrame(articles)
            if df.empty:
                log = {
                    "query_group": query_group,
                    "language": language,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "status": "ok",
                    "rows": 0,
                    "error": "",
                }
                return df, log

            df["requested_lang"] = language
            df["query_group"] = query_group
            df["query_used"] = query
            log = {
                "query_group": query_group,
                "language": language,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "status": "ok",
                "rows": int(len(df)),
                "error": "",
            }
            return df, log
        except Exception as exc:
            last_error = exc
            body = safe_text(exc)
            if is_permanent_query_error(body):
                break
            wait_time = pause_seconds * attempt
            print(
                f"[Retry {attempt}/{retries}] {query_group}/{language} {window_start.date()}->{window_end.date()} -> {exc}"
            )
            time.sleep(wait_time)

    log = {
        "query_group": query_group,
        "language": language,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "status": "failed",
        "rows": 0,
        "error": safe_text(last_error),
    }
    return pd.DataFrame(), log


def get_domain_quality(domain: str) -> str:
    domain = safe_text(domain).lower().replace("www.", "")
    if domain in HIGH_SIGNAL_DOMAINS:
        return "high_signal"
    if domain in LOW_SIGNAL_DOMAINS:
        return "low_signal"
    return "neutral"


def classify_relevance(text: str) -> tuple[str, list[str], list[str], list[str], list[str], list[str]]:
    direct_hits = extract_hits(text, BUSINESS_DIRECT_TERMS)
    anchor_hits = extract_hits(text, COMMODITY_ANCHORS)
    model_signal_hits = extract_hits(text, MODEL_SIGNAL_TERMS)
    industry_hits = extract_hits(text, INDUSTRY_STRUCTURE_TERMS)
    product_hits = extract_hits(text, PRODUCT_NOISE_TERMS)
    event_hits = extract_hits(text, EVENT_NOISE_TERMS)
    finance_hits = extract_hits(text, FINANCE_NOISE_TERMS)

    if direct_hits:
        return "direct", direct_hits, model_signal_hits, industry_hits, product_hits + event_hits, finance_hits
    if anchor_hits and model_signal_hits:
        return "indirect", anchor_hits, model_signal_hits, industry_hits, product_hits + event_hits, finance_hits
    if anchor_hits and industry_hits:
        return "industry", anchor_hits, model_signal_hits, industry_hits, product_hits + event_hits, finance_hits
    if model_signal_hits:
        return "context_only", anchor_hits, model_signal_hits, industry_hits, product_hits + event_hits, finance_hits
    return "other", anchor_hits, model_signal_hits, industry_hits, product_hits + event_hits, finance_hits


def evaluate_title_strength(title: str) -> tuple[list[str], list[str], list[str], list[str], bool, bool]:
    direct_hits = extract_hits(title, BUSINESS_DIRECT_TERMS)
    anchor_hits = extract_hits(title, COMMODITY_ANCHORS)
    signal_hits = extract_hits(title, MODEL_SIGNAL_TERMS)
    industry_hits = extract_hits(title, INDUSTRY_STRUCTURE_TERMS)
    other_metal_hits = extract_hits(title, OTHER_METAL_TERMS)
    title_model_signal = bool(direct_hits) or bool(anchor_hits and signal_hits)
    title_industry_context = bool(anchor_hits and industry_hits)
    return direct_hits, anchor_hits, signal_hits, other_metal_hits, title_model_signal, title_industry_context


def detect_noise(
    text: str,
    relevance: str,
    product_event_hits: list[str],
    finance_hits: list[str],
    domain_quality: str,
) -> tuple[bool, list[str]]:
    noise_hits = sorted(set(extract_hits(text, NOISE_KEYWORDS) + list(product_event_hits) + list(finance_hits)))
    suspected_noise = relevance in {"context_only", "other"}
    if product_event_hits or finance_hits:
        suspected_noise = True
    if domain_quality == "low_signal":
        suspected_noise = True
        noise_hits = sorted(set(noise_hits + ["low_signal_domain"]))
    return suspected_noise, noise_hits


def assign_usage_bucket(row: pd.Series) -> str:
    if row["is_suspected_noise"]:
        return "rejected_noise"
    if row["domain_quality"] == "low_signal":
        return "rejected_noise"
    if row["relevance"] in {"direct", "indirect"} and row["title_model_signal"]:
        return "candidate_model"
    if row["relevance"] in {"direct", "indirect", "industry"} and row["title_industry_context"]:
        return "candidate_readonly"
    return "rejected_noise"


def normalize_news(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    title_col = pick_col(raw, ["title"])
    snippet_col = pick_col(raw, ["description", "snippet"])
    url_col = pick_col(raw, ["url"])
    lang_col = pick_col(raw, ["language", "requested_lang"])
    domain_col = pick_col(raw, ["domain"])
    country_col = pick_col(raw, ["sourcecountry"])
    date_col = pick_col(raw, ["seendate", "date", "published", "pubdate"])

    news = pd.DataFrame()
    news["title"] = raw[title_col] if title_col else ""
    news["snippet"] = raw[snippet_col] if snippet_col else ""
    news["url"] = raw[url_col] if url_col else ""
    news["language"] = raw[lang_col] if lang_col else raw.get("requested_lang", "")
    news["domain"] = raw[domain_col] if domain_col else raw["url"].map(extract_domain)
    news["sourcecountry"] = raw[country_col] if country_col else ""
    news["raw_date"] = raw[date_col] if date_col else None
    news["requested_lang"] = raw.get("requested_lang", "")
    news["query_group"] = raw.get("query_group", "")
    news["query_used"] = raw.get("query_used", "")

    for column in ["title", "snippet", "url", "language", "domain", "sourcecountry", "requested_lang", "query_group", "query_used"]:
        news[column] = news[column].map(safe_text)

    news["domain"] = news["domain"].where(news["domain"].ne(""), news["url"].map(extract_domain))
    news["domain_quality"] = news["domain"].map(get_domain_quality)
    news["news_datetime"] = news["raw_date"].map(parse_news_date)
    news["news_date"] = pd.to_datetime(news["news_datetime"], utc=True).dt.date
    news["text_for_audit"] = (news["title"] + " " + news["snippet"]).str.strip()
    news["title_norm"] = news["title"].map(normalize_text)

    news = news.drop_duplicates(subset=["title", "url"]).reset_index(drop=True)

    classified = news["text_for_audit"].map(classify_relevance)
    news["relevance"] = classified.map(lambda item: item[0])
    news["core_hits"] = classified.map(lambda item: ", ".join(item[1]))
    news["context_hits"] = classified.map(lambda item: ", ".join(item[2]))
    news["industry_hits"] = classified.map(lambda item: ", ".join(item[3]))
    news["product_event_hits"] = classified.map(lambda item: ", ".join(sorted(set(item[4]))))
    news["finance_noise_hits"] = classified.map(lambda item: ", ".join(item[5]))
    news["has_core_term"] = news["core_hits"].ne("")
    news["has_context_term"] = news["context_hits"].ne("")
    news["has_industry_term"] = news["industry_hits"].ne("")
    news["has_product_event_term"] = news["product_event_hits"].ne("")
    news["has_finance_noise_term"] = news["finance_noise_hits"].ne("")

    title_eval = news["title"].map(evaluate_title_strength)
    news["title_direct_hits"] = title_eval.map(lambda item: ", ".join(item[0]))
    news["title_anchor_hits"] = title_eval.map(lambda item: ", ".join(item[1]))
    news["title_signal_hits"] = title_eval.map(lambda item: ", ".join(item[2]))
    news["title_other_metal_hits"] = title_eval.map(lambda item: ", ".join(item[3]))
    news["title_model_signal"] = title_eval.map(lambda item: bool(item[4]))
    news["title_industry_context"] = title_eval.map(lambda item: bool(item[5]))

    noise_info = news.apply(
        lambda row: detect_noise(
            row["text_for_audit"],
            row["relevance"],
            [item.strip() for item in row["product_event_hits"].split(",") if item.strip()],
            [item.strip() for item in row["finance_noise_hits"].split(",") if item.strip()],
            row["domain_quality"],
        ),
        axis=1,
    )
    news["is_suspected_noise"] = noise_info.map(lambda item: item[0])
    news["noise_hits"] = noise_info.map(lambda item: ", ".join(item[1]))
    news["usage_bucket"] = news.apply(assign_usage_bucket, axis=1)

    news = news.sort_values(["news_datetime", "query_group", "language"], ascending=[False, True, True]).reset_index(drop=True)
    return news


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    languages = [item.strip().lower() for item in args.languages.split(",") if item.strip()]
    prefix = args.prefix or f"gdelt_backfill_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"

    raw_path = PROJECT_ROOT / "data/news/raw" / f"{prefix}_raw.csv"
    clean_path = PROJECT_ROOT / "data/news/staging" / f"{prefix}_clean.csv"
    model_path = PROJECT_ROOT / "data/news/staging" / f"{prefix}_candidate_model.csv"
    readonly_path = PROJECT_ROOT / "data/news/staging" / f"{prefix}_candidate_readonly.csv"
    rejected_path = PROJECT_ROOT / "data/news/staging" / f"{prefix}_rejected_noise.csv"
    fetch_log_path = PROJECT_ROOT / "data/news/staging" / f"{prefix}_fetch_log.csv"

    windows = split_windows(start_date, end_date, args.window_days)
    print("Backfill range :", args.start_date, "->", args.end_date)
    print("Languages      :", ", ".join(languages))
    print("Query groups   :", ", ".join(QUERY_MAP.keys()))
    print("Windows        :", len(windows))
    print("Prefix         :", prefix)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    frames: list[pd.DataFrame] = []
    fetch_logs: list[dict] = []

    for query_group in QUERY_MAP:
        for language in languages:
            for idx, (window_start, window_end) in enumerate(windows, start=1):
                print(
                    f"Fetch -> {query_group} | {language} | window {idx}/{len(windows)} | "
                    f"{window_start.date()} -> {(window_end - timedelta(seconds=1)).date()}"
                )
                df_part, log = fetch_gdelt_window(
                    session=session,
                    query_group=query_group,
                    language=language,
                    window_start=window_start,
                    window_end=window_end,
                    maxrecords=args.maxrecords,
                    retries=args.retries,
                    pause_seconds=args.pause_seconds,
                )
                frames.append(df_part)
                fetch_logs.append(log)
                time.sleep(args.pause_seconds)

    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    fetch_log_df = pd.DataFrame(fetch_logs)

    ensure_parent_dir(raw_path)
    ensure_parent_dir(clean_path)
    ensure_parent_dir(model_path)
    ensure_parent_dir(readonly_path)
    ensure_parent_dir(rejected_path)
    ensure_parent_dir(fetch_log_path)

    fetch_log_df.to_csv(fetch_log_path, index=False)
    print("Fetch log saved:", fetch_log_path)

    if raw.empty:
        raw.to_csv(raw_path, index=False)
        print("Tidak ada artikel mentah yang berhasil diambil.")
        print("Raw saved:", raw_path)
        return 0

    raw.to_csv(raw_path, index=False)
    news = normalize_news(raw)
    news.to_csv(clean_path, index=False)

    candidate_model = news[news["usage_bucket"] == "candidate_model"].copy()
    candidate_readonly = news[news["usage_bucket"] == "candidate_readonly"].copy()
    rejected_noise = news[news["usage_bucket"] == "rejected_noise"].copy()

    candidate_model.to_csv(model_path, index=False)
    candidate_readonly.to_csv(readonly_path, index=False)
    rejected_noise.to_csv(rejected_path, index=False)

    summary = pd.DataFrame(
        [
            ("raw_rows", len(raw)),
            ("clean_rows", len(news)),
            ("candidate_model_rows", len(candidate_model)),
            ("candidate_readonly_rows", len(candidate_readonly)),
            ("rejected_noise_rows", len(rejected_noise)),
            ("candidate_model_share", round(float((news["usage_bucket"] == "candidate_model").mean()), 4)),
            ("candidate_readonly_share", round(float((news["usage_bucket"] == "candidate_readonly").mean()), 4)),
            ("noise_share", round(float(news["is_suspected_noise"].mean()), 4)),
            ("low_signal_domain_share", round(float(news["domain_quality"].eq("low_signal").mean()), 4)),
        ],
        columns=["metric", "value"],
    )
    print(summary.to_string(index=False))
    print("Saved raw       :", raw_path)
    print("Saved clean     :", clean_path)
    print("Saved candidate :", model_path)
    print("Saved readonly  :", readonly_path)
    print("Saved rejected  :", rejected_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
