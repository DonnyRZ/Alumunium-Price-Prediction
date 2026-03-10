#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.news.config import build_settings, ensure_parent_dir, has_real_api_key
from src.news.io import load_candidate_news, load_existing_scores, upsert_scores
from src.news.prompts import SYSTEM_PROMPT, build_user_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score candidate aluminium news with Gemini for market-impact sentiment.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV path. Default comes from NEWS_SENTIMENT_INPUT_PATH.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output scored CSV path. Default comes from NEWS_SENTIMENT_SCORED_OUTPUT_PATH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for pilot scoring.",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Rescore even if article_id already exists in output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and print pending rows without calling Gemini.",
    )
    return parser


def build_gemini_endpoint(model: str, api_key: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"


def parse_model_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            preview = text[:500].replace("\n", " ")
            raise ValueError(f"Gagal parse JSON dari output Gemini. Raw output: {preview!r}")
        payload = json.loads(match.group(0))
    required_keys = {
        "market_impact_score",
        "impact_label",
        "impact_channel",
        "confidence",
        "reason_short",
    }
    missing = required_keys.difference(payload)
    if missing:
        raise ValueError(f"Response JSON missing keys: {sorted(missing)}")

    payload["market_impact_score"] = float(payload["market_impact_score"])
    payload["confidence"] = float(payload["confidence"])
    payload["impact_label"] = str(payload["impact_label"]).strip().lower()
    payload["impact_channel"] = str(payload["impact_channel"]).strip().lower()
    payload["reason_short"] = str(payload["reason_short"]).strip()
    return payload


def extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini response has no candidates: {payload}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if not texts:
        raise ValueError(f"Gemini response candidate has no text parts: {payload}")

    return "\n".join(texts).strip()


def score_one_article(api_key: str, model: str, title: str, snippet: str, max_reason_chars: int) -> tuple[dict, str]:
    user_prompt = build_user_prompt(title=title, snippet=snippet, max_reason_chars=max_reason_chars)
    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    endpoint = build_gemini_endpoint(model=model, api_key=api_key)
    generation_config = {
        "temperature": 0.1,
        "maxOutputTokens": 800,
        "responseMimeType": "application/json",
    }
    if model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {
            "thinkingLevel": "LOW",
        }
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": generation_config,
    }
    response = requests.post(endpoint, json=body, timeout=90)
    response.raise_for_status()
    payload = response.json()
    output_text = extract_gemini_text(payload)
    try:
        parsed = parse_model_response(output_text)
    except Exception as exc:
        payload_preview = json.dumps(payload, ensure_ascii=False)[:1200]
        raise ValueError(
            f"Gagal parse output Gemini. Output text={output_text[:300]!r} | payload={payload_preview}"
        ) from exc
    return parsed, output_text


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings()

    input_path = Path(args.input) if args.input else settings.input_path
    output_path = Path(args.output) if args.output else settings.scored_output_path

    input_df = load_candidate_news(input_path)
    existing_df = load_existing_scores(output_path)

    if args.limit is not None:
        input_df = input_df.head(args.limit).copy()

    if args.force_rescore or existing_df.empty:
        pending_df = input_df.copy()
    else:
        done_ids = set(existing_df["article_id"].astype(str))
        pending_df = input_df[~input_df["article_id"].astype(str).isin(done_ids)].copy()

    print("Input path   :", input_path)
    print("Output path  :", output_path)
    print("Input rows   :", len(input_df))
    print("Pending rows :", len(pending_df))
    print("Model        :", settings.gemini_model)
    print("Prompt ver   :", settings.prompt_version)

    if pending_df.empty:
        print("Tidak ada artikel baru untuk discore.")
        return 0

    if args.dry_run:
        print("Dry run aktif. Tidak ada request ke Gemini.")
        return 0

    if not has_real_api_key(settings.gemini_api_key):
        raise SystemExit(
            "GEMINI_API_KEY di .env belum diisi. Ganti placeholder di `.env` lalu jalankan ulang."
        )

    ensure_parent_dir(output_path)
    scored_rows: list[dict] = []

    for index, row in enumerate(pending_df.itertuples(index=False), start=1):
        title = str(getattr(row, "title", "")).strip()
        snippet = str(getattr(row, "snippet", "") or "").strip()[: settings.max_snippet_chars]
        parsed, raw_output = score_one_article(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            title=title,
            snippet=snippet,
            max_reason_chars=settings.max_reason_chars,
        )

        scored_row = {
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
            "prompt_version": settings.prompt_version,
            "scored_model": settings.gemini_model,
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
            "market_impact_score": parsed["market_impact_score"],
            "impact_label": parsed["impact_label"],
            "impact_channel": parsed["impact_channel"],
            "confidence": parsed["confidence"],
            "reason_short": parsed["reason_short"][: settings.max_reason_chars],
            "raw_model_output": raw_output,
        }
        scored_rows.append(scored_row)

        combined_df = upsert_scores(existing_df, scored_rows)
        combined_df.to_csv(output_path, index=False)
        print(
            f"[{index}/{len(pending_df)}] {scored_row['impact_label']:>7} | "
            f"score={scored_row['market_impact_score']:+.2f} | "
            f"conf={scored_row['confidence']:.2f} | {title[:80]}"
        )
        time.sleep(settings.sleep_seconds)

    print("Scoring selesai.")
    print("Saved:", output_path)
    print("Rows :", len(upsert_scores(existing_df, scored_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
