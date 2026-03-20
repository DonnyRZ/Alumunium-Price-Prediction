from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import overwrite_sheet, read_sheet, upsert_sheet
from production.pipeline.common import action_label_id, utc_now_iso
from production.sheet_contract import (
    TAB_AI_BRIEF_HISTORY,
    TAB_AI_BRIEF_LATEST,
    TAB_ARTICLES_SCORED,
    TAB_MARKET_CONTEXT_DAILY,
    TAB_SIGNAL_LATEST,
)
from src.news.config import build_settings, has_real_api_key
from src.news.score_sentiment import build_gemini_endpoint, extract_gemini_text


LOCAL_DASHBOARD_DIR = ROOT / "production" / "data" / "dashboard"
LOCAL_BRIEF_JSON = LOCAL_DASHBOARD_DIR / "ai_market_brief_latest.json"
LOCAL_SIGNAL_JSON = ROOT / "production" / "data" / "model" / "signal_latest_snapshot.json"


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _impact_label_id(value: str) -> str:
    mapping = {
        "bullish": "cenderung naik",
        "bearish": "cenderung turun",
        "neutral": "netral",
    }
    return mapping.get(str(value).strip().lower(), str(value).strip().lower())


def _sanitize_error_message(message: str) -> str:
    return re.sub(r"key=[A-Za-z0-9_\-]+", "key=***", str(message))


def _parse_brief_json(text: str) -> dict:
    payload = json.loads(text)
    required = {
        "executive_summary",
        "why_it_matters",
        "recommended_action",
        "customer_talking_points",
        "internal_caution_note",
        "signal_rationale",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"AI brief response missing keys: {sorted(missing)}")
    if not isinstance(payload["customer_talking_points"], list):
        raise ValueError("customer_talking_points must be a list")
    return payload


def _latest_market_context() -> tuple[dict, list[dict]]:
    signal_df = read_sheet(TAB_SIGNAL_LATEST)
    if signal_df.empty:
        if not LOCAL_SIGNAL_JSON.exists():
            raise ValueError("Sheet signal_latest masih kosong dan artifact lokal signal_latest_snapshot.json belum ada.")
        latest_signal = json.loads(LOCAL_SIGNAL_JSON.read_text())
    else:
        signal_df = _coerce_numeric(
            signal_df,
            ["probability_action", "technical_threshold", "watch_threshold", "current_block_close", "freshness_mean", "freshness_min", "staleness_max"],
        )
        latest_signal = signal_df.iloc[-1].to_dict()

    daily_df = read_sheet(TAB_MARKET_CONTEXT_DAILY)
    if not daily_df.empty:
        daily_df = _coerce_numeric(
            daily_df,
            [
                "news_count_model",
                "market_sentiment_mean",
                "market_sentiment_sum",
                "bullish_ratio",
                "bearish_ratio",
                "high_confidence_ratio",
            ],
        )
        latest_daily = daily_df.sort_values("news_date").iloc[-1].to_dict()
    else:
        latest_daily = {}

    articles_df = read_sheet(TAB_ARTICLES_SCORED)
    if not articles_df.empty:
        articles_df = _coerce_numeric(articles_df, ["market_impact_score", "confidence"])
        articles_df = articles_df.sort_values("news_datetime", ascending=False).head(5)
        top_articles = articles_df[
            ["news_date", "title", "impact_label", "impact_channel", "market_impact_score", "confidence", "reason_short"]
        ].fillna("").to_dict(orient="records")
    else:
        top_articles = []

    return latest_signal, latest_daily, top_articles


def _fallback_brief(latest_signal: dict, latest_daily: dict, top_articles: list[dict]) -> dict:
    probability = float(latest_signal.get("probability_action", 0.0))
    action_level = str(latest_signal.get("action_level", "watch"))
    action_label = action_label_id(action_level)
    freshness_min = latest_signal.get("freshness_min", "")

    sentiment_mean = latest_daily.get("market_sentiment_mean")
    news_count = latest_daily.get("news_count_model", 0)
    if sentiment_mean is None or pd.isna(sentiment_mean):
        sentiment_text = "Belum ada konteks news yang cukup baru."
    elif float(sentiment_mean) >= 0.20:
        sentiment_text = f"News terbaru cenderung positif dengan intensitas {float(sentiment_mean):+.2f}."
    elif float(sentiment_mean) <= -0.20:
        sentiment_text = f"News terbaru cenderung negatif dengan intensitas {float(sentiment_mean):+.2f}."
    else:
        sentiment_text = f"News terbaru cenderung netral dengan intensitas {float(sentiment_mean):+.2f}."

    top_driver_text = ", ".join(
        [item for item in [latest_signal.get("top_driver_1", ""), latest_signal.get("top_driver_2", ""), latest_signal.get("top_driver_3", "")] if item]
    )
    top_article_text = ""
    if top_articles:
        lead = top_articles[0]
        top_article_text = (
            f"Headline terbaru yang paling relevan berarah {_impact_label_id(lead.get('impact_label', 'neutral'))} "
            f"pada topik {lead.get('impact_channel', 'unclear')}."
        )

    recommended_action = {
        "ignore": "Lanjutkan proses marketing normal dan tidak perlu eskalasi khusus.",
        "watch": "Pantau inquiry aktif, cek timing quote, dan siapkan market note singkat bila diperlukan.",
        "act": "Percepat review internal, prioritaskan inquiry penting, dan pertimbangkan validitas quote yang lebih ketat.",
    }.get(action_level, "Pantau pasar dan review konteks bisnis sebelum mengambil keputusan.")

    talking_points = [
        f"Pasar saat ini berada pada status {action_label.lower()} dengan probability {probability:.2f}.",
        sentiment_text,
        "Keputusan pelanggan sebaiknya tidak terlalu lama ditunda bila inquiry sedang aktif." if action_level == "act" else "Belum ada alasan kuat untuk menaikkan urgensi secara agresif.",
    ]
    if top_article_text:
        talking_points.append(top_article_text)

    return {
        "executive_summary": (
            f"Signal marketing berada pada level {action_label.lower()} dengan probability {probability:.2f}. "
            f"Ini berarti tim perlu {'memberi perhatian tambahan' if action_level != 'ignore' else 'menjalankan proses normal'}."
        ),
        "why_it_matters": (
            f"Driver model terbaru: {top_driver_text or 'tidak ada driver dominan yang kuat'}. "
            f"{sentiment_text}"
        ),
        "recommended_action": recommended_action,
        "customer_talking_points": talking_points[:4],
        "internal_caution_note": (
            f"Gunakan signal ini sebagai alat bantu, bukan keputusan tunggal. Freshness minimum data exogenous saat ini: {freshness_min}."
        ),
        "signal_rationale": str(latest_signal.get("signal_reason_short", "")),
    }


def _generate_ai_brief(latest_signal: dict, latest_daily: dict, top_articles: list[dict]) -> tuple[dict, str]:
    settings = build_settings()
    if not has_real_api_key(settings.gemini_api_key):
        return _fallback_brief(latest_signal, latest_daily, top_articles), "fallback_no_api_key"

    context = {
        "latest_signal": latest_signal,
        "latest_daily_sentiment": latest_daily,
        "top_articles": top_articles,
        "instructions": {
            "audience": "tim marketing INALUM",
            "language": "Bahasa Indonesia",
            "goal": "terjemahkan signal model menjadi brief bisnis yang bisa dipakai untuk keputusan marketing",
            "guardrails": [
                "jangan mengklaim harga pasti naik atau turun",
                "jangan menggantikan judgement manusia",
                "fokus pada prioritas, urgensi, dan komunikasi",
            ],
        },
    }

    prompt = f"""
Kamu adalah AI assistant untuk tim marketing INALUM.

Tugasmu adalah membuat brief bisnis yang singkat, hati-hati, dan bisa dipakai untuk pengambilan keputusan marketing.

Gunakan konteks berikut:
{json.dumps(context, ensure_ascii=False, default=str, indent=2)}

Keluarkan JSON valid dengan struktur persis berikut:
{{
  "executive_summary": "...",
  "why_it_matters": "...",
  "recommended_action": "...",
  "customer_talking_points": ["...", "...", "..."],
  "internal_caution_note": "...",
  "signal_rationale": "..."
}}

Aturan:
- gunakan Bahasa Indonesia
- ringkas, jelas, dan non-teknis
- jangan menyebut nama model statistik kecuali benar-benar perlu
- jangan overclaim kepastian harga
- customer_talking_points harus 3 sampai 4 butir
""".strip()

    endpoint = build_gemini_endpoint(settings.gemini_model, settings.gemini_api_key)
    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 900,
        "responseMimeType": "application/json",
    }
    if settings.gemini_model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "LOW"}

    response = requests.post(
        endpoint,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    text = extract_gemini_text(payload)
    return _parse_brief_json(text), settings.gemini_model


def build_ai_brief(write_sheets: bool = True) -> dict:
    generated_at = utc_now_iso()
    latest_signal, latest_daily, top_articles = _latest_market_context()

    try:
        brief, ai_status = _generate_ai_brief(latest_signal, latest_daily, top_articles)
        ai_status_note = ""
    except Exception as exc:
        brief = _fallback_brief(latest_signal, latest_daily, top_articles)
        ai_status = "fallback_after_error"
        ai_status_note = _sanitize_error_message(str(exc))

    latest_df = pd.DataFrame(
        [
            {
                "generated_at_utc": generated_at,
                "brief_date": str(latest_signal.get("latest_base_date", "")),
                "forecast_window_end": str(latest_signal.get("forecast_window_end", "")),
                "action_level": str(latest_signal.get("action_level", "")),
                "action_label_id": action_label_id(str(latest_signal.get("action_level", "watch"))),
                "probability_action": latest_signal.get("probability_action", ""),
                "ai_status": ai_status,
                "ai_status_note": ai_status_note,
                "executive_summary": brief["executive_summary"],
                "why_it_matters": brief["why_it_matters"],
                "recommended_action": brief["recommended_action"],
                "customer_talking_points": " | ".join(brief["customer_talking_points"]),
                "internal_caution_note": brief["internal_caution_note"],
                "signal_rationale": brief["signal_rationale"],
            }
        ]
    )

    history_df = latest_df.copy()
    LOCAL_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_BRIEF_JSON.write_text(json.dumps(latest_df.iloc[0].to_dict(), indent=2, default=str))

    if write_sheets:
        overwrite_sheet(latest_df, TAB_AI_BRIEF_LATEST)
        upsert_sheet(history_df, TAB_AI_BRIEF_HISTORY, key_columns=["brief_date", "forecast_window_end"])

    return {
        "generated_at_utc": generated_at,
        "ai_status": ai_status,
        "ai_status_note": ai_status_note,
        "brief_date": str(latest_signal.get("latest_base_date", "")),
        "forecast_window_end": str(latest_signal.get("forecast_window_end", "")),
        "local_brief_json": str(LOCAL_BRIEF_JSON.relative_to(ROOT)),
    }


if __name__ == "__main__":
    result = build_ai_brief(write_sheets=False)
    print(json.dumps(result, indent=2, default=str))
