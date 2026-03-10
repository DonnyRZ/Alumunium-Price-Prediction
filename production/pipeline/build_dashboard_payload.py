from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.pipeline.common import (
    DASHBOARD_PAYLOAD_PATH,
    MODEL_SNAPSHOT_PATH,
    SENTIMENT_SNAPSHOT_PATH,
    read_json,
    simple_signal_from_delta,
    utc_now_iso,
    write_json,
)


def _freshness_label(latest_model_date: str, latest_news_date: str) -> str:
    if not latest_news_date:
        return "Belum ada news relevan terbaru. Dashboard berjalan dengan model saja."
    if latest_model_date == latest_news_date:
        return "Data model dan sentiment sama-sama terbaru"
    return "Tanggal model dan sentiment berbeda, perlu dibaca dengan hati-hati"


def build_dashboard_payload() -> Path:
    model = read_json(MODEL_SNAPSHOT_PATH)
    sentiment = read_json(SENTIMENT_SNAPSHOT_PATH)

    delta_pct = float(model["delta_pct"])
    model_signal = simple_signal_from_delta(delta_pct)
    latest_daily = sentiment.get("latest_daily") or {
        "tone_label": "Belum ada news",
        "market_sentiment_mean": 0.0,
        "dominant_channel": "Belum ada",
    }

    executive_note = (
        f"Model utama memberi sinyal {model_signal.lower()} untuk {model['forecast_date']} "
        f"dengan perubahan {delta_pct:+.2f}%. "
        f"Sentiment terbaru bersifat {latest_daily['tone_label'].lower()} "
        f"dengan channel utama {latest_daily['dominant_channel'].lower()}."
    )

    payload = {
        "generated_at_utc": utc_now_iso(),
        "executive": {
            "forecast_date": model["forecast_date"],
            "latest_data_date": model["latest_data_date"],
            "current_price": model["current_price"],
            "predicted_price_t1": model["pred_price_final_t1"],
            "predicted_price_p10_t1": model["pred_price_p10_t1"],
            "predicted_price_p90_t1": model["pred_price_p90_t1"],
            "baseline_price_t1": model["baseline_price_t1"],
            "delta_pct": model["delta_pct"],
            "signal": model_signal,
            "sentiment_label": latest_daily["tone_label"],
            "sentiment_score": latest_daily["market_sentiment_mean"],
            "headline_note": executive_note,
        },
        "model": model,
        "sentiment": sentiment,
        "data_health": {
            "freshness_note": _freshness_label(model["latest_data_date"], sentiment["latest_news_date"]),
            "model_snapshot_path": str(MODEL_SNAPSHOT_PATH.relative_to(ROOT)),
            "sentiment_snapshot_path": str(SENTIMENT_SNAPSHOT_PATH.relative_to(ROOT)),
        },
    }
    return write_json(payload, DASHBOARD_PAYLOAD_PATH)


if __name__ == "__main__":
    out = build_dashboard_payload()
    print(f"saved: {out}")
