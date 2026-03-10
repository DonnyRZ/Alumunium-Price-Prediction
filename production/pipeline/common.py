from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = ROOT / "production"
PRODUCTION_DATA_DIR = PRODUCTION_DIR / "data"

MODEL_SNAPSHOT_PATH = PRODUCTION_DATA_DIR / "model" / "latest_snapshot.json"
SENTIMENT_SNAPSHOT_PATH = PRODUCTION_DATA_DIR / "sentiment" / "latest_snapshot.json"
DASHBOARD_PAYLOAD_PATH = PRODUCTION_DATA_DIR / "dashboard" / "latest_dashboard_payload.json"
SENTIMENT_ARTICLES_PATH = PRODUCTION_DATA_DIR / "sentiment" / "latest_articles.csv"
SENTIMENT_DAILY_PATH = PRODUCTION_DATA_DIR / "sentiment" / "latest_daily.csv"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(data: dict, path: Path) -> Path:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def simple_signal_from_delta(delta_pct: float, neutral_band: float = 0.3) -> str:
    if delta_pct >= neutral_band:
        return "Bullish"
    if delta_pct <= -neutral_band:
        return "Bearish"
    return "Netral"
