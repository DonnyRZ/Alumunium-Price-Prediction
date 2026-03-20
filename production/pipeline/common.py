from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def simple_signal_from_delta(delta_pct: float, neutral_band: float = 0.3) -> str:
    if delta_pct >= neutral_band:
        return "Bullish"
    if delta_pct <= -neutral_band:
        return "Bearish"
    return "Netral"


def action_level_from_probability(probability: float | int | None) -> str:
    if probability is None or pd.isna(probability):
        return "watch"
    value = float(probability)
    if value < 0.50:
        return "ignore"
    if value <= 0.65:
        return "watch"
    return "act"


def action_label_id(action_level: str) -> str:
    mapping = {
        "ignore": "Abaikan",
        "watch": "Pantau",
        "act": "Tindak",
    }
    return mapping.get(str(action_level).strip().lower(), "Pantau")


def action_urgency(action_level: str) -> str:
    mapping = {
        "ignore": "Rendah",
        "watch": "Sedang",
        "act": "Tinggi",
    }
    return mapping.get(str(action_level).strip().lower(), "Sedang")
