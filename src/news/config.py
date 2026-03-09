from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_if_available(env_path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        if not env_path.exists():
            return
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return

    load_dotenv(env_path, override=False)


@dataclass(frozen=True)
class NewsSentimentSettings:
    project_root: Path
    env_path: Path
    input_path: Path
    scored_output_path: Path
    daily_output_path: Path
    overlay_output_path: Path
    gemini_api_key: str
    gemini_model: str
    prompt_version: str
    sleep_seconds: float
    max_snippet_chars: int
    max_reason_chars: int
    confidence_threshold: float
    overlay_news_min_count: int
    overlay_min_high_conf_ratio: float
    overlay_medium_sentiment_threshold: float
    overlay_strong_sentiment_threshold: float


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_settings() -> NewsSentimentSettings:
    project_root = get_project_root()
    env_path = project_root / ".env"
    load_dotenv_if_available(env_path)

    return NewsSentimentSettings(
        project_root=project_root,
        env_path=env_path,
        input_path=project_root / os.getenv(
            "NEWS_SENTIMENT_INPUT_PATH",
            "data/news/staging/gdelt_eda_candidate_model_news_v4.csv",
        ),
        scored_output_path=project_root / os.getenv(
            "NEWS_SENTIMENT_SCORED_OUTPUT_PATH",
            "data/news/scored/gdelt_candidate_model_scored_v1.csv",
        ),
        daily_output_path=project_root / os.getenv(
            "NEWS_SENTIMENT_DAILY_OUTPUT_PATH",
            "data/news/features/gdelt_daily_sentiment_features_v1.csv",
        ),
        overlay_output_path=project_root / os.getenv(
            "NEWS_SENTIMENT_OVERLAY_OUTPUT_PATH",
            "data/news/features/gdelt_daily_overlay_signals_v1.csv",
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "PASTE_GEMINI_API_KEY_HERE"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        prompt_version=os.getenv("NEWS_SENTIMENT_PROMPT_VERSION", "pilot_v1"),
        sleep_seconds=float(os.getenv("NEWS_SENTIMENT_SLEEP_SECONDS", "1.0")),
        max_snippet_chars=int(os.getenv("NEWS_SENTIMENT_MAX_SNIPPET_CHARS", "1200")),
        max_reason_chars=int(os.getenv("NEWS_SENTIMENT_MAX_REASON_CHARS", "160")),
        confidence_threshold=float(os.getenv("NEWS_SENTIMENT_CONFIDENCE_THRESHOLD", "0.60")),
        overlay_news_min_count=int(os.getenv("NEWS_OVERLAY_MIN_NEWS_COUNT", "1")),
        overlay_min_high_conf_ratio=float(os.getenv("NEWS_OVERLAY_MIN_HIGH_CONF_RATIO", "0.50")),
        overlay_medium_sentiment_threshold=float(os.getenv("NEWS_OVERLAY_MEDIUM_SENTIMENT_THRESHOLD", "0.30")),
        overlay_strong_sentiment_threshold=float(os.getenv("NEWS_OVERLAY_STRONG_SENTIMENT_THRESHOLD", "0.70")),
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def has_real_api_key(api_key: str) -> bool:
    value = (api_key or "").strip()
    if not value:
        return False
    placeholders = {
        "PASTE_GEMINI_API_KEY_HERE",
        "YOUR_GEMINI_API_KEY",
        "GEMINI_API_KEY_HERE",
    }
    if value in placeholders:
        return False
    return True
