from __future__ import annotations

import os


SPREADSHEET_NAME = os.getenv("PRODUCTION_SPREADSHEET_NAME", "Alumunium_Data_Master")

TAB_XGB_LATEST = "xgb_latest_prediction"
TAB_XGB_HISTORY = "xgb_prediction_history"
TAB_XGB_SUMMARY = "xgb_model_summary"
TAB_SENTIMENT_ARTICLES = "sentiment_articles_scored"
TAB_SENTIMENT_DAILY = "sentiment_daily"
TAB_PIPELINE_STATUS = "pipeline_status"

