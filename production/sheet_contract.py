from __future__ import annotations

import os


SPREADSHEET_NAME = os.getenv("PRODUCTION_SPREADSHEET_NAME", "Alumunium_Data_Master")

# Final workbook layout for the marketing-facing production dashboard.
TAB_RUN_STATUS = "run_status"
TAB_SIGNAL_LATEST = "signal_latest"
TAB_SIGNAL_HISTORY = "signal_history"
TAB_SIGNAL_METRICS = "signal_model_metrics"
TAB_MARKET_CONTEXT_DAILY = "market_context_daily"
TAB_ARTICLES_SCORED = "articles_scored"
TAB_AI_BRIEF_LATEST = "ai_market_brief_latest"
TAB_AI_BRIEF_HISTORY = "ai_market_brief_history"
TAB_MARKETING_PLAYBOOK = "marketing_playbook"
TAB_ACCOUNT_PRIORITY_QUEUE = "account_priority_queue"
TAB_QUOTE_DECISION_LOG = "quote_decision_log"

MARKETING_TABS = [
    TAB_RUN_STATUS,
    TAB_SIGNAL_LATEST,
    TAB_SIGNAL_HISTORY,
    TAB_SIGNAL_METRICS,
    TAB_MARKET_CONTEXT_DAILY,
    TAB_ARTICLES_SCORED,
    TAB_AI_BRIEF_LATEST,
    TAB_AI_BRIEF_HISTORY,
    TAB_MARKETING_PLAYBOOK,
    TAB_ACCOUNT_PRIORITY_QUEUE,
    TAB_QUOTE_DECISION_LOG,
]

# Legacy tabs kept only so old modules do not break if imported accidentally.
TAB_XGB_LATEST = "xgb_latest_prediction"
TAB_XGB_HISTORY = "xgb_prediction_history"
TAB_XGB_SUMMARY = "xgb_model_summary"
TAB_SENTIMENT_ARTICLES = "sentiment_articles_scored"
TAB_SENTIMENT_DAILY = "sentiment_daily"
TAB_PIPELINE_STATUS = "pipeline_status"
