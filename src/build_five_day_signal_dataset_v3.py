from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_PATH = PROCESSED_DIR / "five_day_signal_expanded_v2.csv"
OUTPUT_PATH = PROCESSED_DIR / "five_day_signal_expanded_v3.csv"
METADATA_PATH = PROCESSED_DIR / "five_day_signal_metadata_v3.json"

EPS = 1e-12


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / (den.abs() + EPS)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    df = df.drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    return df


def add_block_shape_features(df: pd.DataFrame) -> list[str]:
    features = []
    df["block_body_to_range"] = safe_div(df["current_block_body"], df["current_block_range"])
    df["block_close_location"] = safe_div(df["current_block_close"] - df["current_block_low"], df["current_block_range"])
    df["block_upper_wick"] = df["current_block_high"] - df[["current_block_open", "current_block_close"]].max(axis=1)
    df["block_lower_wick"] = df[["current_block_open", "current_block_close"]].min(axis=1) - df["current_block_low"]
    df["block_wick_balance"] = df["block_upper_wick"] - df["block_lower_wick"]
    df["block_efficiency"] = safe_div(df["current_block_body"].abs(), df["current_block_range"])
    df["block_up_rate"] = safe_div(df["current_block_up_days"], df["block_size"])
    df["block_vol_adj_body"] = safe_div(df["current_block_body"], df["current_block_realized_vol"])
    df["block_range_to_open"] = safe_div(df["current_block_range"], df["current_block_open"])
    df["block_close_to_open"] = safe_div(df["current_block_close"], df["current_block_open"]) - 1.0
    features.extend(
        [
            "block_body_to_range",
            "block_close_location",
            "block_upper_wick",
            "block_lower_wick",
            "block_wick_balance",
            "block_efficiency",
            "block_up_rate",
            "block_vol_adj_body",
            "block_range_to_open",
            "block_close_to_open",
        ]
    )
    return features


def add_target_history_features(df: pd.DataFrame) -> list[str]:
    features = []
    shifted = df["y"].shift(1)
    for lag in range(1, 6):
        name = f"y_lag{lag}"
        df[name] = df["y"].shift(lag)
        features.append(name)

    for window in (3, 5):
        mean_name = f"y_roll_mean_{window}"
        std_name = f"y_roll_std_{window}"
        abs_mean_name = f"y_abs_roll_mean_{window}"
        trend_name = f"y_trend_strength_{window}"

        df[mean_name] = shifted.rolling(window, min_periods=1).mean()
        df[std_name] = shifted.rolling(window, min_periods=2).std(ddof=0)
        df[abs_mean_name] = shifted.abs().rolling(window, min_periods=1).mean()
        df[trend_name] = safe_div(df[mean_name], df[std_name])

        features.extend([mean_name, std_name, abs_mean_name, trend_name])

    df["y_momentum_3"] = df["y_lag1"] - df["y_lag3"]
    df["y_momentum_5"] = df["y_lag1"] - df["y_lag5"]
    df["y_abs_lag1"] = df["y_lag1"].abs()
    df["y_abs_lag3"] = df["y_lag3"].abs()
    df["y_abs_lag5"] = df["y_lag5"].abs()
    features.extend(["y_momentum_3", "y_momentum_5", "y_abs_lag1", "y_abs_lag3", "y_abs_lag5"])
    return features


def add_market_spreads(df: pd.DataFrame) -> list[str]:
    features = []
    df["spread_shfe_brent_last"] = df["lag1_shfe_price_proxy_last"] - df["lag1_brent_close_last"]
    df["spread_shfe_lme_last"] = df["lag1_shfe_price_proxy_last"] - df["lag1_lme_yesterdayofficialusd_last"]
    df["spread_lme_brent_last"] = df["lag1_lme_yesterdayofficialusd_last"] - df["lag1_brent_close_last"]

    df["spread_shfe_brent_change"] = df["lag1_shfe_price_proxy_change"] - df["lag1_brent_close_change"]
    df["spread_shfe_lme_change"] = df["lag1_shfe_price_proxy_change"] - df["lag1_lme_yesterdayofficialusd_change"]
    df["spread_lme_brent_change"] = df["lag1_lme_yesterdayofficialusd_change"] - df["lag1_brent_close_change"]

    df["opt_call_minus_official_share_last"] = df["lag1_opt_call_share_last"] - df["lag1_opt_official_share_last"]
    df["opt_call_minus_official_share_change"] = df["lag1_opt_call_share_change"] - df["lag1_opt_official_share_change"]
    df["opt_activity_strength_last"] = df["lag1_opt_core_total_last"] * df["lag1_opt_call_share_last"]
    df["opt_activity_strength_change"] = df["lag1_opt_core_total_change"] * df["lag1_opt_call_share_change"]
    df["stock_to_shfe_price_last"] = safe_div(df["lag1_warehouse_stockclose_last"], df["lag1_shfe_price_proxy_last"])
    df["stock_to_lme_price_last"] = safe_div(df["lag1_warehouse_stockclose_last"], df["lag1_lme_yesterdayofficialusd_last"])
    df["fx_brent_product_last"] = df["lag1_fx_eur_last"] * df["lag1_brent_close_last"]
    features.extend(
        [
            "spread_shfe_brent_last",
            "spread_shfe_lme_last",
            "spread_lme_brent_last",
            "spread_shfe_brent_change",
            "spread_shfe_lme_change",
            "spread_lme_brent_change",
            "opt_call_minus_official_share_last",
            "opt_call_minus_official_share_change",
            "opt_activity_strength_last",
            "opt_activity_strength_change",
            "stock_to_shfe_price_last",
            "stock_to_lme_price_last",
            "fx_brent_product_last",
        ]
    )
    return features


def add_freshness_features(df: pd.DataFrame) -> list[str]:
    features = []
    df["freshness_brent"] = 1.0 / (1.0 + df["lag1_brent_age_days_last"])
    df["freshness_shfe"] = 1.0 / (1.0 + df["lag1_shfe_age_days_last"])
    df["freshness_option"] = 1.0 / (1.0 + df["qc_lag1_option_age_days_last"])
    df["freshness_mean"] = df[["freshness_brent", "freshness_shfe", "freshness_option"]].mean(axis=1)
    df["freshness_min"] = df[["freshness_brent", "freshness_shfe", "freshness_option"]].min(axis=1)
    df["staleness_gap_shfe_brent"] = df["lag1_shfe_age_days_last"] - df["lag1_brent_age_days_last"]
    df["staleness_gap_option_shfe"] = df["qc_lag1_option_age_days_last"] - df["lag1_shfe_age_days_last"]
    df["staleness_max"] = df[["lag1_brent_age_days_last", "lag1_shfe_age_days_last", "qc_lag1_option_age_days_last"]].max(axis=1)
    features.extend(
        [
            "freshness_brent",
            "freshness_shfe",
            "freshness_option",
            "freshness_mean",
            "freshness_min",
            "staleness_gap_shfe_brent",
            "staleness_gap_option_shfe",
            "staleness_max",
        ]
    )
    return features


def build_dataset() -> tuple[pd.DataFrame, dict]:
    df = load_frame()
    feature_groups = {
        "block_shape": add_block_shape_features(df),
        "target_history": add_target_history_features(df),
        "market_spreads": add_market_spreads(df),
        "freshness": add_freshness_features(df),
    }

    df["target_next_direction"] = df["target_next_direction"].astype(int)
    df["signal_binary"] = (df["y"].abs() > float(df["y"].iloc[: int(len(df) * 0.7)].abs().median())).astype(int)

    metadata = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "base_dataset": "five_day_signal_expanded_v2",
        "engineered_feature_groups": feature_groups,
        "engineered_feature_count": int(sum(len(v) for v in feature_groups.values())),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_definition": "signal_binary = 1 if abs(y) > median(abs(y)) on first 70% of data",
    }
    return df, metadata


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved engineered dataset to {OUTPUT_PATH}")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
