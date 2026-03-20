from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TARGET_RAW_PATH = DATA_DIR / "LME - Alumunium - daily.csv"
DAILY_CORE_PATH = PROCESSED_DIR / "prophet_daily_core_v2.csv"
DAILY_EXPANDED_PATH = PROCESSED_DIR / "prophet_daily_expanded_v2.csv"

CORE_OUT_PATH = PROCESSED_DIR / "five_day_signal_core_v1.csv"
EXPANDED_OUT_PATH = PROCESSED_DIR / "five_day_signal_expanded_v1.csv"
METADATA_OUT_PATH = PROCESSED_DIR / "five_day_signal_metadata_v1.json"

CORE_OUT_PATH_V2 = PROCESSED_DIR / "five_day_signal_core_v2.csv"
EXPANDED_OUT_PATH_V2 = PROCESSED_DIR / "five_day_signal_expanded_v2.csv"
METADATA_OUT_PATH_V2 = PROCESSED_DIR / "five_day_signal_metadata_v2.json"

BLOCK_SIZE = 5
EPS = 1e-12

DIRTY_BLOCK_OHLC_FEATURES = [
    "current_block_open",
    "current_block_high",
    "current_block_low",
    "current_block_body",
    "current_block_range",
    "current_block_pct_range",
    "current_block_return",
    "current_block_realized_vol",
    "current_block_up_days",
]

CORE_FEATURE_COLS = [
    "target_ohlc_has_placeholder_one",
    "lag1_warehouse_stockclose",
    "lag1_lme_yesterdayofficialusd",
    "lag1_fx_eur",
    "lag1_brent_close",
    "lag1_brent_age_days",
]

EXPANDED_EXTRA_COLS = [
    "lag1_shfe_price_proxy",
    "lag1_shfe_openinterest",
    "qc_lag1_shfe_rows_per_date",
    "qc_lag1_shfe_price_from_settlement",
    "lag1_opt_core_total",
    "lag1_opt_official_share",
    "lag1_opt_call_share",
    "lag1_shfe_age_days",
    "qc_lag1_option_age_days",
]


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_processed_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"]).sort_values("ds")
    df = df.drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    return df


def _load_target_raw() -> pd.DataFrame:
    raw = pd.read_csv(TARGET_RAW_PATH, parse_dates=["startdatetime"]).sort_values("startdatetime")
    raw = raw.drop_duplicates(subset=["startdatetime"], keep="last").reset_index(drop=True)
    raw["target_ohlc_has_placeholder_one_raw"] = (
        raw[["priceopen", "pricehigh", "pricelow", "priceclose"]] == 1
    ).any(axis=1).astype(int)
    raw = raw.rename(columns={"startdatetime": "ds"})
    return raw[
        [
            "ds",
            "priceopen",
            "pricehigh",
            "pricelow",
            "priceclose",
            "target_ohlc_has_placeholder_one_raw",
        ]
    ]


def _merge_target_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    target = _load_target_raw()
    merged = daily.drop(columns=["target_ohlc_has_placeholder_one"], errors="ignore").merge(
        target, on="ds", how="left", validate="one_to_one"
    )

    ohlc_cols = ["priceopen", "pricehigh", "pricelow", "priceclose", "target_ohlc_has_placeholder_one_raw"]
    if merged[ohlc_cols].isna().any().any():
        missing = merged.loc[merged[ohlc_cols].isna().any(axis=1), "ds"].tolist()
        raise ValueError(f"Missing target OHLC rows after merge: {missing[:5]}")

    if "y" in merged.columns:
        if not np.allclose(merged["y"].to_numpy(), merged["priceclose"].to_numpy(), equal_nan=False):
            raise ValueError("Merged daily close does not match processed daily y column.")

    merged["target_ohlc_has_placeholder_one"] = merged["target_ohlc_has_placeholder_one_raw"].astype(int)
    merged = merged.drop(columns=["target_ohlc_has_placeholder_one_raw"])
    return merged


def _pct_change(first: float, last: float) -> float:
    if not np.isfinite(first) or abs(first) < EPS:
        return np.nan
    return (last / first) - 1.0


def _series_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    if clean.isna().any():
        raise ValueError(f"Unexpected missing values while aggregating {prefix}")

    first = float(clean.iloc[0])
    last = float(clean.iloc[-1])

    return {
        f"{prefix}_first": first,
        f"{prefix}_last": last,
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_change": last - first,
        f"{prefix}_pct_change": float(_pct_change(first, last)),
    }


def _block_target_metrics(group: pd.DataFrame) -> dict[str, float]:
    priceopen = group["priceopen"].astype(float)
    pricehigh = group["pricehigh"].astype(float)
    pricelow = group["pricelow"].astype(float)
    priceclose = group["priceclose"].astype(float)
    daily_close_ret = priceclose.pct_change().dropna()

    current_open = float(priceopen.iloc[0])
    current_high = float(pricehigh.max())
    current_low = float(pricelow.min())
    current_close = float(priceclose.iloc[-1])

    return {
        "current_block_open": current_open,
        "current_block_high": current_high,
        "current_block_low": current_low,
        "current_block_close": current_close,
        "current_block_body": current_close - current_open,
        "current_block_range": current_high - current_low,
        "current_block_pct_range": (
            (current_high - current_low) / current_open if abs(current_open) > EPS else np.nan
        ),
        "current_block_return": _pct_change(current_open, current_close),
        "current_block_realized_vol": float(daily_close_ret.std(ddof=0)) if len(daily_close_ret) else 0.0,
        "current_block_up_days": int((priceclose.diff() > 0).sum()),
        "current_block_placeholder_days": int(group["target_ohlc_has_placeholder_one"].sum()),
        "current_block_placeholder_any": int(group["target_ohlc_has_placeholder_one"].max()),
        "current_block_ohlc_dirty": int(group["target_ohlc_has_placeholder_one"].max()),
        "current_block_calendar_span_days": int((group["ds"].iloc[-1] - group["ds"].iloc[0]).days),
        "block_end_year": int(group["ds"].iloc[-1].year),
        "block_end_month": int(group["ds"].iloc[-1].month),
        "block_end_quarter": int(((group["ds"].iloc[-1].month - 1) // 3) + 1),
        "block_end_dow": int(group["ds"].iloc[-1].dayofweek),
    }


def _build_block_dataset(
    daily: pd.DataFrame,
    feature_cols: list[str],
    dataset_name: str,
    clean_dirty_blocks: bool = False,
) -> tuple[pd.DataFrame, dict]:
    work = daily.sort_values("ds").reset_index(drop=True).copy()
    work["block_id"] = np.arange(len(work)) // BLOCK_SIZE

    full_block_ids = (
        work.groupby("block_id", sort=True).size().loc[lambda s: s == BLOCK_SIZE].index.to_list()
    )
    work = work[work["block_id"].isin(full_block_ids)].reset_index(drop=True)

    rows: list[dict] = []
    for block_id, group in work.groupby("block_id", sort=True):
        group = group.sort_values("ds").reset_index(drop=True)
        row: dict[str, float | int | str | pd.Timestamp] = {
            "block_id": int(block_id),
            "ds": group["ds"].iloc[-1],
            "block_start_ds": group["ds"].iloc[0],
            "block_end_ds": group["ds"].iloc[-1],
            "block_size": int(len(group)),
        }
        row.update(_block_target_metrics(group))

        for col in feature_cols:
            row.update(_series_stats(group[col], col))

        rows.append(row)

    blocks = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    blocks["target_next_close"] = blocks["current_block_close"].shift(-1)
    blocks["target_ds"] = blocks["block_end_ds"].shift(-1)
    blocks["target_next_change"] = blocks["target_next_close"] - blocks["current_block_close"]
    blocks["target_next_return"] = blocks["target_next_close"] / blocks["current_block_close"] - 1.0
    blocks["target_next_log_return"] = np.log(
        blocks["target_next_close"] / blocks["current_block_close"]
    )
    blocks["target_next_direction"] = (blocks["target_next_return"] > 0).astype(int)
    blocks["naive_anchor_close"] = blocks["current_block_close"]
    blocks["naive_anchor_return"] = 0.0
    blocks["y"] = blocks["target_next_return"]

    blocks = blocks.dropna(subset=["target_next_close"]).reset_index(drop=True)
    blocks = blocks.fillna(0)

    if clean_dirty_blocks:
        dirty_mask = blocks["current_block_placeholder_any"].astype(int).eq(1)
        blocks["current_block_ohlc_dirty"] = dirty_mask.astype(int)
        for col in DIRTY_BLOCK_OHLC_FEATURES:
            if col in blocks.columns:
                blocks.loc[dirty_mask, col] = np.nan
    else:
        blocks["current_block_ohlc_dirty"] = blocks["current_block_placeholder_any"].astype(int)

    qc_columns = [
        "target_ohlc_has_placeholder_one_first",
        "target_ohlc_has_placeholder_one_last",
        "target_ohlc_has_placeholder_one_mean",
        "target_ohlc_has_placeholder_one_std",
        "target_ohlc_has_placeholder_one_min",
        "target_ohlc_has_placeholder_one_max",
        "target_ohlc_has_placeholder_one_change",
        "target_ohlc_has_placeholder_one_pct_change",
        "current_block_placeholder_days",
        "current_block_placeholder_any",
        "current_block_ohlc_dirty",
    ]

    model_feature_cols = [
        col
        for col in blocks.columns
        if col
        not in {
            "block_id",
            "ds",
            "block_start_ds",
            "block_end_ds",
            "block_size",
            "target_ds",
            "target_next_close",
            "target_next_change",
            "target_next_return",
            "target_next_log_return",
            "target_next_direction",
            "naive_anchor_close",
            "naive_anchor_return",
            "y",
        }
        and col not in qc_columns
        and not col.startswith("target_ohlc_has_placeholder_one_")
    ]

    metadata = {
        "dataset_name": dataset_name,
        "block_size": BLOCK_SIZE,
        "input_rows": int(len(daily)),
        "full_blocks_before_target_shift": int(len(rows)),
        "output_rows": int(len(blocks)),
        "tail_rows_dropped_before_blocking": int(len(daily) % BLOCK_SIZE),
        "tail_rows_dropped_after_target_shift": 1,
        "start_date": blocks["ds"].min().date().isoformat(),
        "end_date": blocks["ds"].max().date().isoformat(),
        "target_start_date": blocks["target_ds"].min().date().isoformat() if len(blocks) else None,
        "target_end_date": blocks["target_ds"].max().date().isoformat() if len(blocks) else None,
        "strategy": "non-overlapping 5-trading-day blocks; target is next block return",
        "target_definition": "y = next block return relative to current block close",
        "naive_anchor": "naive_anchor_close = current_block_close",
        "missing_value_policy": "residual NaNs from block summaries are filled with 0 after aggregation; dirty OHLC-derived block features are masked to NaN in cleaned datasets",
        "label_columns": [
            "target_ds",
            "target_next_close",
            "target_next_change",
            "target_next_return",
            "target_next_log_return",
            "target_next_direction",
            "y",
        ],
        "model_feature_count": int(len(model_feature_cols)),
        "model_feature_columns": model_feature_cols,
        "qc_columns": qc_columns,
        "dirty_block_ohlc_features": DIRTY_BLOCK_OHLC_FEATURES,
        "feature_source_columns": feature_cols,
    }
    return blocks, metadata


def build_core_dataset(version: str = "v1") -> tuple[pd.DataFrame, dict]:
    daily = _load_processed_daily(DAILY_CORE_PATH)
    daily = _merge_target_ohlc(daily)
    clean_dirty_blocks = version == "v2"
    return _build_block_dataset(
        daily,
        CORE_FEATURE_COLS,
        f"five_day_signal_core_{version}",
        clean_dirty_blocks=clean_dirty_blocks,
    )


def build_expanded_dataset(version: str = "v1") -> tuple[pd.DataFrame, dict]:
    daily = _load_processed_daily(DAILY_EXPANDED_PATH)
    daily = _merge_target_ohlc(daily)
    feature_cols = CORE_FEATURE_COLS + EXPANDED_EXTRA_COLS
    clean_dirty_blocks = version == "v2"
    return _build_block_dataset(
        daily,
        feature_cols,
        f"five_day_signal_expanded_{version}",
        clean_dirty_blocks=clean_dirty_blocks,
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    core_v1, core_meta_v1 = build_core_dataset("v1")
    expanded_v1, expanded_meta_v1 = build_expanded_dataset("v1")
    metadata_v1 = {
        "block_size": BLOCK_SIZE,
        "definition": "true 5-trading-day aggregated dataset built from daily observations",
        "core": core_meta_v1,
        "expanded": expanded_meta_v1,
    }

    core_v2, core_meta_v2 = build_core_dataset("v2")
    expanded_v2, expanded_meta_v2 = build_expanded_dataset("v2")
    metadata_v2 = {
        "block_size": BLOCK_SIZE,
        "definition": "true 5-trading-day aggregated dataset built from daily observations with dirty OHLC-derived block features masked to NaN",
        "core": core_meta_v2,
        "expanded": expanded_meta_v2,
    }

    _save_csv(core_v1, CORE_OUT_PATH)
    _save_csv(expanded_v1, EXPANDED_OUT_PATH)
    METADATA_OUT_PATH.write_text(json.dumps(metadata_v1, indent=2, default=str))

    _save_csv(core_v2, CORE_OUT_PATH_V2)
    _save_csv(expanded_v2, EXPANDED_OUT_PATH_V2)
    METADATA_OUT_PATH_V2.write_text(json.dumps(metadata_v2, indent=2, default=str))

    print(f"Saved core dataset to {CORE_OUT_PATH}")
    print(f"Saved expanded dataset to {EXPANDED_OUT_PATH}")
    print(json.dumps(metadata_v1, indent=2, default=str))
    print(f"Saved core dataset to {CORE_OUT_PATH_V2}")
    print(f"Saved expanded dataset to {EXPANDED_OUT_PATH_V2}")
    print(json.dumps(metadata_v2, indent=2, default=str))


if __name__ == "__main__":
    main()
