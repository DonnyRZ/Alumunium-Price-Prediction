from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXOG_DIR = DATA_DIR / "Exogenous"
OUT_DIR = DATA_DIR / "processed"

TARGET_PATH = DATA_DIR / "LME - Alumunium - daily.csv"
CORE_OUT_PATH = OUT_DIR / "prophet_daily_core_v2.csv"
EXPANDED_OUT_PATH = OUT_DIR / "prophet_daily_expanded_v2.csv"
METADATA_OUT_PATH = OUT_DIR / "prophet_dataset_metadata_v2.json"

CORE_START = pd.Timestamp("2011-05-16")
EXPANDED_START = pd.Timestamp("2013-07-01")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_csv(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    return df.reset_index(drop=True)


def build_target() -> pd.DataFrame:
    raw = pd.read_csv(TARGET_PATH, parse_dates=["startdatetime"]).sort_values("startdatetime")
    raw = raw.drop_duplicates(subset=["startdatetime"], keep="last").reset_index(drop=True)

    has_placeholder_one = (raw[["priceopen", "pricehigh", "pricelow", "priceclose"]] == 1).any(axis=1)

    target = raw[["startdatetime", "priceclose"]].rename(
        columns={"startdatetime": "ds", "priceclose": "y"}
    )
    target["target_ohlc_has_placeholder_one"] = has_placeholder_one.astype(int)
    return target


def build_summary_stock_source() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "Summary-Warehouse-Stock.csv", "datestamp")
    return df[["datestamp", "stockclose"]].rename(
        columns={"datestamp": "ds", "stockclose": "warehouse_stockclose"}
    )


def build_lme_official_source() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "LME - Volumes.csv", "datestamp")
    return df[["datestamp", "yesterdayofficialusd"]].rename(
        columns={"datestamp": "ds", "yesterdayofficialusd": "lme_yesterdayofficialusd"}
    )


def build_fx_source() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "LME-Official-FX.csv", "datestamp")
    df.loc[df["eur"] == 0, "eur"] = np.nan
    df = df.dropna(subset=["eur"]).reset_index(drop=True)
    return df[["datestamp", "eur"]].rename(columns={"datestamp": "ds", "eur": "fx_eur"})


def build_brent_source() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "World-Indices-Brent Crude-Oil-Daily-OHLC.csv", "startdatetime")
    return df[["startdatetime", "priceclose"]].rename(
        columns={"startdatetime": "ds", "priceclose": "brent_close"}
    )


def _select_shfe_active(group: pd.DataFrame) -> pd.Series:
    chosen = group.sort_values(
        by=["openinterest", "close", "settlement"],
        ascending=[False, False, False],
    ).iloc[0]
    close_value = chosen["close"]
    price_from_settlement = int(close_value == 0)
    price_proxy = chosen["settlement"] if close_value == 0 else close_value

    return pd.Series(
        {
            "shfe_price_proxy": float(price_proxy),
            "shfe_openinterest": int(chosen["openinterest"]),
            "shfe_rows_per_date": int(len(group)),
            "shfe_price_from_settlement": price_from_settlement,
        }
    )


def build_shfe_source() -> pd.DataFrame:
    raw = pd.read_csv(EXOG_DIR / "SHFE-Alumunium-Active.csv", parse_dates=["datestamp"]).sort_values(
        ["datestamp", "prompt"]
    )
    grouped = raw.groupby("datestamp").apply(_select_shfe_active, include_groups=False)
    grouped = grouped.reset_index().rename(columns={"datestamp": "ds"})
    return grouped.reset_index(drop=True)


def build_option_source() -> pd.DataFrame:
    raw = pd.read_csv(
        EXOG_DIR / "LME-Aluminium- Option -Volumes.csv", parse_dates=["datestamp"]
    ).sort_values("datestamp")

    official_cols = [
        "yesterdayofficialusd",
        "yesterdayofficialgbp",
        "yesterdayofficialeur",
        "yesterdayofficialjpy",
    ]
    unofficial_cols = ["unofficialusd", "unofficialgbp", "unofficialeur", "unofficialjpy"]

    raw["opt_official_total"] = raw[official_cols].sum(axis=1)
    raw["opt_unofficial_total"] = raw[unofficial_cols].sum(axis=1)
    raw["opt_core_total"] = raw["opt_official_total"] + raw["opt_unofficial_total"]
    raw["opt_core_call_total"] = np.where(raw["pcind"] == "C", raw["opt_core_total"], 0.0)

    grouped = raw.groupby("datestamp", as_index=False).agg(
        opt_core_total=("opt_core_total", "sum"),
        opt_official_total=("opt_official_total", "sum"),
        opt_unofficial_total=("opt_unofficial_total", "sum"),
        opt_core_call_total=("opt_core_call_total", "sum"),
    )
    grouped["opt_official_share"] = grouped["opt_official_total"] / grouped["opt_core_total"].replace(
        0, np.nan
    )
    grouped["opt_call_share"] = grouped["opt_core_call_total"] / grouped["opt_core_total"].replace(
        0, np.nan
    )
    grouped = grouped.drop(columns=["opt_core_call_total", "opt_official_total", "opt_unofficial_total"])
    return grouped.rename(columns={"datestamp": "ds"})


def _merge_last_known(
    base: pd.DataFrame,
    source: pd.DataFrame,
    source_name: str,
    *,
    keep_source_date: bool = False,
) -> pd.DataFrame:
    left = base.sort_values("ds").reset_index(drop=True)
    right = source.sort_values("ds").reset_index(drop=True)
    source_date_col = f"{source_name}_source_ds"
    right = right.rename(columns={"ds": source_date_col})

    merged = pd.merge_asof(
        left,
        right,
        left_on="ds",
        right_on=source_date_col,
        direction="backward",
    )

    if not keep_source_date:
        merged = merged.drop(columns=[source_date_col])
    return merged


def build_core_base() -> pd.DataFrame:
    base = build_target()
    base = _merge_last_known(base, build_summary_stock_source(), "warehouse")
    base = _merge_last_known(base, build_lme_official_source(), "lme_volume")
    base = _merge_last_known(base, build_fx_source(), "fx")
    base = _merge_last_known(base, build_brent_source(), "brent", keep_source_date=True)
    return base


def build_expanded_base(core_base: pd.DataFrame) -> pd.DataFrame:
    expanded = _merge_last_known(core_base, build_shfe_source(), "shfe", keep_source_date=True)
    expanded = _merge_last_known(expanded, build_option_source(), "option", keep_source_date=True)
    return expanded


def _make_forecast_safe(
    df: pd.DataFrame,
    feature_cols: list[str],
    start_date: pd.Timestamp,
    *,
    source_age_names: list[str] | None = None,
) -> pd.DataFrame:
    source_age_names = source_age_names or []
    safe = df.sort_values("ds").reset_index(drop=True).copy()
    lag_source_cols = [f"{name}_source_ds" for name in source_age_names]
    safe[feature_cols + lag_source_cols] = safe[feature_cols + lag_source_cols].shift(1)
    rename_map = {col: f"lag1_{col}" for col in feature_cols}
    rename_map.update({col: f"lag1_{col}" for col in lag_source_cols})
    safe = safe.rename(columns=rename_map)

    for name in source_age_names:
        safe[f"lag1_{name}_age_days"] = (
            safe["ds"] - safe[f"lag1_{name}_source_ds"]
        ).dt.days.astype("float")

    safe = safe[safe["ds"] >= start_date].reset_index(drop=True)

    lagged_cols = [f"lag1_{col}" for col in feature_cols] + [
        f"lag1_{name}_age_days" for name in source_age_names
    ]
    safe = safe.dropna(subset=lagged_cols).reset_index(drop=True)

    int_like_cols = [
        col
        for col in [*lagged_cols, *(f"lag1_{col}" for col in lag_source_cols)]
        if col.endswith("_age_days")
        or col.endswith("_rows_per_date")
        or col.endswith("_from_settlement")
    ]
    for col in int_like_cols:
        safe[col] = safe[col].astype(int)

    safe = safe.drop(columns=[f"lag1_{col}" for col in lag_source_cols])
    return safe


def build_core_dataset_v2() -> pd.DataFrame:
    core_base = build_core_base()
    core_features = [
        "warehouse_stockclose",
        "lme_yesterdayofficialusd",
        "fx_eur",
        "brent_close",
    ]
    return _make_forecast_safe(
        core_base,
        core_features,
        CORE_START,
        source_age_names=["brent"],
    )


def build_expanded_dataset_v2() -> pd.DataFrame:
    core_base = build_core_base()
    expanded_base = build_expanded_base(core_base)
    expanded_features = [
        "warehouse_stockclose",
        "lme_yesterdayofficialusd",
        "fx_eur",
        "brent_close",
        "shfe_price_proxy",
        "shfe_openinterest",
        "opt_core_total",
        "opt_official_share",
        "opt_call_share",
        "shfe_rows_per_date",
        "shfe_price_from_settlement",
    ]
    expanded = _make_forecast_safe(
        expanded_base,
        expanded_features,
        EXPANDED_START,
        source_age_names=["brent", "shfe", "option"],
    )
    return expanded.rename(
        columns={
            "lag1_shfe_rows_per_date": "qc_lag1_shfe_rows_per_date",
            "lag1_shfe_price_from_settlement": "qc_lag1_shfe_price_from_settlement",
            "lag1_option_age_days": "qc_lag1_option_age_days",
        }
    )


def build_metadata(core: pd.DataFrame, expanded: pd.DataFrame) -> dict:
    return {
        "core_v2": {
            "rows": int(len(core)),
            "start_date": core["ds"].min().date().isoformat(),
            "end_date": core["ds"].max().date().isoformat(),
            "columns": list(core.columns),
            "strategy": "lag1 regressors on LME target calendar",
            "model_regressors": [
                "lag1_warehouse_stockclose",
                "lag1_lme_yesterdayofficialusd",
                "lag1_fx_eur",
                "lag1_brent_close",
                "lag1_brent_age_days",
            ],
        },
        "expanded_v2": {
            "rows": int(len(expanded)),
            "start_date": expanded["ds"].min().date().isoformat(),
            "end_date": expanded["ds"].max().date().isoformat(),
            "columns": list(expanded.columns),
            "strategy": "lag1 regressors on LME target calendar with last-known SHFE and option data",
            "model_regressors": [
                "lag1_warehouse_stockclose",
                "lag1_lme_yesterdayofficialusd",
                "lag1_fx_eur",
                "lag1_brent_close",
                "lag1_brent_age_days",
                "lag1_shfe_price_proxy",
                "lag1_shfe_openinterest",
                "lag1_shfe_age_days",
                "lag1_opt_core_total",
                "lag1_opt_official_share",
                "lag1_opt_call_share",
            ],
            "qc_columns": [
                "target_ohlc_has_placeholder_one",
                "qc_lag1_shfe_rows_per_date",
                "qc_lag1_shfe_price_from_settlement",
                "qc_lag1_option_age_days",
            ],
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    core = build_core_dataset_v2()
    expanded = build_expanded_dataset_v2()
    metadata = build_metadata(core, expanded)

    _save_csv(core, CORE_OUT_PATH)
    _save_csv(expanded, EXPANDED_OUT_PATH)
    METADATA_OUT_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved core dataset to {CORE_OUT_PATH}")
    print(f"Saved expanded dataset to {EXPANDED_OUT_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
