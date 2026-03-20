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
CORE_OUT_PATH = OUT_DIR / "prophet_daily_core.csv"
EXPANDED_OUT_PATH = OUT_DIR / "prophet_daily_expanded.csv"
METADATA_OUT_PATH = OUT_DIR / "prophet_dataset_metadata.json"

CORE_START = pd.Timestamp("2011-05-16")
EXPANDED_START = pd.Timestamp("2013-07-01")


def _load_csv(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    return df.reset_index(drop=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_target() -> pd.DataFrame:
    raw = pd.read_csv(TARGET_PATH, parse_dates=["startdatetime"]).sort_values("startdatetime")
    raw = raw.drop_duplicates(subset=["startdatetime"], keep="last").reset_index(drop=True)

    has_placeholder_one = (raw[["priceopen", "pricehigh", "pricelow", "priceclose"]] == 1).any(axis=1)

    target = raw[["startdatetime", "priceclose"]].rename(
        columns={"startdatetime": "ds", "priceclose": "y"}
    )
    target["target_ohlc_has_placeholder_one"] = has_placeholder_one.astype(int)
    return target


def build_lme_volumes() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "LME - Volumes.csv", "datestamp")
    df["ring_total"] = df[["ring1", "ring2", "kerb1", "ring3", "ring4", "kerb2"]].sum(axis=1)
    return df[["datestamp", "ring_total", "yesterdayofficialusd", "interofficeusd"]].rename(
        columns={
            "datestamp": "ds",
            "yesterdayofficialusd": "lme_yesterdayofficialusd",
            "interofficeusd": "lme_interofficeusd",
        }
    )


def build_fx() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "LME-Official-FX.csv", "datestamp")
    for col in ["gbp", "eur", "jpy"]:
        df.loc[df[col] == 0, col] = np.nan
    return df[["datestamp", "eur"]].rename(columns={"datestamp": "ds", "eur": "fx_eur"})


def build_summary_stock() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "Summary-Warehouse-Stock.csv", "datestamp")
    return df[["datestamp", "stockclose"]].rename(
        columns={"datestamp": "ds", "stockclose": "stockclose"}
    )


def build_brent() -> pd.DataFrame:
    df = _load_csv(EXOG_DIR / "World-Indices-Brent Crude-Oil-Daily-OHLC.csv", "startdatetime")
    return df[["startdatetime", "priceclose"]].rename(
        columns={"startdatetime": "ds", "priceclose": "brent_close"}
    )


def build_shfe_active() -> pd.DataFrame:
    raw = pd.read_csv(EXOG_DIR / "SHFE-Alumunium-Active.csv", parse_dates=["datestamp"]).sort_values(
        "datestamp"
    )
    raw["close_nonzero"] = raw["close"].replace(0, np.nan)

    grouped = raw.groupby("datestamp", as_index=False).agg(
        shfe_close=("close_nonzero", "max"),
        shfe_settlement=("settlement", "mean"),
        shfe_openinterest=("openinterest", "sum"),
        shfe_rows_per_date=("prompt", "size"),
    )
    return grouped.rename(columns={"datestamp": "ds"})


def build_option_volumes() -> pd.DataFrame:
    raw = pd.read_csv(
        EXOG_DIR / "LME-Aluminium- Option -Volumes.csv", parse_dates=["datestamp"]
    ).sort_values("datestamp")

    raw["opt_all_total"] = raw[
        [
            "yesterdayofficialusd",
            "yesterdayofficialgbp",
            "yesterdayofficialeur",
            "yesterdayofficialjpy",
            "premarketusd",
            "premarketgbp",
            "premarketeur",
            "premarketjpy",
            "morningusd",
            "morninggbp",
            "morningeur",
            "morningjpy",
            "unofficialusd",
            "unofficialgbp",
            "unofficialeur",
            "unofficialjpy",
        ]
    ].sum(axis=1)
    raw["opt_official_total"] = raw[
        ["yesterdayofficialusd", "yesterdayofficialgbp", "yesterdayofficialeur", "yesterdayofficialjpy"]
    ].sum(axis=1)
    raw["opt_premarket_total"] = raw[
        ["premarketusd", "premarketgbp", "premarketeur", "premarketjpy"]
    ].sum(axis=1)
    raw["opt_morning_total"] = raw[
        ["morningusd", "morninggbp", "morningeur", "morningjpy"]
    ].sum(axis=1)
    raw["opt_unofficial_total"] = raw[
        ["unofficialusd", "unofficialgbp", "unofficialeur", "unofficialjpy"]
    ].sum(axis=1)

    raw["opt_call_total"] = np.where(raw["pcind"] == "C", raw["opt_all_total"], 0.0)
    raw["opt_t_total"] = np.where(raw["type"] == "T", raw["opt_all_total"], 0.0)

    grouped = raw.groupby("datestamp", as_index=False).agg(
        opt_all_total=("opt_all_total", "sum"),
        opt_official_total=("opt_official_total", "sum"),
        opt_premarket_total=("opt_premarket_total", "sum"),
        opt_morning_total=("opt_morning_total", "sum"),
        opt_unofficial_total=("opt_unofficial_total", "sum"),
        opt_call_total=("opt_call_total", "sum"),
        opt_t_total=("opt_t_total", "sum"),
    )
    grouped["opt_call_share"] = grouped["opt_call_total"] / grouped["opt_all_total"].replace(0, np.nan)
    grouped["opt_t_share"] = grouped["opt_t_total"] / grouped["opt_all_total"].replace(0, np.nan)
    grouped = grouped.drop(columns=["opt_call_total", "opt_t_total"])
    return grouped.rename(columns={"datestamp": "ds"})


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.merge(frame, on="ds", how="inner")
    return merged.sort_values("ds").reset_index(drop=True)


def build_core_dataset() -> pd.DataFrame:
    core = _merge_frames(
        [
            build_target(),
            build_summary_stock(),
            build_lme_volumes(),
            build_fx(),
            build_brent(),
        ]
    )
    core = core[core["ds"] >= CORE_START].reset_index(drop=True)
    return core


def build_expanded_dataset(core: pd.DataFrame | None = None) -> pd.DataFrame:
    if core is None:
        core = build_core_dataset()

    expanded = _merge_frames(
        [
            core,
            build_shfe_active(),
            build_option_volumes(),
        ]
    )
    expanded = expanded[expanded["ds"] >= EXPANDED_START].reset_index(drop=True)
    return expanded


def build_metadata(core: pd.DataFrame, expanded: pd.DataFrame) -> dict:
    return {
        "core": {
            "rows": int(len(core)),
            "start_date": core["ds"].min().date().isoformat(),
            "end_date": core["ds"].max().date().isoformat(),
            "columns": list(core.columns),
        },
        "expanded": {
            "rows": int(len(expanded)),
            "start_date": expanded["ds"].min().date().isoformat(),
            "end_date": expanded["ds"].max().date().isoformat(),
            "columns": list(expanded.columns),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    core = build_core_dataset()
    expanded = build_expanded_dataset(core=core)
    metadata = build_metadata(core, expanded)

    _save_csv(core, CORE_OUT_PATH)
    _save_csv(expanded, EXPANDED_OUT_PATH)
    METADATA_OUT_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved core dataset to {CORE_OUT_PATH}")
    print(f"Saved expanded dataset to {EXPANDED_OUT_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
