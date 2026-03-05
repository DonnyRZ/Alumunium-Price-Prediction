#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def load_raw(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["is_missing"] = close.isna()
    df["is_stale"] = close.notna() & close.eq(close.shift(1))
    df["price_changed"] = close.notna() & (~df["is_stale"])

    # Streak group based on stale flag only (missing does not count as stale)
    df["streak_group"] = (df["is_stale"] != df["is_stale"].shift(1)).cumsum()
    streak_sizes = df[df["is_stale"]].groupby("streak_group").size()
    df["streak_len"] = df["streak_group"].map(streak_sizes).fillna(0).astype(int)

    if "Volume" in df.columns:
        df["volume_zero"] = df["Volume"] == 0
    else:
        df["volume_zero"] = False

    if set(["Open", "High", "Low", "Close"]).issubset(df.columns):
        valid = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        df["flat_candle"] = (
            (df["Open"] == df["High"])
            & (df["High"] == df["Low"])
            & (df["Low"] == df["Close"])
            & valid
        )
        upper_ref = pd.concat([df["Open"], df["Close"]], axis=1).max(axis=1)
        lower_ref = pd.concat([df["Open"], df["Close"]], axis=1).min(axis=1)
        df["is_ohlc_invalid"] = valid & (
            (df["High"] < upper_ref) | (df["Low"] > lower_ref) | (df["Low"] > df["High"])
        )
    else:
        df["flat_candle"] = False
        df["is_ohlc_invalid"] = False

    return df


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change(fill_method=None)
    df["is_outlier_iqr"], df["is_outlier_mad"] = _compute_outlier_masks(df["Return"])

    return df


def _compute_outlier_masks(ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    # IQR outliers (global)
    q1 = ret.quantile(0.25)
    q3 = ret.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier_iqr = ((ret < lower) | (ret > upper)).fillna(False)

    # Rolling MAD outliers (robust)
    window = 90
    min_periods = 30
    median = ret.rolling(window, min_periods=min_periods).median()
    mad = (ret - median).abs().rolling(window, min_periods=min_periods).median()
    mad = mad.replace(0, np.nan).clip(lower=1e-6)
    robust_z = 0.6745 * (ret - median) / mad
    is_outlier_mad = (robust_z.abs() > 5).fillna(False)

    return is_outlier_iqr.astype(bool), is_outlier_mad.astype(bool)


def flag_suspect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    streak_sizes = df[df["is_stale"]].groupby("streak_group").size()
    long_streak_groups = set(streak_sizes[streak_sizes >= 5].index)
    last_day_mask = df["streak_group"].isin(long_streak_groups) & df["is_stale"] & (
        df["streak_group"] != df["streak_group"].shift(-1)
    )
    next_day_idx = df.index.to_series().shift(-1)[last_day_mask].dropna()
    df["next_day_after_long_streak"] = df.index.isin(next_day_idx.values)

    df["is_suspect_outlier"] = (df["is_outlier_iqr"] | df["is_outlier_mad"]) & (
        df["volume_zero"] | df["next_day_after_long_streak"]
    )
    return df


def recompute_event_outlier_flags(df: pd.DataFrame, gap_threshold: int = 7) -> pd.DataFrame:
    """
    Recompute outlier/suspect flags on event-based data.

    This keeps flags consistent with the final Return series after row removals.
    """
    df = df.copy()

    if "volume_zero" not in df.columns:
        if "Volume" in df.columns:
            df["volume_zero"] = df["Volume"] == 0
        else:
            df["volume_zero"] = False

    df = detect_outliers(df)
    df["gap_days"] = df.index.to_series().diff().dt.days

    # In event-based data stale streak markers are no longer meaningful.
    df["next_day_after_long_streak"] = False
    df["is_suspect_outlier"] = (df["is_outlier_iqr"] | df["is_outlier_mad"]) & (
        df["volume_zero"] | (df["gap_days"] >= gap_threshold)
    )
    return df


def clean_event_based(df: pd.DataFrame) -> pd.DataFrame:
    # Best-practice rules:
    # 1) Drop missing Close
    # 2) Keep only event-based days (price changes)
    # 3) Drop suspect outliers
    # 4) Recompute Return + outlier flags on the final event-based series
    df_clean = df.copy()
    df_clean = df_clean[df_clean["Close"].notna()]
    df_clean = df_clean[~df_clean["is_stale"]]
    df_clean = df_clean[~df_clean["is_suspect_outlier"]]

    # Keep diagnostics in-sync with final Return after row removals.
    df_clean = recompute_event_outlier_flags(df_clean)
    return df_clean


def clean_model_ready_v2(df: pd.DataFrame, max_gap_days: int = 7,
                         drop_residual_suspect: bool = True):
    """
    Build stricter model-ready dataset on top of event-based cleaning.

    Steps:
    1) Run event-based cleaning
    2) Optionally drop residual suspect outliers after recompute
    3) Drop rows after long temporal gaps (gap_days > max_gap_days)
    4) Drop OHLC-invalid rows
    5) Recompute diagnostic flags for final consistency
    """
    event_df = clean_event_based(df)
    # Baseline recompute once on event data (non-cascading reference frame).
    baseline_df = recompute_event_outlier_flags(event_df, gap_threshold=max_gap_days)
    model_df = baseline_df.copy()

    diagnostics = {
        "event_rows": int(len(event_df)),
        "dropped_residual_suspect_pre": 0,
        "dropped_long_gap_rows": 0,
        "dropped_ohlc_invalid_rows": 0,
        "dropped_residual_suspect_post": 0,
        "residual_suspect_iterations": 0,
        "residual_suspect_remaining": 0,
        "residual_gap_rows_remaining": 0,
        "residual_ohlc_invalid_remaining": 0,
        "residual_loop_hit_cap": False,
        "max_gap_days": int(max_gap_days),
    }

    if drop_residual_suspect and "is_suspect_outlier" in model_df.columns:
        mask = model_df["is_suspect_outlier"].fillna(False)
        diagnostics["dropped_residual_suspect_pre"] = int(mask.sum())
        model_df = model_df[~mask]

    # Enforce static constraints once from baseline flags:
    # - long temporal gaps
    # - OHLC invalid
    # - suspect outliers (pre-pass)
    if max_gap_days is not None and "gap_days" in model_df.columns:
        mask_gap = model_df["gap_days"].fillna(0) > int(max_gap_days)
        diagnostics["dropped_long_gap_rows"] = int(mask_gap.sum())
        model_df = model_df[~mask_gap]

    if "is_ohlc_invalid" in model_df.columns:
        mask_ohlc = model_df["is_ohlc_invalid"].fillna(False)
        n_ohlc = int(mask_ohlc.sum())
        if n_ohlc > 0:
            diagnostics["dropped_ohlc_invalid_rows"] = n_ohlc
            model_df = model_df[~mask_ohlc]

    # Recompute after static drops.
    model_df = recompute_event_outlier_flags(model_df, gap_threshold=max_gap_days)

    # Optional single residual suspect pass (no cascade loop).
    if drop_residual_suspect and "is_suspect_outlier" in model_df.columns:
        mask_sus = model_df["is_suspect_outlier"].fillna(False)
        n_sus = int(mask_sus.sum())
        diagnostics["dropped_residual_suspect_post"] = n_sus
        diagnostics["residual_suspect_iterations"] = 1 if n_sus > 0 else 0
        if n_sus > 0:
            model_df = model_df[~mask_sus]
            model_df = recompute_event_outlier_flags(model_df, gap_threshold=max_gap_days)

    # Final non-cascading enforcement for gap/ohlc after recompute.
    final_dropped = 0
    if max_gap_days is not None and "gap_days" in model_df.columns:
        mask_gap = model_df["gap_days"].fillna(0) > int(max_gap_days)
        n_gap = int(mask_gap.sum())
        if n_gap > 0:
            diagnostics["dropped_long_gap_rows"] += n_gap
            model_df = model_df[~mask_gap]
            final_dropped += n_gap

    if "is_ohlc_invalid" in model_df.columns:
        mask_ohlc = model_df["is_ohlc_invalid"].fillna(False)
        n_ohlc = int(mask_ohlc.sum())
        if n_ohlc > 0:
            diagnostics["dropped_ohlc_invalid_rows"] += n_ohlc
            model_df = model_df[~mask_ohlc]
            final_dropped += n_ohlc

    if final_dropped > 0:
        model_df = recompute_event_outlier_flags(model_df, gap_threshold=max_gap_days)

    if "is_suspect_outlier" in model_df.columns:
        diagnostics["residual_suspect_remaining"] = int(model_df["is_suspect_outlier"].fillna(False).sum())
    if max_gap_days is not None and "gap_days" in model_df.columns:
        diagnostics["residual_gap_rows_remaining"] = int((model_df["gap_days"].fillna(0) > int(max_gap_days)).sum())
    if "is_ohlc_invalid" in model_df.columns:
        diagnostics["residual_ohlc_invalid_remaining"] = int(model_df["is_ohlc_invalid"].fillna(False).sum())

    diagnostics["model_rows"] = int(len(model_df))
    return event_df, model_df, diagnostics


def recompute_event_outlier_flags_v3(df: pd.DataFrame, gap_threshold: int = 7) -> pd.DataFrame:
    """
    Recompute diagnostic flags with long-gap-aware return logic.

    For event-based datasets, very long temporal gaps can inflate pct-change and
    trigger unstable cascading drops. v3 breaks return continuity on long gaps:
    - keep the row
    - mark it as long gap
    - set Return to NaN for that row
    """
    df = df.copy()
    gap_threshold = int(gap_threshold)

    if "volume_zero" not in df.columns:
        if "Volume" in df.columns:
            df["volume_zero"] = df["Volume"] == 0
        else:
            df["volume_zero"] = False

    df["gap_days"] = df.index.to_series().diff().dt.days
    df["is_long_gap"] = df["gap_days"].fillna(0) > gap_threshold

    raw_return = df["Close"].pct_change(fill_method=None)
    # Do not score outliers across long temporal gaps.
    df["Return"] = raw_return.where(~df["is_long_gap"], np.nan)
    df["is_outlier_iqr"], df["is_outlier_mad"] = _compute_outlier_masks(df["Return"])

    # In event-based data stale streak markers are no longer meaningful.
    df["next_day_after_long_streak"] = False
    df["is_suspect_outlier"] = (df["is_outlier_iqr"] | df["is_outlier_mad"]) & df["volume_zero"]
    return df


def clean_model_ready_v3(
    df: pd.DataFrame,
    max_gap_days: int = 7,
    drop_residual_suspect: bool = True,
    max_suspect_passes: int = 5,
    max_suspect_drop_pct: float = 5.0,
):
    """
    Build stable model-ready v3 dataset with bounded residual pruning.

    Key differences vs v2:
    1) Long gaps are kept but explicitly marked (`is_long_gap`)
    2) Return across long gaps is set NaN before outlier scoring
    3) Residual suspect outliers are pruned iteratively with a drop budget cap
    """
    event_df = clean_event_based(df)
    model_df = event_df.copy()

    diagnostics = {
        "event_rows": int(len(event_df)),
        "dropped_ohlc_invalid_rows": 0,
        "dropped_residual_suspect": 0,
        "residual_suspect_iterations": 0,
        "residual_suspect_remaining": 0,
        "residual_gap_rows_remaining": 0,
        "residual_ohlc_invalid_remaining": 0,
        "residual_loop_hit_cap": False,
        "max_gap_days": int(max_gap_days),
        "max_suspect_passes": int(max_suspect_passes),
        "max_suspect_drop_pct": float(max_suspect_drop_pct),
        "suspect_drop_budget_rows": 0,
    }

    if "is_ohlc_invalid" in model_df.columns:
        mask_ohlc = model_df["is_ohlc_invalid"].fillna(False)
        n_ohlc = int(mask_ohlc.sum())
        diagnostics["dropped_ohlc_invalid_rows"] = n_ohlc
        if n_ohlc > 0:
            model_df = model_df[~mask_ohlc]

    model_df = recompute_event_outlier_flags_v3(model_df, gap_threshold=max_gap_days)

    budget_rows = int(np.floor(len(model_df) * (float(max_suspect_drop_pct) / 100.0)))
    if float(max_suspect_drop_pct) > 0 and len(model_df) > 0:
        budget_rows = max(1, budget_rows)
    diagnostics["suspect_drop_budget_rows"] = budget_rows

    if drop_residual_suspect:
        dropped_total = 0
        max_passes = max(0, int(max_suspect_passes))

        for it in range(1, max_passes + 1):
            mask_sus = model_df["is_suspect_outlier"].fillna(False)
            n_sus = int(mask_sus.sum())
            if n_sus == 0:
                break

            remaining_budget = budget_rows - dropped_total
            if remaining_budget <= 0:
                diagnostics["residual_loop_hit_cap"] = True
                break

            candidates = model_df[mask_sus].copy()
            candidates["_abs_return"] = candidates["Return"].abs().fillna(-np.inf)
            drop_idx = candidates.sort_values("_abs_return", ascending=False).head(remaining_budget).index
            n_drop = int(len(drop_idx))

            if n_drop == 0:
                break

            model_df = model_df.drop(index=drop_idx)
            dropped_total += n_drop
            diagnostics["dropped_residual_suspect"] += n_drop
            diagnostics["residual_suspect_iterations"] = it
            model_df = recompute_event_outlier_flags_v3(model_df, gap_threshold=max_gap_days)

    diagnostics["residual_suspect_remaining"] = int(model_df["is_suspect_outlier"].fillna(False).sum())
    diagnostics["residual_gap_rows_remaining"] = int((model_df["gap_days"].fillna(0) > int(max_gap_days)).sum())
    diagnostics["residual_ohlc_invalid_remaining"] = int(model_df.get("is_ohlc_invalid", pd.Series(False, index=model_df.index)).fillna(False).sum())
    diagnostics["model_rows"] = int(len(model_df))

    return event_df, model_df, diagnostics


def build_report(raw_df: pd.DataFrame, flagged_df: pd.DataFrame, clean_df: pd.DataFrame,
                 mode: str = "event", model_diag: dict | None = None) -> dict:
    report = {}

    report["mode"] = mode
    report["raw_rows"] = len(raw_df)
    report["raw_missing_close"] = int(raw_df["Close"].isna().sum())
    report["raw_volume_zero"] = int((raw_df["Volume"] == 0).sum()) if "Volume" in raw_df.columns else 0
    report["raw_stale_days"] = int(flagged_df["is_stale"].sum())
    report["raw_long_streak_days"] = int((flagged_df["streak_len"] >= 5).sum())
    report["raw_suspect_outliers"] = int(flagged_df["is_suspect_outlier"].sum())
    report["raw_ohlc_invalid_rows"] = int(flagged_df.get("is_ohlc_invalid", pd.Series(False, index=flagged_df.index)).sum())

    post_basic = flagged_df[flagged_df["Close"].notna() & (~flagged_df["is_stale"])]
    event_rows = int(model_diag["event_rows"]) if (model_diag and "event_rows" in model_diag) else int(len(clean_df))

    report["post_basic_filter_rows"] = len(post_basic)
    report["clean_rows"] = len(clean_df)
    report["dropped_missing"] = int(raw_df["Close"].isna().sum())
    report["dropped_stale"] = int(flagged_df["is_stale"].sum())
    report["dropped_suspect_outliers"] = int(len(post_basic) - event_rows)
    report["clean_suspect_outliers_remaining"] = int(clean_df["is_suspect_outlier"].sum())

    if model_diag:
        for k, v in model_diag.items():
            report[f"model_{k}"] = v

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Process raw ALI=F data with best-practice cleaning.")
    parser.add_argument(
        "--mode",
        choices=["event", "model", "model_v3"],
        default="event",
        help="event: legacy event-clean output, model: stricter model-ready v2 output, model_v3: gap-aware + bounded residual pruning",
    )
    parser.add_argument(
        "--raw",
        default=os.path.join("data", "raw data", "ali_f_raw.csv"),
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for cleaned event-based CSV",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=7,
        help="Gap threshold used in model/model_v3 diagnostics and filtering logic",
    )
    parser.add_argument(
        "--keep-residual-suspect",
        action="store_true",
        help="Keep residual suspect outliers after recompute in model/model_v3 mode",
    )
    parser.add_argument(
        "--max-suspect-passes",
        type=int,
        default=5,
        help="Max residual-suspect pruning passes (model_v3 only)",
    )
    parser.add_argument(
        "--max-suspect-drop-pct",
        type=float,
        default=5.0,
        help="Max percentage rows allowed to drop from residual suspect pruning (model_v3 only)",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path}")

    raw_df = load_raw(raw_path)
    flagged = build_flags(raw_df)
    flagged = detect_outliers(flagged)
    flagged = flag_suspect_outliers(flagged)
    model_diag = None
    if args.mode == "model":
        event_df, clean_df, model_diag = clean_model_ready_v2(
            flagged,
            max_gap_days=args.max_gap_days,
            drop_residual_suspect=(not args.keep_residual_suspect),
        )
        if args.out is None:
            out_path = Path(os.path.join("data", "processed data", "ali_f_event_model_ready_v2.csv"))
        else:
            out_path = Path(args.out)
    elif args.mode == "model_v3":
        event_df, clean_df, model_diag = clean_model_ready_v3(
            flagged,
            max_gap_days=args.max_gap_days,
            drop_residual_suspect=(not args.keep_residual_suspect),
            max_suspect_passes=args.max_suspect_passes,
            max_suspect_drop_pct=args.max_suspect_drop_pct,
        )
        if args.out is None:
            out_path = Path(os.path.join("data", "processed data", "ali_f_event_model_ready_v3.csv"))
        else:
            out_path = Path(args.out)
    else:
        clean_df = clean_event_based(flagged)
        if args.out is None:
            out_path = Path(os.path.join("data", "processed data", "ali_f_event_clean.csv"))
        else:
            out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path)

    report = build_report(raw_df, flagged, clean_df, mode=args.mode, model_diag=model_diag)
    print("PROCESSING REPORT")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k}: {v}")
    print("Saved:", out_path)
    print("Processed at:", datetime.utcnow().isoformat() + "Z")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
