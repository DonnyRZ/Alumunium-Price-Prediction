from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import overwrite_sheet, read_sheet
from production.pipeline.common import utc_now_iso
from production.sheet_contract import TAB_XGB_HISTORY, TAB_XGB_LATEST, TAB_XGB_SUMMARY


SEED = 42
H = 1
START_DATE = "2020-01-01"
DROP_LONG_GAP = True
TRAIN_WINDOW_YEARS = 2
EARLY_STOPPING = 100
INTERVAL_LOW_Q = 0.10
INTERVAL_HIGH_Q = 0.90
MOVE_WEIGHT_MIN = 0.75
MOVE_WEIGHT_MAX = 1.90
MOVE_WEIGHT_CLIP_Q = 0.90
MOVE_WEIGHT_POWER = 1.00
NOHARM_TAU_MULT_MIN = 0.20
MIN_VALID_ROWS = 80

DATA_PATH = ROOT / "data" / "processed data" / "ali_f_event_model_ready_v3.csv"
DECISION_PATH = ROOT / "data" / "processed data" / "xgb_main_h1_decision.json"
SUMMARY_PATH = ROOT / "data" / "processed data" / "xgb_main_h1_summary.csv"
HISTORY_PRED_PATH = ROOT / "data" / "processed data" / "xgb_main_h1_predictions.csv"
PRODUCTION_CONFIG_PATH = ROOT / "production" / "config" / "xgb_main_production_config.json"

FEATURES = [
    "dow",
    "quarter",
    "ret_lag_1",
    "ret_lag_5",
    "ret_roll_mean_5",
    "ret_roll_mean_10",
    "ret_roll_std_10",
    "ret_roll_std_20",
    "gap_lag_1",
    "hl_spread_pct",
    "oc_spread_pct",
    "close_mom_5",
    "close_mom_10",
]

HISTORY_COLUMNS = [
    "base_date",
    "forecast_date",
    "current_price",
    "actual_next_price",
    "model_price_t1",
    "baseline_price_t1",
    "gate_applied",
    "signal",
    "generated_at_utc",
]


def build_price_frame(df_all: pd.DataFrame) -> pd.DataFrame:
    d = df_all.copy()
    d = d[d["Date"] >= pd.Timestamp(START_DATE)].copy()

    if DROP_LONG_GAP:
        d = d[~d["is_long_gap"].fillna(False)].copy()

    d["hl_spread_pct"] = (d["High"] - d["Low"]) / d["Close"]
    d["oc_spread_pct"] = (d["Close"] - d["Open"]) / d["Open"]

    for lag in (1, 2, 3, 5, 8, 10):
        d[f"ret_lag_{lag}"] = d["Return"].shift(lag)
        d[f"vol_lag_{lag}"] = d["Volume"].shift(lag)
        d[f"gap_lag_{lag}"] = d["gap_days"].shift(lag)

    d["ret_roll_mean_5"] = d["Return"].rolling(5, min_periods=3).mean()
    d["ret_roll_mean_10"] = d["Return"].rolling(10, min_periods=5).mean()
    d["ret_roll_std_10"] = d["Return"].rolling(10, min_periods=5).std()
    d["ret_roll_std_20"] = d["Return"].rolling(20, min_periods=8).std()
    d["close_mom_5"] = d["Close"].pct_change(5)
    d["close_mom_10"] = d["Close"].pct_change(10)
    d["quarter"] = d["Date"].dt.quarter
    d["dow"] = d["Date"].dt.dayofweek

    d[f"target_date_t{H}"] = d["Date"].shift(-H)
    d[f"target_close_t{H}"] = d["Close"].shift(-H)
    d[f"target_ret_t{H}"] = (d[f"target_close_t{H}"] / d["Close"]) - 1.0
    d[f"target_logret_t{H}"] = np.log(d[f"target_close_t{H}"] / d["Close"])
    return d


def build_price_baselines(frame: pd.DataFrame, train_mean_ret: float, horizon: int) -> dict[str, np.ndarray]:
    close_t = frame["Close"].values
    drift = close_t * np.exp(train_mean_ret)
    roll5 = close_t * np.exp(horizon * frame["ret_roll_mean_5"].values)
    roll10 = close_t * np.exp(horizon * frame["ret_roll_mean_10"].values)
    mult_mom5 = np.clip(1.0 + frame["close_mom_5"].values, 0.01, None)
    repeat_last_5 = close_t * mult_mom5
    return {
        "drift_mean_ret": drift,
        "rolling_ret5_scaled": roll5,
        "rolling_ret10_scaled": roll10,
        "repeat_last_5event_move": repeat_last_5,
    }


def build_move_sample_weights(target_ret, min_w, max_w, clip_q, power):
    magnitude = np.abs(np.asarray(target_ret, dtype=float))
    clip_value = max(float(np.quantile(magnitude, float(clip_q))), 1e-9)
    scaled = np.clip(magnitude, 0.0, clip_value) / clip_value
    return np.asarray(
        float(min_w) + (float(max_w) - float(min_w)) * np.power(scaled, float(power)),
        dtype=float,
    )


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not np.isfinite(mad) or mad <= 1e-9:
        return max(float(np.std(x)), 1.0)
    return max(1.4826 * mad, 1.0)


def robust_center_scale_ref(x_ref):
    x_ref = np.asarray(x_ref, dtype=float)
    med = float(np.median(x_ref))
    mad = float(np.median(np.abs(x_ref - med)))
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-9 else float(np.std(x_ref))
    return med, max(scale, 1e-6)


def build_regime_mask(vol_ref, vol_eval, z_thr):
    med, scale = robust_center_scale_ref(vol_ref)
    z = (np.asarray(vol_eval, dtype=float) - med) / scale
    return z >= float(z_thr)


def apply_regime_noharm_gate(close_t, pred_price_corr, tau_abs, regime_mask):
    close_t = np.asarray(close_t, dtype=float)
    pred_price_corr = np.asarray(pred_price_corr, dtype=float)
    regime_mask = np.asarray(regime_mask, dtype=bool)
    corr = pred_price_corr - close_t
    allow = regime_mask & (np.abs(corr) >= float(tau_abs))
    pred = np.where(allow, pred_price_corr, close_t)
    return pred, allow


def pick_latest_valid_year(frame_labeled: pd.DataFrame, forecast_year: int) -> int:
    year_counts = frame_labeled["Date"].dt.year.value_counts().to_dict()
    candidates = sorted(
        [year for year, count in year_counts.items() if year < forecast_year and count >= MIN_VALID_ROWS]
    )
    if not candidates:
        raise ValueError("Tidak ada tahun valid yang cukup panjang untuk kalibrasi baseline production.")
    return candidates[-1]


def load_production_contract() -> tuple[dict, list[dict]]:
    if DECISION_PATH.exists():
        decision = json.loads(DECISION_PATH.read_text())
    else:
        decision = json.loads(PRODUCTION_CONFIG_PATH.read_text())

    if SUMMARY_PATH.exists():
        summary_rows = pd.read_csv(SUMMARY_PATH).to_dict(orient="records")
    else:
        summary_rows = decision.get("summary_rows", [])
    return decision, summary_rows


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _coerce_history_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_history_df()

    out = df.copy()
    for col in ["base_date", "forecast_date", "generated_at_utc"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ["current_price", "actual_next_price", "model_price_t1", "baseline_price_t1"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "gate_applied" in out.columns:
        out["gate_applied"] = out["gate_applied"].astype(str).str.lower().map({"true": True, "false": False})

    for col in HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[HISTORY_COLUMNS].copy()


def _seed_history_from_local_artifact() -> pd.DataFrame:
    if not HISTORY_PRED_PATH.exists():
        return _empty_history_df()

    raw = pd.read_csv(HISTORY_PRED_PATH)
    if raw.empty:
        return _empty_history_df()

    seeded = raw.copy()
    seeded["base_date"] = pd.to_datetime(seeded["Date"]).dt.date.astype(str)
    seeded["forecast_date"] = (pd.to_datetime(seeded["Date"]) + BDay(1)).dt.date.astype(str)
    seeded["current_price"] = pd.to_numeric(seeded["close_t"], errors="coerce")
    seeded["actual_next_price"] = pd.to_numeric(seeded["y_true_price_t1"], errors="coerce")
    seeded["model_price_t1"] = pd.to_numeric(seeded["y_pred_p50_t1"], errors="coerce")
    seeded["baseline_price_t1"] = pd.to_numeric(seeded["baseline_price_t1"], errors="coerce")
    seeded["gate_applied"] = seeded["gate_applied"].fillna(False).astype(bool)
    seeded["signal"] = np.where(
        seeded["model_price_t1"] > seeded["current_price"],
        "Bullish",
        np.where(seeded["model_price_t1"] < seeded["current_price"], "Bearish", "Netral"),
    )
    seeded["generated_at_utc"] = utc_now_iso()
    seeded = seeded[HISTORY_COLUMNS].drop_duplicates(subset=["forecast_date"], keep="last")
    seeded = seeded.sort_values("base_date").reset_index(drop=True)
    return seeded


def _load_history_sheet() -> pd.DataFrame:
    history_df = _coerce_history_sheet(read_sheet(TAB_XGB_HISTORY))
    if not history_df.empty:
        return history_df

    seeded = _seed_history_from_local_artifact()
    if not seeded.empty:
        overwrite_sheet(seeded, TAB_XGB_HISTORY)
        return seeded
    return _empty_history_df()


def update_xgb_sheet_outputs(payload: dict) -> list[dict]:
    latest_df = pd.DataFrame(
        [
            {
                "generated_at_utc": payload["generated_at_utc"],
                "latest_data_date": payload["latest_data_date"],
                "forecast_date": payload["forecast_date"],
                "current_price": payload["current_price"],
                "pred_price_final_t1": payload["pred_price_final_t1"],
                "pred_price_corr_t1": payload["pred_price_corr_t1"],
                "pred_price_p10_t1": payload["pred_price_p10_t1"],
                "pred_price_p90_t1": payload["pred_price_p90_t1"],
                "baseline_price_t1": payload["baseline_price_t1"],
                "locked_baseline_name": payload["locked_baseline_name"],
                "delta_abs": payload["delta_abs"],
                "delta_pct": payload["delta_pct"],
                "signal": payload["signal"],
                "gate_applied": payload["gate_applied"],
                "regime_active": payload["regime_active"],
                "noharm_tau_abs": payload["noharm_tau_abs"],
                "train_rows": payload["train_rows"],
                "feature_count": payload["feature_count"],
            }
        ]
    )
    overwrite_sheet(latest_df, TAB_XGB_LATEST)

    summary_df = pd.DataFrame(payload.get("summary_rows", []))
    overwrite_sheet(summary_df, TAB_XGB_SUMMARY)

    history_df = _load_history_sheet()
    latest_data_date = str(payload["latest_data_date"])
    forecast_date = str(payload["forecast_date"])
    current_price = float(payload["current_price"])

    if not history_df.empty and "forecast_date" in history_df.columns:
        mask_actual = history_df["forecast_date"].astype(str).eq(latest_data_date)
        if mask_actual.any():
            history_df.loc[mask_actual, "actual_next_price"] = current_price

    new_row = pd.DataFrame(
        [
            {
                "base_date": latest_data_date,
                "forecast_date": forecast_date,
                "current_price": current_price,
                "actual_next_price": None,
                "model_price_t1": float(payload["pred_price_final_t1"]),
                "baseline_price_t1": float(payload["baseline_price_t1"]),
                "gate_applied": bool(payload["gate_applied"]),
                "signal": str(payload["signal"]),
                "generated_at_utc": payload["generated_at_utc"],
            }
        ]
    )
    history_df = pd.concat([history_df, new_row], ignore_index=True)
    history_df = history_df.drop_duplicates(subset=["forecast_date"], keep="last").sort_values("base_date").reset_index(
        drop=True
    )
    overwrite_sheet(history_df, TAB_XGB_HISTORY)

    recent = history_df.tail(90).copy()
    history_rows = []
    for _, row in recent.iterrows():
        actual_val = row["actual_next_price"]
        history_rows.append(
            {
                "base_date": str(row["base_date"]),
                "forecast_date": str(row["forecast_date"]),
                "current_price": None if pd.isna(row["current_price"]) else float(row["current_price"]),
                "actual_next_price": None if pd.isna(actual_val) else float(actual_val),
                "model_price_t1": None if pd.isna(row["model_price_t1"]) else float(row["model_price_t1"]),
                "baseline_price_t1": None if pd.isna(row["baseline_price_t1"]) else float(row["baseline_price_t1"]),
                "gate_applied": bool(row["gate_applied"]) if pd.notna(row["gate_applied"]) else False,
            }
        )
    return history_rows


def build_xgb_snapshot() -> dict:
    raw = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    decision, summary_rows = load_production_contract()

    frame = build_price_frame(raw).sort_values("Date").reset_index(drop=True)
    labeled = frame.dropna(
        subset=FEATURES + [f"target_date_t{H}", f"target_close_t{H}", f"target_ret_t{H}", f"target_logret_t{H}", "Close"]
    ).copy()
    forecast_rows = frame[frame[f"target_close_t{H}"].isna()].dropna(subset=FEATURES + ["Close"]).copy()
    if forecast_rows.empty:
        raise ValueError("Tidak ada row forecast terbaru untuk membangun prediksi production.")

    forecast_row = forecast_rows.sort_values("Date").tail(1).copy()
    latest_date = pd.Timestamp(forecast_row["Date"].iloc[0])
    forecast_year = int(latest_date.year)

    valid_year = pick_latest_valid_year(labeled, forecast_year=forecast_year)
    cal_train_start = pd.Timestamp(f"{valid_year - TRAIN_WINDOW_YEARS}-01-01")
    cal_train_end = pd.Timestamp(f"{valid_year - 1}-12-31")
    cal_valid_start = pd.Timestamp(f"{valid_year}-01-01")
    cal_valid_end = pd.Timestamp(f"{valid_year}-12-31")

    cal_train = labeled[
        (labeled["Date"] >= cal_train_start)
        & (labeled["Date"] <= cal_train_end)
        & (labeled[f"target_date_t{H}"] <= cal_train_end)
    ].copy()
    cal_valid = labeled[
        (labeled["Date"] >= cal_valid_start)
        & (labeled["Date"] <= cal_valid_end)
        & (labeled[f"target_date_t{H}"] <= cal_valid_end)
    ].copy()
    if cal_train.empty or cal_valid.empty:
        raise ValueError("Kalibrasi baseline production gagal karena split train/valid kosong.")

    train_mean_ret_cal = float(cal_train[f"target_logret_t{H}"].mean())
    baselines_valid = build_price_baselines(cal_valid, train_mean_ret_cal, H)
    locked_baseline_name = min(
        baselines_valid,
        key=lambda key: float(np.mean(np.abs(cal_valid[f"target_close_t{H}"].values - baselines_valid[key]))),
    )

    ytr_cal_corr = (cal_train[f"target_close_t{H}"] - cal_train["Close"]).values
    yva_cal_corr = (cal_valid[f"target_close_t{H}"] - cal_valid["Close"]).values
    wtr_cal = build_move_sample_weights(
        cal_train[f"target_ret_t{H}"].values,
        MOVE_WEIGHT_MIN,
        MOVE_WEIGHT_MAX,
        MOVE_WEIGHT_CLIP_Q,
        MOVE_WEIGHT_POWER,
    )

    cal_params = decision["reg_params_used"].copy()
    cal_params.update({"random_state": SEED, "n_jobs": 1, "early_stopping_rounds": EARLY_STOPPING})
    cal_model = XGBRegressor(**cal_params)
    cal_model.fit(
        cal_train[FEATURES].values,
        ytr_cal_corr,
        sample_weight=wtr_cal,
        eval_set=[(cal_valid[FEATURES].values, yva_cal_corr)],
        verbose=False,
    )

    cal_valid_pred_corr = cal_valid["Close"].values + cal_model.predict(cal_valid[FEATURES].values)
    tau_cal = max(decision["noharm_tau_mult_used"], NOHARM_TAU_MULT_MIN) * robust_scale(ytr_cal_corr)
    reg_mask_cal = build_regime_mask(
        cal_train["ret_roll_std_20"].values,
        cal_valid["ret_roll_std_20"].values,
        decision["regime_vol_z_used"],
    )
    cal_valid_pred_final, _ = apply_regime_noharm_gate(
        cal_valid["Close"].values, cal_valid_pred_corr, tau_cal, reg_mask_cal
    )
    cal_resid = cal_valid[f"target_close_t{H}"].values - cal_valid_pred_final
    res_q10 = float(np.quantile(cal_resid, INTERVAL_LOW_Q))
    res_q90 = float(np.quantile(cal_resid, INTERVAL_HIGH_Q))

    final_train_start = pd.Timestamp(f"{latest_date.year - TRAIN_WINDOW_YEARS}-01-01")
    final_train = labeled[labeled["Date"] >= final_train_start].copy()
    if final_train.empty:
        raise ValueError("Training production kosong setelah filter jendela waktu.")

    ytr_final_corr = (final_train[f"target_close_t{H}"] - final_train["Close"]).values
    wtr_final = build_move_sample_weights(
        final_train[f"target_ret_t{H}"].values,
        MOVE_WEIGHT_MIN,
        MOVE_WEIGHT_MAX,
        MOVE_WEIGHT_CLIP_Q,
        MOVE_WEIGHT_POWER,
    )
    final_params = decision["reg_params_used"].copy()
    final_params.update({"random_state": SEED, "n_jobs": 1})
    final_model = XGBRegressor(**final_params)
    final_model.fit(final_train[FEATURES].values, ytr_final_corr, sample_weight=wtr_final, verbose=False)

    pred_corr = float(forecast_row["Close"].iloc[0] + final_model.predict(forecast_row[FEATURES].values)[0])
    tau_prod = max(decision["noharm_tau_mult_used"], NOHARM_TAU_MULT_MIN) * robust_scale(ytr_final_corr)
    reg_mask_prod = build_regime_mask(
        final_train["ret_roll_std_20"].values,
        forecast_row["ret_roll_std_20"].values,
        decision["regime_vol_z_used"],
    )
    pred_final, gate_prod = apply_regime_noharm_gate(
        forecast_row["Close"].values,
        np.array([pred_corr]),
        tau_prod,
        reg_mask_prod,
    )
    pred_final = float(pred_final[0])
    gate_applied = bool(gate_prod[0])
    regime_active = bool(np.asarray(reg_mask_prod)[0])

    train_mean_ret_final = float(final_train[f"target_logret_t{H}"].mean())
    forecast_baseline_frame = forecast_row[["Close", "ret_roll_mean_5", "ret_roll_mean_10", "close_mom_5"]].copy()
    baseline_map = build_price_baselines(forecast_baseline_frame, train_mean_ret_final, H)
    baseline_pred = float(baseline_map[locked_baseline_name][0])

    pred_p10 = float(pred_final + res_q10)
    pred_p90 = float(pred_final + res_q90)
    current_price = float(forecast_row["Close"].iloc[0])
    delta_abs = pred_final - current_price
    delta_pct = (delta_abs / current_price) * 100 if current_price else 0.0

    payload = {
        "generated_at_utc": utc_now_iso(),
        "model_name": decision.get("model_name", "XGBoost H+1"),
        "data_source": str(DATA_PATH.relative_to(ROOT)),
        "decision_source": (
            str(DECISION_PATH.relative_to(ROOT))
            if DECISION_PATH.exists()
            else str(PRODUCTION_CONFIG_PATH.relative_to(ROOT))
        ),
        "latest_data_date": latest_date.date().isoformat(),
        "forecast_date": (latest_date + BDay(1)).date().isoformat(),
        "train_window_start": final_train_start.date().isoformat(),
        "train_rows": int(len(final_train)),
        "feature_count": len(FEATURES),
        "feature_names": FEATURES,
        "locked_baseline_name": locked_baseline_name,
        "current_price": current_price,
        "baseline_price_t1": baseline_pred,
        "pred_price_corr_t1": pred_corr,
        "pred_price_final_t1": pred_final,
        "pred_price_p10_t1": pred_p10,
        "pred_price_p90_t1": pred_p90,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "signal": "Bullish" if delta_pct > 0 else "Bearish" if delta_pct < 0 else "Netral",
        "gate_applied": gate_applied,
        "regime_active": regime_active,
        "noharm_tau_abs": float(tau_prod),
        "calibration_valid_year": valid_year,
        "summary_rows": summary_rows,
        "production_note": (
            "Prediksi harian dibuat dari setup XGBoost saat ini. "
            "Baseline dikunci dari tahun valid terakhir yang lengkap, lalu model dilatih ulang pada jendela terbaru."
        ),
    }
    payload["recent_history"] = update_xgb_sheet_outputs(payload)
    return payload


if __name__ == "__main__":
    out = build_xgb_snapshot()
    print("xgb production state updated to spreadsheet")
    print(out["forecast_date"], out["pred_price_final_t1"])
