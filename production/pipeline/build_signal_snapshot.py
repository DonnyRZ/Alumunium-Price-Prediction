from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import overwrite_sheet
from production.pipeline.common import action_label_id, action_level_from_probability, action_urgency, utc_now_iso
from production.sheet_contract import (
    TAB_MARKETING_PLAYBOOK,
    TAB_SIGNAL_HISTORY,
    TAB_SIGNAL_LATEST,
    TAB_SIGNAL_METRICS,
)
from src.build_five_day_signal_dataset import (
    BLOCK_SIZE,
    CORE_FEATURE_COLS,
    DAILY_EXPANDED_PATH,
    DIRTY_BLOCK_OHLC_FEATURES,
    EXPANDED_EXTRA_COLS,
    _block_target_metrics,
    _load_processed_daily,
    _merge_target_ohlc,
    _series_stats,
)
from src.build_five_day_signal_dataset_v3 import (
    add_block_shape_features,
    add_freshness_features,
    add_market_spreads,
    add_target_history_features,
)


MODEL_NAME = "Logistic Regression 5D Actionable Signal"
MODEL_VERSION = "signal_v3_l1_liblinear_c1"
TECHNICAL_THRESHOLD = 0.50
WATCH_THRESHOLD = 0.65
MIN_TRAIN_ROWS = 220

LOCAL_MODEL_DIR = ROOT / "production" / "data" / "model"
LOCAL_LATEST_JSON = LOCAL_MODEL_DIR / "signal_latest_snapshot.json"
LOCAL_HISTORY_CSV = LOCAL_MODEL_DIR / "signal_history_snapshot.csv"
LOCAL_METRICS_CSV = LOCAL_MODEL_DIR / "signal_metrics_snapshot.csv"

FEATURE_EXCLUDE = {
    "block_id",
    "ds",
    "block_start_ds",
    "block_end_ds",
    "target_ds",
    "target_next_close",
    "target_next_change",
    "target_next_return",
    "target_next_log_return",
    "target_next_direction",
    "naive_anchor_close",
    "naive_anchor_return",
    "y",
    "signal_binary",
    "current_block_ohlc_dirty",
}
QC_PREFIXES = ("target_ohlc_has_placeholder_one_",)

FEATURE_NAME_MAP = {
    "y_trend_strength_5": "kekuatan tren return 5 blok",
    "y_trend_strength_3": "kekuatan tren return 3 blok",
    "block_close_location": "posisi close dalam range blok",
    "block_body_to_range": "rasio body terhadap range blok",
    "block_efficiency": "efisiensi arah blok",
    "block_up_rate": "proporsi hari naik dalam blok",
    "spread_shfe_lme_last": "spread SHFE vs LME",
    "spread_shfe_brent_last": "spread SHFE vs Brent",
    "spread_lme_brent_last": "spread LME vs Brent",
    "stock_to_shfe_price_last": "rasio stok gudang terhadap harga SHFE",
    "stock_to_lme_price_last": "rasio stok gudang terhadap harga LME",
    "lag1_shfe_openinterest_first": "open interest SHFE awal blok",
    "freshness_mean": "rata-rata kesegaran data exogenous",
    "freshness_min": "kesegaran terlemah data exogenous",
    "staleness_max": "usia data paling stale",
    "lag1_lme_yesterdayofficialusd_change": "perubahan harga resmi LME",
    "lag1_brent_close_change": "perubahan Brent",
    "lag1_fx_eur_change": "perubahan FX EUR",
    "lag1_warehouse_stockclose_pct_change": "perubahan stok gudang",
}


def _humanize_feature_name(name: str) -> str:
    return FEATURE_NAME_MAP.get(name, name.replace("_", " "))


def _format_driver(name: str, contribution: float) -> str:
    direction = "mendorong action" if contribution >= 0 else "menahan action"
    return f"{_humanize_feature_name(name)} ({direction})"


def _build_block_frame_with_latest(daily: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    work = daily.sort_values("ds").reset_index(drop=True).copy()
    work["block_id"] = np.arange(len(work)) // BLOCK_SIZE

    full_block_ids = work.groupby("block_id", sort=True).size().loc[lambda size: size == BLOCK_SIZE].index.to_list()
    work = work[work["block_id"].isin(full_block_ids)].reset_index(drop=True)

    rows: list[dict] = []
    for block_id, group in work.groupby("block_id", sort=True):
        group = group.sort_values("ds").reset_index(drop=True)
        row: dict[str, object] = {
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
    blocks["target_next_log_return"] = np.log(blocks["target_next_close"] / blocks["current_block_close"])
    blocks["target_next_direction"] = np.where(
        blocks["target_next_return"].notna(),
        (blocks["target_next_return"] > 0).astype(int),
        np.nan,
    )
    blocks["naive_anchor_close"] = blocks["current_block_close"]
    blocks["naive_anchor_return"] = 0.0
    blocks["y"] = blocks["target_next_return"]

    dirty_mask = blocks["current_block_placeholder_any"].astype(int).eq(1)
    blocks["current_block_ohlc_dirty"] = dirty_mask.astype(int)
    for col in DIRTY_BLOCK_OHLC_FEATURES:
        if col in blocks.columns:
            blocks.loc[dirty_mask, col] = np.nan

    return blocks


def _build_engineered_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = _load_processed_daily(DAILY_EXPANDED_PATH)
    daily = _merge_target_ohlc(daily)
    feature_cols = CORE_FEATURE_COLS + EXPANDED_EXTRA_COLS
    blocks = _build_block_frame_with_latest(daily, feature_cols)

    add_block_shape_features(blocks)
    add_target_history_features(blocks)
    add_market_spreads(blocks)
    add_freshness_features(blocks)

    historical = blocks[blocks["y"].notna()].copy().reset_index(drop=True)
    latest = blocks[blocks["y"].isna()].copy().tail(1).reset_index(drop=True)

    tau = float(historical["y"].iloc[: int(len(historical) * 0.7)].abs().median())
    historical["signal_binary"] = (historical["y"].abs() > tau).astype(int)
    if not latest.empty:
        latest["signal_binary"] = np.nan
        latest["forecast_window_start"] = latest["block_end_ds"] + BDay(1)
        latest["forecast_window_end"] = latest["block_end_ds"] + BDay(BLOCK_SIZE)
    historical["forecast_window_start"] = historical["block_end_ds"] + BDay(1)
    historical["forecast_window_end"] = historical["target_ds"]

    combined = pd.concat([historical, latest], ignore_index=True, sort=False)
    return combined, historical


def _select_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[[col for col in frame.columns if col not in FEATURE_EXCLUDE]].select_dtypes(include=[np.number]).copy()
    X = X[[col for col in X.columns if not col.startswith("current_block_") and not col.startswith("block_end_")]]
    X = X[[col for col in X.columns if not col.startswith(QC_PREFIXES)]]
    return X


def _make_model() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    penalty="l1",
                    solver="liblinear",
                    C=1.0,
                    class_weight=None,
                ),
            ),
        ]
    )


def _compute_contributions(model: Pipeline, X_row: pd.DataFrame) -> pd.DataFrame:
    imputed = model.named_steps["imp"].transform(X_row)
    scaled = model.named_steps["scaler"].transform(imputed)
    coefs = model.named_steps["model"].coef_[0]
    contributions = scaled[0] * coefs
    return pd.DataFrame(
        {
            "feature": X_row.columns,
            "coefficient": coefs,
            "scaled_value": scaled[0],
            "contribution": contributions,
        }
    ).sort_values("contribution", ascending=False)


def _latest_reason(action_level: str, probability: float, driver_texts: list[str]) -> str:
    action_label = action_label_id(action_level)
    driver_text = "; ".join(driver_texts[:3]) if driver_texts else "tidak ada driver dominan yang jelas"
    return f"Sinyal {action_label.lower()} dengan probabilitas {probability:.2f}. Driver utama: {driver_text}."


def _build_playbook() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probability_range": "< 0.50",
                "action_level": "ignore",
                "label_id": "Abaikan",
                "urgency": "Rendah",
                "recommended_action": "Lanjutkan proses normal dan tidak perlu eskalasi khusus.",
                "customer_guidance": "Gunakan komunikasi standar tanpa tekanan waktu tambahan.",
            },
            {
                "probability_range": "0.50 - 0.65",
                "action_level": "watch",
                "label_id": "Pantau",
                "urgency": "Sedang",
                "recommended_action": "Pantau konteks pasar, review quote aktif, dan siapkan follow-up yang lebih cepat.",
                "customer_guidance": "Boleh beri konteks pasar singkat, tetapi jangan overclaim.",
            },
            {
                "probability_range": "> 0.65",
                "action_level": "act",
                "label_id": "Tindak",
                "urgency": "Tinggi",
                "recommended_action": "Percepat review internal, prioritaskan inquiry penting, dan pertimbangkan validitas quote yang lebih ketat.",
                "customer_guidance": "Gunakan market note yang lebih hati-hati dan arahkan pelanggan agar tidak terlalu lama menunda keputusan.",
            },
        ]
    )


def _threshold_tradeoff(actual: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "section": "threshold_tradeoff",
                "threshold": threshold,
                "acc": float(accuracy_score(actual, pred)),
                "bal_acc": float(balanced_accuracy_score(actual, pred)),
                "f1": float(f1_score(actual, pred, zero_division=0)),
                "precision": float(precision_score(actual, pred, zero_division=0)),
                "recall": float(recall_score(actual, pred, zero_division=0)),
                "flag_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_metrics_df(history_df: pd.DataFrame) -> pd.DataFrame:
    actual = history_df["actual_action_label"].astype(int)
    probabilities = history_df["probability_action"].astype(float)
    pred = (probabilities >= TECHNICAL_THRESHOLD).astype(int)

    summary_rows = [
        {
            "section": "summary",
            "metric": "model_name",
            "value": MODEL_NAME,
            "meaning": "Model klasifikasi final untuk signal marketing 5 hari.",
        },
        {
            "section": "summary",
            "metric": "model_version",
            "value": MODEL_VERSION,
            "meaning": "Konfigurasi final production saat ini.",
        },
        {
            "section": "summary",
            "metric": "accuracy",
            "value": round(float(accuracy_score(actual, pred)), 4),
            "meaning": "Seberapa sering model benar secara total pada evaluasi walk-forward.",
        },
        {
            "section": "summary",
            "metric": "balanced_accuracy",
            "value": round(float(balanced_accuracy_score(actual, pred)), 4),
            "meaning": "Ukuran performa yang lebih adil untuk kelas action dan no-action.",
        },
        {
            "section": "summary",
            "metric": "precision",
            "value": round(float(precision_score(actual, pred, zero_division=0)), 4),
            "meaning": "Dari semua sinyal action yang keluar, berapa banyak yang benar.",
        },
        {
            "section": "summary",
            "metric": "recall",
            "value": round(float(recall_score(actual, pred, zero_division=0)), 4),
            "meaning": "Dari semua kejadian action yang ada, berapa banyak yang berhasil ditangkap.",
        },
        {
            "section": "summary",
            "metric": "f1",
            "value": round(float(f1_score(actual, pred, zero_division=0)), 4),
            "meaning": "Ringkasan keseimbangan precision dan recall.",
        },
        {
            "section": "summary",
            "metric": "auc",
            "value": round(float(roc_auc_score(actual, probabilities)), 4),
            "meaning": "Kemampuan model membedakan action vs no-action.",
        },
        {
            "section": "summary",
            "metric": "technical_threshold",
            "value": TECHNICAL_THRESHOLD,
            "meaning": "Batas teknis utama untuk mengubah probability menjadi action/no-action.",
        },
        {
            "section": "summary",
            "metric": "watch_threshold",
            "value": WATCH_THRESHOLD,
            "meaning": "Batas bisnis untuk membedakan watch dan act.",
        },
        {
            "section": "summary",
            "metric": "history_rows",
            "value": int(len(history_df)),
            "meaning": "Jumlah prediksi walk-forward yang dipakai untuk evaluasi.",
        },
    ]
    return pd.concat([pd.DataFrame(summary_rows), _threshold_tradeoff(actual, probabilities)], ignore_index=True)


def _build_history_and_latest(combined: pd.DataFrame, historical: pd.DataFrame, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_hist = _select_features(historical)
    y_hist = historical["signal_binary"].astype(int).reset_index(drop=True)
    min_train_rows = max(MIN_TRAIN_ROWS, int(len(historical) * 0.35))
    min_train_rows = min(min_train_rows, max(50, len(historical) - 1))

    history_rows: list[dict] = []
    for index in range(min_train_rows, len(historical)):
        X_train = X_hist.iloc[:index]
        y_train = y_hist.iloc[:index]
        X_row = X_hist.iloc[[index]]
        row = historical.iloc[index]

        model = _make_model()
        model.fit(X_train, y_train)
        probability = float(model.predict_proba(X_row)[:, 1][0])
        action_level = action_level_from_probability(probability)
        contributions = _compute_contributions(model, X_row)
        top_drivers = contributions.reindex(contributions["contribution"].abs().sort_values(ascending=False).index).head(3)
        driver_texts = [_format_driver(item.feature, float(item.contribution)) for item in top_drivers.itertuples()]

        history_rows.append(
            {
                "generated_at_utc": generated_at,
                "base_date": pd.Timestamp(row["ds"]).date().isoformat(),
                "forecast_window_start": pd.Timestamp(row["forecast_window_start"]).date().isoformat(),
                "forecast_window_end": pd.Timestamp(row["forecast_window_end"]).date().isoformat(),
                "probability_action": round(probability, 6),
                "action_level": action_level,
                "action_label_id": action_label_id(action_level),
                "urgency": action_urgency(action_level),
                "predicted_action_binary": int(probability >= TECHNICAL_THRESHOLD),
                "actual_action_label": int(row["signal_binary"]),
                "actual_abs_return": round(float(abs(row["y"])), 6),
                "actual_return": round(float(row["y"]), 6),
                "current_block_close": round(float(row["current_block_close"]), 4),
                "top_driver_1": driver_texts[0] if len(driver_texts) > 0 else "",
                "top_driver_2": driver_texts[1] if len(driver_texts) > 1 else "",
                "top_driver_3": driver_texts[2] if len(driver_texts) > 2 else "",
            }
        )

    history_df = pd.DataFrame(history_rows)

    latest_source = combined.tail(1).copy()
    latest_hist_train = historical.copy()
    X_train_latest = _select_features(latest_hist_train)
    y_train_latest = latest_hist_train["signal_binary"].astype(int)
    X_latest = _select_features(latest_source)

    model = _make_model()
    model.fit(X_train_latest, y_train_latest)
    probability_latest = float(model.predict_proba(X_latest)[:, 1][0])
    action_level_latest = action_level_from_probability(probability_latest)
    contributions_latest = _compute_contributions(model, X_latest)
    top_latest = contributions_latest.reindex(
        contributions_latest["contribution"].abs().sort_values(ascending=False).index
    ).head(5)
    latest_driver_texts = [_format_driver(item.feature, float(item.contribution)) for item in top_latest.itertuples()]

    latest_row = latest_source.iloc[0]
    latest_df = pd.DataFrame(
        [
            {
                "generated_at_utc": generated_at,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "latest_base_date": pd.Timestamp(latest_row["ds"]).date().isoformat(),
                "latest_block_start_date": pd.Timestamp(latest_row["block_start_ds"]).date().isoformat(),
                "latest_block_end_date": pd.Timestamp(latest_row["block_end_ds"]).date().isoformat(),
                "forecast_window_start": pd.Timestamp(latest_row["forecast_window_start"]).date().isoformat(),
                "forecast_window_end": pd.Timestamp(latest_row["forecast_window_end"]).date().isoformat(),
                "probability_action": round(probability_latest, 6),
                "technical_threshold": TECHNICAL_THRESHOLD,
                "watch_threshold": WATCH_THRESHOLD,
                "action_level": action_level_latest,
                "action_label_id": action_label_id(action_level_latest),
                "urgency": action_urgency(action_level_latest),
                "signal_reason_short": _latest_reason(action_level_latest, probability_latest, latest_driver_texts),
                "top_driver_1": latest_driver_texts[0] if len(latest_driver_texts) > 0 else "",
                "top_driver_2": latest_driver_texts[1] if len(latest_driver_texts) > 1 else "",
                "top_driver_3": latest_driver_texts[2] if len(latest_driver_texts) > 2 else "",
                "current_block_close": round(float(latest_row["current_block_close"]), 4),
                "block_price_range_pct": round(float(latest_row["current_block_pct_range"]), 6),
                "freshness_mean": round(float(latest_row.get("freshness_mean", np.nan)), 6),
                "freshness_min": round(float(latest_row.get("freshness_min", np.nan)), 6),
                "staleness_max": round(float(latest_row.get("staleness_max", np.nan)), 6),
            }
        ]
    )
    return history_df, latest_df


def _save_local_artifacts(latest_df: pd.DataFrame, history_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LATEST_JSON.write_text(json.dumps(latest_df.iloc[0].to_dict(), indent=2, default=str))
    history_df.to_csv(LOCAL_HISTORY_CSV, index=False)
    metrics_df.to_csv(LOCAL_METRICS_CSV, index=False)


def build_signal_snapshot(write_sheets: bool = True) -> dict:
    generated_at = utc_now_iso()
    combined, historical = _build_engineered_frame()
    history_df, latest_df = _build_history_and_latest(combined, historical, generated_at)
    metrics_df = _build_metrics_df(history_df)
    playbook_df = _build_playbook()

    _save_local_artifacts(latest_df, history_df, metrics_df)

    if write_sheets:
        overwrite_sheet(latest_df, TAB_SIGNAL_LATEST)
        overwrite_sheet(history_df.tail(180).reset_index(drop=True), TAB_SIGNAL_HISTORY)
        overwrite_sheet(metrics_df, TAB_SIGNAL_METRICS)
        overwrite_sheet(playbook_df, TAB_MARKETING_PLAYBOOK)

    latest_payload = latest_df.iloc[0].to_dict()
    return {
        "generated_at_utc": generated_at,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "latest_signal": latest_payload,
        "history_rows": int(len(history_df)),
        "metrics_rows": int(len(metrics_df)),
        "latest_probability_action": float(latest_payload["probability_action"]),
        "latest_action_level": str(latest_payload["action_level"]),
        "latest_label_id": str(latest_payload["action_label_id"]),
        "latest_base_date": str(latest_payload["latest_base_date"]),
        "forecast_window_end": str(latest_payload["forecast_window_end"]),
        "local_latest_json": str(LOCAL_LATEST_JSON.relative_to(ROOT)),
    }


if __name__ == "__main__":
    payload = build_signal_snapshot(write_sheets=False)
    print(json.dumps(payload, indent=2, default=str))
