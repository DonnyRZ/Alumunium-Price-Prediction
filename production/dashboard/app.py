from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import read_sheet
from production.sheet_contract import (
    SPREADSHEET_NAME,
    TAB_PIPELINE_STATUS,
    TAB_SENTIMENT_ARTICLES,
    TAB_SENTIMENT_DAILY,
    TAB_XGB_HISTORY,
    TAB_XGB_LATEST,
    TAB_XGB_SUMMARY,
)


st.set_page_config(
    page_title="INALUM Aluminium Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


CHANNEL_MAP = {
    "price": "Harga pasar",
    "supply": "Pasokan",
    "policy": "Kebijakan",
    "logistics": "Logistik",
    "inventory": "Persediaan",
    "demand": "Permintaan",
    "macro": "Makro",
    "unclear": "Belum jelas",
}

IMPACT_LABEL_MAP = {
    "bullish": "Cenderung naik",
    "bearish": "Cenderung turun",
    "neutral": "Netral",
}


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    column_count = max(1, min(len(items), 5))
    cards_html = "".join(
        f"""
        <div style="background:#f7f7fb;border:1px solid #e7e7ef;border-radius:12px;padding:16px;min-height:96px;">
          <div style="font-size:0.9rem;color:#666;margin-bottom:8px;">{label}</div>
          <div style="font-size:1.35rem;font-weight:700;color:#222;line-height:1.3;">{value}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat({column_count},minmax(0,1fr));gap:12px;">
          {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Belum ada data untuk ditampilkan.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(f'<div style="overflow-x:auto;">{html}</div>', unsafe_allow_html=True)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _dominant_channel(row: pd.Series) -> str:
    mapping = {
        "channel_price_count": "Harga pasar",
        "channel_supply_count": "Pasokan",
        "channel_policy_count": "Kebijakan",
        "channel_logistics_count": "Logistik",
        "channel_inventory_count": "Persediaan",
        "channel_demand_count": "Permintaan",
        "channel_macro_count": "Makro",
        "channel_unclear_count": "Belum jelas",
    }
    available = [col for col in mapping if col in row.index]
    if not available:
        return "Belum jelas"
    numeric = pd.to_numeric(pd.Series({col: row[col] for col in available}), errors="coerce").fillna(0.0)
    if float(numeric.sum()) <= 0:
        return "Belum jelas"
    return mapping[str(numeric.idxmax())]


def _tone_label(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Belum ada news"
    if score >= 0.20:
        return "Positif"
    if score <= -0.20:
        return "Negatif"
    return "Netral"


def render_history_plot(history: pd.DataFrame) -> None:
    if history.empty:
        return
    plot_df = history.copy()
    plot_df["base_date"] = pd.to_datetime(plot_df["base_date"], errors="coerce")
    plot_df["actual_next_price"] = pd.to_numeric(plot_df.get("actual_next_price"), errors="coerce")
    plot_df["model_price_t1"] = pd.to_numeric(plot_df.get("model_price_t1"), errors="coerce")
    plot_df = plot_df.sort_values("base_date").tail(60).reset_index(drop=True)

    latest_forecast = plot_df[plot_df["actual_next_price"].isna()].tail(1).copy()
    history_only = plot_df[plot_df["actual_next_price"].notna()].copy()
    plot_df["x_pos"] = range(len(plot_df))
    history_only["x_pos"] = plot_df.loc[history_only.index, "x_pos"]
    latest_forecast["x_pos"] = plot_df.loc[latest_forecast.index, "x_pos"]

    fig, ax = plt.subplots(figsize=(10, 4))
    if not history_only.empty:
        ax.plot(history_only["x_pos"], history_only["actual_next_price"], label="Actual", linewidth=2)
        ax.plot(history_only["x_pos"], history_only["model_price_t1"], label="XGBoost", linewidth=2, linestyle="--")
    if not latest_forecast.empty:
        latest_row = latest_forecast.iloc[-1]
        ax.scatter(
            [latest_row["x_pos"]],
            [latest_row["model_price_t1"]],
            label="Forecast terbaru",
            s=70,
            color="#ff7f0e",
            zorder=4,
        )
    tick_step = max(1, len(plot_df) // 8)
    tick_positions = plot_df["x_pos"].iloc[::tick_step].tolist()
    if plot_df["x_pos"].iloc[-1] not in tick_positions:
        tick_positions.append(int(plot_df["x_pos"].iloc[-1]))
    tick_labels = [
        plot_df.loc[plot_df["x_pos"] == position, "base_date"].iloc[0].strftime("%Y-%m-%d")
        for position in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_xlabel("Tanggal Dasar")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Sumbu waktu memakai trading day, jadi weekend dan hari libur tidak membuat garis terputus.")


def render_sentiment_plot(recent_daily: pd.DataFrame) -> None:
    if recent_daily.empty:
        return
    recent_daily = recent_daily.copy()
    recent_daily["news_date"] = pd.to_datetime(recent_daily["news_date"], errors="coerce")
    recent_daily = recent_daily.sort_values("news_date").tail(30)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(recent_daily["news_date"], recent_daily["market_sentiment_mean"], label="Sentiment Mean", linewidth=2)
    ax1.set_ylabel("Sentiment")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.bar(recent_daily["news_date"], recent_daily["news_count_model"], alpha=0.2, label="News Count")
    ax2.set_ylabel("News Count")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data(ttl=300, show_spinner=False)
def load_production_state() -> dict:
    latest_df = read_sheet(TAB_XGB_LATEST)
    history_df = read_sheet(TAB_XGB_HISTORY)
    summary_df = read_sheet(TAB_XGB_SUMMARY)
    articles_df = read_sheet(TAB_SENTIMENT_ARTICLES)
    daily_df = read_sheet(TAB_SENTIMENT_DAILY)
    status_df = read_sheet(TAB_PIPELINE_STATUS)

    latest_df = _to_numeric(
        latest_df,
        [
            "current_price",
            "pred_price_final_t1",
            "pred_price_corr_t1",
            "pred_price_p10_t1",
            "pred_price_p90_t1",
            "baseline_price_t1",
            "delta_abs",
            "delta_pct",
            "noharm_tau_abs",
            "train_rows",
            "feature_count",
        ],
    )
    history_df = _to_numeric(
        history_df,
        ["current_price", "actual_next_price", "model_price_t1", "baseline_price_t1"],
    )
    summary_df = _to_numeric(summary_df, [col for col in summary_df.columns if col != "metric"])
    articles_df = _to_numeric(articles_df, ["market_impact_score", "confidence"])
    daily_df = _to_numeric(daily_df, [col for col in daily_df.columns if col != "news_date"])

    if not daily_df.empty:
        daily_df = daily_df.copy()
        daily_df["dominant_channel"] = daily_df.apply(_dominant_channel, axis=1)
        daily_df["tone_label"] = daily_df["market_sentiment_mean"].apply(_tone_label)

    return {
        "latest": latest_df,
        "history": history_df,
        "summary": summary_df,
        "articles": articles_df,
        "daily": daily_df,
        "status": status_df,
    }


def build_view_model(state: dict) -> dict:
    latest_df = state["latest"]
    if latest_df.empty:
        raise ValueError("Sheet xgb_latest_prediction masih kosong. Jalankan updater production dulu.")

    latest_row = latest_df.iloc[-1]
    latest_data_ts = pd.to_datetime(latest_row["latest_data_date"], errors="coerce")
    history_df = state["history"].copy()
    history_rows = []
    if not history_df.empty:
        history_df = history_df.sort_values("base_date")
        history_rows = history_df.tail(90).to_dict(orient="records")

    summary_rows = state["summary"].to_dict(orient="records")

    latest_daily = None
    daily_df = state["daily"]
    if not daily_df.empty:
        latest_daily = daily_df.sort_values("news_date").iloc[-1].to_dict()

    latest_news_date = None if latest_daily is None else str(latest_daily["news_date"])
    tone_label = "Belum ada news" if latest_daily is None else str(latest_daily["tone_label"])
    dominant_channel = "Belum ada" if latest_daily is None else str(latest_daily["dominant_channel"])
    latest_news_ts = pd.to_datetime(latest_news_date, errors="coerce") if latest_news_date else pd.NaT
    news_is_fresh = bool(pd.notna(latest_data_ts) and pd.notna(latest_news_ts) and latest_data_ts.date() == latest_news_ts.date())
    if latest_daily is None:
        sentiment_label = "Belum ada news"
        sentiment_note = "Belum ada news relevan pada refresh sentiment terakhir."
        sentiment_status = "Kosong"
    elif news_is_fresh:
        sentiment_label = tone_label
        sentiment_note = (
            f"Sentiment hari ini bersifat {tone_label.lower()} dengan channel utama {dominant_channel.lower()}."
        )
        sentiment_status = "Fresh"
    else:
        sentiment_label = "Belum ada news hari ini"
        sentiment_note = (
            f"Tidak ada news relevan baru untuk {latest_row['latest_data_date']}. "
            f"Sentiment terakhir {tone_label.lower()} pada {latest_news_date} "
            f"dengan channel utama {dominant_channel.lower()}."
        )
        sentiment_status = "Stale"

    delta_pct = float(latest_row["delta_pct"])
    model_signal = str(latest_row.get("signal", "Netral"))
    executive_note = (
        f"XGBoost memberi sinyal {model_signal.lower()} untuk {latest_row['forecast_date']} "
        f"dengan perubahan {delta_pct:+.2f}%. "
        f"{sentiment_note}"
    )

    top_articles_df = state["articles"].copy()
    top_articles = []
    if not top_articles_df.empty:
        top_articles_df = top_articles_df.sort_values("news_datetime", ascending=False).head(8).copy()
        top_articles_df["impact_channel"] = top_articles_df["impact_channel"].map(CHANNEL_MAP).fillna(
            top_articles_df["impact_channel"]
        )
        top_articles_df["impact_label"] = top_articles_df["impact_label"].map(IMPACT_LABEL_MAP).fillna(
            top_articles_df["impact_label"]
        )
        top_articles = top_articles_df.to_dict(orient="records")

    status_df = state["status"]
    latest_status = status_df.iloc[-1].to_dict() if not status_df.empty else {}
    freshness_note = "Data model dan sentiment berhasil dibaca dari spreadsheet production."
    if latest_status:
        sentiment_status = str(latest_status.get("sentiment_status", "")).strip()
        if sentiment_status and sentiment_status not in {"refreshed", "up_to_date"}:
            freshness_note = (
                f"Sentiment status terakhir: {sentiment_status}. "
                "Dashboard tetap memakai state terbaru yang tersimpan di spreadsheet."
            )

    return {
        "executive": {
            "forecast_date": str(latest_row["forecast_date"]),
            "latest_data_date": str(latest_row["latest_data_date"]),
            "current_price": float(latest_row["current_price"]),
            "predicted_price_t1": float(latest_row["pred_price_final_t1"]),
            "predicted_price_p10_t1": float(latest_row["pred_price_p10_t1"]),
            "predicted_price_p90_t1": float(latest_row["pred_price_p90_t1"]),
            "baseline_price_t1": float(latest_row["baseline_price_t1"]),
            "delta_pct": float(latest_row["delta_pct"]),
            "signal": model_signal,
            "sentiment_label": sentiment_label,
            "sentiment_score": 0.0 if latest_daily is None else float(latest_daily["market_sentiment_mean"]),
            "headline_note": executive_note,
        },
        "model": {
            "latest_data_date": str(latest_row["latest_data_date"]),
            "forecast_date": str(latest_row["forecast_date"]),
            "current_price": float(latest_row["current_price"]),
            "pred_price_final_t1": float(latest_row["pred_price_final_t1"]),
            "pred_price_corr_t1": float(latest_row["pred_price_corr_t1"]),
            "pred_price_p10_t1": float(latest_row["pred_price_p10_t1"]),
            "pred_price_p90_t1": float(latest_row["pred_price_p90_t1"]),
            "baseline_price_t1": float(latest_row["baseline_price_t1"]),
            "signal": model_signal,
            "gate_applied": str(latest_row.get("gate_applied", "")),
            "regime_active": str(latest_row.get("regime_active", "")),
            "locked_baseline_name": str(latest_row.get("locked_baseline_name", "")),
            "summary_rows": summary_rows,
            "recent_history": history_rows,
        },
        "sentiment": {
            "latest_news_date": latest_news_date,
            "latest_model_date": str(latest_row["latest_data_date"]),
            "latest_daily": latest_daily,
            "news_is_fresh": news_is_fresh,
            "sentiment_status": sentiment_status,
            "sentiment_note": sentiment_note,
            "recent_daily": daily_df.tail(60).to_dict(orient="records") if not daily_df.empty else [],
            "top_articles": top_articles,
        },
        "data_health": {
            "freshness_note": freshness_note,
            "spreadsheet_name": SPREADSHEET_NAME,
            "latest_pipeline_status": latest_status,
        },
    }


with st.sidebar:
    st.title("INALUM Dashboard")
    st.caption("XGBoost H+1 + Market Sentiment Context")
    page = st.radio("Menu", ["Executive Summary", "Model Detail", "Market Sentiment", "Data Health"])
    st.markdown("---")
    st.caption("Spreadsheet source")
    st.code(SPREADSHEET_NAME)


try:
    state = load_production_state()
    payload = build_view_model(state)
except Exception as exc:
    st.error("Dashboard belum bisa membaca state production dari Google Sheets.")
    st.info(
        "Pastikan spreadsheet production sudah terisi dan secret `GCP_SERVICE_ACCOUNT_JSON` tersedia di Streamlit Cloud."
    )
    st.code(str(exc))
    st.stop()


executive = payload["executive"]
model = payload["model"]
sentiment = payload["sentiment"]
data_health = payload["data_health"]


if page == "Executive Summary":
    st.title("Executive Summary")
    render_metric_cards(
        [
            ("Tanggal Prediksi", str(executive["forecast_date"])),
            ("Harga Terakhir", f"{executive['current_price']:.2f}"),
            ("Prediksi H+1", f"{executive['predicted_price_t1']:.2f}"),
            ("Delta (%)", f"{executive['delta_pct']:+.2f}%"),
            ("Sentiment", str(executive["sentiment_label"])),
        ]
    )
    st.markdown("---")
    st.info(executive["headline_note"])

    history = pd.DataFrame(model.get("recent_history", []))
    render_history_plot(history)

    top_articles = pd.DataFrame(sentiment.get("top_articles", []))
    if not top_articles.empty:
        st.subheader("Berita Terbaru yang Paling Relevan")
        render_simple_table(
            top_articles[
                ["news_date", "title", "impact_label", "impact_channel", "market_impact_score", "confidence"]
            ].copy()
        )

elif page == "Model Detail":
    st.title("Model Detail")

    model_detail = pd.DataFrame(
        [
            ("Harga terakhir", model["current_price"]),
            ("Prediksi XGBoost H+1", model["pred_price_final_t1"]),
            ("Prediksi baseline H+1", model["baseline_price_t1"]),
            ("Batas bawah P10", model["pred_price_p10_t1"]),
            ("Batas atas P90", model["pred_price_p90_t1"]),
            ("Signal", model["signal"]),
            ("Gate aktif", model["gate_applied"]),
            ("Regime aktif", model["regime_active"]),
            ("Baseline terkunci", model["locked_baseline_name"]),
        ],
        columns=["Metrik", "Nilai"],
    )
    render_simple_table(model_detail)

    st.subheader("Ringkasan Model")
    summary_rows = pd.DataFrame(model.get("summary_rows", []))
    render_simple_table(summary_rows)

elif page == "Market Sentiment":
    st.title("Market Sentiment")
    latest_daily = sentiment.get("latest_daily")
    if latest_daily:
        render_metric_cards(
            [
                ("Tanggal Acuan Harga", str(sentiment["latest_model_date"])),
                ("Tanggal News Terbaru", str(sentiment["latest_news_date"])),
                ("Status News", str(sentiment["sentiment_status"])),
                ("Sentiment Mean", f"{float(latest_daily['market_sentiment_mean']):+.2f}"),
                ("Tone / Channel", f"{latest_daily['tone_label']} / {latest_daily['dominant_channel']}"),
            ]
        )
        if not sentiment.get("news_is_fresh", False):
            st.warning(sentiment["sentiment_note"])
    else:
        st.warning("Belum ada news relevan terbaru pada refresh sentiment terakhir.")

    recent_daily = pd.DataFrame(sentiment.get("recent_daily", []))
    render_sentiment_plot(recent_daily)

    top_articles = pd.DataFrame(sentiment.get("top_articles", []))
    if not top_articles.empty:
        st.subheader("Top Articles")
        render_simple_table(top_articles)

elif page == "Data Health":
    st.title("Data Health")
    latest_status = data_health.get("latest_pipeline_status", {})
    health = pd.DataFrame(
        [
            ("Spreadsheet", data_health["spreadsheet_name"]),
            ("Latest model date", model["latest_data_date"]),
            ("Forecast date", model["forecast_date"]),
            ("Latest news date", sentiment["latest_news_date"]),
            ("News freshness", sentiment.get("sentiment_status", "")),
            ("Freshness note", data_health["freshness_note"]),
            ("Pipeline generated_at", latest_status.get("generated_at_utc", "")),
            ("Sentiment status", latest_status.get("sentiment_status", "")),
            ("Sentiment newly scored rows", latest_status.get("sentiment_newly_scored_rows", "")),
        ],
        columns=["Komponen", "Nilai"],
    )
    render_simple_table(health)
