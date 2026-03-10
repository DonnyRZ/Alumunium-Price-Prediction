from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_PATH = ROOT / "production" / "data" / "dashboard" / "latest_dashboard_payload.json"


st.set_page_config(
    page_title="INALUM Aluminium Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_payload() -> dict | None:
    if not PAYLOAD_PATH.exists():
        return None
    return json.loads(PAYLOAD_PATH.read_text())


payload = load_payload()

with st.sidebar:
    st.title("INALUM Dashboard")
    st.caption("XGBoost H+1 + Market Sentiment Context")
    page = st.radio(
        "Menu",
        ["Executive Summary", "Model Detail", "Market Sentiment", "Data Health"],
    )
    st.markdown("---")
    st.caption("Snapshot source")
    st.code(str(PAYLOAD_PATH.relative_to(ROOT)))


if payload is None:
    st.error("Snapshot dashboard belum tersedia.")
    st.info("Jalankan dulu: `.venv/bin/python production/pipeline/run_daily_pipeline.py`")
    st.stop()


executive = payload["executive"]
model = payload["model"]
sentiment = payload["sentiment"]
data_health = payload["data_health"]


if page == "Executive Summary":
    st.title("Executive Summary")

    cols = st.columns(5)
    cols[0].metric("Tanggal Prediksi", executive["forecast_date"])
    cols[1].metric("Harga Terakhir", f"{executive['current_price']:.2f}")
    cols[2].metric("Prediksi H+1", f"{executive['predicted_price_t1']:.2f}")
    cols[3].metric("Delta (%)", f"{executive['delta_pct']:+.2f}%")
    cols[4].metric("Sentiment", executive["sentiment_label"])

    st.markdown("---")
    st.info(executive["headline_note"])

    history = pd.DataFrame(model.get("recent_history", []))
    if not history.empty:
        history["base_date"] = pd.to_datetime(history["base_date"])
        plot_df = history.sort_values("base_date").tail(40).copy()
        chart_df = plot_df.set_index("base_date")[
            ["actual_next_price", "model_price_t1", "baseline_price_t1"]
        ]
        st.line_chart(chart_df)

    top_articles = pd.DataFrame(sentiment.get("top_articles", []))
    if not top_articles.empty:
        st.subheader("Berita Terbaru yang Paling Relevan")
        st.dataframe(
            top_articles[
                ["news_date", "title", "impact_label", "impact_channel", "market_impact_score", "confidence"]
            ],
            use_container_width=True,
            hide_index=True,
        )


elif page == "Model Detail":
    st.title("Model Detail")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prediksi Hari Ini")
        model_detail = pd.DataFrame(
            [
                ("Harga terakhir", model["current_price"]),
                ("Prediksi final H+1", model["pred_price_final_t1"]),
                ("Prediksi baseline H+1", model["baseline_price_t1"]),
                ("Batas bawah P10", model["pred_price_p10_t1"]),
                ("Batas atas P90", model["pred_price_p90_t1"]),
                ("Signal", model["signal"]),
                ("Gate aktif", model["gate_applied"]),
                ("Regime aktif", model["regime_active"]),
            ],
            columns=["Metrik", "Nilai"],
        )
        st.dataframe(model_detail, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Ringkasan Model")
        summary_rows = pd.DataFrame(model.get("summary_rows", []))
        if not summary_rows.empty:
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)


elif page == "Market Sentiment":
    st.title("Market Sentiment")

    latest_daily = sentiment.get("latest_daily")
    if latest_daily:
        cols = st.columns(4)
        cols[0].metric("Tanggal News Terbaru", sentiment["latest_news_date"])
        cols[1].metric("Jumlah Berita", int(latest_daily["news_count_model"]))
        cols[2].metric("Sentiment Mean", f"{latest_daily['market_sentiment_mean']:+.2f}")
        cols[3].metric("Channel Utama", latest_daily["dominant_channel"])
    else:
        st.warning("Belum ada news relevan terbaru pada refresh sentiment terakhir.")

    recent_daily = pd.DataFrame(sentiment.get("recent_daily", []))
    if not recent_daily.empty:
        recent_daily["news_date"] = pd.to_datetime(recent_daily["news_date"])
        recent_daily = recent_daily.sort_values("news_date")
        chart_df = recent_daily.set_index("news_date")[["market_sentiment_mean", "news_count_model"]]
        st.line_chart(chart_df)

    top_articles = pd.DataFrame(sentiment.get("top_articles", []))
    if not top_articles.empty:
        st.subheader("Top Articles")
        st.dataframe(top_articles, use_container_width=True, hide_index=True)


elif page == "Data Health":
    st.title("Data Health")

    health = pd.DataFrame(
        [
            ("Generated at", payload["generated_at_utc"]),
            ("Latest model date", model["latest_data_date"]),
            ("Forecast date", model["forecast_date"]),
            ("Latest news date", sentiment["latest_news_date"]),
            ("Freshness note", data_health["freshness_note"]),
            ("Model snapshot", data_health["model_snapshot_path"]),
            ("Sentiment snapshot", data_health["sentiment_snapshot_path"]),
        ],
        columns=["Komponen", "Nilai"],
    )
    st.dataframe(health, use_container_width=True, hide_index=True)
