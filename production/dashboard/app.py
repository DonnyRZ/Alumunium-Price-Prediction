from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    cards_html = "".join(
        f"""
        <div style="background:#f7f7fb;border:1px solid #e7e7ef;border-radius:12px;padding:16px;min-height:96px;">
          <div style="font-size:0.9rem;color:#666;margin-bottom:8px;">{label}</div>
          <div style="font-size:1.4rem;font-weight:700;color:#222;">{value}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;">
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
    st.markdown(
        f"""
        <div style="overflow-x:auto;">
          {html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_plot(history: pd.DataFrame) -> None:
    if history.empty:
        return
    history = history.copy()
    history["base_date"] = pd.to_datetime(history["base_date"])
    plot_df = history.sort_values("base_date").tail(40)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["base_date"], plot_df["actual_next_price"], label="Actual", linewidth=2)
    ax.plot(plot_df["base_date"], plot_df["model_price_t1"], label="XGBoost", linewidth=2)
    ax.set_xlabel("Tanggal Dasar")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_sentiment_plot(recent_daily: pd.DataFrame) -> None:
    if recent_daily.empty:
        return
    recent_daily = recent_daily.copy()
    recent_daily["news_date"] = pd.to_datetime(recent_daily["news_date"])
    recent_daily = recent_daily.sort_values("news_date")
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
                ("Tanggal News Terbaru", str(sentiment["latest_news_date"])),
                ("Jumlah Berita", str(int(latest_daily["news_count_model"]))),
                ("Sentiment Mean", f"{latest_daily['market_sentiment_mean']:+.2f}"),
                ("Channel Utama", str(latest_daily["dominant_channel"])),
                ("Tone", str(latest_daily["tone_label"])),
            ]
        )
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
    render_simple_table(health)
