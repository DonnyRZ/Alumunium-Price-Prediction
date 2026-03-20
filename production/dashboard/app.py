from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from production.gsheet_manager import read_sheet
from production.pipeline.common import action_label_id
from production.sheet_contract import (
    SPREADSHEET_NAME,
    TAB_ACCOUNT_PRIORITY_QUEUE,
    TAB_AI_BRIEF_LATEST,
    TAB_ARTICLES_SCORED,
    TAB_MARKET_CONTEXT_DAILY,
    TAB_MARKETING_PLAYBOOK,
    TAB_QUOTE_DECISION_LOG,
    TAB_RUN_STATUS,
    TAB_SIGNAL_HISTORY,
    TAB_SIGNAL_LATEST,
    TAB_SIGNAL_METRICS,
)


st.set_page_config(
    page_title="INALUM Marketing Signal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCAL_SIGNAL_JSON = ROOT / "production" / "data" / "model" / "signal_latest_snapshot.json"
LOCAL_SIGNAL_HISTORY_CSV = ROOT / "production" / "data" / "model" / "signal_history_snapshot.csv"
LOCAL_SIGNAL_METRICS_CSV = ROOT / "production" / "data" / "model" / "signal_metrics_snapshot.csv"
LOCAL_AI_BRIEF_JSON = ROOT / "production" / "data" / "dashboard" / "ai_market_brief_latest.json"


ACTION_STYLE = {
    "ignore": {"label": "Abaikan", "accent": "#6B7280", "bg": "#F3F4F6"},
    "watch": {"label": "Pantau", "accent": "#C67A00", "bg": "#FFF7E6"},
    "act": {"label": "Tindak", "accent": "#A64600", "bg": "#FFF0E8"},
}


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: #F7F4ED;
          --ink: #1E1B16;
          --muted: #6F6A61;
          --line: #E7E0D4;
          --brand: #A64600;
          --brand-soft: #FFF0E8;
          --olive: #53624E;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(166,70,0,0.08), transparent 22%),
            radial-gradient(circle at top right, rgba(83,98,78,0.09), transparent 20%),
            linear-gradient(180deg, #FAF8F2 0%, #F3EEE4 100%);
        }
        .hero-card, .soft-card {
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 18px 20px;
          background: rgba(255,255,255,0.72);
          box-shadow: 0 10px 35px rgba(30,27,22,0.05);
        }
        .hero-title {
          font-size: 0.82rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 10px;
        }
        .hero-value {
          font-size: 2rem;
          font-weight: 800;
          line-height: 1.1;
          color: var(--ink);
        }
        .hero-sub {
          margin-top: 8px;
          color: var(--muted);
          font-size: 0.95rem;
        }
        .badge {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 0.83rem;
          font-weight: 700;
        }
        .section-note {
          color: var(--muted);
          font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value: float | int | str | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def _format_percent(value: float | int | str | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return str(value)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sentiment_label(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Belum ada news"
    if score >= 0.20:
        return "News cenderung positif"
    if score <= -0.20:
        return "News cenderung negatif"
    return "News cenderung netral"


def _action_badge(action_level: str) -> str:
    style = ACTION_STYLE.get(str(action_level).strip().lower(), ACTION_STYLE["watch"])
    return (
        f"<span class='badge' style='background:{style['bg']};color:{style['accent']};"
        f"border:1px solid {style['accent']}33'>{style['label']}</span>"
    )


def _split_talking_points(raw: str | None) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="soft-card">
                  <div class="hero-title">{label}</div>
                  <div class="hero-value" style="font-size:1.35rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Belum ada data untuk ditampilkan.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_signal_history(signal_history: pd.DataFrame) -> None:
    if signal_history.empty:
        st.info("Histori signal belum tersedia.")
        return

    plot_df = signal_history.copy()
    plot_df["base_date"] = pd.to_datetime(plot_df["base_date"], errors="coerce")
    plot_df["probability_action"] = pd.to_numeric(plot_df["probability_action"], errors="coerce")
    plot_df["actual_abs_return"] = pd.to_numeric(plot_df["actual_abs_return"], errors="coerce")
    plot_df = plot_df.sort_values("base_date").tail(36)

    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    ax1.axhspan(0.00, 0.50, color="#F3F4F6", alpha=0.9)
    ax1.axhspan(0.50, 0.65, color="#FFF7E6", alpha=0.9)
    ax1.axhspan(0.65, 1.00, color="#FFF0E8", alpha=0.9)
    ax1.plot(plot_df["base_date"], plot_df["probability_action"], color="#A64600", linewidth=2.2, label="Probability action")
    ax1.axhline(0.50, color="#6B7280", linestyle="--", linewidth=1.1, label="Threshold 0.50")
    ax1.axhline(0.65, color="#C67A00", linestyle=":", linewidth=1.2, label="Threshold 0.65")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Probability")
    ax1.set_title("Perjalanan Signal 5 Hari Terakhir")
    ax1.grid(alpha=0.18)

    ax2 = ax1.twinx()
    ax2.bar(plot_df["base_date"], plot_df["actual_abs_return"], width=3.5, alpha=0.12, color="#53624E", label="Actual |return|")
    ax2.set_ylabel("|Return| aktual")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Area abu-abu = ignore, kuning = watch, oranye = act.")


def render_sentiment_plot(market_context: pd.DataFrame) -> None:
    if market_context.empty:
        st.info("Konteks news harian belum tersedia.")
        return

    plot_df = market_context.copy()
    plot_df["news_date"] = pd.to_datetime(plot_df["news_date"], errors="coerce")
    plot_df["market_sentiment_mean"] = pd.to_numeric(plot_df["market_sentiment_mean"], errors="coerce")
    plot_df["news_count_model"] = pd.to_numeric(plot_df["news_count_model"], errors="coerce")
    plot_df = plot_df.sort_values("news_date").tail(45)

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(plot_df["news_date"], plot_df["market_sentiment_mean"], color="#53624E", linewidth=2.0, label="Sentiment mean")
    ax1.axhline(0.20, color="#A64600", linestyle="--", linewidth=1)
    ax1.axhline(-0.20, color="#A64600", linestyle="--", linewidth=1)
    ax1.set_ylabel("Skor sentiment")
    ax1.set_title("Konteks News Harian")
    ax1.grid(alpha=0.18)

    ax2 = ax1.twinx()
    ax2.bar(plot_df["news_date"], plot_df["news_count_model"], alpha=0.18, color="#C67A00", label="News count")
    ax2.set_ylabel("Jumlah berita")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data(ttl=300, show_spinner=False)
def load_production_state() -> dict:
    signal_latest = read_sheet(TAB_SIGNAL_LATEST)
    signal_history = read_sheet(TAB_SIGNAL_HISTORY)
    signal_metrics = read_sheet(TAB_SIGNAL_METRICS)
    market_context = read_sheet(TAB_MARKET_CONTEXT_DAILY)
    articles = read_sheet(TAB_ARTICLES_SCORED)
    ai_brief = read_sheet(TAB_AI_BRIEF_LATEST)
    playbook = read_sheet(TAB_MARKETING_PLAYBOOK)
    run_status = read_sheet(TAB_RUN_STATUS)
    account_queue = read_sheet(TAB_ACCOUNT_PRIORITY_QUEUE)
    decision_log = read_sheet(TAB_QUOTE_DECISION_LOG)

    if signal_latest.empty and LOCAL_SIGNAL_JSON.exists():
        signal_latest = pd.DataFrame([json.loads(LOCAL_SIGNAL_JSON.read_text())])
    if signal_history.empty and LOCAL_SIGNAL_HISTORY_CSV.exists():
        signal_history = pd.read_csv(LOCAL_SIGNAL_HISTORY_CSV)
    if signal_metrics.empty and LOCAL_SIGNAL_METRICS_CSV.exists():
        signal_metrics = pd.read_csv(LOCAL_SIGNAL_METRICS_CSV)
    if ai_brief.empty and LOCAL_AI_BRIEF_JSON.exists():
        ai_brief = pd.DataFrame([json.loads(LOCAL_AI_BRIEF_JSON.read_text())])

    signal_latest = _to_numeric(
        signal_latest,
        [
            "probability_action",
            "technical_threshold",
            "watch_threshold",
            "current_block_close",
            "block_price_range_pct",
            "freshness_mean",
            "freshness_min",
            "staleness_max",
        ],
    )
    signal_history = _to_numeric(
        signal_history,
        ["probability_action", "actual_abs_return", "actual_return", "current_block_close"],
    )
    signal_metrics = _to_numeric(signal_metrics, ["value", "threshold", "acc", "bal_acc", "f1", "precision", "recall", "flag_rate"])
    market_context = _to_numeric(
        market_context,
        [
            "news_count_model",
            "market_sentiment_mean",
            "market_sentiment_sum",
            "bullish_ratio",
            "bearish_ratio",
            "high_confidence_ratio",
        ],
    )
    articles = _to_numeric(articles, ["market_impact_score", "confidence"])
    run_status = _to_numeric(run_status, ["signal_probability_action"])

    return {
        "signal_latest": signal_latest,
        "signal_history": signal_history,
        "signal_metrics": signal_metrics,
        "market_context": market_context,
        "articles": articles,
        "ai_brief": ai_brief,
        "playbook": playbook,
        "run_status": run_status,
        "account_queue": account_queue,
        "decision_log": decision_log,
    }


def build_view_model(state: dict) -> dict:
    signal_latest_df = state["signal_latest"]
    if signal_latest_df.empty:
        raise ValueError("Sheet signal_latest masih kosong. Jalankan pipeline production terbaru dulu.")

    latest_signal = signal_latest_df.iloc[-1].to_dict()
    market_context_df = state["market_context"]
    latest_context = (
        market_context_df.sort_values("news_date").iloc[-1].to_dict() if not market_context_df.empty else {}
    )
    ai_brief_df = state["ai_brief"]
    latest_brief = ai_brief_df.iloc[-1].to_dict() if not ai_brief_df.empty else {}
    run_status_df = state["run_status"]
    latest_status = run_status_df.iloc[-1].to_dict() if not run_status_df.empty else {}

    probability = float(latest_signal.get("probability_action", 0.0))
    action_level = str(latest_signal.get("action_level", "watch"))
    action_label = action_label_id(action_level)
    sentiment_mean = latest_context.get("market_sentiment_mean")
    sentiment_label = _sentiment_label(sentiment_mean if sentiment_mean not in ("", None) else None)

    return {
        "latest_signal": latest_signal,
        "signal_history": state["signal_history"].copy(),
        "signal_metrics": state["signal_metrics"].copy(),
        "market_context": market_context_df.copy(),
        "articles": state["articles"].copy(),
        "latest_context": latest_context,
        "latest_brief": latest_brief,
        "playbook": state["playbook"].copy(),
        "run_status": latest_status,
        "account_queue": state["account_queue"].copy(),
        "decision_log": state["decision_log"].copy(),
        "headline_note": (
            f"Signal saat ini berada pada level {action_label.lower()} dengan probability {probability:.2f}. "
            f"{latest_brief.get('recommended_action', 'Gunakan signal ini sebagai alat bantu keputusan.')}"
        ),
        "sentiment_label": sentiment_label,
    }


def render_executive_signal(view: dict) -> None:
    latest_signal = view["latest_signal"]
    latest_brief = view["latest_brief"]
    action_level = str(latest_signal.get("action_level", "watch"))
    style = ACTION_STYLE.get(action_level, ACTION_STYLE["watch"])
    probability = float(latest_signal.get("probability_action", 0.0))

    left, right = st.columns([1.3, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="hero-card" style="background:linear-gradient(180deg,{style['bg']} 0%,rgba(255,255,255,0.85) 100%);">
              <div class="hero-title">Status Marketing Hari Ini</div>
              <div class="hero-value" style="color:{style['accent']};">{action_label_id(action_level)}</div>
              <div class="hero-sub">
                Probability action: <b>{probability:.2f}</b><br/>
                Forecast window: <b>{latest_signal.get('forecast_window_start', '-')}</b> s.d. <b>{latest_signal.get('forecast_window_end', '-')}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Semakin tinggi probability, semakin kuat sinyal bahwa periode pasar ini perlu perhatian lebih.")

    with right:
        st.markdown(
            f"""
            <div class="soft-card">
              <div class="hero-title">Inti Rekomendasi</div>
              <div style="font-size:1.02rem;color:#1E1B16;line-height:1.65;">
                {latest_brief.get('recommended_action', latest_signal.get('signal_reason_short', '-'))}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_metric_cards(
        [
            ("Signal", latest_signal.get("action_label_id", "-")),
            ("Urgency", latest_signal.get("urgency", "-")),
            ("Sentiment", view["sentiment_label"]),
            ("Harga Acuan Blok", _format_number(latest_signal.get("current_block_close"))),
        ]
    )

    st.markdown("---")
    st.info(view["headline_note"])

    brief_summary = latest_brief.get("executive_summary")
    if brief_summary:
        st.subheader("Ringkasan untuk Tim Marketing")
        st.write(brief_summary)

    why_it_matters = latest_brief.get("why_it_matters")
    if why_it_matters:
        st.subheader("Kenapa Ini Penting")
        st.write(why_it_matters)

    driver_cols = st.columns(3)
    for idx, key in enumerate(["top_driver_1", "top_driver_2", "top_driver_3"]):
        with driver_cols[idx]:
            st.markdown(
                f"""
                <div class="soft-card">
                  <div class="hero-title">Driver {idx + 1}</div>
                  <div style="font-size:1rem;font-weight:600;color:#1E1B16;">{latest_signal.get(key, '-') or '-'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Trend Signal Terbaru")
    render_signal_history(view["signal_history"])


def render_market_context(view: dict) -> None:
    st.subheader("Konteks News dan Pasar")
    render_sentiment_plot(view["market_context"])

    latest_context = view["latest_context"]
    if latest_context:
        render_metric_cards(
            [
                ("Tanggal News", str(latest_context.get("news_date", "-"))),
                ("News Count", str(int(float(latest_context.get("news_count_model", 0) or 0)))),
                ("Sentiment Mean", _format_number(latest_context.get("market_sentiment_mean"))),
                ("High Confidence Ratio", _format_percent(latest_context.get("high_confidence_ratio"))),
            ]
        )

    st.markdown("---")
    st.subheader("Headline News Terbaru")
    articles = view["articles"].copy()
    if articles.empty:
        st.info("Belum ada artikel hasil scoring yang tersedia.")
    else:
        articles = articles.sort_values("news_datetime", ascending=False).head(8).copy()
        display_df = pd.DataFrame(
            {
                "Tanggal": articles.get("news_date", "-"),
                "Headline": articles.get("title", "-"),
                "Arah": articles.get("impact_label", "-"),
                "Topik": articles.get("impact_channel", "-"),
                "Skor": pd.to_numeric(articles.get("market_impact_score"), errors="coerce").map(
                    lambda value: "-" if pd.isna(value) else f"{value:+.2f}"
                ),
                "Key Insight": articles.get("reason_short", "-").fillna("-"),
            }
        )
        render_dataframe(display_df)


def render_action_center(view: dict) -> None:
    latest_brief = view["latest_brief"]

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        st.subheader("Rekomendasi Tindakan")
        st.markdown(
            f"""
            <div class="soft-card">
              <div class="hero-title">Recommended Action</div>
              <div style="font-size:1.05rem;line-height:1.7;color:#1E1B16;">
                {latest_brief.get('recommended_action', '-')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Caution Internal")
        st.warning(latest_brief.get("internal_caution_note", "Belum ada caution note."))

    with col2:
        st.subheader("Cara Baca Signal")
        playbook = view["playbook"].copy()
        if playbook.empty:
            st.info("Playbook belum tersedia.")
        else:
            render_dataframe(playbook[["label_id", "probability_range", "urgency", "recommended_action"]])

    st.markdown("---")
    st.subheader("Talking Points untuk Pelanggan")
    talking_points = _split_talking_points(latest_brief.get("customer_talking_points"))
    if not talking_points:
        st.info("Belum ada talking points yang tersedia.")
    else:
        for item in talking_points:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.subheader("Prioritas Account")
    account_queue = view["account_queue"].copy()
    if account_queue.empty:
        st.info("Sheet account priority queue masih kosong. Bisa dipakai tim marketing sebagai daftar prioritas manual.")
    else:
        render_dataframe(account_queue.head(20))


def render_ops_and_audit(view: dict) -> None:
    latest_status = view["run_status"]
    metrics = view["signal_metrics"].copy()

    st.subheader("Status Pipeline")
    if latest_status:
        status_df = pd.DataFrame([latest_status])
        render_dataframe(status_df)
    else:
        st.info("Status pipeline belum tersedia.")

    st.subheader("Ringkasan Model")
    if metrics.empty:
        st.info("Metrik model belum tersedia.")
    else:
        summary_df = metrics[metrics["section"].eq("summary")].copy()
        if not summary_df.empty:
            render_dataframe(summary_df[["metric", "value", "meaning"]])

        threshold_df = metrics[metrics["section"].eq("threshold_tradeoff")].copy()
        if not threshold_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            for col, color in [
                ("bal_acc", "#A64600"),
                ("precision", "#53624E"),
                ("recall", "#C67A00"),
                ("flag_rate", "#6B7280"),
            ]:
                ax.plot(threshold_df["threshold"], threshold_df[col], marker="o", linewidth=2, label=col, color=color)
            ax.set_title("Threshold Trade-off")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Skor / Rate")
            ax.grid(alpha=0.18)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Audit Log")
    decision_log = view["decision_log"].copy()
    if decision_log.empty:
        st.info("Quote decision log masih kosong. Ini nantinya dipakai untuk membuktikan manfaat bisnis model.")
    else:
        render_dataframe(decision_log.tail(30))


_apply_theme()

with st.sidebar:
    st.title("INALUM Marketing")
    st.caption("Decision-support dashboard for 5-day signal")
    page = st.radio(
        "Halaman",
        ["Executive Signal", "Market Context", "Action Center", "Ops & Audit"],
    )
    st.markdown("---")
    st.caption("Workbook source")
    st.code(SPREADSHEET_NAME)


try:
    state = load_production_state()
    view = build_view_model(state)
except Exception as exc:
    st.error("Dashboard belum bisa membaca state production baru dari Google Sheets.")
    st.info("Pastikan pipeline production terbaru sudah dijalankan dan workbook baru sudah terisi.")
    st.code(str(exc))
    st.stop()


st.title("INALUM Marketing Decision Dashboard")
st.caption("Fokus dashboard ini adalah membantu tim marketing memutuskan kapan perlu mengabaikan, memantau, atau menindak sinyal pasar.")

if page == "Executive Signal":
    render_executive_signal(view)
elif page == "Market Context":
    render_market_context(view)
elif page == "Action Center":
    render_action_center(view)
else:
    render_ops_and_audit(view)
