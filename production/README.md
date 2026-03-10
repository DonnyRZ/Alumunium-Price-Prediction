# Production

Folder ini adalah area kerja **production** untuk dashboard pengambilan keputusan INALUM.

## Prinsip utama

- **XGBoost H+1** adalah model utama.
- **Sentiment** dipakai sebagai konteks pasar harian.
- **Google Sheets** adalah source of truth production.
- **Streamlit** hanya membaca state production, bukan menjalankan ETL berat saat dibuka.

Spreadsheet production yang dipakai:

- `Alumunium_Data_Master`

## Struktur production

- `production/pipeline/`
  - updater harian untuk model dan sentiment
- `production/dashboard/`
  - aplikasi Streamlit
- `production/config/`
  - kontrak/config production yang ikut repo
- `production/gsheet_manager.py`
  - connector Google Sheets
- `production/sheet_contract.py`
  - daftar nama tab spreadsheet

## Tab spreadsheet yang dipakai

- `xgb_latest_prediction`
  - snapshot 1 row prediksi XGBoost terbaru
- `xgb_prediction_history`
  - histori prediksi harian untuk chart dashboard
- `xgb_model_summary`
  - ringkasan metrik model
- `sentiment_articles_scored`
  - cache artikel yang sudah discore
- `sentiment_daily`
  - agregasi sentiment harian
- `pipeline_status`
  - status updater harian terakhir

## Cara kerja production

### 1. GitHub Actions / updater harian

Menjalankan:

```bash
.venv/bin/python production/pipeline/run_daily_pipeline.py
```

Yang dilakukan updater:

- membangun prediksi `H+1` XGBoost terbaru,
- menulis latest prediction + history ke spreadsheet,
- fetch recent news,
- score **hanya artikel baru** dengan Gemini,
- update agregasi sentiment harian,
- menulis status run terakhir ke spreadsheet.

### 2. Streamlit Cloud

Entry point:

```text
production/dashboard/app.py
```

Dashboard membaca langsung dari spreadsheet production.

## Secret yang dibutuhkan

### GitHub Actions

- `GEMINI_API_KEY`
- `GCP_SERVICE_ACCOUNT_JSON`

### Streamlit Cloud

Karena dashboard membaca spreadsheet langsung, Streamlit Cloud juga perlu:

- `GCP_SERVICE_ACCOUNT_JSON`

Gunakan isi JSON service account mentah sebagai secret/string.

## Catatan implementasi

- `service_account_key.json` boleh dipakai lokal, tapi **jangan di-commit**.
- Workflow harian sudah disiapkan di:
  - `.github/workflows/refresh-production-dashboard.yml`
- Update sentiment harian dibuat **incremental** agar tidak scoring ulang artikel lama setiap run.
- Jika refresh sentiment bermasalah, state lama di spreadsheet tetap dipakai.

## Status scope

Scope production saat ini:

- XGBoost `H+1`
- sentiment context

Masih di luar scope:

- Prophet
- SARIMAX
- ensemble multi-model
