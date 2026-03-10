# Production

Folder ini adalah area kerja **production** untuk dashboard pengambilan keputusan.

Tujuannya sederhana:

- memisahkan file production dari notebook/eksperimen,
- menyediakan **single source of truth** untuk dashboard,
- dan menghindari munculnya banyak file output baru setiap kali update.

## Struktur

- `production/pipeline/`
  - script untuk membentuk snapshot production
- `production/dashboard/`
  - aplikasi Streamlit
- `production/data/model/`
  - snapshot model terbaru
- `production/data/sentiment/`
  - data sentiment terbaru dan snapshot sentiment
- `production/data/dashboard/`
  - payload final yang dibaca dashboard

## Prinsip output

Output production selalu ditulis ke path yang tetap:

- `production/data/model/latest_snapshot.json`
- `production/data/sentiment/latest_articles.csv`
- `production/data/sentiment/latest_daily.csv`
- `production/data/sentiment/latest_snapshot.json`
- `production/data/dashboard/latest_dashboard_payload.json`

Jadi tidak ada file versi baru yang menumpuk setiap update.

## Cara pakai

Bangun snapshot terbaru:

```bash
.venv/bin/python production/pipeline/run_daily_pipeline.py
```

Yang dilakukan pipeline harian:

- membangun prediksi `H+1` XGBoost terbaru,
- refresh berita recent window dan score artikel baru dengan Gemini,
- membangun agregasi sentiment harian terbaru,
- lalu menulis satu payload final untuk dashboard.

Jalankan dashboard:

```bash
.venv/bin/streamlit run production/dashboard/app.py
```

## Deployment yang disarankan

Untuk deployment ke **Streamlit Cloud**, gunakan pendekatan ini:

- **GitHub Actions** menjalankan `production/pipeline/run_daily_pipeline.py`
- workflow hanya meng-commit file final:
  - `production/data/dashboard/latest_dashboard_payload.json`
- **Streamlit Cloud** hanya membaca file payload final itu
- app entrypoint di Streamlit Cloud: `production/dashboard/app.py`

Dengan pendekatan ini:

- dashboard tidak perlu menjalankan training/fetch/scoring saat dibuka,
- secret `GEMINI_API_KEY` cukup disimpan di GitHub Actions,
- dan repo tetap rapi karena file yang dipush hanya payload final terbaru.

Workflow harian sudah disiapkan di:

- `.github/workflows/refresh-production-dashboard.yml`

Runtime Python untuk Streamlit Cloud juga sudah dikunci di:

- `runtime.txt`

## Catatan

- Model utama yang dipakai saat ini adalah **XGBoost H+1**
- Sentiment dipakai sebagai **konteks pasar**
- Prophet dan SARIMAX masih di luar scope production saat ini
