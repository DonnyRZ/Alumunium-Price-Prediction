# News Sentiment Pilot

Tujuan tahap ini adalah pilot sentiment scoring untuk artikel `candidate_model` dari `EDA-NEWS`.

## File utama

- Input kandidat artikel: `data/news/staging/gdelt_eda_candidate_model_news_v4.csv`
- Output artikel terscore: `data/news/scored/gdelt_candidate_model_scored_v1.csv`
- Output fitur harian: `data/news/features/gdelt_daily_sentiment_features_v1.csv`

## Isi `.env`

Isi `GEMINI_API_KEY` di file `.env`, lalu sesuaikan variabel lain jika perlu.

## Jalankan scoring pilot

```bash
.venv/bin/python src/news/score_sentiment.py --limit 10
```

Untuk proses penuh:

```bash
.venv/bin/python src/news/score_sentiment.py
```

Script ini incremental:
- artikel yang sudah punya `article_id` di output akan dilewati
- gunakan `--force-rescore` jika ingin score ulang

Provider yang dipakai adalah **Gemini via REST API**, bukan OpenAI SDK.

## Buat agregasi harian

```bash
.venv/bin/python src/news/aggregate_daily_sentiment.py
```

## Buat overlay signal harian

```bash
.venv/bin/python src/news/build_overlay_signals.py
```

Output default:
- `data/news/features/gdelt_daily_overlay_signals_v1.csv`

Makna overlay saat ini:
- `bullish_conviction` → boleh memperkuat conviction bullish jika model utama sudah bullish
- `bearish_caution` → kurangi conviction bullish / hold for review, bukan hard veto
- `bullish_watch` / `bearish_watch` → ada konteks arah, tapi belum cukup kuat untuk aksi otomatis
- `ignore` → news tidak mengubah keputusan model

## Makna score

- `market_impact_score = -1` → sangat bearish untuk harga aluminium
- `market_impact_score = 0` → netral / tidak jelas
- `market_impact_score = +1` → sangat bullish untuk harga aluminium

## Catatan

- Tahap ini baru untuk `candidate_model`
- `candidate_readonly` sengaja belum masuk ke fitur model
- Integrasi ke XGBoost dilakukan setelah hasil scoring pilot diaudit
