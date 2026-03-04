# EDA Report: ALI=F (Aluminum Futures)

**Dataset:** Yahoo Finance - Aluminum Futures  
**Periode:** 2014-05-06 s/d 2026-03-03  
**Total Data:** 2,979 hari trading (±12 tahun)

---

## Executive Summary

| Aspek | Status | Keterangan |
|-------|--------|------------|
| **Kualitas Data** | Moderate | Missing values 1.31%, settlement-style |
| **Missing Values** | 39 hari | Close NaN pada 2016-2023 |
| **Repaired Data** | 5 hari | 0.17% dari total data |
| **Flat Candle** | 83.15% | Settlement-style data |
| **Stale Behavior** | 16.11% | Return = 0 selama 480 hari |
| **Outliers** | 1 ekstrem | -18.20% pada 2019-06-27 |

**Kesimpulan:** Dataset **LAYAK** untuk Close/return forecasting dengan preprocessing.

---

## 1️. Loading & Inspection

### 1.1 Struktur Data

```
Shape: (2979 rows, 7 columns)
Columns: Adj Close, Close, High, Low, Open, Repaired?, Volume
Date Range: 2014-05-06 s/d 2026-03-03
```

### 1.2 Tipe Data

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| Adj Close | float64 | Harga penutupan yang disesuaikan |
| Close | float64 | Harga penutupan |
| High | float64 | Harga tertinggi |
| Low | float64 | Harga terendah |
| Open | float64 | Harga pembukaan |
| Repaired? | bool | Flag data yang diperbaiki |
| Volume | int64 | Volume trading |

### 1.3 Sample Data (5 Hari Pertama)

| Date | Open | High | Low | Close | Volume | Repaired? |
|------|------|------|-----|-------|--------|-----------|
| 2014-05-06 | 2182.75 | 2205.75 | 2165.00 | 2172.75 | 41 | False |
| 2014-05-07 | 2152.25 | 2152.25 | 2146.00 | 2149.00 | 35 | False |
| 2014-05-08 | 2150.00 | 2150.00 | 2130.00 | 2141.75 | 25 | False |
| 2014-05-09 | 2133.50 | 2133.50 | 2107.25 | 2107.25 | 14 | False |
| 2014-05-12 | 2086.00 | 2088.25 | 2086.00 | 2088.25 | 4 | False |

---

## 2️. Data Quality Check

### 2.1 Missing Values

**Total Missing:** 39 hari (1.31%)

| Kolom | Missing Count | Percentage |
|-------|--------------|------------|
| Adj Close | 39 | 1.31% |
| Close | 39 | 1.31% |
| High | 39 | 1.31% |
| Low | 39 | 1.31% |
| Open | 39 | 1.31% |

**Rentang Tanggal NaN:** 2016-01-18 s/d 2023-11-23

**Pola Missing:**
- Konsentrasi terbesar: Sep-Okt 2016 (29 hari berturut-turut)
- Terserak: Jul 2017 (3 hari), Jan 2018 (1 hari), Nov 2023 (1 hari)

**Interpretasi:** Missing values ini bukan gap tanggal, tetapi harga yang tidak tersedia pada tanggal tertentu. Mudah ditangani dengan drop baris NaN.

---

### 2.2 Repaired Data Flag

**Total Repaired:** 5 hari (0.17%)

| Tanggal | Keterangan |
|---------|------------|
| 2024-11-15 | Diperbaiki oleh yfinance |
| 2025-05-26 | Diperbaiki oleh yfinance |
| 2025-06-19 | Diperbaiki oleh yfinance |
| 2025-07-04 | Diperbaiki oleh yfinance |
| 2026-03-02 | Diperbaiki oleh yfinance |

**Interpretasi:** Sangat sedikit data yang diperbaiki, menunjukkan kualitas data Yahoo Finance untuk ALI=F cukup baik.

---

### 2.3 Settlement-Style Data (Flat Candle) 

**Temuan Utama:** 83.15% hari memiliki Open = High = Low = Close

| Tahun | % Flat Candle | Visualisasi |
|-------|--------------|-------------|
| 2014 | 64.07% | ████████░░ |
| 2015 | 73.41% | █████████░ |
| 2016 | 36.51% | ████░░░░░░ ← Paling "normal" |
| 2017 | 93.23% | ██████████ |
| 2018 | 99.60% | ██████████ ← Hampir semua flat |
| 2019 | 96.03% | ██████████ |
| 2020 | 97.23% | ██████████ |
| 2021 | 97.22% | ██████████ |
| 2022 | 97.21% | ██████████ |
| 2023 | 80.88% | ████████░░ |
| 2024 | 79.76% | ████████░░ |
| 2025 | 79.53% | ████████░░ |
| 2026 | 63.41% | ██████░░░░ |

**Interpretasi:**
- Data ini adalah **settlement price** (1 harga per hari)
- Yahoo Finance menyalin harga settlement ke semua kolom OHLC
- **Bukan** data trading intraday yang sebenarnya

**Implikasi untuk ML:**
- Tidak cocok untuk analisis candlestick
- Tidak cocok untuk fitur berbasis High-Low range (ATR, wick)
- Masih OK untuk forecasting return/price berbasis Close

---

### 2.4 Stale / Plateau Behavior

**Definisi:** Periode di mana harga Close tidak berubah selama beberapa hari berturut-turut.

| Metrik | Nilai |
|--------|-------|
| Run dengan ≥5 hari konstan | **17 run** |
| Streak terpanjang | **~90 hari** (2019-02-20 s/d 2019-06-26) |
| Hari dengan return = 0 | **480 hari (16.11%)** |

**Detail Streak Terpanjang (90 hari):**
```
Periode:    2019-02-20 s/d 2019-06-26
Durasi:     90 hari trading
Harga:      2225.00 (konstan)
Volume:     0 (semua hari)
Akhir:      2019-06-27 (jatuh ke 1820, -18.2%)
```

**Daftar 17 Run (≥5 hari):**

| # | Periode | Durasi | Harga | Keterangan |
|---|---------|--------|-------|------------|
| 1 | 2019-02-20 s/d 2019-06-26 | 90 hari | 2225 | TERPANJANG |
| 2 | 2016-09-06 s/d 2016-10-27 | 30 hari | NaN | Blok NaN |
| 3 | 2017-01-05 s/d 2017-03-10 | 45 hari | 1950 | Streak panjang |
| 4 | 2018-03-12 s/d 2018-04-20 | 30 hari | 2350 | Streak panjang |
| 5 | 2020-05-18 s/d 2020-06-22 | 25 hari | 1650 | Pandemi COVID |
| 6-17 | Berbagai periode | 5-10 hari | Bervariasi | Streak pendek |

---

### 2.5 Outlier Detection

**Statistik Return Harian:**

| Metrik | Nilai |
|--------|-------|
| Mean | +0.0175% / hari |
| Std | ±1.23% / hari |
| Min | **-18.20%** ← OUTLIER EKSTREM |
| Max | +6.91% |
| Return = 0 | 480 hari (16.11%) |

**Outlier Terbesar:**

| Tanggal | Close | Return | Volume | Keterangan |
|---------|-------|--------|--------|------------|
| 2019-06-27 | 1820.00 | **-18.20%** | 0 | Artefak data |

**Konteks Outlier 2019-06-27:**
- Terjadi setelah **90 hari harga konstan** di 2225
- Volume = 0 pada semua hari
- Diduga **stale update/artefak data**, bukan pergerakan normal

**Top 5 Return Terburuk:**

| Tanggal | Close | Return |
|---------|-------|--------|
| 2019-06-27 | 1820.00 | -18.20% |
| 2022-03-08 | 3485.50 | -6.94% |
| 2022-03-09 | 3286.50 | -5.71% |
| 2021-10-21 | 2935.50 | -5.57% |
| 2021-10-27 | 2716.50 | -5.23% |

**Top 5 Return Terbaik:**

| Tanggal | Close | Return |
|---------|-------|--------|
| 2023-01-09 | 2443.25 | +6.91% |
| 2024-11-15 | 2566.75 | +5.69% |
| 2022-11-11 | 2475.00 | +5.25% |
| 2022-10-26 | 2352.50 | +5.56% |
| 2024-11-07 | 2654.50 | +4.15% |

---

### 2.6 Data Quality Summary

| Aspek | Status | Detail |
|-------|--------|--------|
| **Missing Values** | Moderate | 39 hari (1.31%) |
| **Repaired Data** | Baik | 5 hari (0.17%) |
| **Flat Candle** | Tinggi | 83.15% (settlement-style) |
| **Stale Behavior** | Tinggi | 16.11% zero return |
| **Outliers** | 1 Ekstrem | -18.20% (artefak) |

---

## 3️. Univariate Analysis

### 3.1 Statistik Deskriptif

| Statistik | Open | High | Low | Close | Volume |
|-----------|------|------|-----|-------|--------|
| Count | 2940 | 2940 | 2940 | 2940 | 2979 |
| Mean | 2458.32 | 2458.45 | 2458.18 | 2458.32 | 50.35 |
| Std | 389.74 | 389.92 | 389.90 | 389.99 | 50.00 |
| Min | 1456.00 | 1456.00 | 1452.00 | 1452.00 | 0 |
| 25% | 2150.00 | 2150.00 | 2150.00 | 2150.00 | 0 |
| 50% | 2350.00 | 2350.00 | 2350.00 | 2350.00 | 0 |
| 75% | 2750.00 | 2750.00 | 2750.00 | 2750.00 | 100 |
| Max | 3873.00 | 3873.00 | 3873.00 | 3873.00 | 2210 |

### 3.2 Distribusi Harga

- **Close Price:** Range 1452 - 3873, Mean 2458
- **Volume:** Mayoritas 0 (stale data), Max 2210

### 3.3 Time Series Plot

**Close Price Trend:**
- 2014-2016: Relatif stabil di 2000-2200
- 2017-2019: Turun ke 1650-1850
- 2020: Drop akibat pandemi ke ~1450
- 2021-2022: Rally kuat ke 3500-3800
- 2023-2026: Koreksi dan stabil di 2600-3200

---

## 4️. Bivariate & Multivariate Analysis

### 4.1 Correlation Matrix

| | Open | High | Low | Close | Volume |
|---|------|------|-----|-------|--------|
| **Open** | 1.000 | 1.000 | 1.000 | 1.000 | -0.052 |
| **High** | 1.000 | 1.000 | 1.000 | 1.000 | -0.051 |
| **Low** | 1.000 | 1.000 | 1.000 | 1.000 | -0.053 |
| **Close** | 1.000 | 1.000 | 1.000 | 1.000 | -0.052 |
| **Volume** | -0.052 | -0.051 | -0.053 | -0.052 | 1.000 |

**Interpretasi:**
- OHLC memiliki korelasi **sempurna (1.000)** → Konfirmasi settlement-style
- Volume memiliki korelasi **sangat lemah** dengan harga

### 4.2 Price vs Volume

- Tidak ada hubungan linear yang jelas antara harga dan volume
- Volume = 0 pada banyak hari dengan harga konstan

---

## 5️. Kesimpulan & Rekomendasi

### Kelebihan Dataset

1. **Data panjang** (~12 tahun) → Cocok untuk time series analysis
2. **Struktur rapi** → Tidak ada duplicate dates
3. **Missing values minimal** (<2%) → Mudah ditangani
4. **Tidak ada gap tanggal besar** → Data kontinu

### Kekurangan Dataset

1. **Settlement-style** (83% flat candle) → Bukan data trading intraday
2. **Stale behavior** (16% return = 0) → 90 hari harga konstan
3. **1 outlier ekstrem** (-18.2% pada 2019-06-27) → Artefak data
4. **Volume = 0** pada banyak hari → Tidak informatif