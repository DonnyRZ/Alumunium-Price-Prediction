# Berikut adalah rancangan praktikal halaman *dashboard* dan isi *side panel* yang ideal untuk memenuhi kebutuhan INALUM:

## 📱 Menu Navigasi (*Side Panel*)

Menu ini akan berada di sisi kiri layar dan berisi navigasi utama:

* **Executive Summary** (Tampilan utama, fokus pada ringkasan GenAI).


* **Predictive Analytics & Benchmarking** (Ruang kerja utama untuk eksplorasi model *time series* dan XGBoost).


* **Market Sentiment Analysis** (Fokus pada pemrosesan NLP untuk berita dan tren pasar).


* 
**Data Management** (Untuk mengunggah *flat file* pendukung dan mengecek status *API Ingestion*).


* **Settings & User Management** (Pengaturan RBAC dan tata letak *widget*).



---

## 📄 Halaman 1: Executive Dashboard (Fokus untuk Manajemen/Direksi)

Halaman ini dirancang agar pimpinan bisa langsung memahami kondisi pasar tanpa harus melihat metrik statistik yang terlalu rumit.

* **Highlight Angka Utama (KPI Cards):** Menampilkan harga aktual hari ini, prediksi harga rata-rata minggu/bulan depan, dan status sentimen pasar global secara keseluruhan (Positif/Netral/Negatif).


* **Grafik Prediksi Utama:** Visualisasi grafik historis harga aluminium yang bersambung dengan garis proyeksi (hasil rata-rata dari model terbaik atau *ensemble*).


* **Automated Executive Summary (GenAI):** Sebuah panel teks berisi narasi analitis otomatis yang merangkum kondisi harga, metrik komparasi model, dan skor sentimen menjadi bacaan siap saji.


* **Strategic Recommendations:** Poin-poin rekomendasi tindakan berbasis data yang dihasilkan oleh *Artificial Intelligence* (misalnya: saran waktu lindung nilai/ *hedging* atau penyesuaian target penjualan).



## 📄 Halaman 2: Predictive Analytics & Benchmarking (Fokus untuk Data Scientist & Analis)

Halaman ini sangat interaktif dan teknis, dirancang sebagai *sandbox* untuk melakukan *cross-validation* dan membandingkan performa model algoritma.

* **Panel Filter & Konfigurasi:** Pengguna dapat mengaktifkan atau menonaktifkan model yang ingin ditampilkan (Prophet, SARIMAX, XGBoost).


* **Input Variabel Eksogen (Khusus SARIMAX/XGBoost):** *Slider* atau *input field* untuk memasukkan simulasi variabel eksternal seperti kurs USD/IDR, harga batu bara, minyak dunia, dan inflasi.


* **Grafik Benchmarking Interaktif:** Grafik garis yang menampilkan overlay antara harga historis aktual (*ground truth*) dengan garis prediksi dari masing-masing model yang dipilih.


* **Tabel Metrik Akurasi:** Tabel di bawah grafik yang menunjukkan komparasi tingkat akurasi (misal: RMSE, MAE, MAPE) dari Prophet, SARIMAX, dan XGBoost.



## 📄 Halaman 3: Market Sentiment Analysis (Fokus NLP & Teks)

Halaman ini didedikasikan untuk mengelola *pipeline* NLP dan menganalisis variabel kualitatif.

* **Live News Feed:** Daftar berita industri aluminium dan tren pasar global yang ditarik secara *real-time* via API.


* **Sentiment Scoring:** Setiap berita yang masuk akan dilabeli dengan sentimen (Positif, Negatif, Netral) hasil ekstraksi model NLP.


* **Filter Preferensi Berita:** *Dropdown* atau *tag* bagi pengguna untuk mengatur preferensi topik atau kata kunci tertentu (misal: "London Metal Exchange", "Smelter", "Coal").


* **Grafik Distribusi Sentimen:** Visualisasi (seperti *pie chart* atau *bar chart* harian) yang menunjukkan rasio sentimen pasar dalam rentang waktu tertentu, yang nantinya menjadi variabel pendukung dalam pemodelan prediksi.