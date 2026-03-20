# Laporan Akhir Proyek INALUM

Tanggal: 2026-03-20  
Fokus dokumen: bagaimana tim marketing INALUM dapat memanfaatkan hasil model ini secara praktis, aman, dan mudah dipahami.

---

## 1. Ringkasan Singkat

Hasil akhir proyek ini bukan model yang mencoba menebak harga aluminium secara tepat dari hari ke hari. Setelah beberapa iterasi, arah kerja berubah menjadi **signal engine**, yaitu model yang memberi sinyal apakah kondisi pasar sedang cukup penting untuk diperhatikan atau tidak.

Model final yang dipakai adalah:

- dataset: [`five_day_signal_expanded_v3.csv`](/home/sdo/Project/Machine%20Learning/INALUM/data/processed/five_day_signal_expanded_v3.csv)
- target: **binary actionable signal**
- model: **Logistic Regression**
- feature mode final: **feature engineered**
- notebook final: [`Signal-Final.ipynb`](/home/sdo/Project/Machine%20Learning/INALUM/notebooks/Signal-Final.ipynb)

Intinya:

- model ini **bukan penentu harga final**
- model ini **bukan ramalan pasti naik atau turun**
- model ini adalah **alat bantu untuk memutuskan kapan tim marketing perlu lebih waspada, lebih aktif, atau cukup tenang**

Kalau diringkas satu kalimat:

> Model ini membantu tim marketing INALUM memilah hari-hari yang layak diperhatikan lebih serius dari hari-hari yang bisa diperlakukan secara normal.

---

## 2. Kenapa Proyek Ini Berubah Arah

Awalnya proyek ini diarahkan ke forecasting harga aluminium harian dengan Prophet. Masalahnya, harga harian aluminium sangat dekat dengan perilaku **naive forecast**.

### 2.1 Apa itu naive forecast?

Naive forecast adalah baseline paling sederhana:

- harga besok dianggap sama dengan harga terakhir
- atau perubahan berikutnya dianggap nol

Untuk banyak data harga, baseline sederhana ini memang sangat kuat. Kalau model yang lebih rumit hanya sedikit lebih baik atau bahkan kalah, maka model tersebut belum memberi nilai tambah yang jelas.

### 2.2 Kenapa forecasting harga harian kurang cocok?

Karena data harian terlalu noisy. Pergerakan harian sering terlalu kecil, terlalu acak, atau terlalu dipengaruhi banyak faktor yang tidak kita lihat langsung.

Akibatnya:

- model jadi menempel baseline sederhana
- hasilnya sulit dibedakan dari tebakan “diam saja”
- perbedaan kecil di metrik belum tentu berarti berguna untuk bisnis

### 2.3 Apa keputusan perbaikannya?

Kita mengubah tujuan dari:

- **memprediksi harga mentah**

menjadi:

- **mendeteksi apakah sebuah periode pasar cukup penting untuk diberi perhatian**

Itulah alasan kita pindah ke:

- agregasi 5 hari kerja
- target actionable signal
- model tabular yang lebih cocok untuk sinyal

---

## 3. Apa yang Diprediksi Model Ini

Ini bagian paling penting agar tidak salah paham.

### 3.1 Model ini memprediksi apa?

Model ini memprediksi **probabilitas** bahwa kondisi pasar masuk kategori `actionable`.

### 3.2 Apa arti `actionable`?

`Actionable` berarti pergerakan 5 hari berikutnya cukup besar sehingga layak diperhatikan.

Jadi model ini **tidak** mengatakan:

- “harga pasti naik”
- “harga pasti turun”

Melainkan:

- “periode ini kemungkinan akan bergerak cukup besar, jadi tim perlu lebih waspada”

### 3.3 Kenapa ini penting untuk marketing?

Karena tim marketing tidak selalu butuh ramalan harga yang presisi. Yang sering lebih berguna adalah jawaban seperti:

- apakah perlu mempercepat follow-up?
- apakah quote perlu dibuat lebih hati-hati?
- apakah pasar sedang tenang atau sedang bergerak?
- apakah perlu warning internal agar tim tidak terlalu santai?

Model ini memberi sinyal untuk hal-hal seperti itu.

---

## 4. Penjelasan Konsep Dasar

Bagian ini sangat penting untuk membaca output notebook dan laporan ini.

### 4.1 Probability

`Probability` adalah angka antara 0 dan 1 yang menunjukkan seberapa yakin model bahwa suatu baris termasuk kelas `action`.

Contoh:

- `0.20` = model cenderung menganggap ini **no-action**
- `0.65` = model cukup yakin ini **action**
- `0.82` = model sangat yakin ini **action**

### 4.2 Threshold

`Threshold` adalah batas keputusan.

Contoh paling sederhana:

- jika probability `>= 0.50`, model dianggap berkata `action`
- jika probability `< 0.50`, model dianggap berkata `no-action`

Jadi:

- **probability** = tingkat keyakinan
- **threshold** = garis keputusan

### 4.3 Kenapa threshold penting?

Karena model tidak hanya memberi angka probabilitas. Tim bisnis perlu keputusan.

Threshold menentukan apakah model akan:

- lebih sering memberi sinyal
- atau lebih hati-hati memberi sinyal

Kalau threshold rendah:

- sinyal lebih banyak
- coverage tinggi
- false alarm juga lebih banyak

Kalau threshold tinggi:

- sinyal lebih sedikit
- lebih konservatif
- tapi biasanya lebih bersih

### 4.4 Metrik evaluasi

Beberapa istilah penting yang muncul di notebook:

- **Accuracy**: seberapa sering model benar secara total
- **Balanced accuracy**: accuracy yang lebih adil terhadap dua kelas
- **Precision**: dari semua sinyal `action` yang keluar, berapa banyak yang benar
- **Recall**: dari semua kejadian `action` yang memang ada, berapa banyak yang tertangkap
- **F1-score**: ringkasan precision dan recall
- **AUC**: kemampuan model membedakan `action` vs `no-action`

Intuisi sederhananya:

- precision tinggi = sinyal lebih bersih
- recall tinggi = sinyal lebih lengkap
- balanced accuracy tinggi = model lebih seimbang terhadap dua kelas

---

## 5. Data yang Dipakai

### 5.1 Bentuk data final

Data final bukan lagi harga harian mentah. Data diubah menjadi **blok 5 trading days**.

Satu baris mewakili satu blok 5 hari kerja.

### 5.2 Kenapa 5 hari kerja?

Karena:

- noise harian berkurang
- pola pasar lebih stabil
- lebih mudah dipakai untuk sinyal bisnis

### 5.3 Kenapa ada exogenous variables?

`Exogenous variables` adalah variabel tambahan dari luar target utama.

Contoh:

- harga Brent
- FX
- stok gudang
- SHFE
- open interest

Tujuannya adalah memberi konteks pasar yang tidak terlihat hanya dari target utama.

### 5.4 Kenapa feature engineering dipakai?

`Feature engineering` berarti membuat fitur baru dari data yang sudah ada agar model lebih mudah belajar.

Contoh fitur turunan:

- momentum
- rolling mean
- rolling std
- spread antar pasar
- freshness / staleness

Fitur-fitur ini membantu model melihat struktur pasar, bukan cuma angka mentah.

---

## 6. Apa yang Dilakukan Model Secara Praktis

Secara sederhana, alurnya seperti ini:

1. data 5 hari dibentuk
2. fitur tambahan dibuat
3. Logistic Regression menghitung probabilitas `action`
4. probabilitas itu dibandingkan dengan threshold
5. hasilnya menjadi keputusan `ignore`, `watch`, atau `act`

### 6.1 Tiga kelas operasional

Untuk memudahkan penggunaan bisnis, probabilitas model dibaca seperti ini:

- **Ignore**: probabilitas di bawah `0.50`
- **Watch**: probabilitas antara `0.50` dan `0.65`
- **Act**: probabilitas di atas `0.65`

Ini bukan hukum mutlak. Ini adalah panduan operasional yang paling mudah dipakai.

---

## 7. Hasil Model dalam Bahasa yang Mudah

Model final engineered menunjukkan perbaikan dibanding baseline sebelumnya.

### 7.1 Perbandingan baseline dan final

Secara rata-rata:

- baseline v2: balanced accuracy sekitar `0.5241`, AUC sekitar `0.5333`
- final engineered v3: balanced accuracy sekitar `0.5423`, AUC sekitar `0.5439`

Artinya:

- model final **lebih baik**
- tetapi peningkatannya **masih tipis**

### 7.2 Apa arti peningkatan tipis ini?

Artinya model memang menangkap sinyal tambahan dari fitur engineered, tetapi sinyal pasar tetap tidak mudah dipisahkan.

Jadi kita bisa bilang:

- model ini **berguna**
- tetapi belum **sangat kuat**

Karena itu model ini paling tepat dipakai sebagai:

- **signal support**

bukan:

- penentu tunggal
- mesin otomatis penetap keputusan

---

## 8. Bagaimana Tim Marketing INALUM Bisa Memakainya

Ini inti paling penting dari laporan ini.

### 8.1 Sebagai alat prioritas kerja

Model membantu tim marketing menentukan hari mana yang perlu perhatian lebih.

Contoh:

- bila probabilitas rendah, tim bisa memakai proses normal
- bila probabilitas sedang, tim mulai monitor pasar dan customer dengan lebih hati-hati
- bila probabilitas tinggi, tim bisa mempercepat koordinasi internal

### 8.2 Sebagai alarm awal

Model bisa dipakai sebagai alarm awal bahwa pasar sedang bergerak.

Artinya bukan:

- “harga pasti naik”

Melainkan:

- “periode ini sebaiknya tidak dianggap biasa-biasa saja”

### 8.3 Sebagai dukungan timing komunikasi

Dalam pekerjaan marketing, timing sering sama pentingnya dengan isi informasi.

Model ini bisa membantu menjawab:

- apakah sekarang waktu yang baik untuk follow-up?
- apakah perlu memberi catatan hati-hati pada pelanggan?
- apakah quote perlu dipercepat?

### 8.4 Sebagai pengatur tingkat kewaspadaan

Saat model memberi sinyal kuat, tim tidak harus langsung mengubah harga. Tetapi tim bisa:

- mengecek ulang kondisi pasar
- memeriksa stok dan premium
- melihat FX dan energi
- menilai apakah perlu eskalasi internal

Jadi model ini membantu mengubah keputusan dari:

- “reaktif”

menjadi:

- “lebih terstruktur”

---

## 9. Contoh Penggunaan Nyata

### Contoh 1: Probabilitas 0.42

Interpretasi:

- sinyal masih lemah
- pasar tidak menunjukkan kondisi yang cukup penting

Tindakan marketing:

- **ignore**
- jalankan proses normal
- tidak perlu eskalasi khusus

### Contoh 2: Probabilitas 0.58

Interpretasi:

- ada sinyal
- tetapi belum kuat

Tindakan marketing:

- **watch**
- pantau market context
- cek apakah customer perlu respon lebih cepat
- jangan terburu-buru mengambil keputusan final

### Contoh 3: Probabilitas 0.72

Interpretasi:

- model cukup yakin bahwa periode ini penting
- risiko pergerakan besar lebih tinggi

Tindakan marketing:

- **act**
- percepat review internal
- koordinasikan dengan pricing / sales / management jika diperlukan
- gunakan evaluasi pasar yang lebih hati-hati

### Catatan penting

`Act` bukan berarti:

- otomatis naikkan harga
- otomatis ubah quote

`Act` berarti:

- jangan abaikan sinyal
- beri perhatian lebih
- tambah konteks bisnis sebelum memutuskan

---

## 10. Apa Arti Visual Utama di Notebook Final

Notebook final dibuat supaya mudah dipahami secara visual.

### 10.1 Baseline vs Final Model

Visual ini menunjukkan bahwa model engineered final lebih baik dari baseline.

Maknanya:

- feature engineering memang membantu
- tetapi improvement masih moderat

### 10.2 Threshold trade-off

Visual ini menunjukkan hubungan antara threshold dan kualitas sinyal.

Maknanya:

- threshold makin tinggi -> sinyal makin jarang tapi makin bersih
- threshold makin rendah -> sinyal makin banyak tapi lebih berisik

### 10.3 Actual vs Predicted

Visual ini memperlihatkan apakah model benar-benar menandai hari-hari penting.

Maknanya:

- kalau titik prediksi banyak menempel ke area actual action, model cukup berguna
- kalau tidak, model hanya menebak sembarangan

### 10.4 ROC dan Precision-Recall

Kedua grafik ini menunjukkan seberapa baik model memisahkan action vs no-action.

Maknanya:

- ROC membantu melihat kemampuan membedakan kelas
- Precision-Recall membantu melihat apakah sinyal positif cukup bersih

### 10.5 Koefisien fitur

Karena model final adalah Logistic Regression, kita bisa melihat fitur mana yang mendorong `action` dan mana yang mendorong `no-action`.

Ini membantu marketing memahami sinyal apa yang paling sering dipakai model.

---

## 11. Apa yang Paling Berarti untuk Tim Marketing

Kalau harus diringkas secara bisnis, ada tiga manfaat utama:

### 11.1 Prioritization tool

Model membantu memprioritaskan hari dan kasus yang paling layak diperhatikan.

### 11.2 Early warning system

Model memberi peringatan awal ketika kondisi pasar mungkin akan lebih aktif.

### 11.3 Decision support, not decision replacement

Model membantu keputusan, tetapi tidak menggantikan judgment manusia.

Ini penting karena keputusan marketing nyata selalu dipengaruhi banyak hal:

- customer urgency
- kontrak
- inventory
- approval internal
- kondisi kompetitif

Model hanya salah satu input.

---

## 12. Batasan Model

Laporan yang baik harus jujur soal keterbatasan.

### 12.1 Model tidak memprediksi harga pasti

Model ini tidak bilang harga naik atau turun secara spesifik.

### 12.2 Model bukan keputusan final

Model tidak boleh dipakai sendirian untuk memutuskan harga final.

### 12.3 Model sensitif terhadap konteks pasar

Kalau kondisi pasar berubah besar, performa model juga bisa berubah.

### 12.4 Model paling cocok sebagai support signal

Nilai utamanya ada pada:

- screening
- prioritas
- kewaspadaan
- timing

bukan pada presisi harga absolut.

---

## 13. Rekomendasi Cara Pakai di Lapangan

Berikut cara pakai yang paling aman dan paling praktis:

### 13.1 Jika probabilitas < 0.50

- perlakukan sebagai `ignore`
- lanjutkan proses normal
- tidak perlu eskalasi khusus

### 13.2 Jika probabilitas 0.50 sampai 0.65

- perlakukan sebagai `watch`
- cek market context
- pantau customer / quote / timing

### 13.3 Jika probabilitas > 0.65

- perlakukan sebagai `act`
- beri perhatian lebih
- lakukan review tambahan sebelum keputusan penting

### 13.4 Cara terbaik memakainya

Model ini paling baik dipakai bersama:

- harga pasar terbaru
- informasi pelanggan
- inventory
- FX
- energi
- judgement tim marketing

Kalau model dan konteks bisnis sama-sama mendukung, barulah sinyalnya paling berguna.

---

## 14. Kesimpulan Akhir

Kesimpulan utama proyek ini adalah:

1. Forecast harga harian murni terlalu dekat dengan naive forecast.
2. Prophet pada target level tidak cukup memecahkan masalah itu.
3. Mengubah problem menjadi 5-day signal jauh lebih masuk akal.
4. Feature engineering membuat hasil model lebih baik.
5. Logistic Regression final memberi sinyal yang paling sehat dan stabil untuk dipakai sebagai support tool.
6. Untuk tim marketing INALUM, nilai terbesar model ini adalah membantu prioritas, kewaspadaan, dan timing.

Jadi posisi akhir model ini adalah:

- **bukan mesin prediksi harga final**
- **tetapi signal support yang berguna untuk kerja marketing**

---

## 15. Glosarium Singkat

### Actionable signal
Sinyal bahwa pergerakan pasar cukup besar sehingga layak diperhatikan.

### No-action
Sinyal bahwa kondisi pasar belum cukup kuat untuk memicu perhatian khusus.

### Probability
Angka 0 sampai 1 yang menunjukkan tingkat keyakinan model.

### Threshold
Batas probabilitas untuk mengubah angka model menjadi keputusan.

### Logistic Regression
Model klasifikasi sederhana yang memetakan fitur ke probabilitas kelas.

### Feature engineering
Proses membuat fitur baru dari data lama agar model lebih mudah belajar.

### Balanced accuracy
Ukuran performa yang lebih adil untuk dua kelas.

### Precision
Seberapa bersih sinyal positif yang keluar.

### Recall
Seberapa banyak kejadian positif yang berhasil ditangkap.

### AUC
Kemampuan model membedakan kelas positif dan negatif.

### Signal support
Model yang membantu keputusan, tetapi bukan satu-satunya penentu keputusan.

---

## 16. Penutup

Jika tim marketing INALUM ingin menggunakan hasil proyek ini, cara berpikir yang paling tepat adalah:

> gunakan model ini untuk memutuskan kapan perlu lebih waspada, bukan untuk memutuskan harga secara otomatis.

Itulah nilai praktis terbesar dari proyek ini.
