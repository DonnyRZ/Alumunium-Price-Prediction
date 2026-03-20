# Use Case Tim Marketing INALUM untuk Memanfaatkan Hasil Model

Tanggal: 2026-03-20  
Fokus dokumen: bagaimana tim marketing INALUM dapat memakai hasil model sebagai alat bantu keputusan sehari-hari.

---

## 1. Tujuan Dokumen

Dokumen ini menjelaskan secara end-to-end bagaimana tim marketing INALUM dapat memanfaatkan hasil model yang sudah dibangun.

Dokumen ini menjawab pertanyaan seperti:

- model ini dipakai untuk keputusan apa?
- kapan model berguna?
- siapa yang sebaiknya melihat hasil model?
- bagaimana cara membaca probability output?
- kapan harus `ignore`, `watch`, atau `act`?
- bagaimana membuktikan bahwa model benar-benar membantu kerja marketing?

Yang paling penting:

> Model ini bukan penentu harga otomatis. Model ini adalah alat bantu untuk memprioritaskan perhatian, mempercepat respon, dan mengurangi keputusan yang terlalu terlambat atau terlalu santai.

---

## 2. Ringkasan Inti

Hasil akhir proyek ini membentuk **signal engine** berbasis data 5 hari kerja. Output model bukan prediksi harga persis, melainkan probabilitas bahwa periode tertentu masuk kategori **actionable**.

`Actionable` berarti:

- pasar sedang cukup bergerak
- kondisi saat itu layak diperhatikan lebih serius
- tim marketing sebaiknya tidak memperlakukan hari tersebut sebagai hari biasa

Model final paling cocok dipakai untuk:

- menyaring hari-hari yang perlu perhatian lebih
- menentukan kapan follow-up perlu dipercepat
- membantu menentukan kapan quote perlu dibuat lebih hati-hati
- memberi peringatan dini saat pasar berpotensi bergerak
- membantu prioritas internal untuk customer besar atau inquiry besar

Model ini paling berguna ketika dipakai sebagai:

- **signal support**
- **early warning**
- **priority filter**

bukan sebagai:

- mesin penentu harga tunggal
- oracle yang selalu benar
- pengganti judgment manusia

---

## 3. Apa yang Sebenarnya Diprediksi Model

Ini penting supaya tidak salah pakai.

### 3.1 Yang diprediksi model

Model memprediksi **probability** bahwa sebuah periode 5 hari ke depan cukup penting untuk dikategorikan sebagai `actionable`.

### 3.2 Yang tidak diprediksi model

Model **tidak** memprediksi:

- harga akan naik
- harga akan turun
- harga final yang harus dikirim ke pelanggan
- nilai premium secara langsung

### 3.3 Artinya untuk marketing

Untuk tim marketing, ini berarti model berguna sebagai jawaban atas pertanyaan:

- apakah sekarang kondisi pasar cukup penting untuk diberi perhatian?
- apakah kita perlu bergerak lebih cepat?
- apakah perlu extra caution saat memberi quote?
- apakah perlu eskalasi internal?

Jadi model ini lebih dekat ke:

- **kapan harus waspada**

daripada:

- **harga pasti berapa**

---

## 4. Konsep Dasar yang Wajib Dipahami

Bagian ini menjelaskan istilah inti dengan bahasa yang sederhana.

### 4.1 Probability

`Probability` adalah angka antara 0 dan 1 yang menunjukkan seberapa yakin model bahwa suatu periode termasuk `actionable`.

Contoh:

- `0.20` = model cenderung bilang tidak actionable
- `0.55` = model agak condong ke actionable
- `0.80` = model cukup yakin actionable

### 4.2 Threshold

`Threshold` adalah batas yang dipakai untuk mengubah probability menjadi keputusan.

Contoh:

- kalau threshold `0.50`, maka probability `0.52` dianggap `action`
- kalau threshold `0.65`, maka probability `0.52` belum cukup

Jadi:

- **probability** = tingkat keyakinan model
- **threshold** = garis keputusan

### 4.3 Balanced accuracy

Balanced accuracy adalah metrik yang menilai performa model secara lebih seimbang untuk dua kelas:

- `action`
- `no-action`

Ini penting karena data sering tidak seimbang.

### 4.4 Precision dan Recall

- **Precision**: dari semua sinyal `action` yang keluar, berapa banyak yang benar
- **Recall**: dari semua kejadian `action` yang ada, berapa banyak yang berhasil tertangkap

Intuisi singkat:

- precision tinggi = sinyal lebih bersih
- recall tinggi = sinyal lebih lengkap

### 4.5 AUC

AUC menunjukkan kemampuan model membedakan kelas `action` dan `no-action`.

Kalau AUC mendekati 0.5, model hampir seperti tebakan biasa.
Kalau AUC makin tinggi, model makin baik membedakan sinyal.

---

## 5. Bentuk Output yang Paling Berguna untuk Marketing

Output model yang paling berguna bukan angka tunggal, tetapi **kategori keputusan**.

Dalam proyek ini, output dibaca menjadi tiga tingkat:

- **Ignore**
- **Watch**
- **Act**

### 5.1 Ignore

Probability di bawah `0.50`.

Artinya:

- sinyal terlalu lemah
- pasar belum cukup menarik perhatian ekstra
- tim bisa menjalankan proses normal

### 5.2 Watch

Probability sekitar `0.50` sampai `0.65`.

Artinya:

- ada sinyal
- tetapi belum cukup kuat untuk tindakan agresif
- tim sebaiknya memantau

### 5.3 Act

Probability di atas `0.65`.

Artinya:

- sinyal cukup kuat
- tim sebaiknya memberi perhatian ekstra
- perlu review internal yang lebih cepat atau lebih hati-hati

### 5.4 Kenapa tiga tingkat ini penting?

Karena marketing tidak selalu butuh jawaban hitam-putih.

Sering kali yang dibutuhkan adalah:

- jangan lakukan apa-apa
- pantau dulu
- segera tindak lanjuti

Tiga tingkat ini membuat model lebih mudah dipakai dalam praktik.

---

## 6. Bagaimana Tim Marketing Bisa Memakainya dalam Proses Kerja

Di bawah ini adalah alur penggunaan yang paling masuk akal.

### 6.1 Sebelum memberi quote

Gunakan model untuk menjawab:

- apakah sekarang waktu yang aman untuk mengirim quote?
- apakah lebih baik mengirim sekarang atau menunggu?
- apakah quote perlu dibuat lebih konservatif?

#### Contoh

Jika probability model tinggi:

- tim bisa mempercepat review
- memperpendek validity quote
- memberi catatan bahwa kondisi pasar sedang sensitif

Jika probability rendah:

- proses normal bisa dipakai
- tidak perlu reaksi berlebihan

### 6.2 Saat menerima inquiry baru

Gunakan model untuk menentukan prioritas.

#### Contoh

Jika inquiry datang saat model memberi sinyal `act`:

- inquiry besar bisa diprioritaskan lebih cepat
- follow-up bisa dipercepat
- tim bisa lebih siap dengan jawaban pasar

Jika model memberi sinyal `ignore`:

- inquiry diproses normal
- tidak perlu eskalasi khusus

### 6.3 Saat negosiasi harga

Gunakan model untuk melihat apakah timing negosiasi sedang sensitif.

#### Contoh

Jika model memberi `act`:

- jangan terlalu lama menunda closing
- cek apakah ada alasan kuat untuk lock lebih cepat
- siapkan argumen market note yang lebih hati-hati

Jika model memberi `watch`:

- negosiasi tetap berjalan
- tetapi tim sebaiknya memantau kondisi pasar selama proses

### 6.4 Saat memberi market update ke pelanggan besar

Gunakan model untuk memilih kapan perlu memberi update.

#### Contoh

Jika model memberi `act`:

- lebih layak mengirim update pasar singkat
- pelanggan besar bisa diberi konteks tambahan

Jika model memberi `ignore`:

- update tidak perlu dipaksa
- fokus ke account yang memang aktif

### 6.5 Saat melakukan account management

Gunakan model untuk memprioritaskan account mana yang perlu dihubungi lebih dulu.

#### Contoh

- account besar + model `act` = prioritas tinggi
- account kecil + model `ignore` = prioritas biasa

Ini membantu tim mengalokasikan waktu secara lebih efisien.

---

## 7. Contoh Skenario Nyata

### Skenario 1: Probability 0.42

Interpretasi:

- model belum melihat sinyal yang kuat
- pasar tidak terlihat cukup sensitif

Keputusan yang disarankan:

- `ignore`
- lanjutkan proses normal
- tidak perlu langkah tambahan

### Skenario 2: Probability 0.58

Interpretasi:

- ada sinyal menengah
- kondisi pasar perlu dipantau

Keputusan yang disarankan:

- `watch`
- cek update pasar
- pantau customer yang sedang aktif
- siapkan respons jika ada perubahan

### Skenario 3: Probability 0.72

Interpretasi:

- model cukup yakin kondisi pasar penting
- ada kemungkinan periode ini sensitif

Keputusan yang disarankan:

- `act`
- percepat review internal
- pertimbangkan mempercepat follow-up
- pertimbangkan validity quote yang lebih pendek
- komunikasikan risiko pasar dengan lebih hati-hati

### Penting

`Act` **bukan** berarti:

- langsung ubah harga
- langsung ambil keputusan final

`Act` berarti:

- beri perhatian lebih
- jangan biarkan sinyal lewat begitu saja
- gunakan konteks bisnis tambahan sebelum keputusan akhir

---

## 8. Bagaimana Model Ini Membantu Membuat Keputusan yang Lebih Baik

### 8.1 Mengurangi keputusan yang terlalu lambat

Tanpa model, tim bisa saja terlambat menyadari bahwa pasar sedang bergerak.

Model membantu memberi warning lebih awal.

### 8.2 Mengurangi quote yang cepat basi

Kalau pasar sedang sensitif, quote yang terlalu lama menunggu bisa cepat tidak relevan.

Model membantu memilih kapan quote perlu dipercepat atau dipersingkat masa berlakunya.

### 8.3 Mengurangi overreaction

Di sisi lain, model juga mencegah tim bereaksi berlebihan pada hari yang sebenarnya normal.

Kalau sinyal rendah, tim tidak perlu panic mode.

### 8.4 Membantu prioritas kerja

Tim marketing biasanya punya banyak hal sekaligus:

- inquiry masuk
- pelanggan existing
- follow-up
- negosiasi
- update internal

Model membantu menjawab:

- mana yang harus diprioritaskan dulu?

---

## 9. Bagaimana Cara Memakai Model dalam Operasi Harian

Berikut workflow yang sederhana dan realistis.

### 9.1 Langkah 1: Generate score

Model dijalankan pada data terbaru.

Output utama:

- probability `action`

### 9.2 Langkah 2: Terjemahkan ke kategori

Gunakan aturan:

- `< 0.50` = `ignore`
- `0.50 - 0.65` = `watch`
- `> 0.65` = `act`

### 9.3 Langkah 3: Cocokkan dengan konteks bisnis

Sebelum mengambil tindakan, lihat juga:

- urgensi pelanggan
- stok atau ketersediaan
- kondisi premium
- FX
- berita pasar
- approval internal

### 9.4 Langkah 4: Ambil tindakan yang proporsional

Contoh tindakan:

- mempercepat follow-up
- memperpendek validitas quote
- mengirim market note
- menyiapkan eskalasi internal
- menunda keputusan jika sinyal belum kuat

### 9.5 Langkah 5: Catat hasilnya

Kalau model dipakai, hasilnya perlu dicatat:

- apakah model bilang `act` atau `watch`
- apa keputusan yang diambil
- apakah hasilnya berguna

Ini penting untuk evaluasi manfaat nyata.

---

## 10. Bukti Bahwa Model Ini Berguna

Model ini tidak sempurna, tetapi sudah memberi perbaikan nyata dibanding baseline sebelumnya.

### 10.1 Ringkasan performa final

Pada baseline final engineered:

- balanced accuracy sekitar `0.5423`
- AUC sekitar `0.5439`

Ini berarti model sudah sedikit lebih baik dari baseline lama dan bisa dipakai sebagai support signal.

### 10.2 Apa arti angka ini secara bisnis?

Artinya model:

- belum cukup kuat untuk keputusan otomatis
- tetapi cukup berguna untuk triase / penyaringan

### 10.3 Kenapa ini tetap bernilai?

Karena banyak proses marketing tidak membutuhkan prediksi sempurna.

Yang dibutuhkan adalah:

- tahu kapan perlu waspada
- tahu kapan harus segera bergerak
- tahu kapan proses normal sudah cukup

Model ini cocok untuk kebutuhan itu.

---

## 11. Kapan Model Harus Dipakai, dan Kapan Tidak

### 11.1 Gunakan model ketika

- tim ingin memprioritaskan inquiry
- tim sedang menyusun quote
- tim perlu memutuskan apakah perlu follow-up cepat
- tim ingin memberi market note
- tim ingin tahu kapan harus lebih hati-hati

### 11.2 Jangan gunakan model ketika

- ingin menentukan harga final otomatis
- ingin menggantikan judgement komersial
- data input terlalu tidak lengkap
- ada kejadian pasar ekstrem yang belum tercermin di data historis

### 11.3 Prinsip utamanya

Model dipakai untuk:

- mengarahkan perhatian

bukan untuk:

- mengambil alih keputusan bisnis sepenuhnya

---

## 12. Risiko Jika Dipakai Salah

Kalau model dipakai dengan salah, ada beberapa risiko.

### 12.1 Terlalu percaya pada angka

Tim bisa menganggap probability tinggi berarti kepastian.

Padahal model hanya memberi probabilitas, bukan kepastian.

### 12.2 Mengabaikan konteks bisnis

Model bisa bilang `act`, tetapi customer mungkin tidak sensitif waktu, atau kontraknya memang kaku.

Karena itu model harus selalu dipadukan dengan konteks bisnis.

### 12.3 Overreaction

Kalau setiap sinyal kecil dianggap alarm besar, tim bisa terlalu sering bereaksi dan kehilangan fokus.

### 12.4 Underreaction

Kalau model diabaikan sepenuhnya, potensi manfaatnya hilang.

Jadi yang dibutuhkan adalah keseimbangan.

---

## 13. Apa yang Perlu Diuji Kalau Ingin Membuktikan Manfaat Nyata

Untuk menunjukkan bahwa model ini benar-benar berguna, tim marketing bisa melihat beberapa indikator berikut:

### 13.1 Quote turnaround time

Apakah quote lebih cepat diproses saat model bilang `act`?

### 13.2 Quote freshness

Apakah quote yang dikeluarkan lebih jarang basi karena tim lebih waspada saat pasar sensitif?

### 13.3 Follow-up effectiveness

Apakah inquiry yang diprioritaskan oleh model lebih cepat menghasilkan respon atau closing?

### 13.4 Decision quality

Apakah keputusan yang dibantu model terasa lebih terstruktur dan kurang reaktif?

### 13.5 Alarm usefulness

Apakah sinyal `act` benar-benar membantu tim melihat momen penting?

Kalau metrik operasional ini membaik, maka model punya nilai bisnis nyata.

---

## 14. Rekomendasi Implementasi

Jika model ini mau dipakai di tim marketing, saya sarankan format pemakaian seperti ini:

### 14.1 Tampilan ringkas

Tampilkan hanya:

- probability
- label `ignore / watch / act`
- catatan singkat

### 14.2 Frekuensi pemakaian

Karena data dibangun sebagai blok 5 hari, model paling cocok dipakai sebagai:

- review berkala
- market check rutin
- bukan sebagai sinyal intraday

### 14.3 Kombinasi dengan sumber lain

Model paling berguna jika digabung dengan:

- status inquiry
- stok
- premium
- FX
- berita pasar

### 14.4 Escalation rule

Kalau model memberi `act`, maka tim bisa punya aturan:

- review cepat oleh PIC marketing
- jika nilainya besar, eskalasi ke pricing / management

---

## 15. Glossary Singkat

### Actionable
Kondisi pasar yang layak diperhatikan lebih serius.

### Probability
Angka 0 sampai 1 yang menunjukkan keyakinan model.

### Threshold
Batas untuk mengubah probability menjadi keputusan.

### Ignore
Sinyal terlalu lemah untuk memicu tindakan khusus.

### Watch
Sinyal layak dipantau.

### Act
Sinyal cukup kuat untuk mendorong perhatian tambahan atau eskalasi.

### Signal support
Model yang membantu keputusan, tetapi bukan penentu tunggal.

### Feature engineering
Membuat fitur baru dari data yang sudah ada agar model lebih mudah belajar.

### Exogenous variable
Variabel tambahan dari luar target utama yang membantu memberi konteks.

### Balanced accuracy
Ukuran performa yang seimbang untuk dua kelas.

### AUC
Ukuran kemampuan model membedakan dua kelas.

---

## 16. Kesimpulan Akhir

Model ini layak dipakai oleh tim marketing INALUM **bukan karena ia sempurna**, tetapi karena ia sudah cukup baik untuk membantu:

- memilih prioritas
- mengurangi keterlambatan respon
- menghindari quote yang terlalu santai saat pasar sensitif
- memberi alarm awal saat kondisi pasar mulai penting

Kesimpulan praktisnya:

> Jika probability rendah, jalankan proses normal.  
> Jika probability sedang, pantau dengan hati-hati.  
> Jika probability tinggi, perlakukan sebagai sinyal penting dan lakukan review lebih cepat.

Jadi manfaat terbesarnya adalah membuat tim marketing lebih **terarah, lebih waspada, dan lebih terstruktur** dalam mengambil keputusan.
