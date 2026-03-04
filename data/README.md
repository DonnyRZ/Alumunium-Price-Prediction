# Panduan Dataset Bersih Aluminium


## Ringkas Tujuan
File ini adalah **dataset bersih berbasis event** (hanya hari saat harga berubah) untuk harga Aluminium futures (`ALI=F`). Tujuannya membuat data lebih konsisten dan layak untuk modeling.

## Masalah di Data Raw
Data raw mengandung pola yang tidak sehat untuk modeling harian:
- Harga sering **tidak berubah** selama periode panjang (streak).
- Banyak hari dengan **volume = 0**.
- Ada **lonjakan ekstrem** yang kemungkinan artefak data.

## Solusi yang Diterapkan
- **Hapus hari harga tidak berubah** (stale).
- **Hapus baris Close yang kosong** (missing).
- **Hapus outlier yang dicurigai artefak** (volume=0 atau tepat setelah streak panjang).
- **Hitung ulang return** setelah pembersihan.

## Ringkasan Masalah & Solusi (Tabel)

| Aspek | Masalah di Raw | Solusi di Clean |
|---|---|---|
| Harga konstan | Banyak streak harga sama | Hanya simpan hari harga berubah |
| Volume | Banyak hari volume = 0 | Outlier saat volume=0 dibuang |
| Missing | Ada Close kosong | Baris missing dihapus |
| Outlier ekstrem | Terlihat tidak wajar | Outlier suspect dibuang |
| Return | Bias karena data kotor | Return dihitung ulang |

## Fokus Analisis
Kolom yang biasanya dipakai:
- `Date`
- `Close`
- `Return`

Kolom lain adalah **flag kualitas**, bisa dipakai jika ingin analisis kualitas data lebih lanjut.

## Cara Membaca Baris
Setiap baris = **event perubahan harga**, bukan semua hari kalender.  
Ini ideal untuk:
- Analisis pola pergerakan harga
- Prediksi return

Jika butuh seri harian penuh, gunakan data raw atau versi calendar‑based.

## Catatan
Dataset ini bergantung pada sumber publik dan bisa berubah jika data sumber di‑update.  
Jika script dijalankan ulang, hasil bisa sedikit berbeda.
