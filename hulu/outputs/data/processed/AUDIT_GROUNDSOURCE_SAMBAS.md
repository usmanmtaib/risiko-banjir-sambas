# Audit kualitas awal Groundsource — Kabupaten Sambas

## Kesimpulan

Data cukup bersih untuk tahap eksplorasi dan pemetaan awal. Tidak ditemukan nilai kosong, tanggal tidak valid, UUID duplikat, luas tidak positif, atau geometri tanpa data. Data terdiri dari 709 Polygon dan 44 MultiPolygon dengan 753 UUID unik.

## Temuan utama

- Periode kejadian yang tersedia: 2012–2025. Tidak ada kejadian 2010–2011 setelah penyaringan.
- Rentang centroid: longitude 108.9486–109.6180 dan latitude 1.0075–1.9509.
- Median luas area kejadian: 39.98 km².
- Kuartil 75% luas area: 174.56 km².
- Kuartil 90% luas area: 526.55 km².
- Tahun dengan kejadian terbanyak: 2025 (289), 2023 (146), dan 2024 (93).
- Sel 0,1° dengan kejadian terbanyak berada sekitar longitude 109.1–109.3 dan latitude 1.1–1.5.

## Catatan metodologis

Peta menampilkan centroid dari geometri kejadian, bukan batas genangan aktual. Konsentrasi titik dapat mencerminkan kejadian nyata maupun bias cakupan berita. Sebelum Random Forest, perlu dilakukan pemeriksaan spasial lanjutan, deduplikasi lintas berita, serta validasi manual terhadap sampel kejadian.

## Berkas keluaran

- `audit_groundsource_sambas.json`
- `ringkasan_tahunan_groundsource_sambas.csv`
- `sebaran_grid_0p1deg_groundsource_sambas.csv`
- `groundsource-sambas-map.html` pada visualisasi percakapan
