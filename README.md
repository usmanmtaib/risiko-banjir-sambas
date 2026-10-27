# Paket reproduksi: pemetaan risiko laporan banjir Kabupaten Sambas

Penelitian DIPA Politeknik Negeri Sambas 2026, *Pengembangan Sistem Cerdas Pemetaan Kerawanan Banjir Wilayah Perbatasan Menggunakan Pipeline Groundsource dan Machine Learning*.

Paket ini memuat semua skrip analisis, data turunan, dan keluaran yang dipakai dalam laporan akhir. Dengan paket ini, setiap angka, tabel, dan gambar di laporan dapat dihitung ulang. Data mentah berukuran besar tidak ikut dikemas, tetapi bisa diunduh ulang dengan skrip yang tersedia (lihat [Data yang tidak dikemas](#data-yang-tidak-dikemas)).

> **Cara membaca hasil.** Label kejadian berasal dari berita (Groundsource), sehingga dipengaruhi bias pelaporan. Model kecamatan-hari hanya mampu **merangking hari**, dan pada data ini **tidak lebih baik daripada indikator hujan 7 hari**. Dashboard adalah prototipe riset, bukan sistem peringatan dini.

## Mulai cepat

Dibuat dan diuji pada Windows 11 dengan Python 3.12.10.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONUTF8=1           # agar konsol Windows dapat mencetak tanda seperti − dan ≥

python panel_model.py      # panel kecamatan-hari dan validasi lintas-tahun
python panel_final.py      # model final, kalibrasi, validasi DIBI, SHAP
python article_stats.py    # interval kepercayaan bootstrap
streamlit run dashboard\app.py
```

Ketiga skrip di atas hanya memakai data yang sudah ada di paket, tanpa koneksi internet. Semua jalur ditulis relatif terhadap folder paket, jadi paket bisa diletakkan di folder mana pun.

## Isi paket

| Folder / berkas | Isi |
|---|---|
| `*.py` (akar) | Skrip analisis utama (urutan di bawah) |
| `hulu/` | Tahap awal: unduh data pendukung, saring Groundsource ke Sambas, audit, fitur grid 1 km. `hulu/DATA_SOURCES.md` mencatat sumber data |
| `hulu/outputs/data/processed/` | Data olahan tahap awal: 753 kejadian Groundsource Sambas, grid fitur 1 km, batas kabupaten BIG, DEM/lereng/WorldCover terpotong, ringkasan tahunan |
| `kec/` | Batas 19 kecamatan (BIG), irisan kejadian–kecamatan, hujan harian NASA POWER per kecamatan |
| `data/` | Fitur turunan: hidrologi (HAND/TWI), OSM, CHIRPS, jarak pantai, statistik genangan Sentinel-1 per sel dan per adegan (termasuk 6 konfigurasi ambang) |
| `dibi/` | Catatan banjir Sambas dari dasbor publik DIBI BNPB (diakses 21 Okt 2026) |
| `bpbd/` | Tabel kejadian kompilasi yang disampaikan peneliti. **Asal sebagian baris belum terverifikasi** (Lampiran B laporan) |
| `outputs*/` | Keluaran setiap tahap (lihat tabel tahap) |
| `dashboard/` | Aplikasi Streamlit + Folium beserta datanya, berdiri sendiri |
| `requirements.txt` | Versi paket Python yang dipakai |
| `SHA256SUMS.txt` | Checksum setiap berkas di paket |
| `buat_paket.py` | Skrip yang membangun paket ini dari folder kerja peneliti (jalur di dalamnya mengacu ke komputer peneliti) |

## Urutan analisis dan kaitannya dengan laporan

Tahap yang ditandai **(internet)** mengunduh data. Tahap lainnya cukup memakai berkas di paket.

### Tahap 0. Data hulu (Bab 3.2–3.3, 4.1)

| Skrip | Fungsi |
|---|---|
| `hulu/download_supporting_data.py` **(internet)** | Unduh SRTM, sungai OSM, hujan NASA POWER (satu titik), WorldCover, WorldPop |
| `hulu/process_data.py` | Saring Groundsource global (Zenodo 10.5281/zenodo.18647054) ke batas Kabupaten Sambas, 2010–2025: 753 kejadian |
| `hulu/audit_groundsource.py` | Audit cakupan, tanggal, dan luas poligon kejadian |
| `hulu/build_sambas_features.py` | Grid fitur 1 km (`feature_grid_sambas_1km.csv`) |

Skrip `process_data.py` membutuhkan berkas Groundsource global `hulu/outputs/data/raw/groundsource_2026.parquet` (667 MB, MD5 `CD1B5DE6508F7AAD8E1D1D0DD4CECEA6`). Hasil saringannya sudah ada di `hulu/outputs/data/processed/`.

### Tahap 1. Model spasial grid 1 km (Bab 3.4, 4.2)

```
python run_model.py hulu\outputs\data\processed\feature_grid_sambas_1km.csv outputs
python osm_features.py data\osm\jalan_tempat.json hulu\outputs\data\processed\feature_grid_sambas_1km.csv data\osm_features.csv
python hand_twi.py <folder_hgt_SRTM> hulu\outputs\data\processed\feature_grid_sambas_1km.csv data\hidrologi
python fetch_chirps_monthly.py data\chirps_monthly          (internet)
python rain_features.py data\chirps_monthly hulu\outputs\data\processed\feature_grid_sambas_1km.csv data\fitur_hujan_grid.csv
python run_model2.py data\hidrologi\fitur_hidrologi_grid.csv outputs_v3 data\fitur_hujan_grid.csv
python hand_sens.py data\hidrologi\dem_90m.tif hulu\outputs\data\processed\feature_grid_sambas_1km.csv data\hand_sens
python sens_model.py
```

Keluaran:
- `outputs/`: Random Forest dan K-Means awal.
- `outputs_v3/`: model dengan HAND/TWI, dengan dan tanpa populasi. Ini keluaran versi akhir `run_model2.py`.
- `outputs_v2/`: keluaran dari versi lebih awal skrip yang sama, disimpan sebagai arsip.
- `outputs_sens/`: sensitivitas ambang aliran HAND. Data OSM diambil lewat Overpass (`data/osm/fetch.sh`, kueri `q.txt` dan `q2.txt`). `<folder_hgt_SRTM>` berisi tile SRTM `N00E108`–`N01E109`, yang diunduh oleh `hulu/download_supporting_data.py`.

### Tahap 2. Bias pelaporan (Bab 3.5, 4.3)

`python bias_model.py` menghasilkan `outputs_bias/`: model gabungan faktor fisik dan pelaporan, kinerja menurut strata propensitas pelaporan, dan diagnostik tren tahunan.

### Tahap 3. Label genangan Sentinel-1 (Bab 3.6, 4.4, Lampiran D)

```
python fetch_s1.py data\s1                (internet, sekitar 3,6 GB; Microsoft Planetary Computer, Sentinel-1 RTC)
python s1_masks.py
python s1_flood.py calib
python s1_flood.py run -15 4              # konfigurasi dasar
python s1_coast_dist.py
python s1_analyze.py
python s1_model.py
python s1_robust.py
python s1_coast.py
python s1_events.py
python s1_figure.py
```

Sensitivitas ambang (Lampiran D): jalankan `s1_flood.py run` untuk enam konfigurasi lain, lalu `python s1_sens.py`.

```
python s1_flood.py run -13 4 _t13d4
python s1_flood.py run -17 4 _t17d4
python s1_flood.py run -15 3 _t15d3
python s1_flood.py run -15 5 _t15d5
python s1_flood.py run -13 3 _t13d3
python s1_flood.py run -17 5 _t17d5
```

Hasil deteksi per sel dan per adegan untuk ketujuh konfigurasi **sudah ada** di `data/` (`s1_cell_stats.csv`, `s1_cell_scene*.npz`, `s1_pixel_stats*.npz`, `s1_scene_series*.csv`). Karena itu, `s1_analyze.py` sampai `s1_sens.py` dapat dijalankan tanpa mengunduh citra. Hanya `fetch_s1.py`, `s1_masks.py`, `s1_flood.py`, dan `s1_coast_dist.py` yang membutuhkan cache citra di `data/s1/`.

Keluaran: `outputs_s1/` dan `outputs_article/sensitivitas_ambang_radar.csv`.

### Tahap 4. Model kecamatan-hari (Bab 3.7, 4.5)

```
python kec_prep.py overlay     # irisan kejadian Groundsource dengan kecamatan -> kec/_baru/
python kec_prep.py hujan       # (internet) hujan harian NASA POWER per kecamatan -> kec/_baru/
python kec_prep.py cek         # bandingkan dengan kec/
python panel_model.py
python panel_final.py
```

`kec_prep.py` menulis ke `kec/_baru/` supaya berkas di `kec/` tidak tertimpa. Pada 28 Oktober 2026, kedua berkas hasilnya **identik** dengan `kec/event_kec_overlay.csv` (2.115 baris) dan `kec/hujan_kec_power.csv` (97.166 baris).

Keluaran di `outputs_kec/`: panel kecamatan-hari, kinerja lintas-tahun dan ke-depan, model final (`model_final.joblib`), kalibrasi, SHAP, zona kecamatan, dan peringatan tiga tingkat.

### Tahap 5. Validasi eksternal (Bab 3.8, 4.6, Lampiran B)

| Skrip | Fungsi |
|---|---|
| `panel_final.py` | Validasi terhadap catatan DIBI (kabupaten-hari) |
| `bpbd_validasi.py` | Mencocokkan tabel kompilasi peneliti dengan label Groundsource dan skor model |
| `bpbd_csv_cek.py` | Memeriksa tabel per desa terhadap DIBI dan sumber lain |

Data DIBI diambil secara manual dari dasbor publik `https://dibi.bnpb.go.id/superset/dashboard/2/`. Respons API-nya disimpan di `dibi/api_v1_dashboard_2*.json`.

### Tahap 6. Inferensi statistik (Bab 3.9)

`python article_stats.py` menghitung interval kepercayaan bootstrap: blok spasial 10 km untuk analisis grid, dan per tanggal untuk analisis kecamatan-hari. Keluarannya ada di `outputs_article/`.

### Tahap 7. Dashboard (Bab 3.10, 4.7)

```
python prep_dashboard.py       # salin model dan data ke dashboard/data
streamlit run dashboard\app.py
```

Dashboard terbuka di `http://localhost:8501`. Peta dasarnya memakai OpenStreetMap, sehingga perlu internet untuk menampilkan ubin peta.

## Data yang tidak dikemas

| Data | Ukuran | Cara memperoleh |
|---|---|---|
| Groundsource global 2026 | 667 MB | Zenodo 10.5281/zenodo.18647054, berkas `groundsource_2026.parquet` |
| Citra Sentinel-1 RTC (cache per adegan) | 3,6 GB | `python fetch_s1.py data\s1` |
| Tile SRTM, WorldCover, WorldPop mentah | ±200 MB | `hulu/download_supporting_data.py` |
| CHIRPS harian | — | `python fetch_chirps.py data\chirps` (cache kecil sudah ada di `data/chirps/`) |

## Reproduksibilitas

- Semua model memakai *seed* 42.
- Data API eksternal (NASA POWER, CHIRPS, Sentinel-1, OSM) dapat direvisi oleh penyedianya. Unduhan ulang di masa depan mungkin sedikit berbeda dari data di paket.
- Interval bootstrap bergantung pada urutan bilangan acak. Versi numpy atau scikit-learn yang berbeda dapat menggeser batas interval pada digit terakhir.

## Hasil uji ulang

Pada 28 Oktober 2026, paket ini disalin ke folder lain dan skrip berikut dijalankan dari salinan tersebut. Keluarannya dibandingkan dengan keluaran asli yang dipakai di laporan.

| Skrip | Waktu | Hasil |
|---|---|---|
| `kec_prep.py overlay` dan `hujan` | ±3 menit | `event_kec_overlay.csv` dan `hujan_kec_power.csv` identik (selisih 0) |
| `panel_model.py` | 34 detik | Seluruh berkas di `outputs_kec/` identik byte demi byte |
| `panel_final.py` | 4 detik | Model, kalibrasi, SHAP, validasi DIBI, dan `ringkasan_final.json` identik |
| `article_stats.py` | 58 detik | Interval bootstrap (`auc_pembalikan_ci.csv`, `model_vs_hujan_ci.csv`, `ringkasan_statistik.json`) identik |
| `s1_model.py` | 7 detik | `outputs_s1/kinerja_label_s1_vs_groundsource.csv` dan `sel_layak_s1.csv` identik |
| `s1_sens.py` | ±1 menit | `sensitivitas_ambang_radar.csv` (Lampiran D) identik |

Skrip lain dalam paket (tahap grid, bias, dan deteksi radar) **belum diuji ulang dari paket**. Sebagian dari skrip itu membutuhkan unduhan data mentah. Keluarannya tetap disertakan sebagaimana dihasilkan saat analisis.

## Keterbatasan data yang perlu diketahui

- **Groundsource** adalah kejadian berbasis berita dengan resolusi setingkat kecamatan, dan pelaporannya melonjak sejak 2021.
- **Hujan NASA POWER** beresolusi sekitar 0,5°, sehingga hanya ada 4 deret unik untuk 19 kecamatan.
- **DIBI** tidak memuat nama kecamatan dan tidak memuat satu pun catatan tahun 2021.
- **Tabel kompilasi di `bpbd/`** belum seluruhnya terverifikasi. Jangan menyebutnya "data resmi BPBD".

## Lisensi dan atribusi data

Groundsource (Google Research, Zenodo), batas wilayah BIG, NASA POWER, CHIRPS (UCSB CHC), Sentinel-1 RTC (ESA Copernicus melalui Microsoft Planetary Computer), SRTM (NASA), ESA WorldCover 2021, WorldPop, OpenStreetMap (© kontributor OpenStreetMap, ODbL), dan DIBI BNPB. Gunakan setiap sumber sesuai lisensinya masing-masing.
