# Sumber data penelitian kerawanan banjir Kabupaten Sambas

Dokumen ini mencatat sumber data publik yang dipilih untuk tahap awal penelitian.
Data mentah ditempatkan di `work/data/raw/`; data hasil pemotongan wilayah Sambas dan
hasil pembersihan akan ditempatkan terpisah di `work/data/processed/`.

## Paket inti

| Komponen | Sumber | Peran | Catatan |
|---|---|---|---|
| Kejadian banjir | [Groundsource v1, Zenodo DOI 10.5281/zenodo.18647054](https://zenodo.org/records/18647054) | Titik/waktu kejadian historis | Parquet sekitar 667 MB; dataset global berbasis berita dan berfokus pada banjir bandang perkotaan. Perlu audit cakupan Sambas dan deduplikasi. |
| Data kejadian pembanding | [BNPB Data Bencana](https://gis.bnpb.go.id/databencana/tabel/pencarian.php) | Label/validasi independen | Pilih kejadian BANJIR, Kabupaten Sambas, 2010–2025. |
| Elevasi dan lereng | [NASA Earthdata SRTM](https://search.earthdata.nasa.gov/search) | Fitur topografi | Unduh tile yang menutupi Sambas, lalu turunkan slope dari DEM. |
| Batas administrasi | [Geoportal Kabupaten Sambas](https://sambaskab.ina-sdi.or.id/) dan [BIG GIS Services](https://geoservices.big.go.id/) | Clip dan agregasi desa/kecamatan | Prioritaskan sumber BIG/geoportal resmi; simpan versi dan tanggal akses. |
| Curah hujan | [BMKG SACA daily data](https://sacad.bmkg.go.id/dailydata/index.php) | Fitur temporal/iklim | Gunakan stasiun terdekat atau grid; cek kelengkapan dan periode observasi. |
| Tutupan lahan | [ESA WorldCover 2021](https://esa-worldcover.org/en/data-access) | Fitur penggunaan lahan | Resolusi 10 m, 11 kelas; gunakan hanya untuk analisis, bukan peta RGB/WMS. |
| Kepadatan penduduk | [WorldPop Indonesia](https://hub.worldpop.org/doi/10.5258/SOTON/WP00114) | Fitur paparan sosial | Pilih tahun yang konsisten dengan periode analisis; dokumentasikan bahwa ini estimasi raster. |

## Sumber tambahan yang berguna

- [InaRISK BNPB GIS services](https://gis.bnpb.go.id/server/rest/services/inarisk) untuk membandingkan peta bahaya/risiko yang sudah tersedia.
- [BPS Kabupaten Sambas – Peta Tematik 2021](https://sambaskab.bps.go.id/id/publication/2022/10/21/088bed4fcff0c78b0db5308c/peta-tematik-kabupaten-sambas-2021.html) untuk pemeriksaan statistik dan konteks wilayah.
- [OpenStreetMap/Geofabrik Indonesia](https://download.geofabrik.de/asia/indonesia.html) untuk jaringan jalan dan fasilitas pendukung dashboard, bila diperlukan.

## Urutan pengolahan yang akan digunakan

1. Saring Groundsource berdasarkan wilayah geografis Kabupaten Sambas dan periode 2010–2025.
2. Periksa koordinat, tanggal, duplikasi, dan ketidakpastian lokasi.
3. Cocokkan kejadian dengan data BNPB dan berita/laporan lokal yang tersedia.
4. Potong DEM, WorldCover, populasi, dan curah hujan menggunakan batas Sambas.
5. Bentuk unit analisis yang konsisten—disarankan grid 1 km atau desa.
6. Simpan semua metadata, checksum, tanggal akses, lisensi, dan keputusan pembersihan.

## Batasan penting

Groundsource tidak boleh diperlakukan sebagai kebenaran lapangan tanpa validasi. Karena data berasal dari pemberitaan, bias liputan dan ketidakpastian lokasi/waktu harus dilaporkan. Label Random Forest juga harus berasal dari sumber independen agar evaluasi model tidak bersifat sirkular.

## Status lokal

- `outputs/data/raw/groundsource_2026.parquet` — sudah diunduh lengkap, ukuran 667,122,400 byte.
- MD5 terverifikasi: `CD1B5DE6508F7AAD8E1D1D0DD4CECEA6`.
- Data pendukung belum diunduh karena beberapa sumber memerlukan pemilihan tile, wilayah, tahun, atau ekspor melalui layanan GIS/API.

## Data pendukung yang sudah diunduh

- `outputs/data/raw/dem_srtm/` — 6 tile SRTM HGT dan HGT.GZ untuk cakupan Sambas.
- `outputs/data/raw/sungai_osm/sungai_sambas_osm.geojson` — 2.252 fitur sungai/saluran dari OpenStreetMap Overpass.
- `outputs/data/raw/curah_hujan/nasa_power_precipitation_sambas_2010_2025.csv` — 5.844 observasi harian pada titik 1,35 LU; 109,35 BT.
- `outputs/data/raw/tutupan_lahan/ESA_WorldCover_10m_2021_v200_N00E108_Map.tif` — tile WorldCover 2021 resolusi 10 m.
- `outputs/data/raw/populasi/idn_ppp_2020_1km_Aggregated.tif` — populasi WorldPop Indonesia resolusi 1 km tahun 2020.

Curah hujan NASA POWER adalah deret waktu pada satu titik, bukan raster spasial. Untuk pemodelan per-grid, tahap lanjutan sebaiknya memakai beberapa stasiun BMKG atau raster hujan seperti CHIRPS.
