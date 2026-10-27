# Dataset fitur grid Kabupaten Sambas

## Hasil

- Unit analisis: grid 1 km × 1 km.
- Jumlah grid: 6.215.
- Kejadian Groundsource teralokasikan: 753.
- CRS perhitungan jarak: EPSG:32649 (UTM zona 49N).
- Kelas `risk_class_exploratory`: sementara, berdasarkan frekuensi kejadian; belum merupakan label final Random Forest.

## Layer raster

- `dem_sambas_30m.tif`: DEM SRTM yang dipotong batas Sambas.
- `slope_sambas_degrees.tif`: kemiringan lereng dalam derajat.
- `worldcover_sambas_2021_10m.tif`: tutupan lahan ESA WorldCover 2021.
- `population_sambas_2020_1km.tif`: estimasi populasi WorldPop 2020.

## Tabel fitur

- `feature_grid_sambas_1km.geojson`: grid poligon dengan atribut fitur.
- `feature_grid_sambas_1km.csv`: tabel atribut tanpa geometri.
- `curah_hujan_tahunan_sambas_2010_2025.csv`: ringkasan curah hujan tahunan.
- `feature_grid_summary.json`: ringkasan jumlah grid dan statistik proses.

## Atribut utama

`elevation_m`, `slope_deg`, `distance_river_m`, `landcover_code`, `population_2020`, `event_count`, `risk_class_exploratory`, `rain_mean_mm_day`, `rain_max_mm_day`, `rain_total_mm`, `rain_mean_annual_total_mm`, dan `rain_max_annual_total_mm`.

## Batasan

Curah hujan yang digunakan saat ini adalah deret waktu NASA POWER pada satu titik koordinat Sambas dan disalin sebagai fitur wilayah. Untuk model spasial yang lebih kuat, fitur ini perlu diganti atau dilengkapi dengan beberapa stasiun BMKG atau raster hujan.
