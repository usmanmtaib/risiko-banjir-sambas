# Hasil penyaringan dan validasi awal

## Groundsource

- Sumber: Groundsource v1, Zenodo DOI 10.5281/zenodo.18647054.
- Filter spasial: centroid geometri berada di dalam batas Kabupaten Sambas dari BIG.
- Filter waktu: `start_date` tahun 2010–2025.
- Hasil: 753 kejadian.

## BNPB

- Sumber: portal Data Bencana BNPB.
- Filter query: Provinsi Kalimantan Barat, jenis BANJIR, tanggal 01-01-2010 sampai 31-12-2025.
- Hasil query: 17.243 baris.
- Hasil dengan kolom Kabupaten tepat bernilai `Sambas`: 4 baris.

## Berkas

- `groundsource_sambas_2010_2025.parquet`: data Groundsource terfilter.
- `groundsource_sambas_2010_2025.csv`: salinan tabular Groundsource terfilter.
- `bnpb_banjir_sambas_2010_2025.csv`: data BNPB terfilter.
- `validasi_tahunan_groundsource_bnpb.csv`: gabungan jumlah kejadian per tahun.
- `batas_kabupaten_sambas_big.geojson`: batas yang digunakan untuk penyaringan.

## Interpretasi

Perbedaan jumlah Groundsource dan BNPB bukan kesalahan otomatis. Groundsource berasal dari berita dan menangkap kejadian dengan cakupan berbeda, sedangkan portal BNPB hanya memuat kejadian yang tercatat dalam sistemnya. Tabel tahunan ini adalah validasi cakupan awal, bukan klaim bahwa salah satu sumber merupakan kebenaran mutlak.
