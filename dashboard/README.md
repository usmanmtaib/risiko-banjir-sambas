# Dashboard risiko laporan banjir kecamatan-hari, Kabupaten Sambas

Prototipe riset (TKT 4). **Bukan sistem peringatan dini operasional.**

## Menjalankan
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501.

## Isi
- `app.py`: aplikasi Streamlit + Folium (5 tab: peta risiko, ambang hujan, zona kecamatan, riwayat kejadian, metode dan validasi).
- `data/`: panel kecamatan-hari 2012–2025, model final (`model_final.joblib`), batas kecamatan BIG, hasil SHAP dan kalibrasi, catatan DIBI.

## Yang perlu diketahui sebelum memakai
- Label berasal dari berita (Groundsource) dan dipengaruhi bias pelaporan; pelaporan melonjak sejak 2021.
- Pada data ini model **tidak lebih baik** daripada indikator hujan 7 hari sederhana (PR-AUC sama, 0,051). Lihat tab *Metode dan validasi*.
- Kelas risiko ("Meningkat", "Tinggi", "Sangat tinggi") adalah peringkat relatif, bukan peringatan. Presisi tingkat teratas tidak lebih baik daripada tingkat di bawahnya pada uji ke-depan 2025.
- Hujan: NASA POWER (sekitar 0,5°), hanya 4 deret unik untuk 19 kecamatan.
- Mode skenario memperkirakan beberapa fitur hujan turunan dari empat masukan, jadi hanya ilustrasi.

## Mereproduksi data
Skrip penyusun ada di folder induk: `panel_model.py`, `panel_final.py`, `prep_dashboard.py`.
