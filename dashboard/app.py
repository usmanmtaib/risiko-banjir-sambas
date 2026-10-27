"""Dashboard riset: risiko relatif laporan banjir per kecamatan-hari, Kabupaten Sambas (prototipe TKT 4).

Jalankan:  streamlit run app.py
"""
import json
from pathlib import Path

import branca.colormap as cm
import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Risiko Laporan Banjir Kecamatan-Hari, Sambas", layout="wide")

FEATS_RAIN = ["r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7"]


@st.cache_resource
def load_model():
    return joblib.load(DATA / "model_final.joblib"), json.load(open(DATA / "model_final_meta.json", encoding="utf-8"))


@st.cache_data
def load_data():
    panel = pd.read_csv(DATA / "panel.csv.gz", parse_dates=["tanggal"])
    geo = json.load(open(DATA / "kecamatan_sambas.geojson", encoding="utf-8"))
    kec = pd.read_csv(DATA / "kecamatan_dashboard.csv", index_col=0)
    return panel, geo, kec


model, meta = load_model()
panel, geo, kec = load_data()
FEATS = meta["fitur"]
TH = meta["ambang_prob"]  # Waspada/Siaga/Awas = kuantil 90/97/99 -> dinamai ulang netral
LEVELS = [("Sangat tinggi", TH["Awas"], "#7f0000", "8,6%"), ("Tinggi", TH["Siaga"], "#d7301f", "7,2%"),
          ("Meningkat", TH["Waspada"], "#fc8d59", "5,2%"), ("Normal", -1, "#ffffcc", "sekitar 0,5%")]


def level_of(p):
    for name, th, _, _ in LEVELS:
        if p >= th:
            return name


def make_map(df, value_col="p"):
    """df: satu baris per kecamatan dengan kolom kec, p, kelas, dan keterangan."""
    m = folium.Map(location=[1.45, 109.3], zoom_start=9, tiles="OpenStreetMap")
    vals = df.set_index("kec")
    cmap = cm.LinearColormap(["#ffffcc", "#fd8d3c", "#bd0026", "#7f0000"], vmin=0, vmax=max(0.15, float(vals[value_col].max())))
    cmap.caption = "Probabilitas laporan banjir (per kecamatan-hari)"

    def style(f):
        n = f["properties"]["NAMOBJ"]
        v = vals[value_col].get(n, np.nan)
        return {"fillColor": cmap(v) if v == v else "#cccccc", "color": "#555", "weight": 1, "fillOpacity": 0.75}

    tip = folium.GeoJsonTooltip(fields=["NAMOBJ"], aliases=["Kecamatan"])
    folium.GeoJson(geo, style_function=style, tooltip=tip).add_to(m)
    for _, r in vals.reset_index().iterrows():
        f = [x for x in geo["features"] if x["properties"]["NAMOBJ"] == r.kec]
        if not f:
            continue
        from shapely.geometry import shape
        c = shape(f[0]["geometry"]).representative_point()
        html = f"<b>{r.kec}</b><br>Probabilitas: {r.p*100:.2f}%<br>Kelas: {r.kelas}<br>{r.ket}"
        folium.CircleMarker([c.y, c.x], radius=4, color="#222", fill=True, fill_opacity=1, popup=folium.Popup(html, max_width=260)).add_to(m)
    cmap.add_to(m)
    return m


def geo_bounds(g):
    """[[lat_min, lon_min], [lat_max, lon_max]] dari seluruh koordinat GeoJSON."""
    xs, ys = [], []
    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for k in c: walk(k)
    for f in g["features"]:
        walk(f["geometry"]["coordinates"])
    return [[min(ys), min(xs)], [max(ys), max(xs)]] if xs else None


def show_map(m, height=560):
    # Paskan tampilan ke batas kabupaten setiap kali ukuran peta berubah (mis. saat tab dibuka).
    b = geo_bounds(geo)
    if b:
        js = f"""<script>
        (function() {{
          var B = {b}, last = "";
          function fit() {{
            var mp = window["{m.get_name()}"];
            if (!mp) return;
            var c = mp.getContainer(), key = c.clientWidth + "x" + c.clientHeight;
            if (c.clientWidth === 0 || c.clientHeight === 0 || key === last) return;
            last = key; mp.invalidateSize(); mp.fitBounds(B, {{padding: [10, 10]}});
          }}
          setInterval(fit, 300);
        }})();
        </script>"""
        m.get_root().html.add_child(folium.Element(js))
    # HTML lengkap: peta mengisi tinggi bingkai (_repr_html_ memakai rasio 60% lebar, sehingga terpotong di kolom lebar)
    components.html(m.get_root().render(), height=height)


st.title("Risiko laporan banjir per kecamatan-hari, Kabupaten Sambas")
st.warning("**Prototipe riset (TKT 4), bukan sistem peringatan dini operasional.** Label berasal dari berita (Groundsource) dan dipengaruhi "
           "bias pelaporan. Pada data ini, model **tidak lebih baik** daripada indikator hujan 7 hari sederhana (lihat tab *Validasi*). "
           "Angka adalah risiko **relatif**, dan kemampuannya hanya merangking hari, bukan memastikan banjir.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Peta risiko", "Ambang hujan", "Zona kecamatan", "Riwayat kejadian", "Metode dan validasi"])

# ---------------- TAB 1 ----------------
with tab1:
    mode = st.radio("Mode", ["Putar ulang tanggal historis (2012–2025)", "Skenario hujan (eksperimental)"], horizontal=True)
    if mode.startswith("Putar"):
        c1, c2 = st.columns([1, 3])
        with c1:
            d0 = st.date_input("Tanggal", value=pd.Timestamp("2025-01-18"), min_value=pd.Timestamp("2012-02-01"), max_value=pd.Timestamp("2025-12-31"))
            st.caption("Model dilatih pada 2021–2025, jadi untuk tanggal itu skor bersifat dalam-sampel. Hujan: NASA POWER (resolusi sekitar 0,5°).")
        day = panel[panel.tanggal == pd.Timestamp(d0)].copy()
        if day.empty:
            st.info("Tidak ada data untuk tanggal ini.")
        else:
            day["p"] = model.predict_proba(day[FEATS])[:, 1]
            day["kelas"] = day.p.apply(level_of)
            day["ket"] = day.apply(lambda r: f"Hujan 3 hari: {r.r3:.0f} mm<br>Hujan 7 hari: {r.r7:.0f} mm<br>Laporan Groundsource: {'ya' if r.y == 1 else 'tidak'}", axis=1)
            with c1:
                st.metric("Kecamatan 'Meningkat' atau lebih", int((day.kelas != "Normal").sum()))
                st.metric("Kecamatan dengan laporan (Groundsource)", int(day.y.sum()))
                st.metric("Hujan 7 hari rata-rata (mm)", f"{day.r7.mean():.0f}")
            with c2:
                show_map(make_map(day))
            tbl = day[["kec", "p", "kelas", "r1", "r3", "r7", "y"]].sort_values("p", ascending=False)
            tbl.columns = ["Kecamatan", "Probabilitas", "Kelas", "Hujan 1 hari (mm)", "Hujan 3 hari (mm)", "Hujan 7 hari (mm)", "Laporan Groundsource (1=ya)"]
            st.dataframe(tbl.style.format({"Probabilitas": "{:.2%}", "Hujan 1 hari (mm)": "{:.0f}", "Hujan 3 hari (mm)": "{:.0f}", "Hujan 7 hari (mm)": "{:.0f}"}), width="stretch", hide_index=True)
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.caption("Hujan yang sama diterapkan ke seluruh kecamatan. Fitur turunan (hujan 2 hari, 5 hari, maksimum 3 hari, jumlah hari hujan) **diperkirakan** dari empat masukan, jadi hasilnya hanya ilustrasi.")
            r1 = st.slider("Hujan 1 hari terakhir (mm)", 0, 250, 40)
            r3 = st.slider("Hujan 3 hari terakhir (mm)", r1, 400, max(r1, 90))
            r7 = st.slider("Hujan 7 hari terakhir (mm)", r3, 600, max(r3, 150))
            r14 = st.slider("Hujan 14 hari terakhir (mm)", r7, 900, max(r7, 250))
            bulan = st.selectbox("Bulan", list(range(1, 13)), index=0, format_func=lambda i: ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"][i - 1])
        doy = (bulan - 1) * 30.4 + 15
        sc = pd.DataFrame({"kec": kec.index})
        sc = sc.join(kec[["elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border"]], on="kec")
        sc["r1"], sc["r3"], sc["r7"], sc["r14"] = r1, r3, r7, r14
        sc["r2"] = r1 + (r3 - r1) / 2
        sc["r5"] = r3 + (r7 - r3) / 2
        sc["max3"] = max(r1, r3 / 2)
        sc["r30_prev"] = max(r14 - r7, 0) * 1.5
        sc["wet_days7"] = min(7, round(r7 / 20))
        sc["sin"], sc["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        sc["p"] = model.predict_proba(sc[FEATS])[:, 1]
        sc["kelas"] = sc.p.apply(level_of)
        sc["ket"] = f"Skenario: 1h {r1} mm, 3h {r3} mm, 7h {r7} mm"
        with c1:
            st.metric("Probabilitas rata-rata", f"{sc.p.mean()*100:.2f}%")
            st.metric("Kecamatan 'Meningkat' atau lebih", int((sc.kelas != "Normal").sum()))
        with c2:
            show_map(make_map(sc))
        st.dataframe(sc[["kec", "p", "kelas"]].sort_values("p", ascending=False).rename(columns={"kec": "Kecamatan", "p": "Probabilitas", "kelas": "Kelas"}).style.format({"Probabilitas": "{:.2%}"}), width="stretch", hide_index=True)
    st.markdown("**Arti kelas** (hasil validasi lintas-tahun 2021–2025; laju dasar 1,1% kecamatan-hari berlaporan):")
    st.table(pd.DataFrame([{"Kelas": n, "Ambang probabilitas": (f"≥ {t:.1%}" if t > 0 else "di bawah ambang"), "Presisi empiris": pe} for n, t, _, pe in LEVELS]))
    st.caption("Presisi empiris = persentase kecamatan-hari pada kelas itu (atau lebih tinggi) yang memang berlaporan. Peningkatan dari kelas ke kelas kecil dan tidak stabil antar-tahun.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Hujan 7 hari dan laju laporan")
    d21 = panel[panel.tanggal >= "2021-01-01"].copy()
    bins = [-1, 25, 50, 100, 150, 200, 1e9]
    labs = ["< 25", "25–50", "50–100", "100–150", "150–200", "≥ 200"]
    d21["bin"] = pd.cut(d21.r7, bins, labels=labs)
    g = d21.groupby("bin", observed=True).agg(laju=("y", "mean"), n=("y", "size"), laporan=("y", "sum")).reset_index()
    g["laju_pct"] = g.laju * 100
    st.bar_chart(g.set_index("bin").laju_pct, y_label="% kecamatan-hari berlaporan", x_label="Hujan 7 hari (mm)")
    st.dataframe(g.rename(columns={"bin": "Hujan 7 hari (mm)", "laju_pct": "Laju laporan (%)", "n": "Kecamatan-hari", "laporan": "Jumlah laporan"})[["Hujan 7 hari (mm)", "Laju laporan (%)", "Kecamatan-hari", "Jumlah laporan"]].style.format({"Laju laporan (%)": "{:.2f}"}), width="stretch", hide_index=True)
    st.subheader("Hujan 7 hari pada hari berlaporan, per kecamatan (2021–2025)")
    amb = pd.read_csv(DATA / "ambang_hujan_kecamatan.csv")
    a7 = amb[amb["var"] == "r7"].sort_values("n_kejadian", ascending=False)
    st.dataframe(a7.rename(columns={"kec": "Kecamatan", "n_kejadian": "Jumlah hari berlaporan", "median_hujan_kejadian": "Median hujan 7 hari (mm)", "p10_hujan_kejadian": "Persentil 10 (mm)", "p90_hujan_semua_hari": "Persentil 90 semua hari (mm)"}).drop(columns=["var"]), width="stretch", hide_index=True)
    st.caption("Banyak laporan terjadi pada hujan yang jauh lebih kecil daripada median. Hujan berasal dari NASA POWER (sekitar 0,5°; hanya 4 deret unik untuk 19 kecamatan), jadi angka ini indikasi, bukan ambang operasional.")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Zona kecamatan (K-Means, k = 4, silhouette 0,255: struktur klaster lemah)")
    zc = kec.reset_index().rename(columns={"index": "kec"})
    zc = zc.rename(columns={zc.columns[0]: "kec"})
    zc["p"] = zc.zona.astype(float); zc["kelas"] = "Zona " + zc.zona.astype(int).astype(str)
    zc["ket"] = zc.apply(lambda r: f"Laporan/tahun: {r.laporan_per_tahun:.1f}<br>Per 100 km²: {r.laporan_per_100km2:.2f}", axis=1)
    mz = folium.Map(location=[1.45, 109.3], zoom_start=9, tiles="OpenStreetMap")
    pal = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3", 3: "#e7298a"}
    zmap = zc.set_index("kec").zona.astype(int).to_dict()
    zlayer = folium.GeoJson(geo, style_function=lambda f: {"fillColor": pal.get(zmap.get(f["properties"]["NAMOBJ"], 0), "#ccc"), "color": "#444", "weight": 1, "fillOpacity": 0.6},
                   tooltip=folium.GeoJsonTooltip(fields=["NAMOBJ"], aliases=["Kecamatan"])).add_to(mz)
    show_map(mz, 520)
    st.dataframe(zc[["kec", "zona", "laporan_per_tahun", "laporan_per_100km2", "log_popdens", "elev", "coast"]].sort_values(["zona", "kec"]).rename(columns={"kec": "Kecamatan", "zona": "Zona", "laporan_per_tahun": "Laporan/tahun", "laporan_per_100km2": "Laporan/100 km²", "log_popdens": "log kepadatan penduduk", "elev": "Elevasi rata-rata (m)", "coast": "Jarak pantai (km)"}), width="stretch", hide_index=True)
    st.info("Ini **zona pelaporan**, bukan zona bahaya: frekuensi laporan bercampur dengan peluang suatu kejadian diberitakan.")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Kecamatan-hari berlaporan per tahun (Groundsource)")
    yy = panel.groupby(panel.tanggal.dt.year).y.sum()
    st.bar_chart(yy, y_label="Kecamatan-hari berlaporan", x_label="Tahun")
    st.caption("Lonjakan sejak 2021 mencerminkan pertumbuhan pelaporan berita, bukan hanya banjir. Analisis model dibatasi pada 2021–2025.")
    st.subheader("Catatan banjir Sambas di DIBI BNPB (dasbor publik, diakses 21 Okt 2026)")
    db = pd.read_csv(DATA / "dibi_sambas.csv")
    _g = "kecamatan_berlabel_groundsource"
    if _g in db:
        db[_g] = db[_g].map(lambda v: "–" if pd.isna(v) else str(int(v)))
    st.dataframe(db.rename(columns={"tanggal": "Tanggal", "rumah_terendam": "Rumah terendam", "terdampak_mengungsi": "Terdampak/mengungsi", "meninggal": "Meninggal", "kecamatan_berlabel_groundsource": "Kecamatan berlabel Groundsource (±1 hari)"}), width="stretch", hide_index=True)
    st.caption("DIBI tidak memuat kecamatan dan tidak memuat satu pun catatan 2021, jadi tidak lengkap. Tanda – berarti tanggal di luar periode data Groundsource (sebelum 2012). Hanya 52% hari Groundsource dengan ≥ 5 kecamatan berlabel yang punya catatan DIBI dalam ±3 hari.")

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("Perbandingan model dengan indikator hujan sederhana (lintas-tahun 2021–2025)")
    st.dataframe(pd.read_csv(DATA / "perbandingan_model.csv"), width="stretch", hide_index=True)
    st.markdown("""
**Pembacaan jujur.** Hujan 7 hari saja sudah menghasilkan PR-AUC yang sama (0,051) dengan model final. Tambahan ROC-AUC model berasal terutama dari musim
(kosinus hari-dalam-tahun), bukan dari fitur kecamatan. Pada validasi terhadap catatan DIBI (kabupaten-hari), hujan 3 hari maksimum bahkan sedikit lebih baik (ROC-AUC 0,88 vs 0,84).
Dengan sekitar 400 kejadian positif dan 191 hari unik, kompleksitas tambahan tidak terbukti bermanfaat.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SHAP (model final)")
        shp = pd.read_csv(DATA / "shap_importance.csv")
        st.dataframe(shp.rename(columns={"fitur": "Fitur", "mean_abs_shap": "Rata-rata |SHAP|", "koef_std": "Koefisien terstandar"}), width="stretch", hide_index=True)
    with c2:
        st.image(str(DATA / "shap_kalibrasi.png"))
    st.subheader("Kinerja")
    kin = pd.read_csv(DATA / "kinerja_kecamatan_hari.csv")
    st.dataframe(kin, width="stretch", hide_index=True)
    st.markdown("""
**Metode ringkas.** Panel kecamatan-hari (19 kecamatan BIG). Label: ada kejadian Groundsource yang ≥ 50% luasnya berada dalam kecamatan itu. Fitur: akumulasi hujan 1–14 hari, kelembapan awal,
musim, dan karakteristik kecamatan. Model: regresi logistik. Validasi: *leave-one-year-out* dan ke-depan pada 2021–2025.

**Keterbatasan.** Label berbasis berita (bias pelaporan, lonjakan sejak 2021). Hujan kasar (NASA POWER, harian). Sampel efektif 191 hari unik. Peringatan tiga tingkat tidak stabil pada uji ke-depan
(presisi tingkat teratas tidak lebih baik daripada tingkat di bawahnya pada 2025). Validasi eksternal terbatas pada peristiwa besar. Banjir luapan di luar musim hujan dan banjir pasang kurang tertangkap.

**Sumber data.** Groundsource (Google Research, Zenodo), BIG (batas kecamatan), NASA POWER (hujan), SRTM, ESA WorldCover, WorldPop, OpenStreetMap, DIBI BNPB.
""")
