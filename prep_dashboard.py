"""Siapkan folder dashboard/data agar dashboard berdiri sendiri (paket reproduksi)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent
D = W / "dashboard" / "data"; D.mkdir(parents=True, exist_ok=True)
O = W / "outputs_kec"

cols = ["kec", "tanggal", "y", "r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7", "sin", "cos",
        "elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border", "hujan_mm"]
P = pd.read_csv(O / "panel_kecamatan_hari.csv", usecols=cols)
P.to_csv(D / "panel.csv.gz", index=False, compression="gzip")

for f in ["model_final.joblib", "model_final_meta.json", "kecamatan_dashboard.csv", "ambang_hujan_kecamatan.csv",
          "shap_importance.csv", "kalibrasi_desil.csv", "kinerja_kecamatan_hari.csv", "peringatan_ke_depan.csv", "loko_per_kecamatan.csv"]:
    shutil.copy(O / f, D / f)
shutil.copy(O / "shap_kalibrasi.png", D / "shap_kalibrasi.png")
shutil.copy(W / "kec/kecamatan_sambas.geojson", D / "kecamatan_sambas.geojson")

# DIBI + kecocokan dengan Groundsource
dibi = pd.read_csv(W / "dibi/dibi_sambas_banjir_tgl.csv", parse_dates=["tgl"]).drop_duplicates("tgl")
Y = P.pivot(index="tanggal", columns="kec", values="y"); Y.index = pd.to_datetime(Y.index)
rows = []
for _, r in dibi.iterrows():
    t = r.tgl
    if t < pd.Timestamp("2012-01-01") or t > pd.Timestamp("2025-12-31"):
        n = np.nan
    else:
        n = int((Y.loc[t - pd.Timedelta(days=1): t + pd.Timedelta(days=1)].max() >= 1).sum())
    rows.append(dict(tanggal=t.date(), rumah_terendam=int(r["Rumah Terendam"]), terdampak_mengungsi=int(r["menderita_mengungsi"]),
                     meninggal=int(r["Meninggal"]), kecamatan_berlabel_groundsource=n))
pd.DataFrame(rows).to_csv(D / "dibi_sambas.csv", index=False)

# perbandingan model vs indikator hujan sederhana (hasil lintas-tahun 2021-2025; lihat catatan v5)
cmp = pd.DataFrame([
    ["Hujan 3 hari saja (1 fitur)", 0.753, 0.038, "4,6% / 40,5%", "5,2% / 13,7%", "8,1% / 7,4%", 0.878, 0.136],
    ["Hujan 7 hari saja (1 fitur)", 0.793, 0.051, "5,5% / 48,3%", "8,6% / 22,9%", "8,0% / 7,1%", 0.854, 0.108],
    ["Hujan (9 fitur)", 0.794, 0.051, "5,4% / 47,3%", "8,1% / 21,4%", "9,1% / 8,1%", 0.837, 0.087],
    ["Hujan + musim", 0.824, 0.048, "4,7% / 41,7%", "7,6% / 20,1%", "8,0% / 7,1%", 0.837, 0.099],
    ["Hujan + musim + kecamatan (model final)", 0.834, 0.051, "5,2% / 45,8%", "7,2% / 19,1%", "8,6% / 7,6%", 0.836, 0.100],
], columns=["model", "ROC-AUC", "PR-AUC", "presisi/recall 10% teratas", "presisi/recall 3% teratas", "presisi/recall 1% teratas", "DIBI ROC-AUC", "DIBI PR-AUC"])
cmp.to_csv(D / "perbandingan_model.csv", index=False)
print("dashboard/data:", sorted(p.name for p in D.iterdir()))
print("ukuran total MB:", round(sum(p.stat().st_size for p in D.iterdir()) / 1e6, 1))
