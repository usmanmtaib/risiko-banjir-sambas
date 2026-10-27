"""Model final kecamatan-hari: LOKO, kalibrasi, peringatan 3 tingkat, validasi DIBI, SHAP, penyimpanan model."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_kec"
P = pd.read_csv(OUT / "panel_kecamatan_hari.csv", parse_dates=["tanggal"])
RAIN = ["r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7"]
SEAS = ["sin", "cos"]
STATIC = ["elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border"]
FEATS = RAIN + SEAS + STATIC
d = P[P.tahun >= 2021].reset_index(drop=True)
YRS = [2021, 2022, 2023, 2024, 2025]
base_rate = d.y.mean()
res = {"prevalensi_2021_2025": round(float(base_rate), 5), "n_kecamatan_hari": len(d), "positif": int(d.y.sum())}


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3))  # tanpa class_weight -> probabilitas terkalibrasi


def metrics(y, p):
    return dict(roc=round(roc_auc_score(y, p), 3), pr=round(average_precision_score(y, p), 3),
                brier=round(brier_score_loss(y, p), 5), brier_skill=round(1 - brier_score_loss(y, p) / (y.mean() * (1 - y.mean())), 4))


# ---------- 1. lintas-tahun (OOF) ----------
oof = np.zeros(len(d))
for ty in YRS:
    tr, te = d.tahun != ty, d.tahun == ty
    oof[te] = mk().fit(d.loc[tr, FEATS], d.y[tr]).predict_proba(d.loc[te, FEATS])[:, 1]
d["p"] = oof
res["lintas_tahun"] = metrics(d.y.values, oof)
rel = d.assign(bin=pd.qcut(d.p.rank(method="first"), 10, labels=False) + 1).groupby("bin").agg(p_prediksi=("p", "mean"), laju_aktual=("y", "mean"), n=("y", "size")).round(5)
rel.to_csv(OUT / "kalibrasi_desil.csv")
print("== Kalibrasi per desil (lintas-tahun) ==\n", rel.to_string())

# ---------- 2. leave-one-kecamatan-out ----------
oofk = np.zeros(len(d))
for kc in d.kec.unique():
    tr, te = d.kec != kc, d.kec == kc
    oofk[te] = mk().fit(d.loc[tr, FEATS], d.y[tr]).predict_proba(d.loc[te, FEATS])[:, 1]
res["leave_one_kecamatan_out"] = metrics(d.y.values, oofk)
perk = d.assign(pk=oofk).groupby("kec").apply(lambda g: pd.Series({"positif": int(g.y.sum()), "roc": roc_auc_score(g.y, g.pk) if 0 < g.y.sum() < len(g) else np.nan}))
perk.round(3).to_csv(OUT / "loko_per_kecamatan.csv")
res["loko_roc_per_kecamatan_median"] = round(float(perk.roc.median()), 3)
print("\nLOKO:", res["leave_one_kecamatan_out"], "| median ROC per kecamatan:", res["loko_roc_per_kecamatan_median"])

# ---------- 3. peringatan 3 tingkat, uji ke-depan ----------
LV = {"Waspada": 0.90, "Siaga": 0.97, "Awas": 0.99}  # kuantil probabilitas pada data latih
alert_rows = []
for ty in (2024, 2025):
    tr, te = d[d.tahun < ty], d[d.tahun == ty]
    m = mk().fit(tr[FEATS], tr.y)
    ptr, pte = m.predict_proba(tr[FEATS])[:, 1], m.predict_proba(te[FEATS])[:, 1]
    for lv, q in LV.items():
        th = np.quantile(ptr, q); a = pte >= th
        alert_rows.append(dict(tahun_uji=ty, tingkat=lv, ambang_prob=round(float(th), 4), hari_alarm=int(a.sum()),
                               alarm_per_kec_per_tahun=round(a.sum() / 19, 1), presisi=round(float(te.y[a].mean()), 3) if a.any() else np.nan,
                               recall=round(float(te.y[a].sum() / max(te.y.sum(), 1)), 3), laju_dasar=round(float(te.y.mean()), 4)))
A = pd.DataFrame(alert_rows); A.to_csv(OUT / "peringatan_ke_depan.csv", index=False)
print("\n== Peringatan 3 tingkat (latih tahun sebelumnya, uji tahun berikutnya) ==\n", A.to_string(index=False))

# ---------- 4. validasi eksternal terhadap DIBI (kabupaten-hari) ----------
dibi = pd.read_csv(W / "dibi/dibi_sambas_banjir_tgl.csv", parse_dates=["tgl"]).drop_duplicates("tgl")
dd = d.groupby("tanggal").agg(smax=("p", "max"), s3=("p", lambda s: s.nlargest(3).mean()), r3max=("r3", "max")).reset_index()
days = pd.DataFrame({"tanggal": pd.date_range("2021-01-01", "2025-12-31")}).merge(dd, on="tanggal", how="left")
ev = dibi.tgl[(dibi.tgl >= "2021-01-01") & (dibi.tgl <= "2025-12-31")]
days["dibi"] = days.tanggal.apply(lambda x: int(((ev >= x - pd.Timedelta(days=1)) & (ev <= x + pd.Timedelta(days=1))).any()))
ext = {"n_tanggal_dibi": int(len(ev)), "hari_berlabel": int(days.dibi.sum()), "hari": len(days)}
for nm in ("smax", "s3", "r3max"):
    ext[f"roc_{nm}"] = round(roc_auc_score(days.dibi, days[nm]), 3); ext[f"pr_{nm}"] = round(average_precision_score(days.dibi, days[nm]), 3)
res["validasi_dibi_kabupaten_hari"] = ext
print("\nDIBI:", ext, "| prevalensi hari berlabel: %.3f" % days.dibi.mean())

# ---------- 5. model final + SHAP ----------
final = mk().fit(d[FEATS], d.y)
sc, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
Z = sc.transform(d[FEATS])
sv = shap.LinearExplainer(lr, Z).shap_values(Z)
imp = pd.DataFrame({"fitur": FEATS, "mean_abs_shap": np.abs(sv).mean(0), "koef_std": lr.coef_[0]}).sort_values("mean_abs_shap", ascending=False)
imp.round(4).to_csv(OUT / "shap_importance.csv", index=False)
print("\n== SHAP (rata-rata |nilai|) ==\n", imp.round(4).head(12).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
t = imp.head(12)[::-1]; cols = ["#2a6f97" if c > 0 else "#e07a1f" for c in t.koef_std]
ax[0].barh(t.fitur, t.mean_abs_shap, color=cols); ax[0].set_title("SHAP: kontribusi rata-rata |log-odds| (biru = menaikkan risiko)")
a = ax[1]; a.plot(rel.p_prediksi * 100, rel.laju_aktual * 100, "o-", color="#e07a1f", label="model"); mx = max(rel.p_prediksi.max(), rel.laju_aktual.max()) * 100
a.plot([0, mx], [0, mx], "k--", label="ideal"); a.set_xlabel("Probabilitas prediksi (%)"); a.set_ylabel("Laju laporan aktual (%)"); a.set_title("Kalibrasi lintas-tahun (per desil)"); a.legend()
plt.tight_layout(); plt.savefig(OUT / "shap_kalibrasi.png", dpi=130)

# ---------- 6. simpan untuk dashboard ----------
th_all = {lv: float(np.quantile(final.predict_proba(d[FEATS])[:, 1], q)) for lv, q in LV.items()}
joblib.dump(final, OUT / "model_final.joblib")
st = pd.read_csv(OUT / "fitur_kecamatan.csv", index_col=0)
zona = pd.read_csv(OUT / "zona_kecamatan.csv", index_col=0)[["zona", "laporan_per_tahun", "laporan_per_100km2"]]
st.join(zona).to_csv(OUT / "kecamatan_dashboard.csv")
json.dump({"fitur": FEATS, "ambang_prob": th_all, "kuantil": LV, "prevalensi": float(base_rate), "metrik": res}, open(OUT / "model_final_meta.json", "w"), indent=1, ensure_ascii=False)
json.dump(res, open(OUT / "ringkasan_final.json", "w"), indent=1, ensure_ascii=False)
print("\nAmbang probabilitas (model final):", {k: round(v, 4) for k, v in th_all.items()})
print(json.dumps(res, indent=1, ensure_ascii=False))
