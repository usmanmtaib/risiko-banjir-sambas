"""Periksa banjir_sambas_per_desa_2020_2025.csv: kecocokan dengan DIBI, kecocokan daftar kecamatan dengan sumber lain,
dan uji diskriminasi spasial (kecamatan mana yang tergenang pada hari kejadian)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
W = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
c = pd.read_csv(W / "bpbd/banjir_per_desa_dibersihkan.csv", parse_dates=["tgl"])
dibi = pd.read_csv(W / "dibi/dibi_sambas_banjir_tgl.csv", parse_dates=["tgl"])
m = c.merge(dibi[["id", "tgl", "Rumah Terendam", "menderita_mengungsi", "Meninggal"]], left_on="ID_DIBI", right_on="id", how="left", suffixes=("", "_dibi"))
mm = m[m.id.notna()]
print("=== A. Kecocokan dengan DIBI ===")
print(f"baris CSV: {len(c)} ({c.tgl.min().date()} s.d. {c.tgl.max().date()}) | ID ada di DIBI: {len(mm)}")
print("tanggal sama:", int((mm.tgl.dt.date == mm.tgl_dibi.dt.date).sum()), "| rumah terendam sama:", int((mm.Rumah_Terendam == mm["Rumah Terendam"]).sum()),
      "| terdampak sama:", int((mm.Jiwa_Terdampak_Mengungsi == mm.menderita_mengungsi).sum()), "| meninggal sama:", int((mm.Meninggal == mm.Meninggal_dibi).sum()), f"(dari {len(mm)})")
print("baris tanpa ID DIBI:", m[m.id.isna()][["ID_DIBI", "tgl", "Jiwa_Terdampak_Mengungsi", "Rumah_Terendam"]].values.tolist())
print("catatan DIBI 2022+ yang tidak ada di CSV:", dibi[(dibi.tgl >= "2022-01-01") & (~dibi.id.isin(c.ID_DIBI))][["tgl", "Rumah Terendam"]].assign(tgl=lambda d: d.tgl.dt.date).values.tolist())
print("per tahun:", c.Tahun.value_counts().sort_index().to_dict())

# ---------- B. daftar kecamatan CSV vs sumber lain ----------
def S(x): return {t.strip() for t in str(x).split(";") if t.strip()}
csvlist = {r.tgl.date(): S(r.Kecamatan_Terdampak) for r in c.itertuples()}
ref = {  # daftar kecamatan dari sumber lain yang sudah kita miliki
    "2025-12-08": ("Portal lama BNPB (kolom Lokasi)", {"Jawai", "Jawai Selatan", "Paloh", "Pemangkat", "Sajingan Besar", "Sambas", "Sebawi", "Sejangkung", "Selakau", "Tangaran", "Tebas", "Teluk Keramat"}),
    "2025-01-17": ("Portal lama BNPB (kolom Lokasi)", {"Galing", "Jawai Selatan", "Paloh", "Sajad", "Sambas", "Sejangkung", "Selakau", "Selakau Timur", "Subah", "Tangaran", "Tebas"}),
    "2025-01-11": ("Portal lama BNPB (kolom Lokasi)", {"Jawai Selatan", "Pemangkat"}),
    "2023-01-31": ("Berita yang mengutip BNPB (Tirto/iNews)", {"Sejangkung", "Paloh", "Tangaran", "Teluk Keramat", "Galing", "Jawai Selatan", "Sambas", "Jawai"}),
}
print("\n=== B. Daftar kecamatan CSV vs sumber lain ===")
rowsB = []
for dt, (src, rs) in ref.items():
    cs = csvlist[pd.Timestamp(dt).date()]
    inter = cs & rs; jac = len(inter) / len(cs | rs)
    rowsB.append((dt, src, len(cs), len(rs), len(inter), round(jac, 2)))
    print(f"{dt} | {src} | CSV {len(cs)} kec, rujukan {len(rs)} kec, irisan {len(inter)}, Jaccard {jac:.2f} | hanya di CSV: {sorted(cs - rs)} | hanya di rujukan: {sorted(rs - cs)}")
print("Jaccard rata-rata:", round(float(np.mean([r[-1] for r in rowsB])), 2))

# ---------- C. diskriminasi spasial pada hari kejadian ----------
P = pd.read_csv(W / "outputs_kec/panel_kecamatan_hari.csv", parse_dates=["tanggal"])
RAIN = ["r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7"]
F = RAIN + ["sin", "cos", "elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border"]
d = P[P.tahun >= 2021].reset_index(drop=True); o = np.zeros(len(d))
for ty in range(2021, 2026):
    tr, te = d.tahun != ty, d.tahun == ty
    o[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)).fit(d.loc[tr, F], d.y[tr]).predict_proba(d.loc[te, F])[:, 1]
d["p"] = o
KECS = sorted(d.kec.unique())


def within_auc(lists, label):
    out = {k: [] for k in ["model (skor harian)", "label Groundsource", "hujan 3 hari (per kecamatan)", "kepadatan penduduk", "elevasi rendah", "jarak pantai dekat"]}
    prs = []
    for dt, L in lists.items():
        t = pd.Timestamp(dt)
        if t < pd.Timestamp("2021-01-01") or t > pd.Timestamp("2025-12-31"):
            continue
        w = d[(d.tanggal >= t - pd.Timedelta(days=1)) & (d.tanggal <= t + pd.Timedelta(days=1))]
        g = w.groupby("kec").agg(p=("p", "max"), y=("y", "max"), r3=("r3", "max"), pop=("log_popdens", "first"), elev=("elev", "first"), coast=("coast", "first")).reindex(KECS)
        yl = np.array([k in L for k in KECS]).astype(int)
        if yl.sum() == 0 or yl.sum() == len(KECS):
            continue
        def auc(v):
            v = np.asarray(v, float)
            return roc_auc_score(yl, v) if np.nanstd(v) > 0 else 0.5
        out["model (skor harian)"].append(auc(g.p)); out["label Groundsource"].append(auc(g.y)); out["hujan 3 hari (per kecamatan)"].append(auc(g.r3))
        out["kepadatan penduduk"].append(auc(g["pop"])); out["elevasi rendah"].append(auc(-g.elev)); out["jarak pantai dekat"].append(auc(-g.coast))
        gs = set(g.index[g.y == 1])
        prs.append((dt, len(L), len(gs), len(L & gs), len(L & gs) / len(L), (len(L & gs) / len(gs)) if gs else np.nan, len(L) / len(KECS)))
    print(f"\n=== C. {label}: n peristiwa = {len(prs)} ===")
    for k, v in out.items():
        v = np.array(v); bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(2000)]
        print(f"  AUC dalam-hari {k:32s}: rata-rata {v.mean():.3f}  CI95 [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    pr = pd.DataFrame(prs, columns=["tanggal", "n_terdaftar", "n_GS_positif", "irisan", "recall_GS", "presisi_GS", "dasar_acak"])
    print(f"  Groundsource: recall rata-rata {pr.recall_GS.mean():.2f} | presisi rata-rata {pr.presisi_GS.mean():.2f} (dasar bila acak ~{pr.dasar_acak.mean():.2f})")
    return pr


within_auc(csvlist, "Seluruh peristiwa CSV 2022-2025 (daftar kecamatan CSV, asal belum terverifikasi)")
within_auc({k: v[1] for k, v in ref.items()}, "Hanya peristiwa dengan daftar dari sumber lain (portal lama BNPB dan berita)")
