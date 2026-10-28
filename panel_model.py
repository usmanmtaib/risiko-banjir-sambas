"""Panel kecamatan-hari Sambas: label dari Groundsource, fitur hujan dan kecamatan, validasi lintas-tahun."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_kec"; OUT.mkdir(exist_ok=True)
SEED = 42

# ---------------- 1. label ----------------
ov = pd.read_csv(W / "kec/event_kec_overlay.csv", parse_dates=["s"])
pos_B = ov[ov.f_ev >= 0.5][["kec", "s"]].drop_duplicates().assign(y=1)       # kejadian terutama di dalam kecamatan
pos_A = ov[ov.f_kec >= 0.5][["kec", "s"]].drop_duplicates().assign(yA=1)     # kejadian menutupi >=50% kecamatan

# ---------------- 2. hujan ----------------
R = pd.read_csv(W / "kec/hujan_kec_power.csv", parse_dates=["tanggal"]).sort_values(["kec", "tanggal"])
g = R.groupby("kec").hujan_mm
for k in (1, 2, 3, 5, 7, 14, 30):
    R[f"r{k}"] = g.transform(lambda s, k=k: s.rolling(k).sum())
R["max3"] = g.transform(lambda s: s.rolling(3).max())
R["wet_days7"] = g.transform(lambda s: (s >= 20).rolling(7).sum())
R["r30_prev"] = R.r30 - R.r7  # kelembapan awal (hari 8-30)
doy = R.tanggal.dt.dayofyear
R["sin"], R["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
R = R.dropna(subset=["r30"])

# ---------------- 3. fitur statis kecamatan ----------------
k = gpd.read_file(W / "kec/kecamatan_sambas.geojson").set_crs(4326, allow_override=True)[["NAMOBJ", "geometry"]].rename(columns={"NAMOBJ": "kec"})
gr = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv").merge(pd.read_csv(W / "data/osm_features.csv"), on="grid_id").merge(
    pd.read_csv(W / "data/dist_coast.csv"), on="grid_id")
gg = gpd.GeoDataFrame(gr, geometry=gpd.points_from_xy(gr.longitude, gr.latitude), crs=4326)
gg = gpd.sjoin(gg, k, how="inner", predicate="within")
area = k.to_crs(32649).set_index("kec").geometry.area / 1e6
st = gg.groupby("kec").agg(elev=("elevation_m", "mean"), hand=("hand_mean", "mean"), riv=("distance_river_m", "mean"),
                           coast=("dist_coast_km", "mean"), road=("road_km_per_km2", "mean"), pop=("population_2020", "sum"),
                           built=("landcover_code", lambda s: (s == 50).mean()), slope=("slope_deg", "mean"))
st["area_km2"] = area; st["log_popdens"] = np.log1p(st["pop"] / st.area_km2)
st["border"] = st.index.isin(["Sajingan Besar", "Paloh"]).astype(int)
STATIC = ["elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border"]
st.to_csv(OUT / "fitur_kecamatan.csv")

# ---------------- 4. panel ----------------
P = R.merge(pos_B.rename(columns={"s": "tanggal"}), on=["kec", "tanggal"], how="left").merge(
    pos_A.rename(columns={"s": "tanggal"}), on=["kec", "tanggal"], how="left")
P["y"] = P.y.fillna(0).astype(int); P["yA"] = P.yA.fillna(0).astype(int)
P = P.merge(st[STATIC], left_on="kec", right_index=True); P["tahun"] = P.tanggal.dt.year
P.to_csv(OUT / "panel_kecamatan_hari.csv", index=False)
print("panel:", len(P), "| positif B:", int(P.y.sum()), f"({P.y.mean():.4f}) | positif A:", int(P.yA.sum()))
print(P.groupby("tahun").y.agg(["sum", "mean"]).round(4).T.to_string())

RAIN = ["r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7"]
SEAS = ["sin", "cos"]
sets = {"Hujan saja": RAIN, "Hujan + musim": RAIN + SEAS, "Hujan + musim + kecamatan": RAIN + SEAS + STATIC}


def mk_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3))


def mk_hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=60,
                                          l2_regularization=5.0, class_weight="balanced", random_state=SEED)


def recall_at(y, p, rate):
    n = max(int(len(p) * rate), 1)
    top = np.argsort(-p)[:n]
    return y[top].sum() / max(y.sum(), 1), y[top].mean()


def evaluate(train_years, label, df, ycol="y"):
    rows, oofs = [], {}
    for sname, cols in sets.items():
        for mname, mk in [("LogReg", mk_lr), ("HistGB", mk_hgb)]:
            if sname == "Hujan saja" and mname == "HistGB":
                continue
            preds = []
            for ty in train_years:
                tr = df[(df.tahun != ty) & (df.tahun.isin(train_years))]; te = df[df.tahun == ty]
                if label == "ke-depan":
                    tr = df[(df.tahun < ty) & (df.tahun >= train_years[0] - 0)]
                if tr[ycol].sum() < 20 or te[ycol].sum() < 5:
                    continue
                m = mk().fit(tr[cols], tr[ycol])
                p = m.predict_proba(te[cols])[:, 1]
                preds.append(pd.DataFrame({"tahun": ty, "y": te[ycol].values, "p": p, "kec": te.kec.values, "tanggal": te.tanggal.values}))
            if not preds:
                continue
            pr = pd.concat(preds)
            per_year = pr.groupby("tahun").apply(lambda d: pd.Series({"roc": roc_auc_score(d.y, d.p), "pr": average_precision_score(d.y, d.p)}))
            r1 = recall_at(pr.y.values, pr.p.values, 0.01); r5 = recall_at(pr.y.values, pr.p.values, 0.05)
            rows.append(dict(skema=label, set=sname, model=mname, n=len(pr), positif=int(pr.y.sum()), prevalensi=round(pr.y.mean(), 4),
                             roc_auc=round(roc_auc_score(pr.y, pr.p), 3), pr_auc=round(average_precision_score(pr.y, pr.p), 3),
                             roc_per_tahun_sd=round(per_year.roc.std(), 3), recall_top1=round(r1[0], 3), presisi_top1=round(r1[1], 3),
                             recall_top5=round(r5[0], 3), presisi_top5=round(r5[1], 3)))
            oofs[(sname, mname)] = pr
    return pd.DataFrame(rows), oofs


YRS = [2021, 2022, 2023, 2024, 2025]
d2021 = P[P.tahun >= 2021]
res1, oof1 = evaluate(YRS, "lintas-tahun (leave-one-year-out 2021-2025)", d2021)
res2, oof2 = evaluate([2023, 2024, 2025], "ke-depan (latih < tahun uji, 2021+)", d2021)
res3, _ = evaluate(YRS, "lintas-tahun, label ketat A", d2021, ycol="yA")
res = pd.concat([res1, res2, res3])
res.to_csv(OUT / "kinerja_kecamatan_hari.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
print("\n", res.to_string(index=False))

# ---------------- 5. pentingnya fitur (permutation, model terbaik, LOYO) ----------------
best = ("Hujan + musim + kecamatan", "LogReg")
imp = pd.Series(0.0, index=RAIN + SEAS + STATIC); n = 0
for ty in YRS:
    tr, te = d2021[d2021.tahun != ty], d2021[d2021.tahun == ty]
    m = mk_lr().fit(tr[imp.index], tr.y)
    r = permutation_importance(m, te[imp.index], te.y, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp += r.importances_mean; n += 1
imp = (imp / n).sort_values(ascending=False); imp.to_csv(OUT / "importance_logreg.csv"); print("\nImportance (penurunan PR-AUC):"); print(imp.round(4).head(10).to_string())

# ---------------- 6. ambang hujan per kecamatan ----------------
thr = []
for kc, d in d2021.groupby("kec"):
    if d.y.sum() < 5:
        continue
    for var in ("r3", "r7"):
        q = d[var].quantile([0.5, 0.9]).values
        pe = d[d.y == 1][var]
        thr.append(dict(kec=kc, var=var, n_kejadian=int(d.y.sum()), median_hujan_kejadian=round(pe.median(), 1),
                        p10_hujan_kejadian=round(pe.quantile(0.1), 1), p90_hujan_semua_hari=round(q[1], 1)))
pd.DataFrame(thr).to_csv(OUT / "ambang_hujan_kecamatan.csv", index=False)

# ---------------- 7. K-Means zona (hot spot) ----------------
yrs = (2025 - 2021 + 1)
prof = st[STATIC].copy()
prof["laporan_per_tahun"] = d2021.groupby("kec").y.sum() / yrs
prof["laporan_per_100km2"] = prof.laporan_per_tahun / st.area_km2 * 100
prof["hari_hujan_pemicu"] = d2021[d2021.y == 1].groupby("kec").r3.median()
prof = prof.fillna(prof.median())
Xz = StandardScaler().fit_transform(np.log1p(prof[["laporan_per_tahun", "laporan_per_100km2", "log_popdens", "elev", "coast", "hand", "riv", "road"]].clip(lower=0)))
sil = {kk: silhouette_score(Xz, KMeans(kk, n_init=20, random_state=SEED).fit_predict(Xz)) for kk in (3, 4, 5)}
kbest = max(sil, key=sil.get); print("\nK-Means kecamatan: silhouette", {a: round(b, 3) for a, b in sil.items()}, "-> k =", kbest)
prof["zona"] = KMeans(kbest, n_init=20, random_state=SEED).fit_predict(Xz)
prof.round(3).to_csv(OUT / "zona_kecamatan.csv")
print(prof.groupby("zona")[["laporan_per_tahun", "laporan_per_100km2", "log_popdens", "elev", "coast"]].mean().round(2).to_string())
print({z: list(prof.index[prof.zona == z]) for z in sorted(prof.zona.unique())})

# ---------------- 8. gambar ----------------
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
a = ax[0]; s = res1.assign(lab=res1.set + " | " + res1.model).sort_values("pr_auc")
a.barh(s.lab, s.pr_auc, color="#6c757d"); a.axvline(res1.prevalensi.iloc[0], color="r", ls="--", label="prevalensi"); a.legend()
a.set_title("PR-AUC kecamatan-hari (leave-one-year-out 2021-2025)"); a.tick_params(axis="y", labelsize=8)
a = ax[1]; a.barh(imp.head(10)[::-1].index, imp.head(10)[::-1].values, color="#2a6f97"); a.set_title("Pentingnya fitur (regresi logistik, permutation)")
a = ax[2]
pr = oof1[best]; q = pr.assign(bin=pd.qcut(pr.p.rank(method="first"), 10, labels=False) + 1).groupby("bin").y.mean() * 100
a.bar(q.index, q.values, color="#e07a1f"); a.set_xlabel("Desil skor risiko (10 = tertinggi)"); a.set_ylabel("% kecamatan-hari berlaporan")
a.axhline(pr.y.mean() * 100, color="k", ls="--"); a.set_title("Laju laporan per desil skor (regresi logistik penuh)")
plt.tight_layout(); plt.savefig(OUT / "ringkasan_kecamatan.png", dpi=130)
pr.to_csv(OUT / "prediksi_oof_terbaik.csv", index=False)
