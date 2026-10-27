"""Statistik untuk naskah artikel jalur 1: interval kepercayaan bootstrap berblok/berklaster untuk tiga hasil inti.

A. Pembalikan prediktor antar-label (sel layak radar): AUC dan selisih dengan bootstrap blok spasial 10 km.
B. Model kecamatan-hari vs hujan 7 hari: PR-AUC/ROC dengan bootstrap per tanggal (klaster hari).
C. Keterkaitan waktu: persentil hujan 3 hari pada hari mulai kejadian + kontrol bulan-sama.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_article"; OUT.mkdir(exist_ok=True)
FIG = W / "outputs_article"
rng = np.random.default_rng(42)
B = 1000
BLUE, ORG, GRY = "#2a6f97", "#e07a1f", "#8a8f94"
res = {}


def ci(a):
    a = np.asarray(a); a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


# =============== A. pembalikan prediktor ===============
el = pd.read_csv(W / "outputs_s1/sel_layak_pesisir.csv")
LC = [c for c in el.columns if c.startswith("lc_") and c != "lc_50"]
TERR = ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90", "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
TERR = [c for c in TERR if c in el.columns and el[c].nunique() > 1]
RPT = [c for c in ["log_pop", "log_dist_road_m", "log_dist_major_road_m", "log_dist_place_m", "log_dist_town_m", "road_km_per_km2"] if c in el.columns]
sets = {"Terrain + tutupan lahan": TERR + LC, "Terrain/hidrologi": TERR, "Pelaporan": RPT, "Populasi": ["log_pop"]}
print("sel layak:", len(el), "| terrain:", len(TERR), "| LC:", len(LC), "| pelaporan:", len(RPT))
THR = el.frac_flag_mean.quantile(0.90)
labels = {"Radar (Sentinel-1)": (el.frac_flag_mean >= THR).astype(int).values, "Groundsource": (el.event_count > 0).astype(int).values}
xk = el.longitude * 111.32 * np.cos(np.deg2rad(el.latitude)); yk = el.latitude * 110.57


def blocks(rep):
    ox, oy = (rep * 2.0) % 10, (rep * 3.0) % 10
    return (((xk + ox) // 10).astype(int).astype(str) + "_" + ((yk + oy) // 10).astype(int).astype(str)).values


def oof_scores(df, cols, y, reps=10):
    oof = np.zeros(len(df))
    for rep in range(reps):
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(df, y, blocks(rep)[df.index]):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)).fit(df.iloc[tr][cols], y[tr])
            oof[te] += (m.named_steps["standardscaler"].transform(df.iloc[te][cols]) @ m.named_steps["logisticregression"].coef_[0]) / reps
    return oof


subsets = {"Semua sel layak": np.ones(len(el), bool), "Setelah pesisir dan tepi air dikecualikan (> 2 km, air < 2%)": ((el.dist_coast_km > 2) & (el.wc_water_frac < 0.02)).values}
rowsA = []
for sname, mask in subsets.items():
    sub = el[mask].reset_index(drop=True)
    sub_idx = np.where(mask)[0]
    for lname, yfull in labels.items():
        y = yfull[mask]
        if sname != "Semua sel layak" and lname == "Radar (Sentinel-1)":
            pass
        # blok untuk bootstrap (rep 0)
        g0 = blocks(0)[mask]
        ub = np.unique(g0); members = {b: np.where(g0 == b)[0] for b in ub}
        sc = {}
        for pname, cols in sets.items():
            # indeks asli diperlukan oleh blocks(rep)[df.index]; gunakan sub dengan indeks asli
            s2 = el[mask]
            yy = y
            oof = np.zeros(len(s2))
            for rep in range(10):
                gg = blocks(rep)[mask]
                for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(s2, yy, gg):
                    mm = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)).fit(s2.iloc[tr][cols], yy[tr])
                    oof[te] += (mm.named_steps["standardscaler"].transform(s2.iloc[te][cols]) @ mm.named_steps["logisticregression"].coef_[0]) / 10
            sc[pname] = oof
        boot = {p: [] for p in sc}
        for _ in range(B):
            pick = rng.choice(ub, len(ub), replace=True)
            idx = np.concatenate([members[b] for b in pick])
            if y[idx].sum() < 5 or (len(idx) - y[idx].sum()) < 5:
                continue
            for p, o in sc.items():
                boot[p].append(roc_auc_score(y[idx], o[idx]))
        for p, o in sc.items():
            rowsA.append(dict(subset=sname, label=lname, prediktor=p, n=int(len(y)), positif=int(y.sum()), auc=round(roc_auc_score(y, o), 3), ci_lo=ci(boot[p])[0], ci_hi=ci(boot[p])[1]))
        # selisih fisik penuh - pelaporan
        bd = np.array(boot["Terrain + tutupan lahan"]) - np.array(boot["Pelaporan"]); bd2 = np.array(boot["Terrain/hidrologi"]) - np.array(boot["Pelaporan"])
        d = roc_auc_score(y, sc["Terrain + tutupan lahan"]) - roc_auc_score(y, sc["Pelaporan"]); d2 = roc_auc_score(y, sc["Terrain/hidrologi"]) - roc_auc_score(y, sc["Pelaporan"])
        res.setdefault("selisih_AUC", []).append(dict(subset=sname, label=lname, terrain_lc_minus_pelaporan=[round(d, 3)] + ci(bd), terrain_minus_pelaporan=[round(d2, 3)] + ci(bd2)))
    # AUC lintas-label
    yg = labels["Groundsource"][mask]; cont = el.frac_flag_mean.values[mask]
    g0 = blocks(0)[mask]; ub = np.unique(g0); members = {b: np.where(g0 == b)[0] for b in ub}
    bb = []
    for _ in range(B):
        idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
        if yg[idx].sum() >= 5:
            bb.append(roc_auc_score(yg[idx], cont[idx]))
    res.setdefault("auc_radar_untuk_groundsource", []).append(dict(subset=sname, auc=round(roc_auc_score(yg, cont), 3), ci=ci(bb)))
TA = pd.DataFrame(rowsA); TA.to_csv(OUT / "auc_pembalikan_ci.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(TA.to_string(index=False)); print(json.dumps(res, indent=1, ensure_ascii=False))

# =============== B. model vs hujan 7 hari ===============
P = pd.read_csv(W / "outputs_kec/panel_kecamatan_hari.csv", parse_dates=["tanggal"])
RAIN = ["r1", "r2", "r3", "r5", "r7", "r14", "r30_prev", "max3", "wet_days7"]
FEATS = RAIN + ["sin", "cos", "elev", "hand", "riv", "coast", "road", "built", "slope", "log_popdens", "border"]
d = P[P.tahun >= 2021].reset_index(drop=True)
mods = {"Hujan 7 hari saja": ["r7"], "Hujan 3 hari saja": ["r3"], "Model final (hujan + musim + kecamatan)": FEATS}
oofs = {}
for nm, cols in mods.items():
    o = np.zeros(len(d))
    for ty in range(2021, 2026):
        tr, te = d.tahun != ty, d.tahun == ty
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)).fit(d.loc[tr, cols], d.y[tr])
        o[te] = m.predict_proba(d.loc[te, cols])[:, 1]
    oofs[nm] = o
dates = d.tanggal.values; ud = np.unique(dates); by_date = {x: np.where(dates == x)[0] for x in ud}
yv = d.y.values
bootm = {n: {"roc": [], "pr": []} for n in oofs}
for _ in range(B):
    idx = np.concatenate([by_date[x] for x in rng.choice(ud, len(ud), replace=True)])
    if yv[idx].sum() < 20:
        continue
    for n, o in oofs.items():
        bootm[n]["roc"].append(roc_auc_score(yv[idx], o[idx])); bootm[n]["pr"].append(average_precision_score(yv[idx], o[idx]))
rowsB = []
for n, o in oofs.items():
    rowsB.append(dict(model=n, roc=round(roc_auc_score(yv, o), 3), roc_ci=ci(bootm[n]["roc"]), pr=round(average_precision_score(yv, o), 3), pr_ci=ci(bootm[n]["pr"])))
fin, r7 = "Model final (hujan + musim + kecamatan)", "Hujan 7 hari saja"
dpr = np.array(bootm[fin]["pr"]) - np.array(bootm[r7]["pr"]); droc = np.array(bootm[fin]["roc"]) - np.array(bootm[r7]["roc"])
res["model_vs_hujan7"] = dict(selisih_PR=[round(average_precision_score(yv, oofs[fin]) - average_precision_score(yv, oofs[r7]), 4)] + ci(dpr),
                                selisih_ROC=[round(roc_auc_score(yv, oofs[fin]) - roc_auc_score(yv, oofs[r7]), 3)] + ci(droc), n=len(d), positif=int(yv.sum()), hari_unik=int(len(ud)), prevalensi=round(float(yv.mean()), 4))
TB = pd.DataFrame(rowsB); TB.to_csv(OUT / "model_vs_hujan_ci.csv", index=False)
print(TB.to_string(index=False)); print(res["model_vs_hujan7"])

# =============== C. keterkaitan waktu ===============
rain = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/raw/curah_hujan/nasa_power_precipitation_sambas_2010_2025.csv"))
rain["tanggal"] = pd.to_datetime(rain.tanggal.astype(str), format="%Y%m%d"); rr = rain.set_index("tanggal").precipitation_mm_day.asfreq("D").fillna(0)
ev = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/groundsource_sambas_2010_2025.csv"), usecols=["start_date"])
days = pd.to_datetime(ev.start_date).dt.normalize().drop_duplicates()
clim = pd.DataFrame({"r3": rr.rolling(3).sum(), "r7": rr.rolling(7).sum()}).dropna(); clim = clim[clim.index >= "2012-01-01"]
out = {}
for k in ("r3", "r7"):
    v = clim[k].reindex(days).dropna(); pct = np.array([(clim[k] <= x).mean() for x in v])
    out[k] = dict(n_hari=int(len(v)), median_persentil=round(float(np.median(pct) * 100)), frac_ge_p90=round(float((pct >= .9).mean()), 3), frac_ge_p95=round(float((pct >= .95).mean()), 3))
    obs = v.mean(); null = []
    pool = {m: clim.index[clim.index.month == m] for m in range(1, 13)}
    for _ in range(2000):
        pick = [rng.choice(pool[x.month]) for x in v.index]
        null.append(clim.loc[pick, k].mean())
    out[k].update(rata_kejadian=round(float(obs), 1), rata_acak=round(float(np.mean(null)), 1), acak_ci=[round(float(np.percentile(null, 2.5)), 1), round(float(np.percentile(null, 97.5)), 1)], p=round(float(np.mean(np.array(null) >= obs)), 4))
    if k == "r3":
        pct3 = pct
res["keterkaitan_waktu"] = out; print(json.dumps(out, indent=1))

# DIBI
dibi = pd.read_csv(W / "dibi/dibi_sambas_banjir_tgl.csv", parse_dates=["tgl"]).drop_duplicates("tgl")
Y = P.pivot(index="tanggal", columns="kec", values="y"); Y.index = pd.to_datetime(Y.index)
dd = dibi[(dibi.tgl >= "2016-01-01") & (dibi.tgl <= "2025-12-31")]
hit = [(Y.loc[t - pd.Timedelta(days=1): t + pd.Timedelta(days=1)].max() >= 1).sum() for t in dd.tgl]
base = np.mean([(Y.loc[x - pd.Timedelta(days=1): x + pd.Timedelta(days=1)].max() >= 1).sum() >= 1 for x in pd.date_range("2021-01-02", "2025-12-30")])
res["dibi"] = dict(n=int(len(dd)), positif=int(np.sum(np.array(hit) >= 1)), ge3=int(np.sum(np.array(hit) >= 3)), peluang_dasar=round(float(base), 2))
# interval Wilson untuk proporsi
from math import sqrt
k_, n_ = res["dibi"]["positif"], res["dibi"]["n"]; z = 1.96; ph = k_ / n_
lo = (ph + z * z / (2 * n_) - z * sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_))) / (1 + z * z / n_); hi = (ph + z * z / (2 * n_) + z * sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_))) / (1 + z * z / n_)
res["dibi"]["wilson_ci"] = [round(lo, 2), round(hi, 2)]; print(res["dibi"])
json.dump(res, open(OUT / "ringkasan_statistik.json", "w"), indent=1, ensure_ascii=False)

# =============== Gambar 2 dan 3 ===============
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
ax[0].hist(pct3 * 100, bins=10, range=(0, 100), color=BLUE); ax[0].axhline(len(pct3) / 10, color="r", ls="--", lw=1); ax[0].text(2, len(pct3) / 10 + 1.2, "harapan jika tak ada hubungan", color="r", fontsize=7)
ax[0].set_xlabel("Persentil hujan 3 hari pada hari mulai kejadian"); ax[0].set_ylabel("Jumlah hari kejadian"); ax[0].set_title(f"(a) Hari kejadian cenderung hujan lebat\n(n = {len(pct3)} hari unik)")
d21 = P[P.tanggal >= "2021-01-01"].copy(); bins = [-1, 25, 50, 100, 150, 200, 1e9]; lab = ["<25", "25–50", "50–100", "100–150", "150–200", "≥200"]
d21["bin"] = pd.cut(d21.r7, bins, labels=lab); g = d21.groupby("bin", observed=True).y.mean() * 100
ax[1].bar(g.index.astype(str), g.values, color=ORG); ax[1].set_xlabel("Hujan 7 hari (mm)"); ax[1].set_ylabel("% kecamatan-hari berlaporan"); ax[1].set_title("(b) Laju laporan menurut hujan 7 hari\n(2021–2025)")
fig.tight_layout(); fig.savefig(FIG / "naskah_fig2_waktu.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
colors = {"Terrain + tutupan lahan": BLUE, "Terrain/hidrologi": "#5b9bc4", "Pelaporan": ORG, "Populasi": GRY}
for a, (sname, title) in zip(ax, [("Semua sel layak", "(a) Semua sel layak (n = %d)"), (list(subsets)[1], "(b) Setelah pesisir dan tepi air\ndikecualikan (n = %d)")]):
    t = TA[TA.subset == sname]; n_ = int(t.n.iloc[0]); w = 0.2
    for j, (p_, col) in enumerate(colors.items()):
        v = [t[(t.label == lb) & (t.prediktor == p_)].iloc[0] for lb in ["Radar (Sentinel-1)", "Groundsource"]]
        a.bar(np.arange(2) + (j - 1.5) * w, [x.auc for x in v], w, yerr=[[x.auc - x.ci_lo for x in v], [x.ci_hi - x.auc for x in v]], color=col, label=p_, error_kw=dict(lw=0.8, capsize=1.5))
    a.axhline(0.5, color="k", ls="--", lw=1); a.set_xticks([0, 1]); a.set_xticklabels(["Label radar\n(Sentinel-1)", "Label berita\n(Groundsource)"]); a.set_ylim(0.3, 0.95)
    a.set_title(title % n_)
ax[0].set_ylabel("ROC-AUC (CV blok spasial; 95% CI)"); ax[0].legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "naskah_fig3_pembalikan.png", dpi=200); plt.close(fig)
print("selesai")
