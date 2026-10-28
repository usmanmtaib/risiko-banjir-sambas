"""Koreksi bias pelaporan: pisahkan faktor fisik (P) dari faktor pelaporan (R).

Asumsi identifikasi (HARUS dinyatakan di laporan):
  logit P(ada laporan di grid) = f(fisik) + h(pelaporan)
  - Populasi, aksesibilitas, dan permukiman memengaruhi peluang DIBERITAKAN (R).
  - Elevasi, HAND, TWI, sungai, dan tutupan lahan non-permukiman memengaruhi peluang BANJIR (P).
  - Pelanggaran: urbanisasi dapat memengaruhi banjir itu sendiri; fisik dan R saling berkorelasi.
Tanpa label independen, hasilnya adalah asosiasi bersyarat, bukan bukti kausal.

Pemakaian: python bias_model.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_bias"; OUT.mkdir(exist_ok=True)
SEED, BLOCK_KM, N_FOLDS, N_REPS, N_BOOT = 42, 10, 5, 5, 200
log = {}

df = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv")
osm_f = W / "data/osm_features.csv"
has_osm = osm_f.exists()
if has_osm:
    df = df.merge(pd.read_csv(osm_f), on="grid_id", how="left")
log["osm_tersedia"] = has_osm
y = (df.event_count > 0).astype(int).values

# ---- kolom fisik (P) dan pelaporan (R) ----
P = ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90",
     "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
df["landcover_code"] = df.landcover_code.fillna(-1).astype(int)
lc = pd.get_dummies(df.landcover_code, prefix="lc").astype(int)
df = pd.concat([df, lc], axis=1)
P += [c for c in lc.columns if c != "lc_50"]  # lc_50 = lahan terbangun -> sisi pelaporan
df["log_pop"] = np.log1p(df.population_2020.fillna(0))
df["built_up"] = df.get("lc_50", 0)
R = ["log_pop", "built_up"]
if has_osm:
    for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]:
        df["log_" + c] = np.log1p(df[c])
        R.append("log_" + c)
    R.append("road_km_per_km2")
for c in P + R:
    df[c] = df[c].fillna(df[c].median())
P = [c for c in P if df[c].nunique() > 1]
log["kolom_fisik"], log["kolom_pelaporan"] = P, R

x_km = df.longitude * 111.32 * np.cos(np.deg2rad(df.latitude))
y_km = df.latitude * 110.57


def groups(rep):
    ox, oy = (rep * 2.0) % BLOCK_KM, (rep * 3.0) % BLOCK_KM
    return (((x_km + ox) // BLOCK_KM).astype(int).astype(str) + "_" +
            ((y_km + oy) // BLOCK_KM).astype(int).astype(str)).values


def fit_lr(X, yy):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5).fit(sc.transform(X), yy)
    return sc, m


def lin(sc, m, X, cols=None):
    Z = sc.transform(X)
    return Z @ m.coef_[0] if cols is None else Z[:, cols] @ m.coef_[0][cols]


# ---- 1. diagnostik temporal ----
yr = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/ringkasan_tahunan_groundsource_sambas.csv"))
rain = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/curah_hujan_tahunan_sambas_2010_2025.csv"))
tt = pd.DataFrame({"tahun": range(2012, 2026)}).merge(yr[["tahun", "jumlah_kejadian"]], how="left").fillna(0)
tt = tt.merge(rain.rename(columns={"tanggal": "tahun", "sum": "hujan_tahunan_mm", "max": "hujan_harian_maks_mm"})[
    ["tahun", "hujan_tahunan_mm", "hujan_harian_maks_mm"]], on="tahun")
ly = np.log1p(tt.jumlah_kejadian)
slope, icpt = np.polyfit(tt.tahun, ly, 1)
resid = ly - (icpt + slope * tt.tahun)
log["temporal"] = {
    "n_tahun": len(tt),
    "spearman_kejadian_vs_tahun": round(float(spearmanr(tt.jumlah_kejadian, tt.tahun)[0]), 3),
    "spearman_kejadian_vs_hujan_tahunan": round(float(spearmanr(tt.jumlah_kejadian, tt.hujan_tahunan_mm)[0]), 3),
    "spearman_kejadian_vs_hujan_harian_maks": round(float(spearmanr(tt.jumlah_kejadian, tt.hujan_harian_maks_mm)[0]), 3),
    "tren_log_per_tahun": round(float(slope), 3), "faktor_kenaikan_per_tahun": round(float(np.exp(slope)), 2),
    "spearman_residu_tren_vs_hujan_tahunan": round(float(spearmanr(resid, tt.hujan_tahunan_mm)[0]), 3),
}
tt.to_csv(OUT / "diagnostik_temporal.csv", index=False)
print(json.dumps(log["temporal"], indent=1))

# ---- 2. CV blok spasial berulang ----
recs, strat = [], []
oof_acc = {k: np.zeros(len(y)) for k in ["pR", "pP", "pJ", "fP"]}
corr = []
for rep in range(N_REPS):
    folds = list(GroupKFold(N_FOLDS, shuffle=True, random_state=rep).split(df, y, groups(rep)))
    oof = {k: np.zeros(len(y)) for k in ["pR", "pP", "pJ", "fP"]}
    for tr, te in folds:
        for key, cols in [("pR", R), ("pP", P), ("pJ", P + R)]:
            sc, m = fit_lr(df.loc[df.index[tr], cols], y[tr])
            oof[key][te] = lin(sc, m, df.loc[df.index[te], cols])
            if key == "pJ":
                oof["fP"][te] = lin(sc, m, df.loc[df.index[te], cols], cols=list(range(len(P))))
    for key, lab in [("pR", "Pelaporan saja"), ("pP", "Fisik saja"), ("pJ", "Gabungan fisik+pelaporan")]:
        recs.append(dict(rep=rep, model=lab, roc=roc_auc_score(y, oof[key]), pr=average_precision_score(y, oof[key])))
    # strata berdasarkan propensitas pelaporan (OOF pelaporan-saja)
    q = pd.qcut(oof["pR"], 3, labels=["rendah", "sedang", "tinggi"])
    for s in ["rendah", "sedang", "tinggi"]:
        i = np.asarray(q == s)
        if y[i].sum() >= 5:
            strat.append(dict(rep=rep, strata_pelaporan=s, n=int(i.sum()), positif=int(y[i].sum()),
                              laju=y[i].mean(),
                              auc_fisik_saja=roc_auc_score(y[i], oof["pP"][i]),
                              auc_terkoreksi=roc_auc_score(y[i], oof["fP"][i])))
    pop = df.log_pop.values
    corr.append(dict(rep=rep, rho_pelaporan_pop=spearmanr(oof["pR"], pop)[0],
                     rho_fisiksaja_pop=spearmanr(oof["pP"], pop)[0],
                     rho_terkoreksi_pop=spearmanr(oof["fP"], pop)[0]))
    for k in oof_acc:
        oof_acc[k] += oof[k] / N_REPS
    print("ulangan", rep + 1, flush=True)

M = pd.DataFrame(recs).groupby("model").agg(roc=("roc", "mean"), roc_sd=("roc", "std"),
                                            pr=("pr", "mean"), pr_sd=("pr", "std")).round(4)
M.to_csv(OUT / "kinerja_cv_spasial.csv"); print(M)
S = pd.DataFrame(strat).groupby("strata_pelaporan").agg(
    n=("n", "mean"), positif=("positif", "mean"), laju=("laju", "mean"),
    auc_fisik_saja=("auc_fisik_saja", "mean"), auc_fisik_sd=("auc_fisik_saja", "std"),
    auc_terkoreksi=("auc_terkoreksi", "mean"), ulangan_valid=("rep", "count")).round(3)
S = S.reindex(["rendah", "sedang", "tinggi"]); S.to_csv(OUT / "auc_per_strata_pelaporan.csv"); print(S)
C = pd.DataFrame(corr).mean(numeric_only=True).round(3); log["spearman_dengan_log_populasi"] = C.drop("rep").to_dict(); print(C)

# ---- 3. koefisien gabungan + bootstrap blok ----
Xall = df[P + R]
sc, m = fit_lr(Xall, y)
coef = pd.Series(m.coef_[0], index=P + R)
g0 = groups(0)
ub = np.unique(g0); rng = np.random.default_rng(SEED)
members = {b: np.where(g0 == b)[0] for b in ub}
boots = []
for _ in range(N_BOOT):
    pick = rng.choice(ub, len(ub), replace=True)
    idx = np.concatenate([members[b] for b in pick])
    if y[idx].sum() < 20:
        continue
    s2, m2 = fit_lr(Xall.iloc[idx], y[idx])
    boots.append(m2.coef_[0])
B = np.array(boots)
ct = pd.DataFrame({"koef_std": coef, "p2.5": np.percentile(B, 2.5, axis=0), "p97.5": np.percentile(B, 97.5, axis=0)},
                  index=P + R).round(3)
ct["kelompok"] = ["fisik"] * len(P) + ["pelaporan"] * len(R)
ct["CI_tidak_memuat_nol"] = (ct["p2.5"] > 0) | (ct["p97.5"] < 0)
ct.to_csv(OUT / "koefisien_gabungan.csv"); print(ct.to_string())

# ---- 4. keluaran peta/ranking ----
pct = lambda a: pd.Series(a).rank(pct=True).values
out = df[["grid_id", "longitude", "latitude", "event_count"]].copy()
out["skor_pelaporan_pct"] = pct(oof_acc["pR"])
out["skor_fisik_saja_pct"] = pct(oof_acc["pP"])
out["skor_terkoreksi_pct"] = pct(oof_acc["fP"])
out.to_csv(OUT / "prediksi_terkoreksi.csv", index=False)

fig, ax = plt.subplots(2, 2, figsize=(14, 12))
for a, col, ttl in [(ax[0, 0], "skor_pelaporan_pct", "Propensitas pelaporan (persentil, OOF)"),
                    (ax[0, 1], "skor_terkoreksi_pct", "Kerawanan fisik terkoreksi (persentil, OOF)")]:
    sc_ = a.scatter(out.longitude, out.latitude, c=out[col], s=5, cmap="viridis")
    ev = out[out.event_count > 0]; a.scatter(ev.longitude, ev.latitude, s=2, c="r", alpha=.5)
    a.set_title(ttl); a.set_aspect("equal"); plt.colorbar(sc_, ax=a, shrink=.7)
a = ax[1, 0]
w = 0.38; xs = np.arange(3)
a.bar(xs - w / 2, S.auc_fisik_saja, w, yerr=S.auc_fisik_sd, label="Fisik saja", color="#2a6f97")
a.bar(xs + w / 2, S.auc_terkoreksi, w, label="Fisik terkoreksi", color="#e07a1f")
a.axhline(0.5, color="k", ls="--"); a.set_xticks(xs); a.set_xticklabels(S.index)
a.set_ylim(0.4, 0.85); a.set_xlabel("Strata propensitas pelaporan"); a.set_ylabel("ROC-AUC dalam strata"); a.legend()
a.set_title("Apakah faktor fisik membedakan banjir pada tingkat pelaporan setara?")
a = ax[1, 1]
ctp = ct.sort_values("koef_std")
cols_ = ctp.kelompok.map({"fisik": "#2a6f97", "pelaporan": "#e07a1f"})
a.barh(ctp.index, ctp.koef_std, color=cols_, xerr=[ctp.koef_std - ctp["p2.5"], ctp["p97.5"] - ctp.koef_std], ecolor="grey")
a.axvline(0, color="k"); a.tick_params(axis="y", labelsize=7)
a.set_title("Koefisien gabungan terstandar (biru fisik, oranye pelaporan; CI 95% bootstrap blok)")
plt.tight_layout(); plt.savefig(OUT / "ringkasan_bias.png", dpi=130)
json.dump(log, open(OUT / "ringkasan.json", "w"), indent=1, default=str)
