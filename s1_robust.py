"""Uji ketahanan label genangan S1: antar-orbit, antar-musim, tanpa sawah, dan tanpa elevasi."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
Z = np.load(W / "data/s1_cell_scene.npz")
flag, valid = Z["flag_open"].astype(float), Z["valid_open"].astype(float)
dates = pd.to_datetime(Z["scene_date"]); orbit = Z["scene_orbit"]
ok = valid >= 30
frac = np.where(ok, flag / np.maximum(valid, 1), np.nan)


def mean_frac(mask):
    f = frac[mask]
    n = ok[mask].sum(0)
    with np.errstate(all="ignore"):
        m = np.nanmean(f, axis=0)
    return np.where(n >= 15, m, np.nan), n


# 1. antar-orbit
f47, n47 = mean_frac(orbit == 47); f149, n149 = mean_frac(orbit == 149)
both = ~np.isnan(f47) & ~np.isnan(f149)
print(f"1. ANTAR-ORBIT: sel layak orbit 47 = {int((~np.isnan(f47)).sum())}, orbit 149 = {int((~np.isnan(f149)).sum())}, keduanya = {int(both.sum())}")
if both.sum() > 30:
    print("   Spearman frekuensi genangan orbit 47 vs 149: %.3f (n=%d)" % (spearmanr(f47[both], f149[both])[0], both.sum()))
    top47 = f47[both] >= np.nanpercentile(f47[both], 90); top149 = f149[both] >= np.nanpercentile(f149[both], 90)
    print("   tumpang tindih desil teratas: %d dari %d sel (acak ~%.1f)" % ((top47 & top149).sum(), top47.sum(), top47.sum() * 0.1))

# 2. antar-musim (basah Okt-Jan vs lainnya)
mo = dates.month.values
fw, nw = mean_frac(np.isin(mo, [10, 11, 12, 1])); fd, nd = mean_frac(~np.isin(mo, [10, 11, 12, 1]))
b = ~np.isnan(fw) & ~np.isnan(fd)
print(f"\n2. ANTAR-MUSIM: sel layak keduanya = {int(b.sum())}")
print("   Spearman musim basah vs musim lain: %.3f" % spearmanr(fw[b], fd[b])[0])
print("   rata-rata frac. genangan: basah %.4f vs lain %.4f" % (np.nanmean(fw[b]), np.nanmean(fd[b])))

# 3. model tanpa sawah / tanpa elevasi
df = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv").merge(pd.read_csv(W / "data/osm_features.csv"), on="grid_id").merge(
    pd.read_csv(W / "data/s1_cell_stats.csv")[["grid_id", "n_eval", "frac_flag_mean"]], on="grid_id")
P = ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90",
     "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
df["landcover_code"] = df.landcover_code.fillna(-1).astype(int)
lc = pd.get_dummies(df.landcover_code, prefix="lc").astype(int); df = pd.concat([df, lc], axis=1)
P += [c for c in lc.columns if c != "lc_50"]
df["log_pop"] = np.log1p(df.population_2020.fillna(0))
R = ["log_pop"] + [c for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]]
for c in R[1:]:
    df["log_" + c] = np.log1p(df[c])
R = ["log_pop"] + ["log_" + c for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]] + ["road_km_per_km2"]
for c in P + R:
    df[c] = df[c].fillna(df[c].median())
e = df[df.n_eval >= 30].reset_index(drop=True)
print("\n3. KOMPOSISI sel layak menurut tutupan lahan dominan (WorldCover):")
print((e.landcover_code.value_counts(normalize=True) * 100).round(1).to_string())
print("   rata-rata frac. genangan per tutupan:", e.groupby("landcover_code").frac_flag_mean.mean().round(4).to_dict())


def cv_auc(sub, cols, label, reps=10):
    y = (sub.frac_flag_mean >= sub.frac_flag_mean.quantile(0.90)).astype(int).values
    xk = sub.longitude * 111.32 * np.cos(np.deg2rad(sub.latitude)); yk = sub.latitude * 110.57
    out = []
    for rep in range(reps):
        g = (((xk + rep * 2.0 % 10) // 10).astype(int).astype(str) + "_" + ((yk + rep * 3.0 % 10) // 10).astype(int).astype(str)).values
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(sub, y, g):
            sc = StandardScaler().fit(sub.iloc[tr][cols])
            m = LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5).fit(sc.transform(sub.iloc[tr][cols]), y[tr])
            oof[te] = sc.transform(sub.iloc[te][cols]) @ m.coef_[0]
        out.append(roc_auc_score(y, oof))
    return np.mean(out), np.std(out)


print("\n   ROC-AUC (CV blok spasial, label = desil teratas S1):")
no_crop = e[e.landcover_code != 40].reset_index(drop=True)
P_noelev = [c for c in P if c != "elevation_m"]
for lab, sub, cols_p in [("semua sel layak", e, P), ("tanpa sel sawah/ladang dominan", no_crop, P),
                         ("semua sel, tanpa elevasi", e, P_noelev)]:
    ap = cv_auc(sub, cols_p, lab); ar = cv_auc(sub, R, lab)
    print(f"   {lab:34s} n={len(sub):5d}  fisik {ap[0]:.3f}±{ap[1]:.3f}   pelaporan {ar[0]:.3f}±{ar[1]:.3f}")

# 4. fisik per fitur tunggal
print("\n4. ROC-AUC fitur tunggal (CV blok), label S1 desil teratas:")
res = []
for c in ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "frac_hand_lt5m", "twi_mean", "log_upa_max", "log_pop"]:
    res.append((c, *cv_auc(e, [c], c, reps=5)))
for c, a, s in sorted(res, key=lambda x: -x[1]):
    print(f"   {c:18s} {a:.3f} ± {s:.3f}")
