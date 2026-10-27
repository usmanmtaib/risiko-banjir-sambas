"""Bandingkan prediktor fisik vs pelaporan untuk label genangan Sentinel-1 (independen) dan label Groundsource.
Hanya sel 'layak' (cukup piksel lahan terbuka valid pada adegan S1), agar kedua label dinilai pada sel yang sama."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_s1"; OUT.mkdir(exist_ok=True)
N_FOLDS, N_REPS, BLOCK_KM = 5, 10, 10

df = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv").merge(
    pd.read_csv(W / "data/osm_features.csv"), on="grid_id").merge(
    pd.read_csv(W / "data/s1_cell_stats.csv")[["grid_id", "n_eval", "flood_freq", "frac_flag_mean", "pix_open_max"]], on="grid_id")
P = ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90",
     "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
df["landcover_code"] = df.landcover_code.fillna(-1).astype(int)
lc = pd.get_dummies(df.landcover_code, prefix="lc").astype(int)
df = pd.concat([df, lc], axis=1)
P += [c for c in lc.columns if c != "lc_50"]
df["log_pop"] = np.log1p(df.population_2020.fillna(0)); df["built_up"] = df.get("lc_50", 0)
R = ["log_pop", "built_up"]
for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]:
    df["log_" + c] = np.log1p(df[c]); R.append("log_" + c)
R.append("road_km_per_km2")
for c in P + R:
    df[c] = df[c].fillna(df[c].median())

e = df[df.n_eval >= 30].reset_index(drop=True)
P = [c for c in P if e[c].nunique() > 1]
R = [c for c in R if e[c].nunique() > 1]
print("sel layak:", len(e))

targets = {
    "S1: desil teratas frekuensi genangan": (e.frac_flag_mean >= e.frac_flag_mean.quantile(0.90)).astype(int).values,
    "S1: frekuensi genangan > 0,1%": (e.frac_flag_mean > 0.001).astype(int).values,
    "Groundsource: ada laporan": (e.event_count > 0).astype(int).values,
}
for k, v in targets.items():
    print(f"  {k}: positif {int(v.sum())} ({v.mean():.3f})")

x_km = e.longitude * 111.32 * np.cos(np.deg2rad(e.latitude)); y_km = e.latitude * 110.57


def groups(rep):
    ox, oy = (rep * 2.0) % BLOCK_KM, (rep * 3.0) % BLOCK_KM
    return (((x_km + ox) // BLOCK_KM).astype(int).astype(str) + "_" + ((y_km + oy) // BLOCK_KM).astype(int).astype(str)).values


def fit(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5).fit(sc.transform(X), y)


rows = []
cont = e.frac_flag_mean.values
for tname, y in targets.items():
    for rep in range(N_REPS):
        folds = list(GroupKFold(N_FOLDS, shuffle=True, random_state=rep).split(e, y, groups(rep)))
        for mname, cols in [("Pelaporan saja", R), ("Fisik saja", P), ("Gabungan", P + R), ("Populasi saja", ["log_pop"]),
                            ("HAND saja", ["hand_mean", "frac_hand_lt5m", "frac_hand_lt2m"])]:
            oof = np.zeros(len(y))
            for tr, te in folds:
                if y[tr].sum() < 5:
                    oof[te] = 0; continue
                sc, m = fit(e.loc[tr, cols], y[tr])
                oof[te] = sc.transform(e.loc[te, cols]) @ m.coef_[0]
            rows.append(dict(target=tname, model=mname, rep=rep, roc=roc_auc_score(y, oof),
                             pr=average_precision_score(y, oof), rho_kontinu=spearmanr(oof, cont)[0]))
    print(tname, "selesai", flush=True)

R_ = pd.DataFrame(rows)
S = R_.groupby(["target", "model"]).agg(roc=("roc", "mean"), roc_sd=("roc", "std"), pr=("pr", "mean"),
                                         pr_sd=("pr", "std"), rho_s1_kontinu=("rho_kontinu", "mean")).round(3)
S.to_csv(OUT / "kinerja_label_s1_vs_groundsource.csv")
pd.set_option("display.width", 200)
print(S.to_string())

# hubungan langsung antara kedua label
yg = targets["Groundsource: ada laporan"]
print("\nROC-AUC frekuensi genangan S1 (kontinu) untuk memprediksi laporan Groundsource: %.3f" % roc_auc_score(yg, cont))
print("Spearman(frac_flag_mean, log_pop) = %.3f ; Spearman(frac_flag_mean, hand_mean) = %.3f ; Spearman(frac_flag_mean, elevation) = %.3f" % (
    spearmanr(cont, e.log_pop)[0], spearmanr(cont, e.hand_mean)[0], spearmanr(cont, e.elevation_m)[0]))
e[["grid_id", "longitude", "latitude", "event_count", "frac_flag_mean", "flood_freq", "n_eval"]].to_csv(OUT / "sel_layak_s1.csv", index=False)
