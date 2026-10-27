"""Model v2: tambah fitur HAND/TWI (dan hujan spasial bila tersedia), CV blok spasial berulang.

Pemakaian: python run_model2.py <fitur_hidrologi_csv> <out_dir> [fitur_hujan_csv]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, BLOCK_KM, N_FOLDS, N_REPS = 42, 10, 5, 5
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rain_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
if rain_csv is not None:
    df = df.merge(pd.read_csv(rain_csv), on="grid_id", how="left")
y = (df["event_count"] > 0).astype(int).values

# --- fitur ---
base = ["elevation_m", "slope_deg", "distance_river_m"]
hydro = ["hand_mean", "hand_min", "twi_mean", "twi_p90", "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
rain = [c for c in (pd.read_csv(rain_csv).columns if rain_csv else []) if c != "grid_id"]
for c in base + hydro + rain + ["population_2020"]:
    df[c] = df[c].fillna(df[c].median())
df["landcover_code"] = df["landcover_code"].fillna(-1).astype(int)
lc = pd.get_dummies(df["landcover_code"], prefix="lc").astype(int)
lc_cols = list(lc.columns)
df = pd.concat([df, lc], axis=1)

sets = {
    "A. Dasar (+populasi)": base + ["population_2020"] + lc_cols,
    "B. Dasar + HAND/TWI (+populasi)": base + hydro + ["population_2020"] + lc_cols,
    "C. Dasar + HAND/TWI (tanpa populasi)": base + hydro + lc_cols,
}
if rain:
    sets["D. + HAND/TWI + hujan spasial (+populasi)"] = base + hydro + rain + ["population_2020"] + lc_cols
    sets["E. + HAND/TWI + hujan spasial (tanpa populasi)"] = base + hydro + rain + lc_cols

x_km = df["longitude"] * 111.32 * np.cos(np.deg2rad(df["latitude"]))
y_km = df["latitude"] * 110.57


def make_groups(rep):
    ox, oy = (rep * 2.0) % BLOCK_KM, (rep * 3.0) % BLOCK_KM  # geser asal blok tiap ulangan
    return (((x_km + ox) // BLOCK_KM).astype(int).astype(str) + "_" +
            ((y_km + oy) // BLOCK_KM).astype(int).astype(str)).values


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_depth=6,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          min_samples_leaf=40, l2_regularization=5.0,
                                          class_weight="balanced", random_state=SEED)


models = {"LogReg": lr, "RandomForest": rf, "HistGB": hgb}
configs = [("Populasi saja", lr, ["population_2020"])]
for sname, cols in sets.items():
    for mname, mk in models.items():
        configs.append((f"{sname} | {mname}", mk, cols))

recs = []
oof_sum = {}
for rep in range(N_REPS):
    g = make_groups(rep)
    folds = list(GroupKFold(N_FOLDS, shuffle=True, random_state=rep).split(df, y, g))
    for name, mk, cols in configs:
        X = df[cols]
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            p = mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            recs.append(dict(rep=rep, fold=k, model=name,
                             roc=roc_auc_score(y[te], p), pr=average_precision_score(y[te], p)))
        oof_sum[name] = oof_sum.get(name, 0) + oof / N_REPS
    print("ulangan", rep + 1, "selesai", flush=True)

R = pd.DataFrame(recs)
R.to_csv(out / "skor_per_lipatan.csv", index=False)
rep_mean = R.groupby(["model", "rep"])[["roc", "pr"]].mean().reset_index()
summ = rep_mean.groupby("model").agg(roc_mean=("roc", "mean"), roc_sd=("roc", "std"),
                                     pr_mean=("pr", "mean"), pr_sd=("pr", "std")).round(4)
ref = "Populasi saja"
piv = rep_mean.pivot(index="rep", columns="model", values="pr")
summ["dPR_vs_populasi"] = (piv.sub(piv[ref], axis=0)).mean().round(4)
summ["ulangan_lebih_baik"] = (piv.sub(piv[ref], axis=0) > 0).sum().astype(int)
summ = summ.sort_values("pr_mean", ascending=False)
summ.to_csv(out / "ringkasan_skor.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
print(summ)

# --- model terbaik tanpa populasi dan dengan populasi: importance ---
best_nopop = summ[summ.index.str.contains("tanpa populasi")].index[0]
best_pop = summ[~summ.index.str.contains("tanpa populasi") & ~summ.index.str.startswith("Populasi")].index[0]
imps = {}
for lab, name in [("dengan_populasi", best_pop), ("tanpa_populasi", best_nopop)]:
    cols = [c for n, mk, c in configs if n == name][0]
    mk = [mk for n, mk, c in configs if n == name][0]
    g = make_groups(0)
    acc = pd.Series(0.0, index=cols)
    for tr, te in GroupKFold(N_FOLDS, shuffle=True, random_state=0).split(df, y, g):
        m = mk().fit(df[cols].iloc[tr], y[tr])
        r = permutation_importance(m, df[cols].iloc[te], y[te], scoring="average_precision",
                                   n_repeats=5, random_state=SEED, n_jobs=-1)
        acc += pd.Series(r.importances_mean, index=cols) / N_FOLDS
    imps[lab] = (name, acc.sort_values(ascending=False))
    imps[lab][1].to_csv(out / f"importance_{lab}.csv")
    print(lab, name); print(imps[lab][1].head(8).round(4))

# --- prediksi rata-rata OOF dari model terbaik tanpa populasi ---
df["p_oof_tanpa_pop"] = oof_sum[best_nopop]
df["p_oof_dengan_pop"] = oof_sum[best_pop]
df[["grid_id", "longitude", "latitude", "event_count", "p_oof_tanpa_pop", "p_oof_dengan_pop"]].to_csv(
    out / "prediksi_oof_v2.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
s = summ.sort_values("pr_mean")
ax[0].barh(s.index, s.pr_mean, xerr=s.pr_sd, color="#6c757d")
ax[0].axvline(y.mean(), color="r", ls="--"); ax[0].set_title("PR-AUC (CV spasial 5x5)")
ax[0].tick_params(axis="y", labelsize=7)
for a_, lab in zip(ax[1:], ["dengan_populasi", "tanpa_populasi"]):
    n, ser = imps[lab]
    t = ser.head(8)[::-1]
    a_.barh(t.index, t.values, color="#2a6f97"); a_.set_title(f"Importance: {n}", fontsize=9)
plt.tight_layout(); plt.savefig(out / "ringkasan_v2.png", dpi=140)
json.dump(dict(sets=sets, best_pop=best_pop, best_nopop=best_nopop, positives=int(y.sum())),
          open(out / "meta.json", "w"), indent=1)
