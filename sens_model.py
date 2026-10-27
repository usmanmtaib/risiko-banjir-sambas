"""Sensitivitas ambang aliran HAND terhadap kinerja model (CV blok spasial 5x5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
SEED, BLOCK_KM, N_FOLDS, N_REPS = 42, 10, 5, 5
THR = [0.5, 1.0, 2.0, 5.0, 10.0]

df = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv")
y = (df.event_count > 0).astype(int).values
base = ["elevation_m", "slope_deg", "distance_river_m"]
tw = ["twi_mean", "twi_p90", "log_upa_max"]
for c in base + tw + ["population_2020"]:
    df[c] = df[c].fillna(df[c].median())
df["landcover_code"] = df.landcover_code.fillna(-1).astype(int)
lc = pd.get_dummies(df.landcover_code, prefix="lc").astype(int)
lc_cols = list(lc.columns)
df = pd.concat([df.drop(columns=["hand_mean", "hand_min", "frac_hand_lt2m", "frac_hand_lt5m"]), lc], axis=1)

handc = ["hand_mean", "hand_min", "frac_hand_lt2m", "frac_hand_lt5m"]
H = {}
for t in THR:
    h = pd.read_csv(W / f"data/hand_sens/hand_thr_{t:g}km2.csv")
    for c in handc:
        h[c] = h[c].fillna(h[c].median())
    H[t] = h.set_index("grid_id")

x_km = df.longitude * 111.32 * np.cos(np.deg2rad(df.latitude))
y_km = df.latitude * 110.57


def groups(rep):
    ox, oy = (rep * 2.0) % BLOCK_KM, (rep * 3.0) % BLOCK_KM
    return (((x_km + ox) // BLOCK_KM).astype(int).astype(str) + "_" +
            ((y_km + oy) // BLOCK_KM).astype(int).astype(str)).values


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_depth=6,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


configs = []  # (label, ambang, kolom, populasi?)
for use_pop in (False, True):
    pop = ["population_2020"] if use_pop else []
    tag = "dengan pop" if use_pop else "tanpa pop"
    configs.append((f"Kontrol: dasar ({tag})", None, base + pop + lc_cols))
    configs.append((f"Kontrol: dasar + TWI/upa tanpa HAND ({tag})", None, base + tw + pop + lc_cols))
    for t in THR:
        configs.append((f"HAND ambang {t:g} km2 ({tag})", t, base + tw + handc + pop + lc_cols))

recs = []
for rep in range(N_REPS):
    folds = list(GroupKFold(N_FOLDS, shuffle=True, random_state=rep).split(df, y, groups(rep)))
    for label, t, cols in configs:
        X = df[["grid_id"] + [c for c in cols if c not in handc]].copy()
        if t is not None:
            X = X.join(H[t][handc], on="grid_id")
        X = X.drop(columns=["grid_id"])[cols]
        for mname, mk in (("LogReg", lr), ("RandomForest", rf)):
            prs, rocs = [], []
            for tr, te in folds:
                p = mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                prs.append(average_precision_score(y[te], p)); rocs.append(roc_auc_score(y[te], p))
            recs.append(dict(rep=rep, config=label, model=mname, pr=np.mean(prs), roc=np.mean(rocs)))
    print("ulangan", rep + 1, flush=True)

R = pd.DataFrame(recs)
R.to_csv(W / "outputs_sens/skor_ulangan.csv", index=False)
S = R.groupby(["config", "model"]).agg(pr_mean=("pr", "mean"), pr_sd=("pr", "std"),
                                      roc_mean=("roc", "mean")).round(4).reset_index()
S.to_csv(W / "outputs_sens/ringkasan_sensitivitas.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
print(S.pivot(index="config", columns="model", values="pr_mean"))
