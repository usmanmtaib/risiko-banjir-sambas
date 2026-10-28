"""Model kerawanan banjir Sambas: Random Forest dengan validasi blok spasial.

Pemakaian: python run_model.py <feature_grid_csv> <output_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BLOCK_KM = 10
N_FOLDS = 5

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(src)
log = {}

# ---------- 1. Persiapan data ----------
log["n_grid"] = len(df)
log["missing_per_col"] = {k: int(v) for k, v in df.isna().sum().items() if v}
const = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
log["constant_columns_dropped"] = const  # hujan satu titik -> konstan

# Target biner: ada/tidak ada kejadian. Kelas tinggi hanya ~74 grid.
df["y"] = (df["event_count"] > 0).astype(int)
log["positives"] = int(df["y"].sum())
log["positive_rate"] = round(float(df["y"].mean()), 4)
log["class_counts_3"] = df["risk_class_exploratory"].value_counts().to_dict()

num = ["elevation_m", "slope_deg", "distance_river_m", "population_2020"]
for c in num:
    df[c + "_na"] = df[c].isna().astype(int)
    df[c] = df[c].fillna(df[c].median())
df["landcover_code"] = df["landcover_code"].fillna(-1).astype(int)
lc = pd.get_dummies(df["landcover_code"], prefix="lc").astype(int)
X_full = pd.concat([df[num], lc], axis=1)
X_noPop = X_full.drop(columns=["population_2020"])
y = df["y"].values

# Blok spasial ~10 km (koordinat dikonversi ke km)
x_km = df["longitude"] * 111.32 * np.cos(np.deg2rad(df["latitude"]))
y_km = df["latitude"] * 110.57
groups = ((x_km // BLOCK_KM).astype(int).astype(str) + "_" +
          (y_km // BLOCK_KM).astype(int).astype(str)).values
log["n_spatial_blocks"] = int(len(set(groups)))
gkf = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(X_full, y, groups))
log["positives_per_test_fold"] = [int(y[te].sum()) for _, te in folds]


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def cv_scores(make_model, X, name):
    oof = np.zeros(len(y))
    per_fold = []
    for tr, te in folds:
        m = make_model().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        per_fold.append((roc_auc_score(y[te], p), average_precision_score(y[te], p)))
    pf = np.array(per_fold)
    res = {"model": name,
           "roc_auc_pooled": round(roc_auc_score(y, oof), 4),
           "pr_auc_pooled": round(average_precision_score(y, oof), 4),
           "roc_auc_fold_mean": round(pf[:, 0].mean(), 4),
           "roc_auc_fold_sd": round(pf[:, 0].std(), 4),
           "pr_auc_fold_mean": round(pf[:, 1].mean(), 4)}
    return res, oof


# ---------- 2. Model dan pembanding ----------
rows, oofs = [], {}
for name, mk, X in [
    ("Dummy (prior)", lambda: DummyClassifier(strategy="prior"), X_full),
    ("Hanya populasi (LogReg)",
     lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
     df[["population_2020"]]),
    ("Logistic Regression (semua fitur)",
     lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
     X_full),
    ("Random Forest (semua fitur)", rf, X_full),
    ("Random Forest (tanpa populasi)", rf, X_noPop),
    ("Random Forest teregularisasi (leaf=30, depth=6)",
     lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=30, max_depth=6,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
     X_full),
    ("Random Forest teregularisasi (tanpa populasi)",
     lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=30, max_depth=6,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
     X_noPop),
]:
    r, o = cv_scores(mk, X, name)
    rows.append(r)
    oofs[name] = o
metrics = pd.DataFrame(rows)
metrics.to_csv(out / "metrik_cv_spasial.csv", index=False)
print(metrics.to_string(index=False))

# ---------- 3. Evaluasi RF utama (3 kelas dari probabilitas OOF) ----------
p = oofs["Random Forest teregularisasi (leaf=30, depth=6)"]
thr = np.quantile(p, 1 - y.mean())  # ambang: proporsi positif sama dengan data
pred = (p >= thr).astype(int)
log["rf_binary_at_matched_prevalence"] = {
    "threshold": round(float(thr), 4),
    "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
    "f1_positive": round(f1_score(y, pred), 4),
    "confusion_matrix[[TN,FP],[FN,TP]]": confusion_matrix(y, pred).tolist(),
}
# Kelas risiko dari kuantil probabilitas OOF (bukan label pelatihan)
df["p_oof"] = p
df["kelas_prediksi"] = pd.cut(p, [-1, np.quantile(p, 0.80), np.quantile(p, 0.95), 2],
                              labels=["rendah", "sedang", "tinggi"])
share = df.groupby("kelas_prediksi", observed=True)["y"].agg(["size", "sum", "mean"])
log["kejadian_per_kelas_prediksi"] = share.round(4).reset_index().to_dict("records")

# ---------- 4. Pentingnya fitur (permutation, pada data uji tiap fold) ----------
imp = pd.DataFrame(0.0, index=X_full.columns, columns=["imp"])
for tr, te in folds:
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=30, max_depth=6,
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=SEED).fit(X_full.iloc[tr], y[tr])
    r = permutation_importance(m, X_full.iloc[te], y[te], scoring="average_precision",
                               n_repeats=5, random_state=SEED, n_jobs=-1)
    imp["imp"] += r.importances_mean / N_FOLDS
imp = imp.sort_values("imp", ascending=False)
imp.to_csv(out / "feature_importance_permutation.csv")
print(imp.head(8))

# ---------- 5. K-Means eksploratif (tanpa event_count) ----------
Xs = StandardScaler().fit_transform(np.log1p(X_full[num].clip(lower=0)))
ks, sil = list(range(2, 9)), []
for k in ks:
    sil.append(silhouette_score(Xs, KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)))
kbest = ks[int(np.argmax(sil))]
df["cluster"] = KMeans(kbest, n_init=10, random_state=SEED).fit_predict(Xs)
cl = df.groupby("cluster").agg(n=("y", "size"), event_rate=("y", "mean"),
                               elev=("elevation_m", "median"), slope=("slope_deg", "median"),
                               river_m=("distance_river_m", "median"),
                               pop=("population_2020", "median")).round(3)
cl.to_csv(out / "kmeans_profil_klaster.csv")
log["kmeans"] = {"silhouette_by_k": dict(zip(ks, np.round(sil, 3))), "k_terbaik": kbest}
print(cl)

# ---------- 6. Simpan keluaran ----------
df[["grid_id", "longitude", "latitude", "event_count", "y", "p_oof",
    "kelas_prediksi", "cluster"]].to_csv(out / "prediksi_oof_grid.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
top = imp.head(8)[::-1]
ax[0].barh(top.index, top["imp"], color="#2a6f97")
ax[0].set_title("Pentingnya fitur\n(permutation, penurunan PR-AUC)")
sc = ax[1].scatter(df.longitude, df.latitude, c=df.p_oof, s=4, cmap="YlOrRd")
ax[1].set_title("Probabilitas kejadian (prediksi spasial-CV)")
ax[1].set_aspect("equal"); plt.colorbar(sc, ax=ax[1], shrink=.8)
ev = df[df.y == 1]
ax[1].scatter(ev.longitude, ev.latitude, s=2, c="k", alpha=.35)
m_ = metrics.set_index("model")["pr_auc_pooled"]
ax[2].barh(m_.index, m_.values, color="#6c757d")
ax[2].axvline(y.mean(), color="r", ls="--", label=f"prevalensi={y.mean():.3f}")
ax[2].set_title("PR-AUC (validasi blok spasial)"); ax[2].legend()
plt.tight_layout(); plt.savefig(out / "ringkasan_model.png", dpi=150)

(out / "ringkasan.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print(json.dumps(log, indent=2, default=str))
