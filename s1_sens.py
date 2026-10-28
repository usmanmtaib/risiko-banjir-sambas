"""Sensitivitas ambang deteksi radar: apakah pola label radar (fisik vs pelaporan) bergantung pada ambang?

Himpunan sel dasar = 1.754 sel layak pada konfigurasi dasar (VV < -15 dB, penurunan >= 4 dB). Untuk tiap konfigurasi,
label = desil teratas fraksi piksel tergenang pada sel yang tetap layak (>= 30 piksel lahan terbuka valid).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_article"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
B = 500
el = pd.read_csv(W / "outputs_s1/sel_layak_pesisir.csv")
gi = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/feature_grid_sambas_1km.csv"), usecols=["grid_id"])
LC = [c for c in el.columns if c.startswith("lc_") and c != "lc_50"]
TERR = [c for c in ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90", "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"] if c in el.columns and el[c].nunique() > 1]
RPT = [c for c in ["log_pop", "log_dist_road_m", "log_dist_major_road_m", "log_dist_place_m", "log_dist_town_m", "road_km_per_km2"] if c in el.columns]
SETS = {"Terrain + tutupan lahan": TERR + LC, "Terrain/hidrologi": TERR, "Pelaporan": RPT, "Populasi": ["log_pop"]}
xk = el.longitude * 111.32 * np.cos(np.deg2rad(el.latitude)); yk = el.latitude * 110.57
yg_all = (el.event_count > 0).astype(int).values

CFG = [("Dasar: VV < −15 dB, turun ≥ 4 dB", ""), ("Lebih longgar: < −13 dB, ≥ 4 dB", "_t13d4"), ("Lebih ketat: < −17 dB, ≥ 4 dB", "_t17d4"),
       ("< −15 dB, turun ≥ 3 dB", "_t15d3"), ("< −15 dB, turun ≥ 5 dB", "_t15d5"), ("Paling longgar: < −13 dB, ≥ 3 dB", "_t13d3"), ("Paling ketat: < −17 dB, ≥ 5 dB", "_t17d5")]


def blocks(rep, mask):
    ox, oy = (rep * 2.0) % 10, (rep * 3.0) % 10
    return (((xk[mask] + ox) // 10).astype(int).astype(str) + "_" + ((yk[mask] + oy) // 10).astype(int).astype(str)).values


def oof(s, cols, y, mask):
    o = np.zeros(len(s))
    for rep in range(10):
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(s, y, blocks(rep, mask)):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)).fit(s.iloc[tr][cols], y[tr])
            o[te] += (m.named_steps["standardscaler"].transform(s.iloc[te][cols]) @ m.named_steps["logisticregression"].coef_[0]) / 10
    return o


def ci(a):
    a = np.array(a); return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


rows = []
for name, suf in CFG:
    Z = np.load(W / f"data/s1_cell_scene{suf}.npz")
    flag, valid = Z["flag_open"].astype(float), Z["valid_open"].astype(float)
    ok = valid >= 30
    frac = np.where(ok, flag / np.maximum(valid, 1), np.nan)
    with np.errstate(all="ignore"):
        fm = pd.Series(np.nanmean(frac, axis=0), index=gi.grid_id); neval = pd.Series(ok.sum(0), index=gi.grid_id)
    f = el.grid_id.map(fm); ne = el.grid_id.map(neval)
    base_ok = ((ne >= 30) & f.notna()).values
    for sname, sub_mask in [("Semua sel", np.ones(len(el), bool)), ("Tanpa pesisir dan tepi air", ((el.dist_coast_km > 2) & (el.wc_water_frac < 0.02)).values)]:
        mask = base_ok & sub_mask
        s = el[mask].reset_index(drop=True)
        fv = f[mask].values
        y = (fv >= np.quantile(fv, 0.90)).astype(int)
        yg = yg_all[mask]
        if y.sum() < 15:
            continue
        sc = {p: oof(s, cols, y, mask) for p, cols in SETS.items()}
        g0 = blocks(0, mask); ub = np.unique(g0); members = {b: np.where(g0 == b)[0] for b in ub}
        boot = {p: [] for p in sc}; bcross = []
        for _ in range(B):
            idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
            if y[idx].sum() < 5 or (len(idx) - y[idx].sum()) < 5:
                continue
            for p, o in sc.items():
                boot[p].append(roc_auc_score(y[idx], o[idx]))
            if yg[idx].sum() >= 5:
                bcross.append(roc_auc_score(yg[idx], fv[idx]))
        a = {p: roc_auc_score(y, o) for p, o in sc.items()}
        d1 = a["Terrain + tutupan lahan"] - a["Pelaporan"]; d2 = a["Terrain/hidrologi"] - a["Pelaporan"]
        b1 = np.array(boot["Terrain + tutupan lahan"]) - np.array(boot["Pelaporan"]); b2 = np.array(boot["Terrain/hidrologi"]) - np.array(boot["Pelaporan"])
        cross = roc_auc_score(yg, fv)
        rows.append(dict(konfigurasi=name, subset=sname, n=len(s), positif_radar=int(y.sum()),
                         terrain_lc=round(a["Terrain + tutupan lahan"], 3), terrain=round(a["Terrain/hidrologi"], 3), pelaporan=round(a["Pelaporan"], 3), populasi=round(a["Populasi"], 3),
                         selisih_terrainLC_minus_pelaporan=f"{d1:+.3f} [{ci(b1)[0]:+.3f}, {ci(b1)[1]:+.3f}]", selisih_terrain_minus_pelaporan=f"{d2:+.3f} [{ci(b2)[0]:+.3f}, {ci(b2)[1]:+.3f}]",
                         auc_radar_untuk_groundsource=f"{cross:.3f} [{ci(bcross)[0]:.3f}, {ci(bcross)[1]:.3f}]"))
        print(name, "|", sname, "| n", len(s), "| selesai", flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT / "sensitivitas_ambang_radar.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_columns", 20)
print(T.to_string(index=False))
