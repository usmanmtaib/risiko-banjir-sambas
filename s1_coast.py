"""Uji ulang label S1 dengan mengecualikan zona pesisir dan tambak/lahan basah OSM."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

W = Path(__file__).resolve().parent
OUT = W / "outputs_s1"
osm_f = W / "data/osm/tambak_pantai.json"

# ---------- fraksi badan air WorldCover per sel (proksi tambak/tepi sungai) ----------
import sys
_argv = sys.argv; sys.argv = ["x", "none"]
exec(open(W / "s1_flood.py", encoding="utf-8").read().split("mode = sys.argv[1]")[0])
sys.argv = _argv
_ok = cell >= 0
_n = np.bincount(cell[_ok], minlength=NCELL).clip(1)
_wf = np.bincount(cell[_ok], weights=masks["frac_water"][_ok], minlength=NCELL) / _n
wc_water = pd.DataFrame({"grid_id": gi.grid_id.values, "wc_water_frac": _wf})
print("sel dgn fraksi air WorldCover >= 2%%: %d ; >= 5%%: %d (dari %d)" % ((_wf >= .02).sum(), (_wf >= .05).sum(), len(_wf)))

# ---------- fitur data ----------
df = pd.read_csv(W / "data/hidrologi/fitur_hidrologi_grid.csv").merge(
    pd.read_csv(W / "data/osm_features.csv"), on="grid_id").merge(
    pd.read_csv(W / "data/s1_cell_stats.csv")[["grid_id", "n_eval", "frac_flag_mean"]], on="grid_id").merge(
    pd.read_csv(W / "data/dist_coast.csv"), on="grid_id").merge(wc_water, on="grid_id")

# ---------- tambak / lahan basah dari OSM ----------
has_osm = osm_f.exists() and osm_f.stat().st_size > 1000
if has_osm:
    els = json.load(open(osm_f, encoding="utf-8"))["elements"]
    t = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    aqua, wet, coast = [], [], []
    for e in els:
        if e["type"] != "way" or "geometry" not in e or len(e["geometry"]) < 3:
            continue
        tg = e.get("tags", {})
        xy = [t.transform(p["lon"], p["lat"]) for p in e["geometry"]]
        closed = xy[0] == xy[-1] and len(xy) >= 4
        if tg.get("natural") == "coastline":
            coast.append(LineString(xy)); continue
        if not closed:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if tg.get("landuse") in ("aquaculture", "salt_pond") or tg.get("water") in ("pond", "basin", "reservoir", "lagoon"):
            aqua.append(poly)
        elif tg.get("natural") in ("wetland", "mud"):
            wet.append(poly)
    print(f"OSM: poligon tambak/kolam = {len(aqua)}, lahan basah/lumpur = {len(wet)}, garis pantai = {len(coast)}")
    ids = df.grid_id.str.split("_", expand=True)
    gx, gy = ids[1].astype(float).values, ids[2].astype(float).values

    def cover(polys):
        out = np.zeros(len(df))
        if not polys:
            return out
        tr = STRtree(polys)
        for i, (x, y) in enumerate(zip(gx, gy)):
            b = box(x, y, x + 1000, y + 1000)
            for j in tr.query(b):
                out[i] += polys[j].intersection(b).area / 1e6
        return np.clip(out, 0, 1)
    df["tambak_frac"] = cover(aqua); df["wetland_frac"] = cover(wet)
    if coast:
        tr = STRtree(coast)
        cxy = np.c_[gx + 500, gy + 500]
        import shapely
        _, dd = tr.query_nearest(shapely.points(cxy[:, 0], cxy[:, 1]), return_distance=True, all_matches=False)
        df["dist_osm_coast_km"] = dd / 1000.0
        ok = df.dist_osm_coast_km.notna() & df.dist_coast_km.notna()
        print("korelasi jarak pantai radar vs OSM coastline: Spearman %.3f" % spearmanr(df.dist_coast_km[ok], df.dist_osm_coast_km[ok])[0])
    print("sel dgn tambak/kolam >=5%%: %d ; lahan basah/lumpur >=20%%: %d" % ((df.tambak_frac >= 0.05).sum(), (df.wetland_frac >= 0.2).sum()))
else:
    df["tambak_frac"] = 0.0; df["wetland_frac"] = 0.0
    print("OSM tambak tidak tersedia; hanya jarak pantai yang dipakai")

# ---------- kolom ----------
P = ["elevation_m", "slope_deg", "distance_river_m", "hand_mean", "hand_min", "twi_mean", "twi_p90",
     "log_upa_max", "frac_hand_lt2m", "frac_hand_lt5m"]
df["landcover_code"] = df.landcover_code.fillna(-1).astype(int)
lc = pd.get_dummies(df.landcover_code, prefix="lc").astype(int); df = pd.concat([df, lc], axis=1)
LC = [c for c in lc.columns if c != "lc_50"]
df["log_pop"] = np.log1p(df.population_2020.fillna(0))
for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]:
    df["log_" + c] = np.log1p(df[c])
R = ["log_pop"] + ["log_" + c for c in ["dist_road_m", "dist_major_road_m", "dist_place_m", "dist_town_m"]] + ["road_km_per_km2"]
for c in P + R + LC:
    df[c] = df[c].fillna(df[c].median())

elig = df[(df.n_eval >= 30) & (~df.in_sea_mask.astype(bool))].reset_index(drop=True)
THR = elig.frac_flag_mean.quantile(0.90)  # ambang TETAP dari seluruh sel layak
print(f"sel layak (tanpa sel laut): {len(elig)} ; ambang label tetap frac_flag_mean >= {THR:.5f}")


def cv(sub, cols, ylab, reps=10, min_pos=20):
    y = ylab(sub)
    if y.sum() < min_pos or (len(y) - y.sum()) < min_pos:
        return np.nan, np.nan
    xk = sub.longitude * 111.32 * np.cos(np.deg2rad(sub.latitude)); yk = sub.latitude * 110.57
    aucs = []
    for rep in range(reps):
        g = (((xk + rep * 2.0 % 10) // 10).astype(int).astype(str) + "_" + ((yk + rep * 3.0 % 10) // 10).astype(int).astype(str)).values
        nf = min(5, len(set(g)))
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(nf, shuffle=True, random_state=rep).split(sub, y, g):
            if y[tr].sum() < 5:
                oof[te] = 0; continue
            sc = StandardScaler().fit(sub.iloc[tr][cols])
            m = LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5).fit(sc.transform(sub.iloc[tr][cols]), y[tr])
            oof[te] = sc.transform(sub.iloc[te][cols]) @ m.coef_[0]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


y_s1 = lambda s: (s.frac_flag_mean >= THR).astype(int).values
y_gs = lambda s: (s.event_count > 0).astype(int).values
terrain = P  # tanpa tutupan lahan
full = P + LC
strata = [("semua sel layak", np.ones(len(elig), bool)),
          ("jarak pantai > 1 km", elig.dist_coast_km > 1),
          ("jarak pantai > 2 km", elig.dist_coast_km > 2),
          ("jarak pantai > 5 km", elig.dist_coast_km > 5),
          ("jarak pantai > 10 km", elig.dist_coast_km > 10)]
strata.append(("> 2 km & fraksi air WorldCover < 2%", (elig.dist_coast_km > 2) & (elig.wc_water_frac < 0.02)))
strata.append(("> 5 km & fraksi air WorldCover < 2%", (elig.dist_coast_km > 5) & (elig.wc_water_frac < 0.02)))
strata.append(("> 2 km & fraksi air WorldCover < 0,5%", (elig.dist_coast_km > 2) & (elig.wc_water_frac < 0.005)))

rows = []
for name, m in strata:
    sub = elig[np.asarray(m)].reset_index(drop=True)
    ys = y_s1(sub); yg = y_gs(sub)
    r = dict(strata=name, n=len(sub), pos_s1=int(ys.sum()), pos_gs=int(yg.sum()))
    for lab, cols in [("terrain", terrain), ("fisik_penuh", full), ("pelaporan", R), ("populasi", ["log_pop"])]:
        a = cv(sub, cols, y_s1); r[f"S1_{lab}"] = a[0]
    for lab, cols in [("fisik_penuh", full), ("pelaporan", R), ("populasi", ["log_pop"])]:
        a = cv(sub, cols, y_gs, min_pos=15); r[f"GS_{lab}"] = a[0]
    r["AUC_S1kontinu_utk_GS"] = roc_auc_score(yg, sub.frac_flag_mean) if 15 <= yg.sum() < len(yg) else np.nan
    rows.append(r)
    print(f"  selesai: {name}", flush=True)
T = pd.DataFrame(rows)
T.to_csv(OUT / "uji_pesisir_tambak.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).to_string(index=False))
elig.to_csv(OUT / "sel_layak_pesisir.csv", index=False)
