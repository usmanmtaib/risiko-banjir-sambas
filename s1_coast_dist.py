"""Jarak ke pantai per sel 1 km, dari median VV orbit 149 (laut = air luas, sungai sempit dibuang dengan opening)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.argv = ["x", "none"]
src = open(str(Path(__file__).resolve().parent / "s1_flood.py"), encoding="utf-8").read().split("mode = sys.argv[1]")[0]
exec(src)

dates, st = load_group(149)
ref, nval = chunked_median(st)
del st
water = (ref < -15) & (nval >= 15)
r_px = 6  # opening ~ 6 piksel (480 m) membuang sungai sempit
opened = ndi.binary_opening(water, structure=np.ones((2 * r_px + 1, 2 * r_px + 1)))
lab, n = ndi.label(opened)
sizes = ndi.sum(opened, lab, range(1, n + 1))
keep = np.isin(lab, 1 + np.where(sizes >= 20000)[0])  # komponen besar = laut/estuari lebar (>= 128 km2)
print("piksel air luas:", int(keep.sum()), f"({keep.mean():.3f} dari grid) | komponen:", int((sizes >= 20000).sum()))
dist_px = ndi.distance_transform_edt(~keep)  # jarak (piksel) ke air luas terdekat
dist_km = dist_px * c.RES / 1000.0

gi = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/feature_grid_sambas_1km.csv"), usecols=["grid_id"])
ids = gi.grid_id.str.split("_", expand=True)
cx = ids[1].astype(float).values + 500.0; cy = ids[2].astype(float).values + 500.0
col = np.clip(((cx - c.X0) / c.RES).astype(int), 0, Wd - 1); row = np.clip(((c.Y1 - cy) / c.RES).astype(int), 0, H - 1)
d = dist_km[row, col]
out = pd.DataFrame({"grid_id": gi.grid_id, "dist_coast_km": d, "in_sea_mask": keep[row, col]})
out.to_csv(W / "data/dist_coast.csv", index=False)
np.save(W / "data/sea_mask.npy", keep)

e = pd.read_csv(W / "data/s1_cell_stats.csv").merge(out, on="grid_id")
elig = e.n_eval >= 30
bins = [-0.1, 1, 2, 5, 10, 20, 100]
e["bin"] = pd.cut(e.dist_coast_km, bins)
t = e.groupby("bin", observed=True).agg(sel=("grid_id", "size"), layak=("n_eval", lambda s: (s >= 30).sum()))
t["frac_flag_mean_layak"] = e[elig].groupby("bin", observed=True).frac_flag_mean.mean()
t["laju_laporan_layak"] = e[elig].groupby("bin", observed=True).event_count.apply(lambda s: (s > 0).mean())
print(t.round(4).to_string())
print("sel di dalam masker laut:", int(out.in_sea_mask.sum()))
