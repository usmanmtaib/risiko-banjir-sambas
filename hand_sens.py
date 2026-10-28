"""Sensitivitas ambang aliran untuk HAND. Hitung routing sekali, HAND per ambang.

Pemakaian: python hand_sens.py <dem_90m.tif> <feature_grid_csv> <out_dir>
(dem_90m.tif dari hand_twi.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from pysheds.grid import Grid
from rasterio.transform import xy

if not hasattr(np, "in1d"):
    np.in1d = np.isin

dem_f, grid_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
THRESH = [0.5, 1.0, 2.0, 5.0, 10.0]  # km2

with rasterio.open(dem_f) as s:
    tr90, shape = s.transform, (s.height, s.width)
    d90 = s.read(1).astype("float64")
d90[d90 <= -9000] = np.nan

grid = Grid.from_raster(str(dem_f))
dem = grid.read_raster(str(dem_f))
infl = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
fdir = grid.flowdir(infl)
acc = grid.accumulation(fdir)
acc_a = np.asarray(acc, dtype="float64")
lat_c = np.asarray(xy(tr90, np.arange(shape[0]), np.zeros(shape[0], int))[1])
cell_km2 = (90.0 * np.cos(np.deg2rad(lat_c)) * 90.0 / 1e6)[:, None] * np.ones(shape[1])[None, :]
med_area = float(np.nanmedian(cell_km2))

rows, cols = np.indices(shape)
lon, lat = xy(tr90, rows.ravel(), cols.ravel())
t = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
E, N = t.transform(np.asarray(lon), np.asarray(lat))
key = pd.DataFrame({"E": (np.floor(E / 1000) * 1000).astype(int), "N": (np.floor(N / 1000) * 1000).astype(int)})

g = pd.read_csv(grid_csv)
ids = g.grid_id.str.split("_", expand=True)
g["E"], g["N"] = ids[1].astype(int), ids[2].astype(int)

summary = []
for thr in THRESH:
    streams = acc > (thr / med_area)
    n_str = int((acc_a >= thr / med_area).sum())
    hand = np.asarray(grid.compute_hand(fdir, infl, streams), dtype="float64")
    hand[np.isnan(d90)] = np.nan
    px = key.copy()
    px["hand"] = hand.ravel()
    px = px.dropna(subset=["hand"])
    px["low2"] = (px.hand < 2).astype(float)
    px["low5"] = (px.hand < 5).astype(float)
    agg = px.groupby(["E", "N"]).agg(hand_mean=("hand", "mean"), hand_min=("hand", "min"),
                                     frac_hand_lt2m=("low2", "mean"),
                                     frac_hand_lt5m=("low5", "mean")).reset_index()
    m = g[["grid_id", "E", "N"]].merge(agg, on=["E", "N"], how="left").drop(columns=["E", "N"])
    f = out / f"hand_thr_{thr:g}km2.csv"
    m.to_csv(f, index=False)
    summary.append(dict(ambang_km2=thr, sel_sungai=n_str, hand_mean_median=round(m.hand_mean.median(), 2),
                        frac_lt5m_median=round(m.frac_hand_lt5m.median(), 3)))
    print(summary[-1], flush=True)
pd.DataFrame(summary).to_csv(out / "ringkasan_ambang.csv", index=False)
