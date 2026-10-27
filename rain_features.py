"""Fitur hujan spasial per grid dari CHIRPS bulanan (npy per bulan).

Pemakaian: python rain_features.py <dir_chirps_monthly> <feature_grid_csv> <out_csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

d, grid_csv, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
col0, row0, ncol, nrow = np.load(d / "_window.npy")
years = range(2010, 2023)  # 13 tahun penuh; Jul 2023-Des 2025 gagal diunduh (server membatasi laju)
stack = np.full((len(years), 12, nrow, ncol), np.nan, dtype="float32")
missing = []
for i, y in enumerate(years):
    for m in range(1, 13):
        f = d / f"{y}{m:02d}.npy"
        if f.exists():
            stack[i, m - 1] = np.load(f)
        else:
            missing.append((y, m))
print("bulan hilang:", len(missing), missing[:10])
if missing:
    raise SystemExit("Data belum lengkap; fitur tidak dibuat.")

annual = np.nansum(stack, axis=1)  # (tahun, y, x)
valid = ~np.isnan(stack).all(axis=(0, 1))
annual[:, ~valid] = np.nan
clim = np.nanmean(stack, axis=0)  # (12, y, x): klimatologi bulanan
feats = {
    "rain_annual_mean_mm": np.nanmean(annual, axis=0),
    "rain_annual_cv": np.nanstd(annual, axis=0) / np.nanmean(annual, axis=0),
    "rain_month_max_mean_mm": np.nanmean(np.nanmax(stack, axis=1), axis=0),
    "rain_month_p95_mm": np.nanpercentile(stack.reshape(-1, nrow, ncol), 95, axis=0),
    "rain_driest_month_mm": np.nanmin(clim, axis=0),
    "rain_seasonality_cv": np.nanstd(clim, axis=0) / np.nanmean(clim, axis=0),
}

lon_c = -180 + 0.05 * (col0 + np.arange(ncol) + 0.5)
lat_c = 50 - 0.05 * (row0 + np.arange(nrow) + 0.5)  # menurun
g = pd.read_csv(grid_csv)[["grid_id", "longitude", "latitude"]]
res = {"grid_id": g.grid_id}
for k, a in feats.items():
    a = a.copy()
    bad = np.isnan(a)
    if bad.any():  # isi sel laut/tak valid dengan nilai valid terdekat
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    f = RegularGridInterpolator((lat_c[::-1], lon_c), a[::-1], bounds_error=False, fill_value=None)
    res[k] = f(np.c_[g.latitude.values, g.longitude.values])
R = pd.DataFrame(res)
R.to_csv(out_csv, index=False)
print(R.describe().round(2).T[["min", "mean", "max", "std"]])
print("sel CHIRPS valid:", int(valid.sum()), "dari", valid.size)
