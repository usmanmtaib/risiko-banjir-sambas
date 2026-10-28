"""HAND, TWI, dan luas tangkapan dari SRTM (di-downsample 90 m) lalu agregasi ke grid 1 km.

Pemakaian: python hand_twi.py <dir_hgt> <feature_grid_csv> <out_dir>
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from pysheds.grid import Grid
from rasterio.merge import merge
from rasterio.transform import Affine, xy
from rasterio.windows import from_bounds

if not hasattr(np, "in1d"):  # pysheds 0.5 masih memakai np.in1d (dihapus di NumPy baru)
    np.in1d = np.isin

hgt_dir, grid_csv, out =Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
STREAM_KM2 = 2.0  # ambang aliran: luas tangkapan >= 2 km2
BBOX = (108.80, 0.85, 109.85, 2.18)  # batas Sambas + 0,1 derajat
FACTOR = 3  # 30 m -> 90 m

# 1. Mosaik dan potong
srcs = [rasterio.open(f) for f in sorted(glob.glob(str(hgt_dir / "*.hgt")))]
mos, tr = merge(srcs, bounds=BBOX, nodata=-32768)
a = mos[0].astype("float32")
a[a <= -32000] = np.nan
print("mosaik", a.shape, "NaN", float(np.isnan(a).mean()))

# 2. Downsample 3x3 (rata-rata)
h, w = (a.shape[0] // FACTOR) * FACTOR, (a.shape[1] // FACTOR) * FACTOR
b = a[:h, :w].reshape(h // FACTOR, FACTOR, w // FACTOR, FACTOR)
with np.errstate(all="ignore"):
    d90 = np.nanmean(b, axis=(1, 3))
tr90 = Affine(tr.a * FACTOR, 0, tr.c, 0, tr.e * FACTOR, tr.f)
prof = dict(driver="GTiff", height=d90.shape[0], width=d90.shape[1], count=1, dtype="float32",
            crs="EPSG:4326", transform=tr90, nodata=-9999.0)
dem_f = out / "dem_90m.tif"
with rasterio.open(dem_f, "w", **prof) as dst:
    dst.write(np.where(np.isnan(d90), -9999.0, d90).astype("float32"), 1)
print("DEM 90 m", d90.shape, "NaN", round(float(np.isnan(d90).mean()), 3))

# 3. Hidrologi
grid = Grid.from_raster(str(dem_f))
dem = grid.read_raster(str(dem_f))
pit = grid.fill_pits(dem)
flood = grid.fill_depressions(pit)
infl = grid.resolve_flats(flood)
fdir = grid.flowdir(infl)
acc = grid.accumulation(fdir)
acc_a = np.asarray(acc, dtype="float64")

lat_c = xy(tr90, np.arange(d90.shape[0]), np.zeros(d90.shape[0], int))[1]
lat_c = np.asarray(lat_c)
dx = 90.0 * np.cos(np.deg2rad(lat_c))[:, None] * np.ones(d90.shape[1])[None, :]  # lebar sel (m)
dy = 90.0
cell_area_km2 = (dx * dy) / 1e6
upa_km2 = acc_a * cell_area_km2
streams_np = upa_km2 >= STREAM_KM2
print("sel sungai", int(streams_np.sum()))
thr_cells = STREAM_KM2 / float(np.nanmedian(cell_area_km2))
streams = acc > thr_cells  # tetap bertipe Raster pysheds
hand = np.asarray(grid.compute_hand(fdir, infl, streams), dtype="float64")

# Slope (m/m) dari DEM yang sudah diisi
fl = np.asarray(infl, dtype="float64")
gy, gx = np.gradient(fl, dy, 1.0)
gx = np.gradient(fl, axis=1) / dx
tanb = np.maximum(np.hypot(gx, gy), np.tan(np.deg2rad(0.1)))
sca = acc_a * dx / dx  # luas tangkapan per lebar kontur = (acc*dx*dy)/dx = acc*dy
sca = acc_a * dy
twi = np.log(sca / tanb)

invalid = np.isnan(d90)
for arr in (hand, twi):
    arr[invalid] = np.nan
upa_km2[invalid] = np.nan

# 4. Agregasi ke grid 1 km (UTM 49N)
rows, cols = np.indices(d90.shape)
lon, lat = xy(tr90, rows.ravel(), cols.ravel())
t = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
E, N = t.transform(np.asarray(lon), np.asarray(lat))
px = pd.DataFrame({"E": (np.floor(E / 1000) * 1000).astype(int),
                   "N": (np.floor(N / 1000) * 1000).astype(int),
                   "hand": hand.ravel(), "twi": twi.ravel(), "upa": upa_km2.ravel()})
px = px.dropna(subset=["hand"])
px["low2"] = (px.hand < 2).astype(float)
px["low5"] = (px.hand < 5).astype(float)
agg = px.groupby(["E", "N"]).agg(hand_mean=("hand", "mean"), hand_min=("hand", "min"),
                                 twi_mean=("twi", "mean"), twi_p90=("twi", lambda s: s.quantile(.9)),
                                 upa_max_km2=("upa", "max"), frac_hand_lt2m=("low2", "mean"),
                                 frac_hand_lt5m=("low5", "mean")).reset_index()
agg["log_upa_max"] = np.log10(agg.upa_max_km2.clip(lower=0.01))

# 5. Cocokkan dengan grid_id (cek sudut kiri-bawah vs pusat sel)
g = pd.read_csv(grid_csv)
ids = g.grid_id.str.split("_", expand=True)
g["E"], g["N"] = ids[1].astype(int), ids[2].astype(int)
inv = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
for off in (0, 500):
    lo, la = inv.transform((g.E + off).values, (g.N + off).values)
    print(f"offset {off}: MAE lon {np.mean(np.abs(lo - g.longitude)):.5f}, lat {np.mean(np.abs(la - g.latitude)):.5f}")
m = g.merge(agg, on=["E", "N"], how="left")
cov = m.hand_mean.notna().mean()
print("cakupan grid:", round(float(cov), 4))
m.drop(columns=["E", "N"]).to_csv(out / "fitur_hidrologi_grid.csv", index=False)
print(m[["hand_mean", "twi_mean", "log_upa_max", "frac_hand_lt2m"]].describe().round(3))
