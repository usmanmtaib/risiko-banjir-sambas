"""Siapkan raster pendukung pada grid 80 m UTM 49N: kelas dominan WorldCover, fraksi air, lereng.
Sengaja TIDAK memakai HAND/TWI/elevasi agar label tidak sirkular terhadap prediktor model."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import s1_common as c

P = str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed")
D = str(Path(__file__).resolve().parent / "data")


def warp(src_path, resampling, dtype, band_fn=None, nodata=0):
    dst = np.full((c.NROW, c.NCOL), nodata, dtype=dtype)
    with rasterio.open(src_path) as s:
        arr = s.read(1)
        if band_fn is not None:
            arr = band_fn(arr)
        reproject(arr, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=c.GRID_TR,
                  dst_crs="EPSG:32649", resampling=resampling, src_nodata=None, dst_nodata=nodata)
    return dst


wc = P + r"\worldcover_sambas_2021_10m.tif"
dom = warp(wc, Resampling.mode, "uint8")
fr = {}
for name, code in [("water", 80), ("built", 50), ("tree", 10), ("crop", 40)]:
    fr[name] = warp(wc, Resampling.average, "float32", band_fn=lambda a, k=code: (a == k).astype("float32"), nodata=0)
has_wc = warp(wc, Resampling.average, "float32", band_fn=lambda a: (a > 0).astype("float32"), nodata=0)
slope = warp(P + r"\slope_sambas_degrees.tif", Resampling.average, "float32",
             band_fn=lambda a: np.where(a < -9000, np.nan, a), nodata=np.nan)
np.savez_compressed(D + r"\s1_masks.npz", dominant=dom, frac_water=fr["water"], frac_built=fr["built"],
                    frac_tree=fr["tree"], frac_crop=fr["crop"], has_wc=has_wc, slope=slope)
print("grid", dom.shape, "cakupan WorldCover", float((has_wc > 0.5).mean()).__round__(3))
for k, v in [("air", fr["water"]), ("terbangun", fr["built"]), ("pohon", fr["tree"]), ("sawah/ladang", fr["crop"])]:
    print(f"fraksi {k}: {float(np.nanmean(v[has_wc > 0.5])):.3f}")
print("lereng median", float(np.nanmedian(slope)))
