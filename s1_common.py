"""Utilitas bersama Sentinel-1 RTC: grid 80 m UTM 49N, pembacaan satu adegan ke grid bersama."""
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [108.90, 0.96, 109.75, 2.08]
RES = 80.0
FACTOR = 8  # 10 m -> 80 m

_x0, _y0, _x1, _y1 = transform_bounds("EPSG:4326", "EPSG:32649", *BBOX)
X0 = np.floor(_x0 / RES) * RES
Y1 = np.ceil(_y1 / RES) * RES
NCOL = int(np.ceil((_x1 - X0) / RES))
NROW = int(np.ceil((Y1 - _y0) / RES))
GRID_TR = Affine(RES, 0, X0, 0, -RES, Y1)


def search_items(start="2015-01-01", end="2025-12-31"):
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    s = cat.search(collections=["sentinel-1-rtc"], bbox=BBOX, datetime=f"{start}/{end}", max_items=5000)
    return list(s.items())


def read_item(item):
    """Kembalikan array (NROW, NCOL) int16 dB*100 (-32768 = tak valid) pada grid bersama, atau None."""
    item = pc.sign(item)
    href = item.assets["vv"].href
    out = np.full((NROW, NCOL), -32768, dtype="int16")
    with rasterio.open(href) as s:
        if s.crs.to_epsg() != 32649:
            return None
        # irisan antara grid bersama dan adegan, disejajarkan ke kelipatan FACTOR piksel adegan
        bx0, by0, bx1, by1 = s.bounds
        gx0, gy0 = max(X0, bx0), max(Y1 - NROW * RES, by0)
        gx1, gy1 = min(X0 + NCOL * RES, bx1), min(Y1, by1)
        if gx1 <= gx0 or gy1 <= gy0:
            return None
        c0 = int(np.floor((gx0 - X0) / RES)); c1 = int(np.floor((gx1 - X0) / RES))
        r0 = int(np.floor((Y1 - gy1) / RES)); r1 = int(np.floor((Y1 - gy0) / RES))
        c1, r1 = min(c1, NCOL), min(r1, NROW)
        if c1 <= c0 or r1 <= r0:
            return None
        # jendela dalam piksel adegan (10 m)
        col_off = int(round((X0 + c0 * RES - s.transform.c) / s.transform.a))
        row_off = int(round((s.transform.f - (Y1 - r0 * RES)) / -s.transform.e))
        w = Window(col_off, row_off, (c1 - c0) * FACTOR, (r1 - r0) * FACTOR)
        a = s.read(1, window=w, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.average,
                   boundless=True, fill_value=-32768).astype("float32")
    valid = (a > 0) & (a != -32768)
    db = np.full(a.shape, np.nan, dtype="float32")
    db[valid] = 10 * np.log10(a[valid])
    q = np.where(np.isnan(db), -32768, np.clip(db * 100, -6000, 3000)).astype("int16")
    out[r0:r1, c0:c1] = q
    return out
