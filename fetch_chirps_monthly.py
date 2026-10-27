"""Ambil CHIRPS v2.0 bulanan (0,05 derajat, mm/bulan) untuk bbox Sambas, 2010-2025.
Sopan terhadap server: 2 utas, jeda, dan berhenti jika terkena 403 berulang.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".cog"

cache = Path(sys.argv[1]); cache.mkdir(parents=True, exist_ok=True)
URL = "/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/chirps-v2.0.{y}.{m:02d}.cog"
LON0, LAT0, LON1, LAT1 = 108.80, 0.85, 109.85, 2.18
col0 = int(np.floor((LON0 + 180) / 0.05)); col1 = int(np.ceil((LON1 + 180) / 0.05))
row0 = int(np.floor((50 - LAT1) / 0.05)); row1 = int(np.ceil((50 - LAT0) / 0.05))
WIN = Window(col0, row0, col1 - col0, row1 - row0)
np.save(cache / "_window.npy", np.array([col0, row0, col1 - col0, row1 - row0]))
blocked = {"n": 0}


def get(ym):
    y, m = ym
    f = cache / f"{y}{m:02d}.npy"
    if f.exists():
        return ym, "cached"
    if blocked["n"] >= 1:
        return ym, "SKIP (diblokir)"
    err = ""
    for attempt in range(2):
        try:
            time.sleep(4.0)
            with rasterio.open(URL.format(y=y, m=m)) as s:
                a = s.read(1, window=WIN).astype("float32")
            a[a < 0] = np.nan
            np.save(f, a)
            blocked["n"] = 0
            return ym, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)[:80]
            if "403" in err or "429" in err:
                blocked["n"] += 1  # berhenti total, jangan mengulang
                break
            time.sleep(5)
    return ym, "FAIL " + err


months = [(y, m) for y in range(2010, 2026) for m in range(1, 13)]
res = {}
with ThreadPoolExecutor(1) as ex:
    for i, (ym, st) in enumerate(ex.map(get, months), 1):
        res[ym] = st
        if i % 24 == 0:
            print(i, "/", len(months), flush=True)
bad = {k: v for k, v in res.items() if v.startswith(("FAIL", "SKIP"))}
print("SELESAI. ok/cached:", len(res) - len(bad), "bermasalah:", len(bad))
for k, v in list(bad.items())[:20]:
    print(k, v)
