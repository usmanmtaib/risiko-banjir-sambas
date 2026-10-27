"""Ambil CHIRPS v2.0 harian (0,05 derajat) untuk bounding box Sambas, 2010-2025.
Menyimpan cache per hari (.npy) sehingga dapat dilanjutkan.
"""
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".cog"
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"

cache = Path(sys.argv[1]); cache.mkdir(parents=True, exist_ok=True)
BASE = "/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.cog"
# bbox Sambas +0,1 derajat: lon 108.80-109.85, lat 0.85-2.18 (grid global 0,05 deg, origin -180,50)
LON0, LAT0, LON1, LAT1 = 108.80, 0.85, 109.85, 2.18
col0 = int(np.floor((LON0 + 180) / 0.05)); col1 = int(np.ceil((LON1 + 180) / 0.05))
row0 = int(np.floor((50 - LAT1) / 0.05)); row1 = int(np.ceil((50 - LAT0) / 0.05))
WIN = Window(col0, row0, col1 - col0, row1 - row0)
meta = dict(col0=col0, row0=row0, ncol=col1 - col0, nrow=row1 - row0)
np.save(cache / "_window.npy", np.array([col0, row0, col1 - col0, row1 - row0]))


def get(day):
    f = cache / f"{day:%Y%m%d}.npy"
    if f.exists():
        return day, "cached"
    url = BASE.format(y=day.year, m=day.month, d=day.day)
    for attempt in range(3):
        try:
            with rasterio.open(url) as s:
                a = s.read(1, window=WIN).astype("float32")
            a[a < 0] = np.nan
            np.save(f, a)
            return day, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)[:80]
    return day, "FAIL " + err


days = [dt.date(2010, 1, 1) + dt.timedelta(n) for n in range((dt.date(2025, 12, 31) - dt.date(2010, 1, 1)).days + 1)]
fails, done = [], 0
with ThreadPoolExecutor(16) as ex:
    for day, st in ex.map(get, days):
        done += 1
        if st.startswith("FAIL"):
            fails.append((str(day), st))
        if done % 250 == 0:
            print(done, "/", len(days), "gagal:", len(fails), flush=True)
print("SELESAI", done, "hari; gagal:", len(fails))
for x in fails[:20]:
    print(x)
(cache / "_gagal.txt").write_text("\n".join(f"{a} {b}" for a, b in fails), encoding="utf-8")
