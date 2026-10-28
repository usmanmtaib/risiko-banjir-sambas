"""Ambil seluruh adegan Sentinel-1 RTC (VV) Sambas ke grid 80 m; cache per adegan; dapat dilanjutkan."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import s1_common as c

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
items = c.search_items()
meta = pd.DataFrame([dict(id=i.id, datetime=i.datetime.isoformat(), tanggal=str(i.datetime.date()),
                          orbit=i.properties.get("sat:orbit_state"), rel_orbit=i.properties.get("sat:relative_orbit"),
                          platform=i.properties.get("platform")) for i in items])
meta.to_csv(out / "_meta.csv", index=False)
print("item:", len(items), flush=True)


def work(it):
    f = out / f"{it.id}.npy"
    if f.exists():
        return it.id, "cache"
    for k in range(3):
        try:
            a = c.read_item(it)
            if a is None:
                np.save(f, np.full((1, 1), -32768, dtype="int16")); return it.id, "kosong"
            np.save(f, a)
            return it.id, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)[:90]
            if any(x in err for x in ("403", "429")):
                return it.id, "STOP " + err  # jangan menekan server
            time.sleep(5 * (k + 1))
    return it.id, "GAGAL " + err


res, t0 = {}, time.time()
with ThreadPoolExecutor(4) as ex:
    for i, (iid, st) in enumerate(ex.map(work, items), 1):
        res[iid] = st
        if st.startswith("STOP"):
            print("BERHENTI:", iid, st, flush=True)
        if i % 25 == 0:
            print(i, "/", len(items), f"{(time.time() - t0) / 60:.1f} menit", flush=True)
bad = {k: v for k, v in res.items() if v.startswith(("GAGAL", "STOP"))}
print("SELESAI. bermasalah:", len(bad), "dari", len(res))
for k, v in list(bad.items())[:15]:
    print(k, v)
