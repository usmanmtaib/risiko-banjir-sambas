"""Bangun dua masukan panel kecamatan-hari (dipakai panel_model.py):

1. kec/event_kec_overlay.csv : irisan poligon kejadian Groundsource Sambas dengan 19 kecamatan BIG
   (f_ev = bagian luas kejadian di dalam kecamatan; f_kec = bagian luas kecamatan yang tertutup kejadian).
2. kec/hujan_kec_power.csv   : hujan harian NASA POWER (PRECTOTCORR, mm/hari) di titik representatif tiap kecamatan.

Pemakaian:
    python kec_prep.py overlay [path_groundsource_sambas.parquet]
    python kec_prep.py hujan            # butuh internet (API NASA POWER)
    python kec_prep.py cek              # bandingkan hasil baru (kec/_baru/) dengan berkas di kec/

Hasil ditulis ke kec/_baru/ agar berkas asli tidak tertimpa; salin manual bila sudah dicek.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

W = Path(__file__).resolve().parent
KEC = W / "kec" / "kecamatan_sambas.geojson"
GS_DEFAULT = Path(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/groundsource_sambas_2010_2025.parquet"))
NEW = W / "kec" / "_baru"
UTM = 32649  # UTM zona 49N
START, END = "20120101", "20251231"


def kecamatan_utm():
    k = gpd.read_file(KEC).set_crs(4326, allow_override=True)[["NAMOBJ", "geometry"]]
    return k.rename(columns={"NAMOBJ": "kec"}).to_crs(UTM)


def overlay(src=GS_DEFAULT):
    g = pd.read_parquet(src)  # 753 kejadian Sambas hasil saringan, berurutan menurut start_date
    ev = gpd.GeoDataFrame(
        {"eid": range(len(g)), "s": pd.to_datetime(g.start_date).dt.strftime("%Y-%m-%d").values},
        geometry=shapely.from_wkb(g.geometry.values), crs=4326,
    ).to_crs(UTM)
    ev["area_ev"] = ev.area
    k = kecamatan_utm()
    k["area_kec"] = k.area
    ov = gpd.overlay(ev, k, how="intersection", keep_geom_type=True)
    ov["inter"] = ov.area
    ov = ov[ov.inter > 0]
    ov["f_ev"] = ov.inter / ov.area_ev
    ov["f_kec"] = ov.inter / ov.area_kec
    ov = ov.sort_values(["eid", "kec"])[["eid", "s", "area_ev", "kec", "area_kec", "geometry", "inter", "f_ev", "f_kec"]]
    NEW.mkdir(parents=True, exist_ok=True)
    ov.to_csv(NEW / "event_kec_overlay.csv", index=False)
    print("overlay:", len(ov), "baris,", ov.eid.nunique(), "kejadian,", ov.kec.nunique(), "kecamatan")


def titik_kecamatan():
    k = kecamatan_utm()
    p = gpd.GeoSeries(k.geometry.representative_point(), crs=UTM).to_crs(4326)
    return pd.DataFrame({"kec": k.kec, "lat": p.y.round(4), "lon": p.x.round(4)})


def hujan():
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG"
           "&longitude={lon}&latitude={lat}&start={a}&end={b}&format=JSON")
    rows = []
    for r in titik_kecamatan().itertuples():
        for i in range(5):
            try:
                with urllib.request.urlopen(url.format(lon=r.lon, lat=r.lat, a=START, b=END), timeout=120) as f:
                    d = json.load(f)["properties"]["parameter"]["PRECTOTCORR"]
                break
            except Exception as e:  # API kadang menolak sementara
                print("  ulang", r.kec, e); time.sleep(10 * (i + 1))
        else:
            raise SystemExit(f"gagal mengambil hujan {r.kec}")
        s = pd.Series(d)
        rows.append(pd.DataFrame({"kec": r.kec, "tanggal": pd.to_datetime(s.index).strftime("%Y-%m-%d"),
                                  "hujan_mm": s.values, "lat": r.lat, "lon": r.lon}))
        print(" ", r.kec, len(s), "hari")
    out = pd.concat(rows).sort_values(["kec", "tanggal"])
    NEW.mkdir(parents=True, exist_ok=True)
    out.to_csv(NEW / "hujan_kec_power.csv", index=False)
    print("hujan:", len(out), "baris")


def cek():
    for f, key, num in [("event_kec_overlay.csv", ["eid", "kec"], ["area_ev", "area_kec", "inter", "f_ev", "f_kec"]),
                        ("hujan_kec_power.csv", ["kec", "tanggal"], ["hujan_mm", "lat", "lon"])]:
        a, b = W / "kec" / f, NEW / f
        if not b.exists():
            print(f, ": belum dibuat"); continue
        x = pd.read_csv(a).drop(columns="geometry", errors="ignore").set_index(key).sort_index()
        y = pd.read_csv(b).drop(columns="geometry", errors="ignore").set_index(key).sort_index()
        same_keys = x.index.equals(y.index)
        print(f, ": baris", len(x), "vs", len(y), "| kunci sama:", same_keys)
        if same_keys:
            for c in num:
                print(f"   {c}: selisih maks {abs(x[c] - y[c]).max():.3g}")
        else:
            print("   hanya di asli:", len(x.index.difference(y.index)), "| hanya di baru:", len(y.index.difference(x.index)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "overlay":
        overlay(Path(sys.argv[2]) if len(sys.argv) > 2 else GS_DEFAULT)
    elif cmd == "hujan":
        hujan()
    elif cmd == "cek":
        cek()
    else:
        print(__doc__)
