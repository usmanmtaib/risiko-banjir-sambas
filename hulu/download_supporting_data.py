import csv
import gzip
import json
import os
import ssl
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = (Path(__file__).resolve().parent)
OUT = ROOT / "outputs" / "data" / "raw"
for name in ["dem_srtm", "sungai_osm", "curah_hujan", "tutupan_lahan", "populasi"]:
    (OUT / name).mkdir(parents=True, exist_ok=True)
CTX = ssl._create_unverified_context()
HEADERS = {"User-Agent": "Sambas-flood-research/1.0"}


def download(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=CTX, timeout=180) as response, open(dest, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(dest)


def dem():
    # The BIG boundary spans parts of tiles N00-N02 and E108-E109.
    for lat in range(0, 3):
        for lon in range(108, 110):
            name = f"N{lat:02d}E{lon:03d}"
            gz = OUT / "dem_srtm" / f"{name}.hgt.gz"
            hgt = OUT / "dem_srtm" / f"{name}.hgt"
            download(f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/N{lat:02d}/{name}.hgt.gz", gz)
            with gzip.open(gz, "rb") as src, open(hgt, "wb") as dst:
                dst.write(src.read())


def rivers():
    # Sambas administrative extent with a small margin, in south, west, north, east order.
    query = '[out:json][timeout:120];(way["waterway"~"^(river|stream|canal|drain)$"](0.90,108.85,2.15,109.85););out geom;'
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(url, headers=HEADERS)
    raw = json.loads(urllib.request.urlopen(req, context=CTX, timeout=180).read().decode("utf-8"))
    features = []
    for element in raw.get("elements", []):
        geom = element.get("geometry", [])
        coords = [[p["lon"], p["lat"]] for p in geom]
        if len(coords) < 2:
            continue
        tags = element.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"osm_id": element.get("id"), "waterway": tags.get("waterway"), "name": tags.get("name")},
        })
    dest = OUT / "sungai_osm" / "sungai_sambas_osm.geojson"
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")
    print(dest, len(features))


def rainfall():
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": "109.35",
        "latitude": "1.35",
        "start": "20100101",
        "end": "20251231",
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    raw = json.loads(urllib.request.urlopen(req, context=CTX, timeout=180).read().decode("utf-8"))
    values = raw["properties"]["parameter"]["PRECTOTCORR"]
    dest = OUT / "curah_hujan" / "nasa_power_precipitation_sambas_2010_2025.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tanggal", "precipitation_mm_day", "latitude", "longitude", "source"])
        for date, value in sorted(values.items()):
            writer.writerow([date, value, 1.35, 109.35, "NASA POWER PRECTOTCORR"])
    print(dest, len(values))


def landcover():
    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N00E108_Map.tif"
    download(url, OUT / "tutupan_lahan" / "ESA_WorldCover_10m_2021_v200_N00E108_Map.tif")


def population():
    url = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IDN/idn_ppp_2020_1km_Aggregated.tif"
    download(url, OUT / "populasi" / "idn_ppp_2020_1km_Aggregated.tif")


if __name__ == "__main__":
    dem()
    rivers()
    rainfall()
    landcover()
    population()
