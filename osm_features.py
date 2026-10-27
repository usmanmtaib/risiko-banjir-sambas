"""Fitur aksesibilitas dan permukiman per grid dari OSM (jalan + tempat).

Pemakaian: python osm_features.py <jalan_tempat.json> <feature_grid_csv> <out_csv>
"""
import json
import sys

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

src, grid_csv, out_csv = sys.argv[1:4]
els = json.load(open(src, encoding="utf-8"))["elements"]
t = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)

roads, major, places, towns = [], [], [], []
for e in els:
    tags = e.get("tags", {})
    if e["type"] == "way" and "geometry" in e and len(e["geometry"]) >= 2:
        xy = [t.transform(p["lon"], p["lat"]) for p in e["geometry"]]
        ln = LineString(xy)
        roads.append(ln)
        if tags.get("highway") in ("motorway", "trunk", "primary", "secondary"):
            major.append(ln)
    elif e["type"] == "node" and "place" in tags:
        p = Point(t.transform(e["lon"], e["lat"]))
        places.append(p)
        if tags["place"] in ("city", "town"):
            towns.append(p)
print("jalan:", len(roads), "jalan utama:", len(major), "tempat:", len(places), "kota/kecamatan:", len(towns))

g = pd.read_csv(grid_csv)
ids = g.grid_id.str.split("_", expand=True)
cx = ids[1].astype(float).values + 500.0
cy = ids[2].astype(float).values + 500.0
pts = shapely.points(cx, cy)


def dist(geoms):
    if not geoms:
        return np.full(len(pts), np.nan)
    tree = STRtree(geoms)
    _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    return d


res = pd.DataFrame({"grid_id": g.grid_id,
                    "dist_road_m": dist(roads),
                    "dist_major_road_m": dist(major),
                    "dist_place_m": dist(places),
                    "dist_town_m": dist(towns)})
# kepadatan jalan: jumlah panjang jalan (km) dalam sel 1 km
from shapely.geometry import box
rt = STRtree(roads)
dens = np.zeros(len(pts))
for i, (x, y) in enumerate(zip(cx - 500, cy - 500)):
    b = box(x, y, x + 1000, y + 1000)
    idx = rt.query(b)
    if len(idx):
        dens[i] = sum(roads[j].intersection(b).length for j in idx) / 1000.0
res["road_km_per_km2"] = dens
res.to_csv(out_csv, index=False)
print(res.describe().round(1).T[["min", "50%", "mean", "max"]])
