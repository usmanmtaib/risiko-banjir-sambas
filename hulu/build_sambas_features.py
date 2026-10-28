import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer


ROOT = (Path(__file__).resolve().parent)
RAW = ROOT / "outputs" / "data" / "raw"
PROC = ROOT / "outputs" / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
BOUNDARY_PATH = PROC / "batas_kabupaten_sambas_big.geojson"


def write_raster_clip(src_path, out_path, boundary_geom, nodata=None):
    with rasterio.open(src_path) as src:
        clipped, clipped_transform = mask(src, [mapping(boundary_geom)], crop=True, nodata=nodata)
        profile = src.profile.copy()
        profile.update({"height": clipped.shape[1], "width": clipped.shape[2], "transform": clipped_transform, "compress": "deflate"})
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped)
        return clipped[0], clipped_transform, profile


def build_dem(boundary_geom):
    dem_dir = RAW / "dem_srtm"
    sources = sorted(dem_dir.glob("N*.hgt"))
    datasets = [rasterio.open(p) for p in sources]
    try:
        mosaic, tx = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": tx, "compress": "deflate", "nodata": -9999})
        temp = PROC / "_dem_mosaic_tmp.tif"
        with rasterio.open(temp, "w", **profile) as dst:
            dst.write(mosaic)
        full_dem = mosaic[0].astype("float32")
        full_valid = full_dem != -9999
        full_dem[~full_valid] = np.nan
        lat0 = tx.f + tx.e * (full_dem.shape[0] / 2)
        pixel_y = abs(tx.e) * 111320.0
        pixel_x = abs(tx.a) * 111320.0 * math.cos(math.radians(lat0))
        dz_dy, dz_dx = np.gradient(full_dem, pixel_y, pixel_x)
        full_slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))).astype("float32")
        full_slope[~full_valid] = -9999
        slope_temp = PROC / "_slope_mosaic_tmp.tif"
        slope_profile = profile.copy()
        slope_profile.update({"dtype": "float32", "count": 1, "nodata": -9999})
        with rasterio.open(slope_temp, "w", **slope_profile) as dst:
            dst.write(full_slope, 1)
    finally:
        for ds in datasets:
            ds.close()
    dem_path = PROC / "dem_sambas_30m.tif"
    dem, dem_tx, dem_profile = write_raster_clip(temp, dem_path, boundary_geom, nodata=-9999)
    slope_path = PROC / "slope_sambas_degrees.tif"
    write_raster_clip(slope_temp, slope_path, boundary_geom, nodata=-9999)
    try:
        temp.unlink()
        slope_temp.unlink()
    except OSError:
        pass
    return dem_path, slope_path


def clip_rasters(boundary_geom):
    worldcover = RAW / "tutupan_lahan" / "ESA_WorldCover_10m_2021_v200_N00E108_Map.tif"
    population = RAW / "populasi" / "idn_ppp_2020_1km_Aggregated.tif"
    wc_path = PROC / "worldcover_sambas_2021_10m.tif"
    pop_path = PROC / "population_sambas_2020_1km.tif"
    write_raster_clip(worldcover, wc_path, boundary_geom, nodata=0)
    write_raster_clip(population, pop_path, boundary_geom, nodata=0)
    return wc_path, pop_path


def make_grid(boundary_geom):
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True).transform
    from_utm = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True).transform
    metric_boundary = transform(to_utm, boundary_geom)
    minx, miny, maxx, maxy = metric_boundary.bounds
    startx = math.floor(minx / 1000) * 1000
    starty = math.floor(miny / 1000) * 1000
    cells = []
    for x in np.arange(startx, maxx, 1000):
        for y in np.arange(starty, maxy, 1000):
            cell = box(float(x), float(y), float(x + 1000), float(y + 1000))
            if metric_boundary.intersects(cell):
                clipped_cell = cell.intersection(metric_boundary)
                center = clipped_cell.representative_point()
                lon, lat = from_utm(center.x, center.y)
                cells.append({"grid_id": f"SAMBAS_{int(x)}_{int(y)}", "geom_metric": clipped_cell, "longitude": lon, "latitude": lat})
    return cells, to_utm, from_utm


def sample_raster(path, coords):
    with rasterio.open(path) as src:
        return [float(v[0]) if v[0] is not None else np.nan for v in src.sample(coords)]


def build_features(boundary_geom, dem_path, slope_path, wc_path, pop_path):
    cells, to_utm, from_utm = make_grid(boundary_geom)
    centers = [(c["longitude"], c["latitude"]) for c in cells]
    dem = sample_raster(dem_path, centers)
    slope = sample_raster(slope_path, centers)
    worldcover = sample_raster(wc_path, centers)
    population = sample_raster(pop_path, centers)

    rivers_path = RAW / "sungai_osm" / "sungai_sambas_osm.geojson"
    with open(rivers_path, encoding="utf-8") as f:
        rivers_json = json.load(f)
    rivers = [shape(ft["geometry"]) for ft in rivers_json["features"]]
    metric_river = unary_union([transform(to_utm, g) for g in rivers])

    events = pd.read_csv(PROC / "groundsource_sambas_2010_2025.csv")
    event_xy = events[["longitude", "latitude"]].dropna().to_numpy()
    metric_events = [Point(*to_utm(float(x), float(y))) for x, y in event_xy]
    rain = pd.read_csv(RAW / "curah_hujan" / "nasa_power_precipitation_sambas_2010_2025.csv")
    rain["precipitation_mm_day"] = pd.to_numeric(rain["precipitation_mm_day"], errors="coerce")
    rain = rain[rain["precipitation_mm_day"] >= 0]
    rain["tanggal"] = pd.to_datetime(rain["tanggal"].astype(str), format="%Y%m%d", errors="coerce")
    rain_year = rain.groupby(rain["tanggal"].dt.year)["precipitation_mm_day"].agg(["mean", "max", "sum"]).reset_index()
    rain_stats = {
        "rain_mean_mm_day": float(rain["precipitation_mm_day"].mean()),
        "rain_max_mm_day": float(rain["precipitation_mm_day"].max()),
        "rain_total_mm": float(rain["precipitation_mm_day"].sum()),
        "rain_mean_annual_total_mm": float(rain_year["sum"].mean()),
        "rain_max_annual_total_mm": float(rain_year["sum"].max()),
    }

    features = []
    for i, cell in enumerate(cells):
        event_count = sum(cell["geom_metric"].contains(p) or cell["geom_metric"].touches(p) for p in metric_events)
        distance = float(cell["geom_metric"].centroid.distance(metric_river)) if not metric_river.is_empty else np.nan
        features.append({
            "type": "Feature",
            "geometry": mapping(transform(from_utm, cell["geom_metric"])) ,
            "properties": {
                "grid_id": cell["grid_id"],
                "longitude": round(cell["longitude"], 6),
                "latitude": round(cell["latitude"], 6),
                "elevation_m": None if dem[i] == -9999 else round(dem[i], 3),
                "slope_deg": None if slope[i] == -9999 else round(slope[i], 3),
                "distance_river_m": round(distance, 2),
                "landcover_code": None if worldcover[i] in (0, -9999) else int(worldcover[i]),
                "population_2020": None if population[i] == -9999 else round(population[i], 3),
                "event_count": int(event_count),
                "risk_class_exploratory": "rendah" if event_count == 0 else ("sedang" if event_count <= 3 else "tinggi"),
                **rain_stats,
            },
        })

    collection = {"type": "FeatureCollection", "name": "feature_grid_sambas_1km", "features": features}
    geojson_path = PROC / "feature_grid_sambas_1km.geojson"
    geojson_path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    rows = [ft["properties"] for ft in features]
    pd.DataFrame(rows).to_csv(PROC / "feature_grid_sambas_1km.csv", index=False)
    rain_year.to_csv(PROC / "curah_hujan_tahunan_sambas_2010_2025.csv", index=False)
    summary = {
        "grid_cells": len(features),
        "event_points": len(metric_events),
        "events_assigned_to_grid": int(sum(r["event_count"] for r in rows)),
        "risk_class_counts": pd.Series([r["risk_class_exploratory"] for r in rows]).value_counts().to_dict(),
        "rainfall_features": rain_stats,
        "crs_grid_metric": "EPSG:32649",
        "grid_size_m": 1000,
        "note": "risk_class_exploratory is a provisional frequency class and must be validated before supervised modeling.",
    }
    (PROC / "feature_grid_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    with open(BOUNDARY_PATH, encoding="utf-8") as f:
        boundary = shape(json.load(f)["features"][0]["geometry"])
    dem_path, slope_path = build_dem(boundary)
    wc_path, pop_path = clip_rasters(boundary)
    build_features(boundary, dem_path, slope_path, wc_path, pop_path)
