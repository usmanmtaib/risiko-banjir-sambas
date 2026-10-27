import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import shape, mapping
from shapely import from_wkb, centroid, get_x, get_y


ROOT = (Path(__file__).resolve().parent)
DATA = ROOT / "outputs" / "data" / "processed"
VIZ = (Path(__file__).resolve().parent / "outputs" / "viz")
VIZ.mkdir(parents=True, exist_ok=True)


def main():
    parquet_path = DATA / "groundsource_sambas_2010_2025.parquet"
    boundary_path = DATA / "batas_kabupaten_sambas_big.geojson"
    df = pq.read_table(parquet_path).to_pandas()
    pf = pq.ParquetFile(parquet_path)

    dates = pd.to_datetime(df["start_date"], errors="coerce")
    areas = pd.to_numeric(df["area_km2"], errors="coerce")
    geom = from_wkb(df["geometry"].tolist())
    centers = centroid(geom)
    lon = get_x(centers)
    lat = get_y(centers)

    audit = {
        "rows_csv": int(len(df)),
        "rows_parquet": int(pf.metadata.num_rows),
        "columns": list(df.columns),
        "uuid_unique": int(df["uuid"].nunique(dropna=True)),
        "uuid_duplicates": int(df["uuid"].duplicated(keep=False).sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "invalid_start_date": int(dates.isna().sum()),
        "invalid_end_date": int(pd.to_datetime(df["end_date"], errors="coerce").isna().sum()),
        "nonpositive_area_km2": int((areas <= 0).fillna(False).sum()),
        "missing_area_km2": int(areas.isna().sum()),
        "geometry_types": pd.Series([g.geom_type for g in geom]).value_counts().to_dict(),
        "centroid_lon_range": [float(lon.min()), float(lon.max())],
        "centroid_lat_range": [float(lat.min()), float(lat.max())],
        "year_range": [int(dates.dt.year.min()), int(dates.dt.year.max())],
        "area_km2_quantiles": {str(q): float(areas.quantile(q)) for q in [0.5, 0.75, 0.9, 0.99]},
    }
    (DATA / "audit_groundsource_sambas.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    yearly = df.assign(tahun=dates.dt.year).groupby("tahun", as_index=False).agg(
        jumlah_kejadian=("uuid", "size"),
        median_area_km2=("area_km2", "median"),
        total_area_km2=("area_km2", "sum"),
    )
    yearly.to_csv(DATA / "ringkasan_tahunan_groundsource_sambas.csv", index=False)

    grid = df.assign(
        tahun=dates.dt.year,
        longitude=lon,
        latitude=lat,
        grid_lat=(lat // 0.1) * 0.1,
        grid_lon=(lon // 0.1) * 0.1,
    )
    grid_summary = grid.groupby(["grid_lat", "grid_lon"], as_index=False).agg(
        jumlah_kejadian=("uuid", "size"),
        tahun_awal=("tahun", "min"),
        tahun_akhir=("tahun", "max"),
    ).sort_values("jumlah_kejadian", ascending=False)
    grid_summary.to_csv(DATA / "sebaran_grid_0p1deg_groundsource_sambas.csv", index=False)

    with open(boundary_path, encoding="utf-8") as f:
        boundary = json.load(f)
    boundary_geom = shape(boundary["features"][0]["geometry"]).simplify(0.002, preserve_topology=True)
    points = [
        {"lon": round(float(x), 5), "lat": round(float(y), 5), "year": int(yr), "area": None if pd.isna(ar) else round(float(ar), 2)}
        for x, y, yr, ar in zip(lon, lat, dates.dt.year, areas)
        if pd.notna(yr)
    ]
    point_features = []
    for _, row in df.iterrows():
        point_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
            "properties": {
                "uuid": str(row["uuid"]),
                "start_date": str(row["start_date"]),
                "end_date": str(row["end_date"]),
                "area_km2": float(row["area_km2"]),
            },
        })
    (DATA / "groundsource_sambas_2010_2025_points.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": point_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    html = f'''<div id="groundsource-sambas-map" role="img" aria-label="Peta sebaran kejadian banjir Groundsource di Kabupaten Sambas tahun 2010 sampai 2025">
  <style>
    #groundsource-sambas-map {{ font-family: inherit; color: var(--foreground); }}
    #groundsource-sambas-map svg {{ width: 100%; height: auto; display: block; }}
    #groundsource-sambas-map .boundary {{ fill: color-mix(in srgb, var(--viz-series-1) 10%, transparent); stroke: var(--foreground); stroke-width: 1.2; }}
    #groundsource-sambas-map .point {{ fill: var(--viz-series-2); opacity: .55; stroke: var(--background); stroke-width: .35; }}
    #groundsource-sambas-map .grid {{ stroke: var(--border); stroke-width: .5; opacity: .6; }}
    #groundsource-sambas-map .caption {{ fill: var(--muted-foreground); font-size: 12px; }}
  </style>
  <h3>Peta sebaran kejadian Groundsource — Kabupaten Sambas</h3>
  <svg viewBox="0 0 736 560" role="img" aria-label="753 titik kejadian banjir di dalam batas Kabupaten Sambas">
    <title>753 kejadian Groundsource di Kabupaten Sambas</title>
    <desc>Titik menunjukkan centroid geometri kejadian dan warna digunakan untuk membedakan sebaran titik dari batas administratif.</desc>
  </svg>
  <div class="caption">Periode 2010–2025 · 753 kejadian · titik = centroid geometri · area titik tidak diskalakan</div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {{
      const root = document.getElementById('groundsource-sambas-map');
      const svg = d3.select(root.querySelector('svg'));
      const boundary = {json.dumps(mapping(boundary_geom), ensure_ascii=False)};
      const points = {json.dumps(points, ensure_ascii=False)};
      const projection = d3.geoMercator().fitExtent([[30, 25], [706, 510]], boundary);
      const path = d3.geoPath(projection);
      svg.append('path').datum(boundary).attr('class', 'boundary').attr('d', path);
      svg.append('g').selectAll('circle').data(points).join('circle')
        .attr('class', 'point').attr('cx', d => projection([d.lon, d.lat])[0])
        .attr('cy', d => projection([d.lon, d.lat])[1]).attr('r', 2.5)
        .attr('data-tooltip', d => `${{d.year}} · ${{d.area === null ? 'luas tidak tersedia' : d.area + ' km²'}}`);
    }})();
  </script>
</div>'''
    viz_path = VIZ / "groundsource-sambas-map.html"
    viz_path.write_text(html, encoding="utf-8")
    print(json.dumps({"audit": audit, "visualization": str(viz_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
