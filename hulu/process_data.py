from pathlib import Path
import os
import json
import ssl
import urllib.parse
import urllib.request
from io import StringIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import centroid, contains_xy, from_wkb, get_x, get_y


ROOT = str(Path(__file__).resolve().parent)
RAW = os.path.join(ROOT, "outputs", "data", "raw", "groundsource_2026.parquet")
OUT = os.path.join(ROOT, "outputs", "data", "processed")
os.makedirs(OUT, exist_ok=True)


def filter_groundsource():
    boundary_path = os.path.join(OUT, "batas_kabupaten_sambas_big.geojson")
    if not os.path.exists(boundary_path):
        raise FileNotFoundError("Batas BIG belum tersedia: " + boundary_path)
    with open(boundary_path, encoding="utf-8") as f:
        boundary_json = json.load(f)
    from shapely.geometry import shape

    boundary = shape(boundary_json["features"][0]["geometry"])
    source = pq.ParquetFile(RAW)
    batches = []
    total = 0
    for rg in range(source.num_row_groups):
        table = source.read_row_group(rg)
        total += table.num_rows
        geoms = from_wkb(table["geometry"].to_pylist())
        points = centroid(geoms)
        xs = get_x(points)
        ys = get_y(points)
        mask = contains_xy(boundary, xs, ys)
        if mask.any():
            selected = table.filter(pa.array(mask))
            selected = selected.append_column("longitude", pa.array(xs[mask]))
            selected = selected.append_column("latitude", pa.array(ys[mask]))
            batches.append(selected)

    if not batches:
        raise RuntimeError("Tidak ada kejadian Groundsource pada bounding box Sambas.")
    result = pa.concat_tables(batches, promote_options="default").to_pandas()
    dates = pd.to_datetime(result["start_date"], errors="coerce")
    df = result[(dates.dt.year >= 2010) & (dates.dt.year <= 2025)].copy()
    df["start_date"] = dates.loc[df.index].dt.strftime("%Y-%m-%d")
    out_parquet = os.path.join(OUT, "groundsource_sambas_2010_2025.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=True), out_parquet, compression="zstd")
    df.to_csv(os.path.join(OUT, "groundsource_sambas_2010_2025.csv"), index=False)
    return total, len(df), out_parquet


def fetch_bnpb():
    url = "https://gis.bnpb.go.id/databencana/tabel/pencarian.php"
    payload = urllib.parse.urlencode(
        {
            "s_provinsi": "KALIMANTAN BARAT",
            "s_kejadian": "BANJIR",
            "tgl_awal": "01-01-2010",
            "tgl_akhir": "31-12-2025",
            "submit": "Cari",
        }
    ).encode()
    request = urllib.request.Request(url, data=payload, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    html = urllib.request.urlopen(request, context=context, timeout=120).read().decode("utf-8", "ignore")
    table = pd.read_html(StringIO(html))[0]
    table["Tanggal Kejadian"] = pd.to_datetime(table["Tanggal Kejadian"], errors="coerce")
    sambas = table[table["Kabupaten"].astype(str).str.strip().eq("Sambas")].copy()
    sambas["tahun"] = sambas["Tanggal Kejadian"].dt.year.astype("Int64")
    sambas.to_csv(os.path.join(OUT, "bnpb_banjir_sambas_2010_2025.csv"), index=False)
    return len(table), len(sambas), sambas


def merge_annual_validation(gs_csv, bnpb):
    gs = pd.read_csv(gs_csv)
    gs["tahun"] = pd.to_datetime(gs["start_date"], errors="coerce").dt.year
    a = gs.groupby("tahun", dropna=True).size().rename("groundsource_count")
    b = bnpb.groupby("tahun", dropna=True).size().rename("bnpb_count")
    merged = pd.concat([a, b], axis=1).fillna(0).astype(int).reset_index()
    merged["selisih_groundsource_minus_bnpb"] = merged["groundsource_count"] - merged["bnpb_count"]
    merged.to_csv(os.path.join(OUT, "validasi_tahunan_groundsource_bnpb.csv"), index=False)
    return merged


if __name__ == "__main__":
    total, selected, gs_path = filter_groundsource()
    bnpb_total, bnpb_selected, bnpb = fetch_bnpb()
    gs_csv = os.path.join(OUT, "groundsource_sambas_2010_2025.csv")
    merged = merge_annual_validation(gs_csv, bnpb)
    print(f"Groundsource: {selected} baris terpilih dari {total} baris")
    print(f"BNPB: {bnpb_selected} baris Sambas dari {bnpb_total} baris hasil query")
    print(f"Groundsource parquet: {gs_path}")
    print(f"Validasi tahunan: {len(merged)} tahun")
