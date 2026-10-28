"""Analisis hasil deteksi genangan S1: respons terhadap hujan, musim, statistik per sel."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

W = Path(__file__).resolve().parent
Z = np.load(W / "data/s1_cell_scene.npz", allow_pickle=False)
flag, valid = Z["flag_open"].astype(float), Z["valid_open"].astype(float)
dates = pd.to_datetime(Z["scene_date"]); orbit = Z["scene_orbit"]
print("param deteksi (T_ABS, DELTA):", Z["params"], "| adegan-tanggal:", len(dates))

# ---------- A. deret adegan vs hujan harian ----------
ser = pd.read_csv(W / "data/s1_scene_series.csv", parse_dates=["tanggal"])
rain = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/raw/curah_hujan/nasa_power_precipitation_sambas_2010_2025.csv"))
rain["tanggal"] = pd.to_datetime(rain.tanggal.astype(str), format="%Y%m%d")
r = rain.set_index("tanggal").precipitation_mm_day.asfreq("D").fillna(0)
for k in (1, 3, 7, 14):
    ser[f"hujan_{k}d"] = ser.tanggal.map(r.rolling(k).sum())
ser["bulan"] = ser.tanggal.dt.month
print("\n== A. frac. piksel terdeteksi genangan (lahan terbuka) vs hujan sebelum akuisisi ==")
for o, g in ser.groupby("rel_orbit"):
    g = g[g.n_valid_px > 2000].dropna(subset=["hujan_3d"])
    out = {f"hujan_{k}d": round(spearmanr(g.area_flag_frac_open, g[f"hujan_{k}d"], nan_policy="omit")[0], 3) for k in (1, 3, 7, 14)}
    print(f"orbit {o}: n={len(g)}, Spearman:", out)
    q = g.hujan_7d >= g.hujan_7d.quantile(0.9)
    print(f"   rata-rata frac. genangan: hujan 7 hari top-10% = {g[q].area_flag_frac_open.mean():.4f} vs lainnya = {g[~q].area_flag_frac_open.mean():.4f}")
g = ser[ser.n_valid_px > 2000]
print("\nPer bulan (rata-rata frac. genangan lahan terbuka, semua orbit):")
print((g.groupby("bulan").area_flag_frac_open.mean() * 100).round(3).to_string())
print("\n10 adegan dengan frac. genangan tertinggi:")
print(g.nlargest(10, "area_flag_frac_open")[["tanggal", "rel_orbit", "area_flag_frac_open", "hujan_3d", "hujan_7d"]].round(3).to_string(index=False))
ser.to_csv(W / "outputs_s1_series_hujan.csv" if False else W / "data/s1_series_hujan.csv", index=False)

# ---------- B. statistik per sel ----------
ok = valid >= 30  # sel punya >=30 piksel (80 m) lahan terbuka valid pada adegan itu
frac = np.where(ok, flag / np.maximum(valid, 1), np.nan)
n_eval = ok.sum(0)
n_flood = (ok & (frac >= 0.20)).sum(0)
cells = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/feature_grid_sambas_1km.csv"), usecols=["grid_id", "longitude", "latitude", "event_count"])
cells["n_eval"], cells["n_flood"] = n_eval, n_flood
cells["flood_freq"] = np.where(n_eval > 0, n_flood / np.maximum(n_eval, 1), np.nan)
cells["pix_open_max"] = valid.max(0)
cells["frac_flag_mean"] = np.nanmean(frac, axis=0)
cells.to_csv(W / "data/s1_cell_stats.csv", index=False)
print("\n== B. sel ==")
print("sel total:", len(cells), "| dengan >=1 adegan evaluasi:", int((n_eval > 0).sum()),
      "| >=30 adegan:", int((n_eval >= 30).sum()), "| >=60 adegan:", int((n_eval >= 60).sum()))
e = cells[cells.n_eval >= 30]
print("flood_freq pada sel layak (>=30 adegan): median %.4f  p75 %.4f  p90 %.4f  p95 %.4f  p99 %.4f  maks %.3f" % tuple(np.percentile(e.flood_freq, [50, 75, 90, 95, 99, 100])))
for th in (0.01, 0.02, 0.05, 0.10):
    print(f"   sel dgn flood_freq >= {th:.2f}: {(e.flood_freq >= th).sum()} ({(e.flood_freq >= th).mean():.3f})")
print("sel layak dgn >=1 laporan Groundsource:", int((e.event_count > 0).sum()), "dari", len(e), f"({(e.event_count > 0).mean():.3f})")
print("   vs semua sel: ", f"{(cells.event_count > 0).mean():.3f}")
