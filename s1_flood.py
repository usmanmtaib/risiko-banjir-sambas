"""Deteksi genangan Sentinel-1 VV (perubahan terhadap median referensi per orbit) -> statistik per grid 1 km.

Pemakaian:
  python s1_flood.py calib            # kalibrasi ambang dari distribusi air vs darat
  python s1_flood.py run T_ABS DELTA  # jalankan deteksi (dB)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import s1_common as c

W = Path(__file__).resolve().parent
S1 = W / "data/s1"
meta = pd.read_csv(S1 / "_meta.csv")
masks = np.load(W / "data/s1_masks.npz")
H, Wd = c.NROW, c.NCOL

# ---- peta piksel -> sel grid 1 km (kunci kiri-bawah UTM, sama dengan grid_id) ----
E = c.X0 + (np.arange(Wd) + 0.5) * c.RES
N = c.Y1 - (np.arange(H) + 0.5) * c.RES
gi = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/feature_grid_sambas_1km.csv"), usecols=["grid_id"])
ids = gi.grid_id.str.split("_", expand=True)
key2idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ids[1], ids[2]))}
ke = (np.floor(E / 1000) * 1000).astype(int)
kn = (np.floor(N / 1000) * 1000).astype(int)
# vektorisasi: gunakan pencarian dictionary per pasangan unik
ue = {e: None for e in np.unique(ke)}
un = {n: None for n in np.unique(kn)}
lut = np.full((len(ue), len(un)), -1, dtype="int32")
ue_idx = {e: i for i, e in enumerate(sorted(ue))}
un_idx = {n: i for i, n in enumerate(sorted(un))}
for (a, b), i in key2idx.items():
    if a in ue_idx and b in un_idx:
        lut[ue_idx[a], un_idx[b]] = i
ei = np.array([ue_idx[e] for e in ke]); ni = np.array([un_idx[n] for n in kn])
cell = lut[ei[None, :], ni[:, None]]  # (H, W)
NCELL = len(gi)


def load_group(orbit_key):
    """Muat tumpukan satu orbit relatif: gabungkan bingkai pada tanggal yang sama."""
    m = meta[meta.rel_orbit == orbit_key].sort_values("datetime")
    by_date = {}
    for _, r in m.iterrows():
        f = S1 / f"{r.id}.npy"
        if not f.exists():  # adegan polarisasi tunggal VH (tanpa VV) tidak diunduh
            continue
        a = np.load(f)
        if a.shape != (H, Wd):
            continue
        d = r.tanggal
        if d in by_date:
            fill = by_date[d] == -32768
            by_date[d][fill] = a[fill]
        else:
            by_date[d] = a.copy()
    dates = sorted(by_date)
    return dates, np.stack([by_date[d] for d in dates]) if dates else (dates, None)


def chunked_median(stack):
    ref = np.full((H, Wd), np.nan, dtype="float32")
    nval = (stack != -32768).sum(axis=0).astype("int16")
    for r0 in range(0, H, 100):
        blk = stack[:, r0:r0 + 100].astype("float32") / 100.0
        blk[stack[:, r0:r0 + 100] == -32768] = np.nan
        with np.errstate(all="ignore"):
            ref[r0:r0 + 100] = np.nanmedian(blk, axis=0)
    return ref, nval


mode = sys.argv[1]
orbits = sorted(meta.rel_orbit.unique())

if mode == "calib":
    water = masks["frac_water"] > 0.9
    tree = (masks["frac_tree"] > 0.9) & (masks["has_wc"] > 0.5)
    crop = masks["frac_crop"] > 0.9
    print("piksel air murni:", int(water.sum()), "| pohon murni:", int(tree.sum()), "| sawah/ladang murni:", int(crop.sum()))
    for o in orbits:
        dates, st = load_group(o)
        v = st != -32768
        print(f"\norbit {o}: {len(dates)} tanggal, cakupan valid rata-rata {v.mean():.3f}")
        for name, mk in [("air", water), ("pohon", tree), ("sawah/ladang", crop)]:
            x = st[:, mk]; x = x[x != -32768] / 100.0
            if x.size:
                print(f"  {name:13s} n={x.size:>10d}  p5 {np.percentile(x,5):6.1f}  p25 {np.percentile(x,25):6.1f}  median {np.median(x):6.1f}  p75 {np.percentile(x,75):6.1f}  p95 {np.percentile(x,95):6.1f}")
        x = st[:, water]; x = x[x != -32768] / 100.0
        y = st[:, tree | crop]; y = y[y != -32768] / 100.0
        best, bt = -1, None
        for t in np.arange(-24, -8, 0.5):  # ambang yang memaksimalkan balanced accuracy air vs darat
            ba = 0.5 * ((x < t).mean() + (y >= t).mean())
            if ba > best:
                best, bt = ba, t
        print(f"  ambang optimum air-vs-darat: {bt} dB (balanced accuracy {best:.3f})")

elif mode == "run":
    T_ABS, DELTA = float(sys.argv[2]), float(sys.argv[3])
    SUF = sys.argv[4] if len(sys.argv) > 4 else ""
    T_REF = T_ABS + DELTA  # piksel referensi harus bukan air permanen
    slope_ok = np.nan_to_num(masks["slope"], nan=0) < 10
    not_water = masks["frac_water"] < 0.5
    not_built = masks["frac_built"] < 0.5
    base_mask = slope_ok & not_water & not_built
    open_mask = base_mask & (masks["frac_tree"] < 0.5)
    rows = []
    cellflag, cellvalid, cellflag_open, cellvalid_open = [], [], [], []
    pix_flag = np.zeros((H, Wd), dtype="int16"); pix_valid = np.zeros((H, Wd), dtype="int16")
    for o in orbits:
        dates, st = load_group(o)
        ref, nval = chunked_median(st)
        usable = base_mask & (nval >= 15) & (ref >= T_REF)  # referensi = darat, bukan air/lahan basah permanen
        print(f"orbit {o}: {len(dates)} tanggal; piksel layak {int(usable.sum())} ({usable.mean():.3f} dari grid)", flush=True)
        for t, d in enumerate(dates):
            a = st[t].astype("float32") / 100.0
            valid = (st[t] != -32768) & usable
            flag = valid & (a < T_ABS) & ((ref - a) >= DELTA)
            pix_flag += flag.astype("int16"); pix_valid += valid.astype("int16")
            ci = cell
            ok = ci >= 0
            nv = np.bincount(ci[valid & ok], minlength=NCELL)
            nf = np.bincount(ci[flag & ok], minlength=NCELL)
            vo = valid & open_mask; fo = flag & open_mask
            nvo = np.bincount(ci[vo & ok], minlength=NCELL); nfo = np.bincount(ci[fo & ok], minlength=NCELL)
            cellflag.append(nf); cellvalid.append(nv); cellflag_open.append(nfo); cellvalid_open.append(nvo)
            rows.append(dict(tanggal=d, rel_orbit=o, area_flag_frac=flag.sum() / max(valid.sum(), 1),
                             area_flag_frac_open=fo.sum() / max((valid & open_mask).sum(), 1),
                             n_valid_px=int(valid.sum())))
    pd.DataFrame(rows).to_csv(W / f"data/s1_scene_series{SUF}.csv", index=False)
    np.savez_compressed(W / f"data/s1_cell_scene{SUF}.npz", flag=np.array(cellflag, dtype="int16"),
                        valid=np.array(cellvalid, dtype="int16"), flag_open=np.array(cellflag_open, dtype="int16"),
                        valid_open=np.array(cellvalid_open, dtype="int16"),
                        scene_date=np.array([r["tanggal"] for r in rows]), scene_orbit=np.array([r["rel_orbit"] for r in rows]),
                        params=np.array([T_ABS, DELTA]))
    np.savez_compressed(W / f"data/s1_pixel_stats{SUF}.npz", pix_flag=pix_flag, pix_valid=pix_valid)
    print("selesai:", len(rows), "adegan-tanggal")
