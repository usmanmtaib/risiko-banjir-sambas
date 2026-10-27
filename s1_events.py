"""Uji dinamis: apakah genangan S1 naik pada jendela waktu & lokasi kejadian Groundsource?"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

W = Path(__file__).resolve().parent
Z = np.load(W / "data/s1_cell_scene.npz")
flag, valid = Z["flag_open"].astype(float), Z["valid_open"].astype(float)
sd = pd.to_datetime(Z["scene_date"])
cells = pd.read_csv(W / "data/s1_cell_stats.csv")
xk = (cells.longitude * 111.32 * np.cos(np.deg2rad(cells.latitude))).values
yk = (cells.latitude * 110.57).values
ev = pd.read_csv(str(Path(__file__).resolve().parent / "hulu" / "outputs/data/processed/groundsource_sambas_2010_2025.csv"),
                 usecols=["uuid", "area_km2", "start_date", "end_date", "longitude", "latitude"])
ev["s"] = pd.to_datetime(ev.start_date); ev["e"] = pd.to_datetime(ev.end_date)
ev = ev[(ev.s >= "2015-03-01")].copy()
print("kejadian 2015+ :", len(ev), "| luas median %.1f km2" % ev.area_km2.median())

R_KM, PRE, POST = 3.0, 1, 4  # radius sel, hari sebelum start dan sesudah end
rows = []
for _, r in ev.iterrows():
    x = r.longitude * 111.32 * np.cos(np.deg2rad(r.latitude)); y = r.latitude * 110.57
    near = np.where(np.hypot(xk - x, yk - y) <= R_KM)[0]
    if len(near) == 0:
        continue
    inwin = ((sd >= r.s - pd.Timedelta(days=PRE)) & (sd <= r.e + pd.Timedelta(days=POST)))
    if not inwin.any():
        continue
    v = valid[:, near]; f = flag[:, near]
    ok = v >= 30
    if ok[inwin].sum() == 0:
        continue
    far = ~((sd >= r.s - pd.Timedelta(days=30)) & (sd <= r.e + pd.Timedelta(days=30)))  # baseline: di luar +-30 hari
    if (ok[far]).sum() < 30:
        continue
    fe = (f[inwin] * ok[inwin]).sum() / (v[inwin] * ok[inwin]).sum()
    fb = (f[far] * ok[far]).sum() / (v[far] * ok[far]).sum()
    rows.append(dict(uuid=r.uuid, tahun=r.s.year, area=r.area_km2, n_scene_win=int(inwin.sum()), frac_event=fe, frac_base=fb, ncell=len(near)))
D = pd.DataFrame(rows)
D["selisih"] = D.frac_event - D.frac_base
D.to_csv(W / "outputs_s1/uji_kejadian.csv", index=False)
print("kejadian dengan >=1 adegan S1 valid di jendela & sel layak:", len(D))
if len(D) >= 10:
    print("frac. genangan rata-rata: jendela kejadian %.5f vs baseline sel yang sama %.5f" % (D.frac_event.mean(), D.frac_base.mean()))
    print("median selisih %.5f ; kejadian dgn selisih>0: %d (%.1f%%), selisih<0: %d, sama: %d" % (
        D.selisih.median(), (D.selisih > 0).sum(), 100 * (D.selisih > 0).mean(), (D.selisih < 0).sum(), (D.selisih == 0).sum()))
    nz = D[D.selisih != 0]
    if len(nz) >= 10:
        w = wilcoxon(nz.selisih)
        print("Wilcoxon (selisih != 0): n=%d, p=%.4f" % (len(nz), w.pvalue))
    # per tahun (volume pelaporan berubah)
    print(D.groupby(D.tahun >= 2023).agg(n=("selisih", "size"), selisih_mean=("selisih", "mean"), pos=("selisih", lambda s: (s > 0).mean())).round(5).to_string())
    big = D[D.area <= 50]
    print("hanya kejadian lokal (luas <= 50 km2): n=%d, selisih>0: %.1f%%, rata-rata selisih %.5f" % (len(big), 100 * (big.selisih > 0).mean(), big.selisih.mean()))

# ---- uji permutasi: geser jendela kelipatan 365 hari (musim sama, tahun berbeda) ----
rng = np.random.default_rng(7)
ev2 = ev.set_index("uuid")
shifts = [k * 365 for k in range(-9, 10) if k != 0]
act, nulls = [], []
for _, d in D.iterrows():
    r = ev2.loc[d.uuid]
    x = r.longitude * 111.32 * np.cos(np.deg2rad(r.latitude)); y = r.latitude * 110.57
    near = np.where(np.hypot(xk - x, yk - y) <= R_KM)[0]
    v = valid[:, near]; f = flag[:, near]; ok = v >= 30
    def frac_in(s0, e0):
        w = (sd >= s0 - pd.Timedelta(days=PRE)) & (sd <= e0 + pd.Timedelta(days=POST))
        den = (v[w] * ok[w]).sum()
        return (f[w] * ok[w]).sum() / den if den > 0 else np.nan
    a = frac_in(r.s, r.e)
    nl = [frac_in(r.s + pd.Timedelta(days=k), r.e + pd.Timedelta(days=k)) for k in shifts]
    nl = [t for t in nl if not np.isnan(t)]
    if len(nl) >= 5 and not np.isnan(a):
        act.append(a); nulls.append(np.array(nl))
act = np.array(act)
obs = act.mean()
perm = np.array([np.mean([rng.choice(n) for n in nulls]) for _ in range(2000)])
print("\nUJI PERMUTASI (jendela musim-sama tahun lain), n kejadian = %d" % len(act))
print("rata-rata frac. genangan: kejadian %.5f | jendela acak %.5f (p2.5-p97.5: %.5f-%.5f)" % (obs, perm.mean(), *np.percentile(perm, [2.5, 97.5])))
print("p-satu-sisi (jendela acak >= kejadian): %.4f" % (np.mean(perm >= obs)))
rk = np.array([(n < a_).mean() + 0.5 * (n == a_).mean() for a_, n in zip(act, nulls)])
print("persentil rata-rata kejadian terhadap jendela acak-nya: %.3f (0,5 = tak ada hubungan)" % rk.mean())
print("kejadian dengan frac > semua jendela acak: %.1f%% ; dengan frac = 0: %.1f%%" % (100 * np.mean([(a_ > n.max()) for a_, n in zip(act, nulls)]), 100 * np.mean(act == 0)))
