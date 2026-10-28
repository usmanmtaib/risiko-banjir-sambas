from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent
e = pd.read_csv(W / "outputs_s1/sel_layak_s1.csv")
K = pd.read_csv(W / "outputs_s1/kinerja_label_s1_vs_groundsource.csv")
allc = pd.read_csv(W / "data/s1_cell_stats.csv")

fig, ax = plt.subplots(1, 3, figsize=(21, 7))
a = ax[0]
a.scatter(allc.longitude, allc.latitude, s=3, c="#e6e6e6")
v = e.frac_flag_mean.clip(lower=1e-4)
m = a.scatter(e.longitude, e.latitude, c=np.log10(v), s=9, cmap="Blues", vmin=-4, vmax=-1.5)
g = e[e.event_count > 0]
a.scatter(g.longitude, g.latitude, s=14, facecolors="none", edgecolors="r", linewidths=.7, label="Ada laporan Groundsource")
plt.colorbar(m, ax=a, shrink=.7, label="log10 fraksi piksel terdeteksi genangan (rata-rata per adegan)")
a.set_aspect("equal"); a.legend(loc="lower right"); a.set_title(f"Genangan Sentinel-1 pada {len(e):,} sel layak (abu-abu = tak layak)")

a = ax[1]
tg = ["S1: desil teratas frekuensi genangan", "Groundsource: ada laporan"]
labs = ["Sentinel-1\n(independen)", "Groundsource\n(berita)"]
mods = [("Fisik saja", "#2a6f97"), ("Pelaporan saja", "#e07a1f"), ("Populasi saja", "#8a8a8a")]
w = 0.26
for j, (mname, col) in enumerate(mods):
    vals = [K[(K.target == t) & (K.model == mname)].roc.iloc[0] for t in tg]
    sds = [K[(K.target == t) & (K.model == mname)].roc_sd.iloc[0] for t in tg]
    a.bar(np.arange(2) + (j - 1) * w, vals, w, yerr=sds, color=col, label=mname)
a.axhline(0.5, color="k", ls="--"); a.set_xticks(range(2)); a.set_xticklabels(labs); a.set_ylim(0.4, 0.85)
a.set_ylabel("ROC-AUC (CV blok spasial, sel layak yang sama)"); a.legend()
a.set_title("Prediktor yang menang berbalik antar-label")

a = ax[2]
t = e.assign(q=pd.qcut(e.frac_flag_mean.rank(method="first"), 10, labels=False) + 1)
r = t.groupby("q").event_count.apply(lambda s: (s > 0).mean() * 100)
a.bar(r.index, r.values, color="#6c757d"); a.axhline((e.event_count > 0).mean() * 100, color="r", ls="--", label="rata-rata sel layak")
a.set_xlabel("Desil frekuensi genangan radar (10 = tertinggi)"); a.set_ylabel("% sel dengan laporan Groundsource")
a.set_xticks(range(1, 11)); a.legend(); a.set_title("Laporan berita tidak naik bersama genangan radar (AUC 0,53)")
plt.tight_layout(); plt.savefig(W / "outputs_s1/ringkasan_s1.png", dpi=130)
print("ok")
