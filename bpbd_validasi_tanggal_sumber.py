"""Varian bpbd_validasi.py dengan tanggal dari sumber asli (28 Okt 2026): Galing 6-7 Nov 2022 (Sambas Times), Maret 2023 pada 2-4 Mar (DIBI)."""
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent
P = pd.read_csv(W / "outputs_kec/panel_kecamatan_hari.csv", parse_dates=["tanggal"])
O = pd.read_csv(W / "outputs_kec/prediksi_oof_terbaik.csv", parse_dates=["tanggal"])
KECS = sorted(P.kec.unique())
D = pd.date_range("2012-01-01", "2025-12-31")
Y = P.pivot(index="tanggal", columns="kec", values="y").reindex(D).fillna(0)
S = O.pivot(index="tanggal", columns="kec", values="p").reindex(pd.date_range("2021-01-01", "2025-12-31"))
R3 = P.pivot(index="tanggal", columns="kec", values="r3")

# Peristiwa bertanggal dan kecamatan yang DISEBUT NAMANYA di tabel pengguna
EV = [
    ("2021-02-07", "2021-02-09", ["Teluk Keramat"], 13, "Feb 2021 (51 desa, 13 kec.)"),
    ("2022-08-05", "2022-08-05", ["Sambas"], 1, "Agu 2022 (Semangau)"),
    ("2022-11-06", "2022-11-07", ["Galing"], 1, "Nov 2022 (Tempapan Hulu)"),
    ("2023-01-29", "2023-02-02", ["Tangaran"], 8, "Jan-Feb 2023 (8 kec.)"),
    ("2023-03-02", "2023-03-04", ["Sejangkung"], 15, "Mar 2023 (63.519 jiwa)"),
    ("2025-01-18", "2025-01-22", ["Tebas", "Subah", "Sejangkung", "Paloh", "Selakau Timur", "Galing"], 12, "Puncak 18-22 Jan 2025"),
    ("2025-01-13", "2025-02-12", ["Tebas", "Subah", "Sejangkung", "Paloh", "Selakau Timur", "Galing"], 12, "Seluruh 13 Jan-12 Feb 2025"),
]
PRE, POST = 1, 1


def hit(kec, a, b):
    return Y.loc[a - pd.Timedelta(days=PRE): b + pd.Timedelta(days=POST), kec].max() >= 1


rows = []
for s, e, kecs, ntot, lab in EV:
    a, b = pd.Timestamp(s), pd.Timestamp(e)
    L = (b - a).days
    for k in kecs:
        h = hit(k, a, b)
        # dasar: jendela musim-sama, tahun lain, dalam rezim pelaporan 2021-2025
        base = []
        for sh in [-4, -3, -2, -1, 1, 2, 3, 4]:
            aa, bb = a + pd.Timedelta(days=365 * sh), b + pd.Timedelta(days=365 * sh)
            if aa >= pd.Timestamp("2021-01-01") and bb <= pd.Timestamp("2025-12-31") and abs(aa.year - a.year) >= 1:
                base.append(float(hit(k, aa, bb)))
        row = dict(peristiwa=lab, kec=k, groundsource_positif=bool(h), dasar_musim_sama=round(np.mean(base), 2) if base else np.nan, n_dasar=len(base))
        if a >= pd.Timestamp("2021-01-01"):
            win = S.loc[a - pd.Timedelta(days=PRE): b + pd.Timedelta(days=POST), k]
            mx = win.max()
            allmax = S[k].rolling(L + 1 + PRE + POST).max().dropna()
            row["persentil_skor_maks"] = round(float((allmax <= mx).mean() * 100))
            r = R3.loc[a - pd.Timedelta(days=PRE): b + pd.Timedelta(days=POST), k].max()
            row["hujan3h_maks_mm"] = round(float(r))
        rows.append(row)
T = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("=== A. Kecamatan yang disebut namanya, dengan tanggal ===")
print(T.to_string(index=False))
sub = T[~T.peristiwa.str.startswith("Seluruh")]
print(f"\nKecamatan-peristiwa (tanpa baris 'seluruh periode'): {len(sub)} ; Groundsource positif: {int(sub.groundsource_positif.sum())} ({sub.groundsource_positif.mean()*100:.0f}%)")
print("Peluang dasar rata-rata (jendela musim-sama di tahun lain, 2021-2025): %.2f" % sub.dasar_musim_sama.mean())
if sub.persentil_skor_maks.notna().any():
    print("Persentil skor model (maks dalam jendela), median: %.0f ; >=90: %d dari %d" % (sub.persentil_skor_maks.median(), (sub.persentil_skor_maks >= 90).sum(), sub.persentil_skor_maks.notna().sum()))

# peristiwa tingkat kabupaten
print("\n=== B. Peristiwa tingkat kabupaten (tanpa kecamatan spesifik) ===")
for s, e, lab in [("2020-11-18", "2020-11-19", "Nov 2020 (pesisir/bantaran, rob)"), ("2025-12-08", "2025-12-13", "Des 2025 (8-13 Des)")]:
    a, b = pd.Timestamp(s), pd.Timestamp(e)
    w = Y.loc[a - pd.Timedelta(days=1): b + pd.Timedelta(days=1)]
    print(f"{lab}: kecamatan dengan positif Groundsource = {int((w.max() >= 1).sum())}/19")

# 2024 rekapitulasi (tanpa tanggal)
print("\n=== C. Rekapitulasi 2024 (BPBD: 17 kejadian banjir; 47 desa di 12 kecamatan) vs Groundsource 2024 ===")
y24 = Y.loc["2024-01-01":"2024-12-31"]
hari = y24.index[(y24.sum(axis=1) > 0)]
print("Groundsource 2024: hari unik berlabel positif =", len(hari), "; kecamatan dengan >=1 hari positif =", int((y24.max() >= 1).sum()), "dari 19")
print("   per kecamatan (jumlah hari positif):", {k: int(v) for k, v in y24.sum().sort_values(ascending=False).items() if v > 0})
named = ["Tebas", "Subah", "Sejangkung", "Sajad", "Sebawi", "Selakau Timur"]
print("   enam kecamatan yang disebut namanya di tabel pengguna: positif di Groundsource 2024 ->", {k: int(y24[k].sum()) for k in named})
# kelompok kejadian: hitung gerombol hari berurutan (jeda <=3 hari)
d = list(hari); cl = 0; last = None
for x in d:
    if last is None or (x - last).days > 3:
        cl += 1
    last = x
print("   gerombol hari positif (jeda <= 3 hari) =", cl, "(pembanding: 17 kejadian menurut tabel pengguna)")
