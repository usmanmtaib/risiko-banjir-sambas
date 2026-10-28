"""Bangun paket reproduksi mandiri dari folder kerja ini (folder kerja tidak diubah).

    python buat_paket.py [folder_tujuan]

Default tujuan: ..\\paket_reproduksi_sambas (di samping folder ini) dan zip-nya.
Di salinan, jalur absolut C:\\Users\\... diganti jalur relatif terhadap folder paket:
  sambas_model\\...                         -> <paket>\\...
  Codex\\2026-08-27\\bn\\outputs\\data\\...      -> <paket>\\hulu\\outputs\\data\\...
"""
import hashlib
import re
import shutil
import zipfile
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
HULU_SRC = Path(r"C:\Users\USER\Documents\Codex\2026-08-27\bn")
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC.parent / "paket_reproduksi_sambas"

HULU_SCRIPTS = ["download_supporting_data.py", "process_data.py", "audit_groundsource.py", "build_sambas_features.py"]
COPY_DIRS = ["kec", "dibi", "bpbd", "data", "dashboard", "outputs", "outputs_v2", "outputs_v3", "outputs_bias",
             "outputs_sens", "outputs_s1", "outputs_kec", "outputs_article"]
SKIP = re.compile(r"(__pycache__|\.bak|server\.(log|err)$|[\\/]_baru([\\/]|$)|^data[\\/]s1([\\/]|$)|\.venv)")
HERE = "Path(__file__).resolve().parent"


def rel(s):
    return s.replace("\\", "/").lstrip("/")


def fix_paths(code, hulu=False):
    sm = r"C:\Users\USER\Documents\sambas_model"
    cx = r"C:\Users\USER\Documents\Codex\2026-08-27\bn"
    up = f"{HERE}.parent" if hulu else HERE  # skrip hulu berada di <paket>/hulu
    code = code.replace(f'Path(r"{sm}")', HERE)
    code = re.sub(re.escape(f'r"{sm}') + r'(\\[^"]*)"', lambda m: f'str({HERE} / "{rel(m.group(1))}")', code)
    cx_root = HERE if hulu else f'{HERE} / "hulu"'
    code = code.replace(f'Path(r"{cx}")', f"({cx_root})")
    code = code.replace(f'r"{cx}"', f"str({cx_root})")
    code = re.sub(re.escape(f'r"{cx}') + r'(\\[^"]*)"', lambda m: f'str({cx_root} / "{rel(m.group(1))}")', code)
    # gambar naskah: folder laporan tidak dikemas, simpan bersama keluaran statistik
    code = code.replace('FIG = W / "laporan/fig"', 'FIG = W / "outputs_article"')
    codex_viz = re.escape('Path(r"C:\\Users\\USER\\.codex') + r'[^"]*"\)'
    code = re.sub(codex_viz, f'({HERE} / "outputs" / "viz")', code)  # folder gambar audit Codex
    # string hasil str(...) dipakai dengan P + r"\nama.tif": tetap sah di Windows
    if "Path(__file__)" in code and not re.search(r"^from pathlib import .*Path|^import pathlib", code, re.M):
        lines = code.split("\n")
        i = next(n for n, l in enumerate(lines) if l.startswith(("import ", "from ")))
        lines.insert(i, "from pathlib import Path")
        code = "\n".join(lines)
    return code


def copy_tree(src, dst):
    for f in src.rglob("*"):
        r = f.relative_to(SRC if src.is_relative_to(SRC) else src.parent)
        if f.is_dir() or SKIP.search(str(r)):
            continue
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)


def main():
    # kosongkan tujuan, tetapi pertahankan riwayat git (.git), .gitignore, dan .gitattributes
    # (.gitattributes berisi "* -text" agar repo menyimpan byte yang sama dengan SHA256SUMS.txt)
    DST.mkdir(parents=True, exist_ok=True)
    for p in DST.iterdir():
        if p.name in (".git", ".gitignore", ".gitattributes"):
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    n = 0
    for f in sorted(SRC.glob("*.py")):
        if f.name == Path(__file__).name:
            shutil.copy2(f, DST / f.name)
        else:
            (DST / f.name).write_text(fix_paths(f.read_text(encoding="utf-8")), encoding="utf-8")
        n += 1
    for d in COPY_DIRS:
        copy_tree(SRC / d, DST / d)
    # tahap hulu: skrip + data olahan kecil (data mentah besar diunduh ulang, lihat README)
    (DST / "hulu").mkdir()
    for s in HULU_SCRIPTS:
        (DST / "hulu" / s).write_text(fix_paths((HULU_SRC / "work" / s).read_text(encoding="utf-8"), hulu=True), encoding="utf-8")
    shutil.copy2(HULU_SRC / "work" / "data" / "DATA_SOURCES.md", DST / "hulu" / "DATA_SOURCES.md")
    copy_tree(HULU_SRC / "outputs" / "data" / "processed", DST / "hulu" / "outputs" / "data" / "processed")
    copy_tree(HULU_SRC / "outputs" / "data" / "raw" / "curah_hujan", DST / "hulu" / "outputs" / "data" / "raw" / "curah_hujan")
    for f in ["README.md", "requirements.txt"]:
        if (SRC / "paket" / f).exists():
            shutil.copy2(SRC / "paket" / f, DST / f)
    # daftar checksum
    # urutan tidak peka huruf besar-kecil, sama di semua versi Python
    files = sorted((p for p in DST.rglob("*") if p.is_file() and ".git" not in p.relative_to(DST).parts
                    and p.name not in (".gitignore", ".gitattributes")),
                   key=lambda p: p.relative_to(DST).as_posix().lower())
    rows = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(DST).as_posix()}" for f in files]
    # LF agar sha256sum -c dapat membaca nama berkas (CRLF menambahkan \r ke nama)
    (DST / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    files.append(DST / "SHA256SUMS.txt")
    z = DST.parent / f"{DST.name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, (Path(DST.name) / f.relative_to(DST)).as_posix())
    size = sum(f.stat().st_size for f in files)
    print(f"skrip {n}, berkas {len(rows)}, ukuran {size / 1e6:.1f} MB, zip {z.stat().st_size / 1e6:.1f} MB -> {z}")


if __name__ == "__main__":
    main()
