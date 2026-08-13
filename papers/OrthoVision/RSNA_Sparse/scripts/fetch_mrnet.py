#!/usr/bin/env python3
"""Fetch / unpack MRNet-v1.0 into data/raw.

MRNet is access-controlled. Stanford emails a (time-limited) download link after you
sign the research-use agreement:

    https://stanfordmlgroup.github.io/competitions/mrnet/

Then, from the repo root:

    python scripts/fetch_mrnet.py '<the emailed url>'
    python scripts/fetch_mrnet.py ~/Downloads/MRNet-v1.0.zip    # already downloaded

The archive is ~5.7 GB. The link is signed and expires, so quote it - it contains '&'.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PLANES = ("axial", "coronal", "sagittal")


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"already downloaded: {dest} ({dest.stat().st_size / 2**30:.2f} GB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading -> {tmp}")
    t0 = time.time()

    def hook(blocks, bs, total):
        got = blocks * bs
        if total > 0:
            pct, gb = 100 * got / total, total / 2**30
            rate = got / max(time.time() - t0, 1e-6) / 2**20
            print(f"\r  {pct:5.1f}% of {gb:.2f} GB  ({rate:.1f} MB/s)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    tmp.rename(dest)
    return dest


def unpack(zip_path: Path, out: Path) -> None:
    print(f"unpacking {zip_path.name} -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        for i, m in enumerate(members, 1):
            z.extract(m, out)
            if i % 2000 == 0 or i == len(members):
                print(f"\r  {i}/{len(members)} entries", end="", flush=True)
    print()


def flatten(out: Path) -> None:
    """The zip carries its own MRNet-v1.0/ top level; lift it into data/raw."""
    if any((out / s).is_dir() for s in ("train", "valid")):
        return
    nested = [d for d in out.iterdir()
              if d.is_dir() and any((d / s).is_dir() for s in ("train", "valid"))]
    if len(nested) == 1:
        print(f"lifting {nested[0].name}/ into {out}")
        for item in list(nested[0].iterdir()):
            shutil.move(str(item), str(out / item.name))
        nested[0].rmdir()


def verify(out: Path) -> bool:
    ok = True
    for split in ("train", "valid"):
        if not (out / split).is_dir():
            print(f"  MISSING  {split}/")
            ok = False
            continue
        for plane in PLANES:
            d = out / split / plane
            n = len(list(d.glob("*.npy"))) if d.is_dir() else 0
            print(f"  {'ok ' if n else 'MISSING'}  {split}/{plane:9s} {n:5d} volumes")
            ok &= n > 0
        for name in ("abnormal", "acl", "meniscus"):
            f = out / f"{split}-{name}.csv"
            print(f"  {'ok ' if f.exists() else 'MISSING'}  {f.name}")
            ok &= f.exists()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?",
                    help="Stanford download URL, or a path to an already-downloaded zip")
    ap.add_argument("--out", type=Path, default=RAW, help=f"destination (default {RAW})")
    ap.add_argument("--keep-zip", action="store_true", help="keep the archive after unpacking")
    args = ap.parse_args()

    if args.source is None:
        print(__doc__)
        return 1

    if args.source.startswith(("http://", "https://")):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        zip_path = download(args.source, args.out.parent / "MRNet-v1.0.zip")
    else:
        zip_path = Path(args.source).expanduser()
        if not zip_path.exists():
            print(f"no such file: {zip_path}")
            return 1

    unpack(zip_path, args.out)
    flatten(args.out)

    print("\nverifying layout:")
    if not verify(args.out):
        print("\nlayout incomplete - check the archive contents.")
        return 1

    if not args.keep_zip and zip_path.parent == args.out.parent:
        print(f"\nremoving {zip_path.name} (pass --keep-zip to keep it)")
        zip_path.unlink()

    print("\nready. Run the notebook: it will pick up data/raw automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
