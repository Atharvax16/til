#!/usr/bin/env python3
"""Remove PNGs in data/images/ that the current data/metadata.csv no longer references.

Switching corpora (synthetic -> real) leaves the previous corpus's PNGs behind. They are
unreferenced, but they still occupy disk. This deletes exactly the unreferenced ones.

    python scripts/clean_images.py           # dry run, lists what would go
    python scripts/clean_images.py --yes     # actually delete
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images"
META = ROOT / "data" / "metadata.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="perform the deletion")
    args = ap.parse_args()

    if not META.exists():
        print(f"no {META} - nothing to compare against; refusing to delete blind.")
        return 1

    keep = set(pd.read_csv(META)["image_id"])
    on_disk = {p.name for p in IMG.glob("*.png")}
    stale = sorted(on_disk - keep)
    missing = sorted(keep - on_disk)

    print(f"referenced by metadata : {len(keep)}")
    print(f"present on disk        : {len(on_disk)}")
    print(f"unreferenced (stale)   : {len(stale)}")
    if missing:
        print(f"WARNING: {len(missing)} referenced images are missing from disk, "
              f"e.g. {missing[:3]} - rebuild before trusting a run.")

    if not stale:
        print("nothing to do.")
        return 0

    freed = sum((IMG / f).stat().st_size for f in stale)
    print(f"would free             : {freed / 2**20:.1f} MB")
    print("examples:", ", ".join(stale[:5]))

    if not args.yes:
        print("\ndry run. re-run with --yes to delete.")
        return 0

    for f in stale:
        (IMG / f).unlink()
    print(f"deleted {len(stale)} files, freed {freed / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
