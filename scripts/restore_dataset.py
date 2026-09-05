#!/usr/bin/env python3
"""Restore and VERIFY the College Experience archive against its provenance.

The archive is 2.76 GB, is not redistributable, and is therefore absent from
every fresh checkout. Without it nothing in this project can be re-run. This
script does two things and refuses to do a third:

    python3 scripts/restore_dataset.py            # verify what is on disk
    python3 scripts/restore_dataset.py --download # fetch it, then verify

It NEVER invents, approximates or substitutes data. If the archive is absent or
its digests disagree with ``data/raw/college-experience/PROVENANCE.json``, it
says so and exits non-zero.

CREDENTIALS ARE YOURS TO SUPPLY. ``--download`` shells out to the Kaggle CLI,
which reads ``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME / KAGGLE_KEY). This
script does not read, write, prompt for or transmit your credentials; if they
are not configured, the Kaggle CLI will say so and this script stops.

Exit codes:
    0  the archive is present and every recorded digest matches
    4  the archive is present but a digest DISAGREES with the provenance record
    6  the archive is absent
    7  --download was requested and the download failed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "college-experience"
PROVENANCE = DATA / "PROVENANCE.json"


def sha256_prefix(path: Path, limit: int) -> str:
    """Digest of the first ``limit`` bytes — the convention PROVENANCE uses."""
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while read < limit:
            chunk = f.read(min(1 << 20, limit - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def verify(record: dict) -> int:
    digests = record.get("key_file_digests", {})
    if not digests:
        print("PROVENANCE.json records no digests; nothing to verify against.",
              file=sys.stderr)
        return 4

    missing, mismatched, ok = [], [], []
    for rel, want in digests.items():
        p = DATA / rel
        if not p.exists():
            missing.append(rel)
            continue
        size = p.stat().st_size
        if size != want["bytes"]:
            mismatched.append((rel, f"size {size:,} vs recorded "
                                    f"{want['bytes']:,}"))
            continue
        got = sha256_prefix(p, int(want["hashed_bytes"]))
        if got != want["sha256_first_bytes"]:
            mismatched.append((rel, f"sha256 {got[:16]}… vs recorded "
                                    f"{want['sha256_first_bytes'][:16]}…"))
        else:
            ok.append(rel)

    for rel in ok:
        print(f"  OK        {rel}")
    for rel in missing:
        print(f"  MISSING   {rel}")
    for rel, why in mismatched:
        print(f"  MISMATCH  {rel}  ({why})")

    if missing and not mismatched and not ok:
        print("\nThe archive is not on disk. Nothing was estimated or "
              "substituted.", file=sys.stderr)
        return 6
    if missing or mismatched:
        print("\nThe archive on disk does NOT match the recorded provenance. "
              "Do not run experiments against it: results would not be "
              "comparable to the pre-registered run.", file=sys.stderr)
        return 4
    print(f"\nAll {len(ok)} recorded files match. The archive is the one the "
          "original experiment used.")
    return 0


def download(record: dict) -> int:
    slug = "subigyanepal/college-experience-dataset"
    src = record.get("official_source", "")
    if "kaggle.com/datasets/" in src:
        slug = src.split("kaggle.com/datasets/", 1)[1].strip("/")
    if shutil.which("kaggle") is None:
        print("The Kaggle CLI is not on PATH. Install it with "
              "`pip install kaggle`, then configure credentials.",
              file=sys.stderr)
        return 7
    DATA.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA),
           "--unzip"]
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\nThe download failed. The usual cause is missing credentials: "
              "place kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > "
              "Create New Token), or set KAGGLE_USERNAME and KAGGLE_KEY. "
              "Accepting the dataset's terms on its Kaggle page is also "
              "required before the API will serve it.", file=sys.stderr)
        return 7
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="fetch from the source recorded in PROVENANCE.json")
    args = ap.parse_args()

    if not PROVENANCE.exists():
        print(f"missing {PROVENANCE}", file=sys.stderr)
        return 6
    record = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    print(f"dataset: {record.get('dataset')}")
    print(f"source:  {record.get('official_source')}")
    print(f"doi:     {record.get('doi')}")
    print(f"recorded: {record.get('file_count')} files, "
          f"{record.get('total_bytes', 0) / 1e9:.2f} GB\n")

    if args.download:
        rc = download(record)
        if rc:
            return rc
        print()

    return verify(record)


if __name__ == "__main__":
    raise SystemExit(main())
