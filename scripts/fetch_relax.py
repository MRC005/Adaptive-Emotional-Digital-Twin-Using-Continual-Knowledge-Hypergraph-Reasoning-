#!/usr/bin/env python3
"""Reproducible partial acquisition of the RELAX dataset (Zenodo 20701999).

    python scripts/fetch_relax.py --root data/raw/relax

WHY PARTIAL. The published archive is a single 16.5 GB zip. Roughly 15.9 GB of
that is 52 Hz accelerometer data which this analysis does not use. This script
uses HTTP range requests to read the remote zip's central directory and then
pull only the members that are scientifically required (~0.5 GB):

    questionnaire_responses.xlsx   the repeated self-reports
    metadata/questionnaires.xlsx   the item and ANSWER-LABEL definitions
    metadata/README.md             the data dictionary
    data/<pid>/ibi_data.parquet    interbeat intervals, 31 participants

Pass --with-acc to additionally fetch the accelerometer files (16.5 GB total).

Source: Halmich C., Jung O., Schmoigl-Tonis M., Schranz C., Kremser W.,
Kunas B. & Laireiter A.-R. (2026). "A six-week longitudinal dataset of wearable
and self-reported stress measurements in working adults". Scientific Data.
Zenodo DOI 10.5281/zenodo.20701999. Licence CC-BY-4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

RECORD = "20701999"
API = f"https://zenodo.org/api/records/{RECORD}"
ZIP_NAME = "RELAXDataset.zip"
URL = f"{API}/files/{ZIP_NAME}/content"
DOI = "10.5281/zenodo.20701999"
LICENCE = "CC-BY-4.0"


def curl_range(start: int, length: int, out: Path, attempts: int = 6) -> bool:
    """Zenodo throttles bursts, so back off and retry rather than failing."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for a in range(attempts):
        r = subprocess.run(
            ["curl", "-sS", "--max-time", "1800", "--retry", "4",
             "--retry-delay", "5", "--retry-all-errors",
             "-r", f"{start}-{start + length - 1}", "-o", str(out), URL],
            capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size == length:
            return True
        wait = 5 * (2 ** a)
        print(f"      retry {a + 1}/{attempts} in {wait}s", flush=True)
        time.sleep(wait)
    return False


def remote_size() -> int:
    r = subprocess.run(["curl", "-sSIL", "--max-time", "120", URL],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("could not determine the remote archive size")


def read_central_directory(size: int) -> list[tuple[str, int, int, int]]:
    """(name, uncompressed, compressed, local_header_offset) for every member."""
    tail = Path("/tmp/_relax_tail.bin")
    span = min(2_000_000, size)
    if not curl_range(size - span, span, tail):
        raise RuntimeError("could not read the archive tail")
    data = tail.read_bytes()
    base = size - len(data)

    j = data.rfind(b"PK\x06\x07")                    # zip64 EOCD locator
    if j >= 0:
        _, _, z64, _ = struct.unpack("<IIQI", data[j:j + 20])
        rec = struct.unpack("<IQHHIIQQQQ", data[z64 - base: z64 - base + 56])
        nrec, cd_size, cd_off = rec[7], rec[8], rec[9]
    else:
        i = data.rfind(b"PK\x05\x06")
        f = struct.unpack("<IHHHHIIH", data[i:i + 22])
        nrec, cd_size, cd_off = f[4], f[5], f[6]

    cd = data[cd_off - base: cd_off - base + cd_size]
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        h = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        nlen, elen, clen = h[10], h[11], h[12]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        csize, usize, lho = h[8], h[9], h[16]
        extra = cd[p + 46 + nlen: p + 46 + nlen + elen]
        if 0xFFFFFFFF in (usize, csize, lho):        # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                tid, tsz = struct.unpack("<HH", extra[q:q + 4])
                body, r = extra[q + 4:q + 4 + tsz], 0
                if tid == 0x0001:
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
                q += 4 + tsz
        out.append((name, usize, csize, lho))
        p += 46 + nlen + elen + clen
    if len(out) != nrec:
        raise RuntimeError(f"parsed {len(out)} of {nrec} central-directory entries")
    return out


def extract(name: str, usize: int, csize: int, lho: int, dest: Path) -> int:
    tmp = Path(str(dest) + ".part")
    if not curl_range(lho, 30 + 4096 + csize, tmp):
        raise RuntimeError(f"range fetch failed for {name}")
    raw = tmp.read_bytes()
    if raw[:4] != b"PK\x03\x04":
        raise RuntimeError(f"not a local file header for {name}")
    method, = struct.unpack("<H", raw[8:10])
    nlen, elen = struct.unpack("<HH", raw[26:30])
    body = raw[30 + nlen + elen: 30 + nlen + elen + csize]
    data = body if method == 0 else zlib.decompress(body, -15)
    if len(data) != usize:
        raise RuntimeError(f"size mismatch for {name}: {len(data)} != {usize}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    tmp.unlink(missing_ok=True)
    return len(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/relax")
    ap.add_argument("--with-acc", action="store_true",
                    help="also fetch the 15.9 GB accelerometer files")
    ap.add_argument("--pause", type=float, default=3.0,
                    help="seconds between files (Zenodo throttles bursts)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    print(f"RELAX acquisition   DOI {DOI}   licence {LICENCE}")
    size = remote_size()
    print(f"remote archive: {size / 1e9:.2f} GB")
    entries = read_central_directory(size)
    print(f"archive contains {len(entries)} members")

    wanted = []
    for name, u, c, l in entries:
        base = name.split("RELAXDataset/", 1)[-1]
        keep = (base in ("questionnaire_responses.xlsx",
                         "metadata/questionnaires.xlsx", "metadata/README.md")
                or base.endswith("ibi_data.parquet")
                or (a.with_acc and base.endswith("acc_data.parquet")))
        if keep and u > 0:
            wanted.append((name, u, c, l, root / base))
    total = sum(w[1] for w in wanted)
    print(f"fetching {len(wanted)} members, {total / 1e6:.0f} MB uncompressed "
          f"({'including' if a.with_acc else 'EXCLUDING'} accelerometer)\n")

    manifest, failed = [], []
    for i, (name, u, c, l, dest) in enumerate(wanted, 1):
        if dest.exists() and dest.stat().st_size == u:
            print(f"[{i}/{len(wanted)}] {dest.name} present")
        else:
            try:
                extract(name, u, c, l, dest)
                print(f"[{i}/{len(wanted)}] {dest.relative_to(root)} "
                      f"{u / 1e6:.1f} MB")
            except Exception as exc:
                print(f"[{i}/{len(wanted)}] {name} FAILED: {exc}")
                failed.append(name)
                continue
            time.sleep(a.pause)
        manifest.append({
            "member": name,
            "path": str(dest.relative_to(root)),
            "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()
            if dest.stat().st_size < 60_000_000 else None,
        })

    prov = {
        "dataset": "RELAX",
        "title": ("A six-week longitudinal dataset of wearable and "
                  "self-reported stress measurements in working adults"),
        "doi": DOI, "zenodo_record": RECORD, "licence": LICENCE,
        "citation": ("Halmich, C., Jung, O., Schmoigl-Tonis, M., Schranz, C., "
                     "Kremser, W., Kunas, B., & Laireiter, A.-R. (2026). "
                     "Scientific Data. https://doi.org/10.1038/s41597-026-07711-4"),
        "archive_bytes": size,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acquisition_method": ("partial extraction from the remote zip via HTTP "
                               "range requests; accelerometer excluded"
                               if not a.with_acc else "full archive"),
        "n_members_fetched": len(manifest),
        "failed_members": failed,
        "files": manifest,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(f"\nprovenance written to {root / 'PROVENANCE.json'}")
    if failed:
        print(f"WARNING: {len(failed)} members failed; re-run to resume.")
        return 1
    print("\nNext:  python scripts/audit_dataset.py --dataset relax --root "
          f"{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
