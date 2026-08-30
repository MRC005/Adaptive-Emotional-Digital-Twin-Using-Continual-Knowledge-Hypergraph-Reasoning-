#!/usr/bin/env python3
"""Convert the StudentLife RDS repackaging into the canonical interim CSVs.

    python scripts/convert_studentlife_rds.py \
        --zip data/raw/studentlife/dataset_rds.zip \
        --out data/interim/studentlife

WHY THIS EXISTS. The Dartmouth StudentLife release is JSON + CSV. A widely
circulated repackaging (``dataset_rds.zip``) stores the same study as R ``.Rds``
objects. This script extracts ONLY the two tables the frozen analysis needs and
writes them as CSV, so the adapter never has to depend on an R runtime and the
raw archive is never modified.

    EMA/Stress.Rds          -> stress_ema.csv       (uid, timestamp, response)
    sensing/conversation.Rds -> conversation.csv    (uid, start, end)

IT REQUIRES R. ``Rscript`` reads the objects natively, which avoids trusting a
third-party binary reader with the scientific record.

⚠ READ THIS BEFORE USING THE OUTPUT. The RDS conversion is **defective for the
stress EMA**. In the archive audited here, ``Stress.Rds`` carries its response
in a column literally named ``null`` which is 88% NA and additionally contains
GPS coordinate strings. Only 122 of 2017 rows parse as a 1-5 response, a
maximum of 6 per participant, against roughly 735 EMA responses per student in
the published descriptor. Other EMA tables in the same archive (PAM, Mood,
Sleep) converted correctly with named columns, so this is specific to Stress
and is a conversion defect, not a property of StudentLife.

This script therefore reports what it found and does NOT repair anything. To
run the frozen primary analysis you need the ORIGINAL Dartmouth release, which
also ships ``EMA_definition.json`` — the label-text codebook the specification
requires and which this repackaging omits entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MEMBERS = {
    "stress": "dataset_rds/EMA/Stress.Rds",
    "conversation": "dataset_rds/sensing/conversation.Rds",
}

R_SCRIPT = r'''
args <- commandArgs(trailingOnly=TRUE)
stress_in <- args[1]; conv_in <- args[2]; outdir <- args[3]
s <- readRDS(stress_in)
# The response column name is not guaranteed: the defective repackaging calls
# it "null". Take whatever single column is neither timestamp nor uid.
resp <- setdiff(names(s), c("timestamp","uid"))
if (length(resp) < 1) stop("Stress table has no response column at all")
out <- data.frame(uid=s$uid, timestamp=s$timestamp,
                  response=as.character(s[[resp[1]]]),
                  response_column_name=resp[1], stringsAsFactors=FALSE)
write.csv(out, file.path(outdir,"stress_ema.csv"), row.names=FALSE, na="")
cv <- readRDS(conv_in)
write.csv(data.frame(uid=cv$uid, start_timestamp=cv$start_timestamp,
                     end_timestamp=cv$end_timestamp),
          file.path(outdir,"conversation.csv"), row.names=FALSE, na="")
cat(sprintf("stress_rows=%d response_col=%s conversation_rows=%d\n",
            nrow(s), resp[1], nrow(cv)))
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/studentlife/dataset_rds.zip")
    ap.add_argument("--out", default="data/interim/studentlife")
    a = ap.parse_args(argv)

    src = Path(a.zip)
    if not src.exists():
        print(f"REAL DATA UNAVAILABLE - STUDENTLIFE: no archive at {src}")
        return 6
    if not shutil.which("Rscript"):
        print("Rscript not found. The RDS repackaging can only be read with R.\n"
              "Install R (https://cran.r-project.org) or obtain the ORIGINAL\n"
              "Dartmouth release, which needs no R at all.")
        return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "_rds"
    stage.mkdir(exist_ok=True)

    with zipfile.ZipFile(src) as z:
        names = set(z.namelist())
        missing = [m for m in MEMBERS.values() if m not in names]
        if missing:
            print(f"DECISION REQUIRED: the archive lacks {missing}. This is not "
                  "the expected StudentLife RDS repackaging.")
            return 2
        for key, member in MEMBERS.items():
            dest = stage / Path(member).name
            if not dest.exists():
                with z.open(member) as fsrc, open(dest, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
            print(f"  extracted {member}  ({dest.stat().st_size/1e6:.2f} MB)")

    rs = stage / "_convert.R"
    rs.write_text(R_SCRIPT)
    r = subprocess.run(
        ["Rscript", str(rs), str(stage / "Stress.Rds"),
         str(stage / "conversation.Rds"), str(out)],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print("R conversion failed:\n" + (r.stderr or "")[-1500:])
        return 2
    print("  " + r.stdout.strip())

    prov = {
        "dataset": "StudentLife (RDS repackaging)",
        "citation": ("Wang, R., Chen, F., Chen, Z., et al. (2014). StudentLife: "
                     "assessing mental health, academic performance and "
                     "behavioral trends of college students using smartphones. "
                     "UbiComp."),
        "official_source": "https://studentlife.cs.dartmouth.edu/dataset.html",
        "archive": {"path": str(src), "bytes": src.stat().st_size,
                    "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
        "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter": "scripts/convert_studentlife_rds.py via Rscript",
        "members_extracted": list(MEMBERS.values()),
        "warning": ("This repackaging is DEFECTIVE for the stress EMA: the "
                    "response column is unnamed and ~88% NA. The frozen primary "
                    "analysis requires the ORIGINAL Dartmouth release, which "
                    "also ships EMA_definition.json (the label-text codebook "
                    "this repackaging omits)."),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}/stress_ema.csv, {out}/conversation.csv, PROVENANCE.json")
    print(f"\nNext:  python scripts/audit_dataset.py --dataset studentlife --root {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
