#!/usr/bin/env python3
"""Compute the predictability ceiling from the real archive. No literals.

Every figure the website quotes under "the ceiling" — and the four cohort
descriptors beside them — used to be typed by hand into
``scripts/export_findings.py``. This script computes them, writes
``results/twin/ceiling.json``, and records how each one was defined so a reader
can disagree with a definition rather than guess at one.

    python3 scripts/run_ceiling_analysis.py [--quick] [--batch 64]

It reads the archive and NOTHING from the primary experiment. It does not
re-run, re-fit or reinterpret the pre-registered twin-vs-persistence result;
that result stays exactly as recorded.

Exit codes follow the project's convention:
    0  wrote results/twin/ceiling.json
    6  the real archive is unavailable — nothing is estimated or approximated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aedt.audit.ceiling import (MAX_GAP_DAYS,  # noqa: E402
                                MIN_OBS_PER_PERSON, ceiling_statistics)

DATA = ROOT / "data" / "raw" / "college-experience"
OUT = ROOT / "results" / "twin"
TARGET = "stress"

#: Columns in Sensing/sensing.csv that are keys, not behaviour.
SENSING_KEYS = ("uid", "day", "is_ios")


def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise. Never raises.

    ``DATA`` is overridable (the smoke test points it elsewhere), so a bare
    ``relative_to(ROOT)`` turns a clean "data is missing" refusal into a
    ValueError traceback -- an error path that itself errors.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _sha256(path: Path, limit: int = 64 * 1024 * 1024) -> dict:
    """Digest of (at most) the first ``limit`` bytes, matching PROVENANCE.json."""
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while read < limit:
            chunk = f.read(min(1 << 20, limit - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return {"bytes": path.stat().st_size, "sha256_first_bytes": h.hexdigest(),
            "hashed_bytes": read}


# ------------------------------------------------------------------- loading
def load_reports(root: Path) -> pd.DataFrame:
    """Long frame of non-null stress reports: participant_id, day_ord, stress."""
    ema = pd.read_csv(root / "EMA" / "general_ema.csv", usecols=["uid", "day", TARGET])
    ema = ema.dropna(subset=[TARGET])
    t = pd.to_datetime(ema["day"].astype(int).astype(str), format="%Y%m%d")
    out = pd.DataFrame({
        "participant_id": ema["uid"].astype(str),
        "day_ord": (t - pd.Timestamp("1970-01-01")).dt.days.astype(int),
        TARGET: ema[TARGET].astype(float),
    })
    return out.sort_values(["participant_id", "day_ord"]).reset_index(drop=True)


def cohort_descriptors(reports: pd.DataFrame) -> dict:
    """Participants, reports, span in years — measured, not remembered."""
    span_days = int(reports["day_ord"].max() - reports["day_ord"].min())
    return {
        "participants": int(reports["participant_id"].nunique()),
        "reports": int(len(reports)),
        "years": round(span_days / 365.25, 2),
        "span_days": span_days,
        "first_day": str(pd.Timestamp("1970-01-01")
                         + pd.Timedelta(days=int(reports["day_ord"].min())))[:10],
        "last_day": str(pd.Timestamp("1970-01-01")
                        + pd.Timedelta(days=int(reports["day_ord"].max())))[:10],
    }


# ---------------------------------------------------------- behaviour screen
def behaviour_screen(root: Path, pairs: pd.DataFrame, batch: int = 64,
                     quick: bool = False) -> dict:
    """Strongest association between ANY daily sensing column and next stress.

    ``pairs`` carries one row per prediction opportunity with columns
    ``participant_id``, ``day_ord`` (the prediction day) and ``target`` (the
    NEXT report's stress). Sensing is taken from the day BEFORE the prediction
    day, exactly as ``aedt/twin/prediction_data.py`` aligns it, so this screen
    describes the information the model could actually have used.

    Every sensing column is screened, so the reported column count is a
    measurement rather than a recollection. Columns are read in batches to keep
    peak memory bounded on a 500 MB file.
    """
    path = root / "Sensing" / "sensing.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    feature_cols = [c for c in header if c not in SENSING_KEYS]
    if quick:
        feature_cols = feature_cols[:batch]

    # sensing from the previous day: shift the sensing day forward by one so it
    # joins onto the prediction day.
    key = pairs[["participant_id", "day_ord", "target"]].copy()

    best = {"column": None, "r": 0.0, "n": 0}
    per_column = []
    screened = 0
    for i in range(0, len(feature_cols), batch):
        cols = feature_cols[i:i + batch]
        chunk = pd.read_csv(path, usecols=["uid", "day", *cols])
        t = pd.to_datetime(chunk["day"].astype(int).astype(str), format="%Y%m%d")
        chunk["participant_id"] = chunk["uid"].astype(str)
        # +1 day: behaviour observed on day d informs the prediction made on d+1
        chunk["day_ord"] = ((t - pd.Timestamp("1970-01-01")).dt.days + 1).astype(int)
        merged = key.merge(chunk[["participant_id", "day_ord", *cols]],
                           on=["participant_id", "day_ord"], how="inner")
        for c in cols:
            v = pd.to_numeric(merged[c], errors="coerce")
            keep = v.notna() & merged["target"].notna()
            n = int(keep.sum())
            screened += 1
            if n < 30 or float(v[keep].std()) < 1e-12:
                per_column.append({"column": c, "r": None, "n": n,
                                   "skipped": "constant or too few rows"})
                continue
            r = float(np.corrcoef(v[keep], merged["target"][keep])[0, 1])
            per_column.append({"column": c, "r": r, "n": n})
            if abs(r) > abs(best["r"]):
                best = {"column": c, "r": r, "n": n}
        del chunk, merged
    return {
        "n_sensing_columns_screened": screened,
        "n_sensing_columns_in_file": len(feature_cols),
        "strongest_behaviour_column": best["column"],
        "strongest_behaviour_r": abs(best["r"]),
        "strongest_behaviour_r_signed": best["r"],
        "strongest_behaviour_n": best["n"],
        "behaviour_variance_explained": best["r"] ** 2,
        "alignment": "sensing from day d joined to the prediction made on day d+1",
        "per_column": per_column,
    }


def prediction_pairs(reports: pd.DataFrame,
                     max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """(participant, prediction day, next report's stress) within the horizon.

    Reproduces the pairing rule of the pre-registration independently of the
    modelling frame, so the count can be cross-checked against it.
    """
    g = reports.groupby("participant_id", sort=False)
    out = reports.copy()
    out["target"] = g[TARGET].shift(-1)
    out["next_day"] = g["day_ord"].shift(-1)
    out["gap"] = out["next_day"] - out["day_ord"]
    out = out[out["target"].notna() & (out["gap"] > 0)
              & (out["gap"] <= max_gap_days)]
    return out.reset_index(drop=True)


# ------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="screen only the first batch of sensing columns")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ema_path = DATA / "EMA" / "general_ema.csv"
    sens_path = DATA / "Sensing" / "sensing.csv"
    missing = [_display(p) for p in (ema_path, sens_path) if not p.exists()]
    if missing:
        print("REAL DATA UNAVAILABLE - nothing was estimated or approximated.",
              file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        print("Obtain the archive from the source recorded in "
              f"{_display(DATA)}/PROVENANCE.json", file=sys.stderr)
        return 6

    t0 = time.time()
    print("Reading reports...", flush=True)
    reports = load_reports(DATA)
    cohort = cohort_descriptors(reports)
    print(f"  participants {cohort['participants']} | reports {cohort['reports']:,}"
          f" | span {cohort['years']} years", flush=True)

    pairs = prediction_pairs(reports)
    cohort["prediction_pairs"] = int(len(pairs))
    print(f"  prediction pairs (gap 1-{MAX_GAP_DAYS} days) {len(pairs):,}", flush=True)

    print("Ceiling, PRIMARY definition (pairs within the task horizon)...",
          flush=True)
    primary = ceiling_statistics(reports, value_col=TARGET, day_col="day_ord",
                                 max_gap_days=MAX_GAP_DAYS)
    print(f"  within-person r {primary.within_person_autocorrelation:.4f}"
          f" | ICC {primary.icc_between_person:.4f}"
          f" | n {primary.n_participants_analysed}", flush=True)

    print("Ceiling, SENSITIVITY definition (all consecutive pairs)...", flush=True)
    allpairs = ceiling_statistics(reports, value_col=TARGET, day_col="day_ord",
                                  max_gap_days=None)
    print(f"  within-person r {allpairs.within_person_autocorrelation:.4f}"
          f" | n {allpairs.n_participants_analysed}", flush=True)

    print("Screening every sensing column against next-report stress...",
          flush=True)
    beh = behaviour_screen(DATA, pairs, batch=args.batch, quick=args.quick)
    print(f"  screened {beh['n_sensing_columns_screened']} columns | strongest "
          f"|r| {beh['strongest_behaviour_r']:.4f} "
          f"({beh['strongest_behaviour_column']})", flush=True)

    payload = {
        "_generated_by": "scripts/run_ceiling_analysis.py",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_runtime_seconds": round(time.time() - t0, 1),
        "_inputs": {
            "EMA/general_ema.csv": _sha256(ema_path),
            "Sensing/sensing.csv": _sha256(sens_path),
        },
        "_declared": {
            "min_obs_per_person": MIN_OBS_PER_PERSON,
            "max_gap_days": MAX_GAP_DAYS,
            "primary_definition": "pairs restricted to the task horizon",
            "note": ("Both definitions are reported. The primary matches the "
                     "prediction task defined in the pre-registration; the "
                     "sensitivity uses every consecutive pair. Neither was "
                     "selected after seeing the other."),
        },
        "cohort": cohort,
        "ceiling": {**primary.to_dict(),
                    "strongest_behaviour_r": beh["strongest_behaviour_r"],
                    "behaviour_variance_explained":
                        beh["behaviour_variance_explained"]},
        "ceiling_all_pairs": allpairs.to_dict(),
        "behaviour_screen": {k: v for k, v in beh.items() if k != "per_column"},
        "quick": bool(args.quick),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ceiling.json").write_text(json.dumps(payload, indent=2),
                                      encoding="utf-8")
    (OUT / "ceiling_behaviour_columns.json").write_text(
        json.dumps(beh["per_column"], indent=2), encoding="utf-8")
    print(f"\nwritten: {OUT/'ceiling.json'} in {payload['_runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
