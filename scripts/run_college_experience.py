#!/usr/bin/env python3
"""Run the frozen AEDT protocol on the College Experience Study.

PROTOCOL, FIXED BEFORE ANY RESULT WAS INSPECTED
-----------------------------------------------
Primary
    report   general EMA `stress`, 1-5, ascending per the published data
             dictionary (direction confirmed from the codebook, not the data)
    sensor   conversation minutes per day (audio_convo_duration_ep_0 / 60)
    windows  halves of each participant's own observation span
    screen   the unchanged pre-specified screen: >= 60 observations per window,
             >= 2 categories used, sensor SD >= 0.10, |beta| >= 0.02 with the
             same sign in both windows, window variance ratio in [0.25, 4]
    gate     contiguous split-half placebo on the baseline window; if the
             placebo rejects, the primary is not reported
    interval participant-cluster bootstrap, 2000 resamples, seed 20260828

Sensitivity (reported alongside, never substituted for the primary)
    S1  windows = observation halves (equal counts rather than equal time)
    S2  windows = cohort calendar median
    S3  sensor  = phone unlock minutes per day
    S4  report  = social_level

Nothing here selects a configuration by its outcome. Every row of the
sensitivity table is printed whatever it says.

    python3 scripts/run_college_experience.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aedt.audit.eligibility import filter_eligible, screen_cohort   # noqa: E402
from aedt.constants import DataStatus                               # noqa: E402
from aedt.estimators.slope_ratio import estimate_rho_star           # noqa: E402
from aedt.inference.placebo import placebo_split_half               # noqa: E402
from aedt.io import CollegeExperienceAdapter                        # noqa: E402
from aedt.preprocess.epochs import assign_epochs                    # noqa: E402

DATA = ROOT / "data" / "raw" / "college-experience"
OUT = ROOT / "data" / "processed"
SEED = 20260828


def one_run(report: str, sensor: str, rule: str, *, n_boot: int, label: str) -> dict:
    """One complete pass: load -> windows -> screen -> placebo -> estimate."""
    adapter = CollegeExperienceAdapter(report=report, sensor=sensor)
    loaded = adapter.load(DATA)
    df = assign_epochs(loaded.frame, rule=rule)
    K = loaded.n_categories

    screened = screen_cohort(df, sensor, K)
    n_eligible = sum(r.eligible for r in screened)
    row = {"label": label, "report": report, "sensor": sensor, "windows": rule,
           "n_participants": int(df["pid"].nunique()),
           "n_observations": int(len(df)), "n_eligible": n_eligible}

    if n_eligible < 10:
        row["status"] = "INSUFFICIENT_EVIDENCE"
        row["reason"] = (f"{n_eligible} eligible participants; the screen "
                         f"requires 10 for a participant-clustered interval")
        return row

    keep = filter_eligible(df, screened)
    placebo = placebo_split_half(keep, sensor, K, seed=SEED)
    row["placebo_rejected"] = bool(getattr(placebo, "rejected", False))
    row["placebo_detail"] = str(getattr(placebo, "summary", "") or
                                getattr(placebo, "message", ""))[:200]
    if row["placebo_rejected"]:
        row["status"] = "DATA_QUALITY_ISSUE"
        row["reason"] = ("the split-half placebo on the baseline window "
                         "rejected; the primary is gated and not reported")
        return row

    est = estimate_rho_star(keep, sensor, K, bootstrap=True, n_resamples=n_boot,
                            seed=SEED, data_status=DataStatus.REAL,
                            eligibility_status="SCREENED")
    unc = est.uncertainty
    lo, hi = float(unc.ci_low), float(unc.ci_high)
    row.update({
        "rho_star": float(est.rho_star), "ci_low": float(lo), "ci_high": float(hi),
        "n_used": int(est.n_participants_used),
        "excludes_one": not (lo <= 1.0 <= hi),
    })
    row["status"] = "DRIFT_DETECTED" if row["excludes_one"] else "NO_MEANINGFUL_DRIFT"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="200 bootstrap resamples instead of 2000, for a smoke run")
    args = ap.parse_args()
    n_boot = 200 if args.quick else 2000

    plan = [
        ("PRIMARY", "stress", "conversation_minutes", "own_span_halves"),
        ("S1 observation halves", "stress", "conversation_minutes", "observation_halves"),
        ("S2 calendar median", "stress", "conversation_minutes", "calendar_median"),
        ("S3 unlock minutes", "stress", "unlock_minutes", "own_span_halves"),
        ("S4 social_level", "social_level", "conversation_minutes", "own_span_halves"),
    ]

    rows = []
    for label, report, sensor, rule in plan:
        print(f"\n--- {label}: report={report} sensor={sensor} windows={rule} ---",
              flush=True)
        try:
            r = one_run(report, sensor, rule, n_boot=n_boot, label=label)
        except Exception as exc:                      # reported, never swallowed
            r = {"label": label, "report": report, "sensor": sensor,
                 "windows": rule, "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(r)
        for k in ("n_participants", "n_observations", "n_eligible", "status",
                  "rho_star", "ci_low", "ci_high", "reason"):
            if k in r:
                v = r[k]
                print(f"    {k:16s} {v:.4f}" if isinstance(v, float) else f"    {k:16s} {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "college_experience_results.csv", index=False)
    (OUT / "college_experience_results.json").write_text(
        json.dumps({"seed": SEED, "n_bootstrap": n_boot, "runs": rows},
                   indent=2, default=str))

    print("\n================ SUMMARY ================")
    cols = [c for c in ("label", "n_eligible", "status", "rho_star", "ci_low", "ci_high")
            if c in tbl.columns]
    print(tbl[cols].to_string(index=False))
    print(f"\nwritten: {OUT / 'college_experience_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
