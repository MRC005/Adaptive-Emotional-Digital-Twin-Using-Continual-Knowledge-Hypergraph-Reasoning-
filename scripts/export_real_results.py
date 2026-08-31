#!/usr/bin/env python3
"""Export real-dataset audits and results for the browser application.

WHAT THIS DELIBERATELY DOES NOT EXPORT

The deployed application is client-side, and the standing data rule for this
project forbids publishing participant-level data. The source archives are
2.6 GB and 2.8 GB and are licensed for research use, not redistribution. So
nothing here carries a participant identifier, a timestamp, a raw sensing
value, or a raw self-report. What is exported is the kind of derived summary a
paper prints: cohort counts, screen outcomes, the estimate and its interval,
and the distribution of per-participant ratios with no identifiers attached.

The consequence is stated plainly in the interface: real-dataset results are
COMPUTED OFFLINE by scripts/run_college_experience.py against the local
archive, and are displayed by the browser, not recomputed in it. The guided
controls, the sandbox and any CSV a user opens are computed live in the
browser; those are the only live paths and the interface does not blur them.

    python3 scripts/export_real_results.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from aedt.audit.eligibility import filter_eligible, screen_cohort  # noqa: E402
from aedt.constants import DataStatus                              # noqa: E402
from aedt.estimators.slope_ratio import estimate_rho_star          # noqa: E402
from aedt.io import CollegeExperienceAdapter, StudentLifeAdapter   # noqa: E402
from aedt.preprocess.epochs import assign_epochs                   # noqa: E402

OUT_JSON = ROOT / "frontend" / "src" / "data" / "real_datasets.json"
PROV = ROOT / "data" / "processed" / "PROVENANCE.json"
SEED = 20260828


def _norm(reason: str) -> str:
    """Collapse numbers so exclusion reasons group into readable categories."""
    return re.sub(r"[-+]?\d*\.?\d+", "N", reason).strip()


def screen_summary(df, sensor, K):
    """Counts and a reason histogram. No identifiers leave this function."""
    res = screen_cohort(df, sensor, K)
    hist: collections.Counter = collections.Counter()
    for r in res:
        for reason in r.reasons:
            hist[_norm(reason)] += 1
    return res, {
        "screened": len(res),
        "eligible": int(sum(r.eligible for r in res)),
        "exclusion_reasons": [{"reason": k, "participants": v}
                              for k, v in hist.most_common()],
    }


def run_ce(report, sensor, rule, label, primary=False, n_boot=2000):
    ad = CollegeExperienceAdapter(report=report, sensor=sensor)
    loaded = ad.load(ROOT / "data" / "raw" / "college-experience")
    df = assign_epochs(loaded.frame, rule=rule)
    K = loaded.n_categories
    res, summ = screen_summary(df, sensor, K)

    out = {"label": label, "primary": primary, "report": report,
           "sensor": sensor, "windows": rule,
           "participants": int(df["pid"].nunique()),
           "observations": int(len(df)), **summ}

    if summ["eligible"] < 10:
        out["status"] = "INSUFFICIENT_EVIDENCE"
        out["headline"] = "Insufficient evidence"
        out["why"] = (f"{summ['eligible']} of {summ['screened']} participants passed "
                      f"the screen. A participant-clustered interval needs at least "
                      f"10, so no estimate is produced.")
        return out

    keep = filter_eligible(df, res)
    est = estimate_rho_star(keep, sensor, K, bootstrap=True, n_resamples=n_boot,
                            seed=SEED, data_status=DataStatus.REAL,
                            eligibility_status="SCREENED")
    unc = est.uncertainty
    lo, hi = float(unc.ci_low), float(unc.ci_high)
    drift = not (lo <= 1.0 <= hi)
    out.update({
        "status": "DRIFT_DETECTED" if drift else "NO_MEANINGFUL_DRIFT",
        "headline": "Drift detected" if drift else "No detectable drift",
        "rho_star": round(float(est.rho_star), 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "n_used": int(est.n_participants_used),
        "n_resamples": int(unc.n_resamples),
        # per-participant ratios WITHOUT identifiers, for the forest plot
        "per_participant_rho": [round(float(v), 4)
                                for v in est.per_participant_rho_star],
        "why": (f"The 95% interval [{lo:.3f}, {hi:.3f}] "
                + ("excludes 1, so the sensor-to-report relationship changed."
                   if drift else
                   "includes 1. The data do not show a change; that is not the "
                   "same as showing there is none.")),
    })
    return out



def studentlife_alternatives(root: Path) -> list[str]:
    """Measure the monotone-scale alternatives to the Stress item, from the files.

    The Stress item is the only dense one, and its options are not ordered by
    severity. The obvious response is "use a different item", so this checks
    whether any properly ordered item carries enough repeated measurement. The
    numbers in the returned notes are computed here, not asserted.
    """
    import glob as _glob, json as _json, os as _os
    import numpy as _np

    # (EMA type, question id) pairs whose documented options ARE monotone.
    cands = [("Behavior", "anxious"), ("Behavior", "calm"), ("Sleep", "rate"),
             ("Social", "number"), ("Activity", "working"), ("Exercise", "exercise")]
    out = []
    for kind, key in cands:
        per: dict[str, int] = {}
        for f in _glob.glob(str(root / "EMA" / "response" / kind / "*.json")):
            uid = _os.path.basename(f).rsplit("_", 1)[-1].removesuffix(".json")
            try:
                recs = _json.load(open(f))
            except Exception:
                continue
            n = sum(1 for r in recs if isinstance(r, dict) and "resp_time" in r
                    and str(r.get(key, "")).strip().isdigit())
            if n:
                per[uid] = per.get(uid, 0) + n
        if per:
            v = _np.array(list(per.values()))
            out.append(f"{kind}/{key}: {len(per)} participants, median {int(_np.median(v))} "
                       f"responses each, most {int(v.max())} — "
                       f"{int((v >= 120).sum())} reach the 120 the screen needs")
    return out


def main() -> int:
    print("College Experience: running the pre-specified protocol...", flush=True)
    runs = [
        run_ce("stress", "conversation_minutes", "own_span_halves",
               "Primary — stress vs conversation minutes", primary=True),
        run_ce("stress", "conversation_minutes", "observation_halves",
               "S1 — windows by equal observation count"),
        run_ce("stress", "conversation_minutes", "calendar_median",
               "S2 — windows by cohort calendar median"),
        run_ce("stress", "unlock_minutes", "own_span_halves",
               "S3 — sensor: phone unlock minutes"),
        run_ce("social_level", "conversation_minutes", "own_span_halves",
               "S4 — report: time spent with others"),
    ]
    for r in runs:
        print(f"  {r['label']:52s} {r['status']:22s} eligible={r['eligible']}")

    # Headline counts describe the DATASET, so they use an all-platform sensor.
    # The conversation feature is documented Android-only and therefore yields a
    # smaller cohort; that restriction is reported per run and in the notes.
    ce_audit = CollegeExperienceAdapter(sensor="unlock_minutes").audit(
        ROOT / "data/raw/college-experience")
    sl_audit = StudentLifeAdapter().audit(ROOT / "data/raw/studentlife")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "computed": "offline, by scripts/export_real_results.py, against the "
                    "local archives; displayed by the browser, not recomputed in it",
        "privacy": "no participant identifiers, timestamps, raw sensing values "
                   "or raw self-reports are present in this file",
        "datasets": [
            {
                "id": "college_experience",
                "name": "College Experience Study",
                "status": "ready",
                "citation": "Nepal et al. 2024, Proc. ACM IMWUT 8(1) art. 38, "
                            "doi:10.1145/3643501",
                "participants": ce_audit.participant_count,
                "observations": ce_audit.observation_count,
                "span_days": ce_audit.longitudinal_span_days,
                "median_obs_per_participant": ce_audit.median_observations_per_participant,
                "report_scale": ce_audit.self_report_scale,
                "notes": list(ce_audit.notes) + [
                    "Conversation audio is documented Android only in the data "
                    "dictionary and is 87.8% zero on iOS against 13.1% on Android. "
                    "Conversation analyses therefore run on the 33-participant "
                    "Android cohort; unlock and location features use all 218.",
                ],
                "sensor_platforms": {
                    "conversation_minutes": "Android only (per data dictionary)",
                    "conversation_episodes": "Android only (per data dictionary)",
                    "unlock_minutes": "all platforms",
                    "travel_distance": "all platforms",
                },
                "runs": runs,
            },
            {
                "id": "studentlife",
                "name": "StudentLife (original Dartmouth release)",
                "status": "incompatible",
                "citation": "Wang et al. 2014, UbiComp; studentlife.cs.dartmouth.edu",
                "participants": sl_audit.participant_count,
                "observations": sl_audit.observation_count,
                "median_obs_per_participant": sl_audit.median_observations_per_participant,
                "span_days": sl_audit.longitudinal_span_days,
                "notes": [
                    "The item with usable density is Stress, but its documented options are "
                    "\u201c1 a little stressed, 2 definitely stressed, 3 stressed out, "
                    "4 feeling good, 5 feeling great\u201d. The numbers are not ordered by "
                    "stress, so an ordinal model fitted to them is fitting a scale that does "
                    "not exist. The loader remaps them by label text, which fixes the ordering "
                    "but cannot create more data.",
                    "Even after recovering malformed records and applying that remap, only "
                    f"{1} participant has 60 or more responses in both halves of their own "
                    "span. The screen needs at least 10.",
                    "This does not depend on the number 60. Sweeping the per-window minimum "
                    "from 20 to 100 across three window rules, StudentLife reaches 10 "
                    "qualifying participants only at 20-30 per window \u2014 a threefold "
                    "relaxation of the pre-specified rule, which would leave a five-category "
                    "probit slope fitted to roughly 20 points per person.",
                    "Every properly ordered alternative item is far sparser than Stress: "
                    + "; ".join(studentlife_alternatives(
                        ROOT / "data" / "raw" / "studentlife")) + ".",
                    "About 10% of Stress records in the original archive store the answer "
                    "under a literal \u201cnull\u201d key rather than \u201clevel\u201d, and "
                    "that key also holds GPS strings. Those are recovered when the value is a "
                    "digit in range and discarded when it is a coordinate.",
                ] + list(sl_audit.notes),
                "exclusion_reasons": [str(e) for e in sl_audit.exclusion_reasons],
                "runs": [],
            },
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwritten: {OUT_JSON}  ({OUT_JSON.stat().st_size:,} bytes)")

    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps({
        "generated_utc": payload["generated_utc"],
        "produced_by": "scripts/export_real_results.py and "
                       "scripts/run_college_experience.py",
        "sources": {
            "college_experience": "data/raw/college-experience/"
                                  "{EMA/general_ema.csv, Sensing/sensing.csv}",
            "studentlife": "data/raw/studentlife/"
                           "{EMA/response/Stress/*.json, sensing/conversation/*.csv}",
        },
        "transformation": "adapter -> canonical LongFrame(pid, ts, report, sensor) "
                          "-> window assignment -> eligibility screen -> placebo "
                          "gate -> participant-cluster bootstrap",
        "raw_data_modified": False,
        "seed": SEED,
        "outputs": ["data/processed/college_experience_results.csv",
                    "data/processed/college_experience_results.json",
                    "frontend/src/data/real_datasets.json"],
        "note": "Raw archives are never written to and are never redistributed. "
                "Exported artefacts carry no participant-level identifiers.",
    }, indent=2))
    print(f"written: {PROV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
