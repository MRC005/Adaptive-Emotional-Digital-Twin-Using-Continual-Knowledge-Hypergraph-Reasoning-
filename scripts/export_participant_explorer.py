#!/usr/bin/env python3
"""DISABLED ON THIS BRANCH: running this script reads and writes nothing (exit 7).
The application ships simulated people written by make_synthetic_demo_data.py;
the files under frontend/src/data/ are NOT produced by this script and are not
research results. What follows describes what the script did when it was enabled.

Generate the participant-level record the Explorer reads. No hand-typed data.

WHAT THIS IS FOR
The website could show the study's aggregate result but not the people behind
it. This disaggregates the frozen experiment's own test evaluation into one
record per held-out participant, so a reader can see what the aggregate is made
of -- including that some participants were better served by the twin and most
were not.

WHAT IT DOES NOT DO
It fits no new model, defines no new metric, and changes nothing. It imports
the frozen experiment's frame builder, split, warm-up rule, scorer and twin
predictor and calls them unmodified. Every number it writes is the same
quantity the study already reports, computed the same way, simply not summed
across people. The aggregate is unchanged and is not recomputed here.

    python3 scripts/export_participant_explorer.py [--limit-series N]

REDISTRIBUTION
The archive's own PROVENANCE.json says it is NOT redistributed. Anonymisation
does not by itself make redistribution acceptable, so the two outputs are
separated deliberately:

  results/explorer/participants_full.json   EVERY participant's full observation
      series. Lives under results/, which is gitignored, so it stays on the
      machine that generated it and is never committed or shipped.
  frontend/src/data/participants.json       Summary STATISTICS for all
      participants -- derived values of the kind a paper publishes in a table --
      plus the observation series for a small ILLUSTRATIVE SUBSET only, so the
      charts have something real to draw.

The subset is chosen by a fixed, stated rule (see ILLUSTRATIVE_RULE) over
autocorrelation, never by how the twin performed, so it cannot flatter the
result. Participants outside it carry `series_published: false` and the
interface says so rather than drawing nothing and looking broken.

PRIVACY
Source identifiers are never written. Participants are renumbered by a stable
sort over their own id digest, so the labels carry no information about the
source ids and do not change between runs. Timestamps become day offsets from
each participant's own first report, so no calendar date is published. This is
the same policy the existing trajectories.json states, applied by code rather
than by hand.

Exit codes:
    0  wrote frontend/src/data/participants.json
    6  the real archive is unavailable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aedt.audit.ceiling import (MAX_GAP_DAYS,  # noqa: E402
                                NEAR_UNPREDICTABLE_BELOW,
                                WELL_PREDICTABLE_ABOVE,
                                person_autocorrelation)
from aedt.reporting.experiment_record import (  # noqa: E402
    write_experiment_record)
from aedt.twin.prediction_data import assert_no_leakage  # noqa: E402
from scripts.run_twin_experiment import (DATA, SEED,  # noqa: E402
                                         build_prediction_frame, fit_global,
                                         make_splits, predict_twin, score,
                                         split_warmup)

OUT_JSON = ROOT / "frontend" / "src" / "data" / "participants.json"
OUT_RESULTS = ROOT / "results" / "explorer"

K_PRIMARY = 20
K_GRID = (K_PRIMARY, 80)

#: How the illustrative subset is chosen. Deterministic, stated, and computed
#: over PREDICTABILITY -- the axis the pages are about -- never over twin or
#: persistence performance.
ILLUSTRATIVE_RULE = ("the scored participants at the minimum, median and "
                     "maximum within-person autocorrelation")
N_ILLUSTRATIVE = 3

HIST_COLUMNS = {"current", "current_social", "current_pam", "prev_1", "prev_2",
                "prev_3", "hist_mean", "hist_sd", "hist_n", "ewma",
                "days_since_prev"}

#: The ONLY categorical rule used anywhere in the Explorer. It is not invented
#: here: these two constants are the ones aedt/audit/ceiling.py already uses to
#: report frac_near_unpredictable and frac_well_predictable, and the published
#: ceiling figures (13% and 17%) are defined by them. The rule is written into
#: the output so the interface can display it verbatim.
BAND_RULE = (f"r < {NEAR_UNPREDICTABLE_BELOW} near-unpredictable · "
             f"{NEAR_UNPREDICTABLE_BELOW} <= r <= {WELL_PREDICTABLE_ABOVE} "
             f"typical · r > {WELL_PREDICTABLE_ABOVE} well-predictable")


def jsonable(v):
    """NaN is not JSON. A metric that is undefined is written as null.

    ``score()`` returns NaN for quadratic-weighted kappa when a participant's
    predictions collapse to a single class -- kappa is genuinely undefined
    there. Writing a bare NaN produces a file every JSON parser rejects, and
    writing 0.0 would invent a value. null says "not defined", which is true.
    """
    return None if (isinstance(v, float) and not np.isfinite(v)) else v


def band(r: float | None) -> str | None:
    """Three-way label from the ceiling analysis's own published thresholds."""
    if r is None or not np.isfinite(r):
        return None
    if r < NEAR_UNPREDICTABLE_BELOW:
        return "near-unpredictable"
    if r > WELL_PREDICTABLE_ABOVE:
        return "well-predictable"
    return "typical"


def anon_labels(pids) -> dict:
    """Stable, information-free labels. Sorted by digest, not by source id."""
    ordered = sorted(pids, key=lambda p: hashlib.sha256(
        str(p).encode()).hexdigest())
    width = max(2, len(str(len(ordered))))
    return {p: f"P{i + 1:0{width}d}" for i, p in enumerate(ordered)}


def participant_record(g, models, features, ctx_features, k_grid) -> dict:
    """Everything the Explorer shows for one participant, from real output."""
    g = g.sort_values("prediction_time")
    _, model_full = models

    observed = g["current"].to_numpy(dtype=float)
    t = g["prediction_time"]
    day0 = t.min()
    days = (t - day0).dt.days.to_numpy(dtype=int)

    r = person_autocorrelation(observed, days, MAX_GAP_DAYS)
    r = float(r) if np.isfinite(r) else None

    rec = {
        "n_observations": int(len(g)),
        "span_days": int(days.max() - days.min()),
        "autocorrelation": None if r is None else round(r, 4),
        "predictability_band": band(r),
        "variability_sd": round(float(np.std(observed, ddof=1)), 4)
                          if len(observed) > 1 else None,
        "mean_stress": round(float(np.mean(observed)), 4),
        "by_K": {},
        "series": [],
    }

    # ---- per-K held-out performance, using the frozen scorer ---------------
    series_marks = {}
    for K in k_grid:
        warm, ev = split_warmup(g, K)
        if len(ev) < 5:                        # the frozen protocol's own rule
            rec["by_K"][str(K)] = {"scored": False,
                                   "reason": "fewer than 5 scored observations "
                                             "after warm-up (frozen protocol)"}
            continue
        y = ev["target"].to_numpy()
        persistence = ev["current"].to_numpy()
        twin = predict_twin(model_full, warm, ev, features, ctx_features)
        st, sp = score(y, twin), score(y, persistence)
        rec["by_K"][str(K)] = {
            "scored": True,
            "n_warmup": int(len(warm)), "n_eval": int(len(ev)),
            "twin": {k: jsonable(round(float(v), 6)) for k, v in st.items()
                     if k != "n"},
            "persistence": {k: jsonable(round(float(v), 6))
                            for k, v in sp.items() if k != "n"},
            "twin_minus_persistence_macro_f1":
                round(float(st["macro_f1"] - sp["macro_f1"]), 6),
            "twin_better": bool(st["macro_f1"] > sp["macro_f1"]),
        }
        if K == K_PRIMARY:
            series_marks = {
                int(d): {"predicted": int(p), "actual_next": int(a),
                         "baseline": int(b)}
                for d, p, a, b in zip(
                    (ev["prediction_time"] - day0).dt.days.to_numpy(),
                    twin, ev["target"].to_numpy(), persistence)
            }
            rec["warmup_k"] = K

    # ---- the series, anonymised to day offsets ----------------------------
    for d, v in zip(days, observed):
        row = {"day": int(d), "observed": int(v),
               "warmup": int(d) not in series_marks}
        row.update(series_marks.get(int(d), {}))
        rec["series"].append(row)
    return rec


def main() -> int:
    # PUBLIC SNAPSHOT: this exporter writes participant-level research records
    # into frontend/src/data/. No such record is distributed with this
    # application, so the exporter is disabled here. The application ships
    # synthetic demonstration data instead (scripts/make_synthetic_demo_data.py).
    print("Disabled in the public snapshot: this script writes participant-level "
          "research records into the web application. Nothing was read or written. "
          "See DATA_PROVENANCE.md.", file=sys.stderr)
    return 7


def _disabled_original_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", choices=("illustrative", "all", "none"),
                    default="illustrative",
                    help="which observation series reach the FRONTEND. "
                         "'illustrative' (default) publishes only the subset "
                         "chosen by ILLUSTRATIVE_RULE; 'all' would redistribute "
                         "every series and is not the default for that reason; "
                         "'none' ships statistics only. The full local file "
                         "under results/ always contains everything.")
    args = ap.parse_args()

    started = time.time()
    if not (DATA / "EMA" / "general_ema.csv").exists():
        print("REAL DATA UNAVAILABLE - nothing was estimated or approximated.",
              file=sys.stderr)
        return 6

    print("Building the prediction frame (frozen builder, unmodified)...",
          flush=True)
    data = build_prediction_frame(DATA)
    assert_no_leakage(data.frame)
    train_pids, val_pids, test_pids = make_splits(data)
    print(f"  train {len(train_pids)} | validation {len(val_pids)} | "
          f"test {len(test_pids)}", flush=True)

    features = data.feature_columns
    ctx_features = [c for c in features if c not in HIST_COLUMNS]
    train_df = data.for_participants(train_pids)

    print("Fitting the global models on TRAINING participants only...",
          flush=True)
    models = (fit_global(train_df, ctx_features),
              fit_global(train_df, features))

    labels = anon_labels(test_pids)
    frame = data.frame[data.frame.participant_id.isin(set(test_pids))]

    records = []
    for pid, g in frame.groupby("participant_id"):
        rec = participant_record(g, models, features, ctx_features, K_GRID)
        rec["id"] = labels[pid]
        records.append(rec)
    records.sort(key=lambda r: r["id"])

    # ---- the FULL local record, written first and never shipped -----------
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "participants_full.json").write_text(
        json.dumps({"_what_this_is": "LOCAL ONLY. Every participant's full "
                                     "observation series. results/ is "
                                     "gitignored; this file is not shipped.",
                    "participants": records}, indent=1), encoding="utf-8")

    # ---- choose the illustrative subset, by predictability alone -----------
    scored_recs = [r for r in records
                   if r["by_K"].get(str(K_PRIMARY), {}).get("scored")
                   and r["autocorrelation"] is not None]
    by_r = sorted(scored_recs, key=lambda r: r["autocorrelation"])
    chosen = set()
    if by_r:
        chosen = {by_r[0]["id"], by_r[len(by_r) // 2]["id"], by_r[-1]["id"]}

    import copy
    records = copy.deepcopy(records)
    if args.series != "all":
        keep = chosen if args.series == "illustrative" else set()
        for rec in records:
            if rec["id"] not in keep:
                rec["series"] = []
                rec["series_published"] = False
            else:
                rec["series_published"] = True

    scored = [r for r in records
              if r["by_K"].get(str(K_PRIMARY), {}).get("scored")]
    better = [r for r in scored
              if r["by_K"][str(K_PRIMARY)]["twin_better"]]
    print(f"\n  participants written: {len(records)}")
    print(f"  scored at K={K_PRIMARY}: {len(scored)}")
    print(f"  twin better than persistence for {len(better)} of {len(scored)}")

    payload = {
        "_what_this_is": (
            "One record per HELD-OUT TEST participant from the frozen "
            "twin-vs-persistence experiment, disaggregated. No new model, no "
            "new metric, no aggregate recomputed."),
        "_generated_by": "scripts/export_participant_explorer.py",
        "_provenance": (
            "REAL ANONYMISED LONGITUDINAL DATA - College Experience Study. "
            "Predictions are the frozen experiment's own, strictly "
            "out-of-sample: these participants were never trained on."),
        "_anonymisation": (
            "participants renumbered by a stable digest sort; source "
            "identifiers dropped entirely; timestamps converted to day "
            "offsets from each person's own first report, so no calendar date "
            "is published"),
        "_definitions": {
            "autocorrelation": (
                "lag-1 Pearson correlation between a participant's successive "
                "stress reports, restricted to consecutive pairs 1-7 days "
                "apart, as defined in aedt/audit/ceiling.py. Descriptive "
                "evidence of temporal dependence; not a causal measure."),
            "predictability_band": BAND_RULE,
            "band_source": (
                "the thresholds are aedt/audit/ceiling.py's own "
                "NEAR_UNPREDICTABLE_BELOW and WELL_PREDICTABLE_ABOVE, the "
                "constants behind the published 13% / 17% figures. No new "
                "threshold was invented."),
            "variability_sd": "sample standard deviation of the participant's "
                              "observed stress values",
            "macro_f1": "the study's pre-registered primary metric, computed "
                        "per participant by the frozen experiment's scorer",
            "warm_up": "the first K observations personalise only and are "
                       "never scored, per the frozen protocol",
        },
        "_uncertainty": (
            "No per-participant confidence intervals are published. The "
            "study's intervals come from a participant-clustered bootstrap "
            "over the cohort, which does not yield an interval for a single "
            "person. Per-participant values are point estimates on small "
            "numbers of observations and must be read as such."),
        "_redistribution": (
            "The source archive is not redistributed. This file carries "
            "summary STATISTICS for every participant and observation series "
            "for an illustrative subset only; the complete series stay in "
            "results/, which is gitignored and never shipped."),
        "series_mode": args.series,
        "illustrative_rule": ILLUSTRATIVE_RULE,
        "illustrative_ids": sorted(chosen),
        "n_series_published": sum(1 for r in records if r.get("series")),
        "split": "test", "K_primary": K_PRIMARY, "K_grid": list(K_GRID),
        "seed": SEED,
        "n_test_participants": len(records),
        "n_scored_primary": len(scored),
        "n_twin_better_primary": len(better),
        "participants": records,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    write_experiment_record(
        OUT_RESULTS, experiment="participant_explorer_export",
        dataset="college-experience", seed=SEED,
        config={"K_grid": list(K_GRID), "K_primary": K_PRIMARY,
                "split": "test", "fits_new_model": False,
                "defines_new_metric": False,
                "series_mode": args.series,
                "illustrative_ids": sorted(chosen),
                "source": "scripts/run_twin_experiment.py (imported, "
                          "unmodified)"},
        started=started, data_root=DATA, output_dir=OUT_RESULTS)

    size = OUT_JSON.stat().st_size / 1e6
    print(f"\nwritten: {OUT_JSON} ({size:.2f} MB)")
    print(f"         {OUT_RESULTS/'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
