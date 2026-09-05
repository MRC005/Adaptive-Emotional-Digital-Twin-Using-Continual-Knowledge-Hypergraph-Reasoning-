#!/usr/bin/env python3
"""ORACLE HEADROOM: is there anything for a router to win?

Protocol: docs/preregistration_selective_personalisation.md, the feasibility
read that precedes any gate. Run on the **43 validation participants only**.
The test split is not read, not scored, and not touched.

THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT
A gate cannot rescue the twin by choosing when to deploy it unless choosing
well is worth something in the first place. The oracle router knows, for each
participant, which of twin or persistence actually scored higher, and picks it.
It is not deployable -- it reads outcomes no system can have at prediction time
-- so it is a CEILING, never a result. What it measures is how much a perfect
per-participant router would gain over always using persistence.

    headroom = mean_p[ max(twin_p, persistence_p) ] - mean_p[ persistence_p ]
             = mean_p[ max(0, twin_p - persistence_p) ]

If that is near zero, no gate can help, and the study's answer is already
written. If it is large, the live question becomes what share of it a gate
using only prior information can recover.

This script fits NOTHING and selects NOTHING. It computes no evidence
features, no coefficients and no thresholds. It reuses the frozen experiment's
own functions, unmodified, by importing them.

    python3 scripts/run_oracle_headroom.py

Writes results/gate/oracle_headroom.json and run_metadata.json beside it.

Exit codes:
    0  wrote the read-out
    6  the real archive is unavailable
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aedt.reporting.experiment_record import (  # noqa: E402
    write_experiment_record)
from aedt.twin.prediction_data import assert_no_leakage  # noqa: E402
# The frozen experiment is imported, never edited. Its splits, its warm-up
# rule, its scorer and its bootstrap are the ones used here.
from scripts.run_twin_experiment import (DATA, SEED, build_prediction_frame,
                                         fit_global, make_splits,
                                         participant_bootstrap, predict_twin,
                                         score, split_warmup)

OUT = ROOT / "results" / "gate"

#: K=20 is the pre-registered primary; K=80 the secondary sensitivity.
K_PRIMARY = 20
K_GRID = (K_PRIMARY, 80)

#: Matches the frozen experiment's history/context partition exactly.
HIST_COLUMNS = {"current", "current_social", "current_pam", "prev_1", "prev_2",
                "prev_3", "hist_mean", "hist_sd", "hist_n", "ewma",
                "days_since_prev"}


def per_participant_scores(data, pids, models, K: int) -> dict:
    """macro-F1 of the twin and of persistence, per participant, at one K."""
    model_ctx, model_full, ctx_features, features = models
    out = {}
    frame = data.frame[data.frame.participant_id.isin(set(pids))]
    for pid, g in frame.groupby("participant_id"):
        warm, ev = split_warmup(g, K)
        if len(ev) < 5:                       # frozen rule: too little to score
            continue
        y = ev["target"].to_numpy()
        persistence = ev["current"].to_numpy()
        twin = predict_twin(model_full, warm, ev, features, ctx_features)
        out[pid] = {
            "n_warmup": int(len(warm)), "n_eval": int(len(ev)),
            "persistence": score(y, persistence),
            "twin": score(y, twin),
        }
    return out


def headroom(per_person: dict, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Oracle gain over always-persistence, with a participant-clustered CI.

    Resamples PARTICIPANTS, as every interval in this project does. The oracle
    value for a participant is the better of the two realised scores, so the
    gain is the mean of the POSITIVE PART of the per-participant advantage.
    """
    pids = sorted(per_person)
    t = np.array([per_person[p]["twin"]["macro_f1"] for p in pids])
    q = np.array([per_person[p]["persistence"]["macro_f1"] for p in pids])
    adv = t - q
    gain = np.maximum(adv, 0.0)

    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(pids), len(pids)) for _ in range(n_boot)]

    def ci(vals):
        boots = np.array([vals[i].mean() for i in idx])
        return [float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5))]

    oracle = np.maximum(t, q)
    order = np.argsort(-gain)
    total = float(gain.sum())
    concentration = {
        f"top_{k}_share": (float(gain[order[:k]].sum() / total)
                           if total > 0 else None)
        for k in (1, 3, 5)
    }
    return {
        "n_participants": len(pids),
        "R0_persistence": {"macro_f1": float(q.mean()), "ci": ci(q)},
        "R1_twin": {"macro_f1": float(t.mean()), "ci": ci(t)},
        "R5_oracle": {"macro_f1": float(oracle.mean()), "ci": ci(oracle)},
        "headroom": {"mean": float(gain.mean()), "ci": ci(gain),
                     "definition": "mean_p[max(0, twin_p - persistence_p)]"},
        "twin_minus_persistence": {
            "mean": float(adv.mean()), "ci": ci(adv),
            "median": float(np.median(adv)),
            "iqr": [float(np.percentile(adv, 25)),
                    float(np.percentile(adv, 75))],
            "range": [float(adv.min()), float(adv.max())],
        },
        "twin_wins": {
            "n": int((adv > 0).sum()), "fraction": float((adv > 0).mean()),
            "n_ties": int((adv == 0).sum()),
        },
        "gain_concentration": concentration,
        "per_participant": [
            {"participant_id": str(p),
             "n_eval": per_person[p]["n_eval"],
             "twin": round(per_person[p]["twin"]["macro_f1"], 6),
             "persistence": round(per_person[p]["persistence"]["macro_f1"], 6),
             "advantage": round(float(a), 6)}
            for p, a in zip(pids, adv)
        ],
    }


def main() -> int:
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
    assert not (set(train_pids) & set(val_pids))
    assert not (set(val_pids) & set(test_pids))
    print(f"  train {len(train_pids)} | validation {len(val_pids)} | "
          f"test {len(test_pids)} (NOT READ)", flush=True)

    features = data.feature_columns
    ctx_features = [c for c in features if c not in HIST_COLUMNS]
    train_df = data.for_participants(train_pids)

    print("Fitting the global models on TRAINING participants only...",
          flush=True)
    model_ctx = fit_global(train_df, ctx_features)
    model_full = fit_global(train_df, features)
    models = (model_ctx, model_full, ctx_features, features)

    results = {}
    for K in K_GRID:
        pp = per_participant_scores(data, val_pids, models, K)
        r = headroom(pp)
        results[str(K)] = r
        h = r["headroom"]
        print(f"\nK={K}  validation participants scored: {r['n_participants']}")
        print(f"  R0 always persistence  {r['R0_persistence']['macro_f1']:.4f}"
              f"  [{r['R0_persistence']['ci'][0]:.4f}, {r['R0_persistence']['ci'][1]:.4f}]")
        print(f"  R1 always twin         {r['R1_twin']['macro_f1']:.4f}"
              f"  [{r['R1_twin']['ci'][0]:.4f}, {r['R1_twin']['ci'][1]:.4f}]")
        print(f"  R5 oracle routing      {r['R5_oracle']['macro_f1']:.4f}"
              f"  [{r['R5_oracle']['ci'][0]:.4f}, {r['R5_oracle']['ci'][1]:.4f}]")
        print(f"  HEADROOM               {h['mean']:.4f}"
              f"  [{h['ci'][0]:.4f}, {h['ci'][1]:.4f}]")
        print(f"  twin wins for {r['twin_wins']['n']} of {r['n_participants']}"
              f" ({r['twin_wins']['fraction']:.1%})")
        c = r["gain_concentration"]
        if c["top_3_share"] is not None:
            print(f"  gain concentration: top 1 {c['top_1_share']:.1%},"
                  f" top 3 {c['top_3_share']:.1%}, top 5 {c['top_5_share']:.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "_what_this_is": (
            "Oracle routing headroom on the VALIDATION split only. The oracle "
            "reads outcomes and is a CEILING, never a deployable result. The "
            "test split was not read."),
        "protocol": "docs/preregistration_selective_personalisation.md",
        "split": "validation", "n_validation_participants": len(val_pids),
        "test_split_read": False,
        "K_primary": K_PRIMARY, "seed": SEED,
        "results_by_K": results,
    }
    (OUT / "oracle_headroom.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    write_experiment_record(
        OUT, experiment="oracle_headroom_validation",
        dataset="college-experience", seed=SEED,
        config={"K_grid": list(K_GRID), "K_primary": K_PRIMARY,
                "split": "validation", "n_bootstrap": 2000,
                "fits_nothing": True, "selects_nothing": True,
                "test_split_read": False,
                "protocol": "docs/preregistration_selective_personalisation.md"},
        started=started, data_root=DATA, output_dir=OUT)

    print(f"\nwritten: {OUT/'oracle_headroom.json'} and run_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
