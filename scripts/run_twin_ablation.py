#!/usr/bin/env python3
"""Ablation: which parts of the twin actually contribute?

Same split, same K, same budget as the primary experiment. K=20 chosen in the
pre-registration as the mid-point of the grid, before results were seen.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aedt.twin.prediction_data import build_prediction_frame
from scripts.run_twin_experiment import (SEED, fit_global, make_splits,
                                         participant_bootstrap, score,
                                         split_warmup, predict_twin)

DATA = ROOT / "data" / "raw" / "college-experience"
OUT = ROOT / "results" / "twin"
K = 20

HIST = {"current","current_social","current_pam","prev_1","prev_2","prev_3",
        "hist_mean","hist_sd","hist_n","ewma","days_since_prev"}
TRAJ = {"prev_1","prev_2","prev_3","ewma","hist_mean","hist_sd"}

def main():
    data = build_prediction_frame(DATA)
    feats = data.feature_columns
    tr, va, te = make_splits(data)
    train = data.for_participants(tr)
    beh = [c for c in feats if c not in HIST]

    ARMS = {
        "A1_global_only":        [c for c in feats if c not in HIST],
        "A3_global_plus_history": feats,
        "A5_no_trajectory":      [c for c in feats if c not in TRAJ],
        "A6_no_behaviour":       [c for c in feats if c in HIST],
    }
    models = {n: fit_global(train, cols) for n, cols in ARMS.items()}
    res = {}
    for name, cols in ARMS.items():
        pp = {}
        for pid, g in data.for_participants(te).groupby("participant_id"):
            warm, ev = split_warmup(g, K)
            if len(ev) < 5: continue
            # static: no online adaptation
            p = models[name].predict(ev[cols])
            pp[pid] = score(ev["target"].to_numpy(), p)
        m, lo, hi = participant_bootstrap(pp, "macro_f1")
        res[name] = {"macro_f1": m, "ci": [lo, hi],
                     "accuracy": float(np.mean([v["accuracy"] for v in pp.values()])),
                     "mae": float(np.mean([v["mae"] for v in pp.values()])),
                     "n_features": len(cols)}
    # A4 = full twin WITH online adaptation (from the primary run)
    full = fit_global(train, feats)
    pp = {}
    for pid, g in data.for_participants(te).groupby("participant_id"):
        warm, ev = split_warmup(g, K)
        if len(ev) < 5: continue
        pp[pid] = score(ev["target"].to_numpy(),
                        predict_twin(full, warm, ev, feats, beh))
    m, lo, hi = participant_bootstrap(pp, "macro_f1")
    res["A4_full_twin_online"] = {"macro_f1": m, "ci": [lo, hi],
        "accuracy": float(np.mean([v["accuracy"] for v in pp.values()])),
        "mae": float(np.mean([v["mae"] for v in pp.values()])), "n_features": len(feats)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"twin_ablation.json").write_text(json.dumps({"K":K,"seed":SEED,"arms":res}, indent=2))
    print(f"{'arm':<24} {'feats':>6} {'macroF1':>8} {'95% CI':>18} {'acc':>7} {'MAE':>7}")
    print("-"*76)
    for n, v in sorted(res.items(), key=lambda kv:-kv[1]["macro_f1"]):
        print(f"{n:<24} {v['n_features']:>6} {v['macro_f1']:>8.4f} "
              f"[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]".rjust(19) +
              f"{v['accuracy']:>7.4f} {v['mae']:>7.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
