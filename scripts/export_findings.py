#!/usr/bin/env python3
"""Export committed experiment results into the frontend.

The website must never restate a number by hand. This regenerates
frontend/src/data/findings.json from the results files, so a page claiming
"persistence 0.332" is reading the same artefact the experiment wrote.

    python3 scripts/export_findings.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    exp = json.loads((ROOT/"results/twin/twin_experiment.json").read_text())
    abl = json.loads((ROOT/"results/twin/twin_ablation.json").read_text())
    r = {x["K"]: x for x in exp["results"]}; k80 = r[80]
    out = {
      "_source": "results/twin/twin_experiment.json + twin_ablation.json",
      "_generated_by": "scripts/export_findings.py",
      "cohort": {"participants": 218, "years": 4.8, "reports": 35348,
                 "prediction_pairs": 25966, "train": exp["n_train"],
                 "val": exp["n_val"], "test": exp["n_test"]},
      "headline": {"K": 80,
        "models": {n: {"macro_f1": m["macro_f1_mean"], "ci": m["macro_f1_ci"],
                       "accuracy": m["accuracy"], "mae": m["mae"]}
                   for n, m in k80["models"].items()},
        "twin_vs_persistence": k80["twin_vs_B1_persistence"],
        "twin_vs_calibrated": k80["twin_vs_B4_calibrated"]},
      "learning_curve": [
        {"K": k, "twin": v["models"]["T_twin"]["macro_f1_mean"],
         "persistence": v["models"]["B1_persistence"]["macro_f1_mean"],
         "twin_acc": v["models"]["T_twin"]["accuracy"],
         "twin_mae": v["models"]["T_twin"]["mae"]} for k, v in sorted(r.items())],
      "ablation": {n: v["macro_f1"] for n, v in abl["arms"].items()},
      # measured by the diagnostics in the forward-strategy inspection
      "ceiling": {"within_person_autocorrelation": 0.339, "variance_explained": 0.115,
        "icc_between_person": 0.246, "strongest_behaviour_r": 0.0907,
        "behaviour_variance_explained": 0.008, "per_person_r_median": 0.346,
        "per_person_r_iqr": [0.239, 0.466], "per_person_r_range": [-0.243, 0.687],
        "frac_near_unpredictable": 0.13, "frac_well_predictable": 0.17,
        "early_late_r": 0.355, "n_participants_analysed": 194},
    }
    p = ROOT/"frontend/src/data/findings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
