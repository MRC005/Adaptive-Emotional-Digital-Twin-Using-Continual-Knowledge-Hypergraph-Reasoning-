#!/usr/bin/env python3
"""Export committed experiment results into the frontend.

The website must never restate a number by hand. This regenerates
frontend/src/data/findings.json from the result files, so a page claiming
"persistence 0.332" is reading the same artefact the experiment wrote.

    python3 scripts/export_findings.py

WHAT CHANGED, AND WHY IT MATTERED
Until now this file *itself* broke that rule. The twelve "ceiling" statistics
and four cohort descriptors were literals typed here, while the emitted
``_source`` field claimed they came from the experiment's result files. They
did not: no result file contained them and no script computed them. They are
now read from ``results/twin/ceiling.json``, written by
``scripts/run_ceiling_analysis.py``.

If a required result file is missing this REFUSES to write. A partial
findings.json that silently drops the ceiling would take the numbers off the
website without anyone noticing, which is the failure mode this change exists
to end.

Exit codes:
    0  wrote frontend/src/data/findings.json
    6  a required result file is missing — run the script it names first
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every input, and the command that produces it if it is absent.
REQUIRED = {
    "results/twin/twin_experiment.json": "python3 scripts/run_twin_experiment.py",
    "results/twin/twin_ablation.json": "python3 scripts/run_twin_ablation.py",
    "results/twin/ceiling.json": "python3 scripts/run_ceiling_analysis.py",
}

#: Keys the frontend reads out of the ceiling block. Kept explicit so that a
#: rename upstream fails here rather than rendering "undefined" on the site.
CEILING_KEYS = (
    "within_person_autocorrelation", "variance_explained", "icc_between_person",
    "strongest_behaviour_r", "behaviour_variance_explained",
    "per_person_r_median", "per_person_r_iqr", "per_person_r_range",
    "frac_near_unpredictable", "frac_well_predictable", "early_late_r",
    "n_participants_analysed",
)

COHORT_KEYS = ("participants", "years", "reports", "prediction_pairs")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    missing = [(p, cmd) for p, cmd in REQUIRED.items() if not (ROOT / p).exists()]
    if missing:
        print("REFUSING to write findings.json - a required result file is "
              "missing, and the website must not display numbers that no "
              "artefact backs.", file=sys.stderr)
        for p, cmd in missing:
            print(f"  missing {p}\n    produce it with: {cmd}", file=sys.stderr)
        return 6

    exp = json.loads((ROOT / "results/twin/twin_experiment.json").read_text(
        encoding="utf-8"))
    abl = json.loads((ROOT / "results/twin/twin_ablation.json").read_text(
        encoding="utf-8"))
    cei = json.loads((ROOT / "results/twin/ceiling.json").read_text(
        encoding="utf-8"))

    r = {x["K"]: x for x in exp["results"]}
    k80 = r[80]

    ceiling = {k: cei["ceiling"][k] for k in CEILING_KEYS}
    cohort = {k: cei["cohort"][k] for k in COHORT_KEYS}
    cohort.update({"train": exp["n_train"], "val": exp["n_val"],
                   "test": exp["n_test"]})

    out = {
        "_generated_by": "scripts/export_findings.py",
        "_provenance": {
            "cohort": "results/twin/ceiling.json :: cohort",
            "headline": "results/twin/twin_experiment.json :: results[K=80]",
            "learning_curve": "results/twin/twin_experiment.json :: results[*]",
            "ablation": "results/twin/twin_ablation.json :: arms",
            "ceiling": "results/twin/ceiling.json :: ceiling",
            "splits": "results/twin/twin_experiment.json :: n_train/n_val/n_test",
            "input_digests": {p: _digest(ROOT / p) for p in REQUIRED},
            "ceiling_definition": cei["ceiling"].get("definition", {}),
            "ceiling_generated_utc": cei.get("_generated_utc"),
        },
        "cohort": cohort,
        "headline": {
            "K": 80,
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
        "ceiling": ceiling,
    }

    p = ROOT / "frontend/src/data/findings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
