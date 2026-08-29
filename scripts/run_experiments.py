#!/usr/bin/env python3
"""Reproduce the validated simulation studies, in the frozen execution order.

    python scripts/run_experiments.py --what all
    python scripts/run_experiments.py --what scenarios
    python scripts/run_experiments.py --what failure
    python scripts/run_experiments.py --what envelope
    python scripts/run_experiments.py --what ablation

EVERYTHING THIS SCRIPT PRODUCES IS SYNTHETIC and is stamped as such. It is the
evidence behind the "Preliminary Results" table, and behind the claim that the
natural affine approach fabricates a -0.107 null bias.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aedt.audit.envelope import bias_envelope
from aedt.config import load_config
from aedt.constants import SEED, DataStatus
from aedt.estimators.affine_did import failure_table
from aedt.hypergraph.ablation import ablation_verdict, run_context_ablation
from aedt.logging_setup import setup_logging
from aedt.reporting.tables import sensitivity_table, write_table
from aedt.simulate.generator import cohort_to_long_frame, simulate_cohort
from aedt.simulate.scenarios import scenario_table

W = 78


def banner(t: str) -> None:
    print()
    print("=" * W)
    print(f"  {t}    [SYNTHETIC]")
    print("=" * W)


def do_scenarios(out: Path, args) -> None:
    banner("E3 - MISSPECIFICATION SCENARIOS (13 conditions x 2 true rho)")
    rows = scenario_table(rhos=(1.00, 0.85), n_participants=args.participants,
                          n_per_epoch=args.per_epoch,
                          n_resamples=args.bootstrap)
    df = pd.DataFrame(rows)
    df["data_status"] = DataStatus.SYNTHETIC.value
    print(df[["scenario", "true_rho", "rho_star", "error", "ci_low", "ci_high",
              "convergence_rate"]].to_string(index=False,
                                             float_format=lambda v: f"{v:7.3f}"))
    print("\n  The NULL rows decide usability: a method that reports scale")
    print("  change where there is none cannot be used on real data.")
    write_table(df, out / "e3_misspecification_scenarios",
                title="E3 - estimator behaviour under 13 misspecification "
                      "conditions")


def do_failure(out: Path, args) -> None:
    banner("THE FAILURE ANALYSIS OF THE NATURAL AFFINE APPROACH")
    rows = [r.to_dict() for r in failure_table(nrep=args.replications * 10)]
    df = pd.DataFrame(rows)
    df["data_status"] = DataStatus.SYNTHETIC.value
    print(df[["mode", "n_categories", "true_rho", "rho_hat", "bias",
              "boundary_rate_pct"]].to_string(index=False,
                                              float_format=lambda v: f"{v:8.3f}"))
    print("\n  per-person K=5, true rho=1.00 must reproduce bias = -0.107.")
    print("  That fabricated ~10% scale compression, when NOTHING has changed,")
    print("  is why the ordinal slope-ratio construction exists.")
    print("  per-anchor reproduces the WITHDRAWN -0.19 harness artefact, kept")
    print("  so the Round-14 self-correction stays verifiable.")
    write_table(df, out / "e2_affine_failure_analysis",
                title="Failure analysis of the natural affine approach "
                      "(a documented NEGATIVE result)")


def do_envelope(out: Path, args) -> None:
    banner("E6 - BIAS ENVELOPE UNDER THE TRUE NULL")
    env = bias_envelope(n_participants=args.participants,
                        n_per_epoch=args.per_epoch,
                        n_replications=args.replications, seed=SEED)
    for k, v in env.rho_star_by_scenario.items():
        print(f"  {k:<28} rho* = {v:6.3f}")
    print(f"\n  ENVELOPE [5th, 95th pct] = "
          f"[{env.envelope_low:.3f}, {env.envelope_high:.3f}]")
    print(f"\n  {env.interpretation}")
    write_table(sensitivity_table(env), out / "e6_bias_envelope",
                title="E6 - bias envelope under the enumerated assumption "
                      "violations")


def do_ablation(out: Path, args) -> None:
    banner("ABLATION 1 - CONTEXT REPRESENTATION (the title-critical experiment)")
    cfg = load_config("simulation")
    df = cohort_to_long_frame(simulate_cohort(
        float(cfg.get("simulation.true_rho", 0.85)),
        n_participants=args.participants, n_per_epoch=args.per_epoch,
        seed=SEED))
    ctx = [c for c in cfg.get("context.features", []) if c in df.columns]
    tab = run_context_ablation(df, "conversation_minutes", 5, ctx_cols=ctx,
                               true_rho=float(cfg.get("simulation.true_rho",
                                                      0.85)),
                               n_resamples=args.bootstrap)
    print(tab[["representation", "rho_star", "ci_low", "ci_high", "ci_width",
               "effect_retention", "convergence_rate", "placebo_rejects"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    verdict = ablation_verdict(tab)
    print(f"\n  VERDICT: {verdict}")
    print("\n  Judged on effect retention at matched calibration, CI width,")
    print("  placebo rejection and convergence - NOT on effect magnitude.")
    tab = tab.copy()
    tab["data_status"] = DataStatus.SYNTHETIC.value
    tab["verdict"] = verdict
    write_table(tab, out / "e9_context_ablation",
                title="Ablation 1 - continuous vs feature-vector vs n-ary "
                      "hyperedge")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "scenarios", "failure", "envelope",
                             "ablation"])
    ap.add_argument("--out", default="tables")
    ap.add_argument("--participants", type=int, default=40)
    ap.add_argument("--per-epoch", type=int, default=200)
    ap.add_argument("--replications", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=399)
    a = ap.parse_args(argv)
    setup_logging("WARNING")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = {"scenarios": do_scenarios, "failure": do_failure,
            "envelope": do_envelope, "ablation": do_ablation}
    for name in (jobs if a.what == "all" else [a.what]):
        jobs[name](out, a)
    print()
    print("=" * W)
    print(f"  EXPERIMENTS COMPLETE in {time.time() - t0:.0f}s. "
          "EVERY RESULT ABOVE IS SYNTHETIC.")
    print("  No dataset file has been opened by this project.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
