"""THE REVIEW-2 DEMONSTRATION. One command, offline, a few minutes.

    python scripts/run_demo.py --dataset synthetic
    python -m aedt.demo --dataset synthetic --participant p07

The same interface accepts studentlife | relax | wesad | pmdata | synthetic.
If the real files are absent it prints, for example:

    REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN

and stops. It NEVER substitutes synthetic data for missing real data.

The nine stages, with the one line to say for each (ROUND-17 §P):

  1 Ingest      "One person, one term."
  2 Reports     "Most days sit at one end of the scale."
  3 Sensor      "The passive signal, and where we cut the term in two."
  4 Context     "A hyperedge is several conditions holding at once."
  5 Twin        "This models their measuring instrument, not their mood."
  6 MODEL       "If the curve got flatter, the same behaviour now earns a
                 different number."                    <- THE MONEY SHOT
  7 Audit       "Before believing it, we check whether we're allowed to."
  8 Result      "If the estimate doesn't clear the band, we say so."
  9 Twin update "The twin now remembers this epoch and how much to trust it."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .constants import DataStatus
from .errors import NoEligibleParticipants, ScientificError
from .logging_setup import setup_logging
from .pipeline import run_pipeline
from .reporting.metadata import make_run_metadata, new_run_dir, write_metadata

log = logging.getLogger(__name__)
W = 78


def rule(ch: str = "-") -> None:
    print(ch * W)


def head(n: int, title: str, say: str) -> None:
    print()
    rule("=")
    print(f"STAGE {n}  {title}")
    print(f'          "{say}"')
    rule("=")


def _run_s1_sensitivity(dataset, root, cfg, n_resamples) -> int:
    """PRE-SPECIFIED sensitivity analysis S1 (frozen spec §15).

    Relaxes MIN_REPORTS_PER_EPOCH from 60 to 40. This is declared IN ADVANCE in
    docs/frozen_scientific_specification.md, the deviation is detected by
    Config.deviations_from_frozen() and written into the run folder, and the
    result is reported ALONGSIDE the primary -- never as the primary.
    """
    from .config import load_config as _load

    rule("=")
    print("  PRE-SPECIFIED SENSITIVITY ANALYSIS S1")
    print("  MIN_REPORTS_PER_EPOCH: 60 (frozen) -> 40")
    print("  This is NOT the primary endpoint and must never be reported as one.")
    rule("=")
    scfg = _load("simulation" if dataset == "synthetic" else dataset,
                 overrides={"eligibility.min_reports_per_epoch": 40})
    print(f"  declared deviations: {scfg.deviations_from_frozen()}")
    try:
        r = run_pipeline(dataset, root=root, config=scfg,
                         n_resamples=n_resamples, build_twins=False,
                         halt_on_placebo_failure=False)
    except ScientificError as exc:
        print(f"\n  S1 ALSO FAILS: {exc}")
        rule("=")
        return exc.exit_code
    print(f"\n  eligible at 40/epoch: {r.n_eligible}/{len(r.eligibility)}")
    if r.placebo is not None:
        print(f"  placebo: rho*={r.placebo.rho_star:.3f} "
              f"[{r.placebo.ci_low:.3f}, {r.placebo.ci_high:.3f}] "
              f"-> {r.placebo.verdict[:60]}")
    if r.primary is not None and r.primary.uncertainty is not None:
        u = r.primary.uncertainty
        print(f"  S1 rho* = {r.primary.rho_star:.4f}  "
              f"95% CI [{u.ci_low:.4f}, {u.ci_high:.4f}]  "
              f"n={r.primary.n_participants_used}")
        print("  ^ SENSITIVITY ONLY. Underpowered and below the frozen "
              "eligibility floor.")
    else:
        print("  S1 produced no estimate either.")
    for b in r.blocking_reasons:
        print(f"  BLOCKING: {b}")
    rule("=")
    return 3


def run_demo(dataset: str = "synthetic", *, root: str | None = None,
             participant: str | None = None, out_base: str = "results",
             n_resamples: int | None = 999, quiet: bool = False,
             sensitivity: bool = False) -> int:
    t0 = time.time()
    cfg = load_config("simulation" if dataset == "synthetic" else dataset)
    setup_logging("WARNING" if quiet else "INFO")

    print()
    rule("=")
    print("  ADAPTIVE EMOTIONAL DIGITAL TWIN USING CONTINUAL KNOWLEDGE "
          "HYPERGRAPH REASONING")
    print(f"  Review-2 demonstration    dataset = {dataset}")
    rule("=")

    # ------------------------------------------------------------- pipeline
    try:
        res = run_pipeline(dataset, root=root, config=cfg,
                           n_resamples=n_resamples,
                           twin_pids=[participant] if participant else None,
                           halt_on_placebo_failure=False)
    except NoEligibleParticipants as exc:
        rule("!")
        print(f"  {exc}")
        rule("!")
        print()
        print("  NO PRIMARY RESULT IS PRODUCED. The frozen eligibility screen")
        print("  excluded every participant, and the screen is not relaxed to")
        print("  make a dataset fit.")
        if sensitivity:
            print()
            code = _run_s1_sensitivity(dataset, root, cfg, n_resamples)
            return code
        print()
        print("  Re-run with --sensitivity to execute the PRE-SPECIFIED S1")
        print("  sensitivity analysis (MIN_REPORTS_PER_EPOCH = 40), which is")
        print("  reported ALONGSIDE the primary and never in place of it.")
        rule("!")
        return exc.exit_code
    except ScientificError as exc:
        rule("!")
        print(f"  {exc}")
        rule("!")
        return exc.exit_code

    status = res.data_status
    if res.frame is None:
        rule("!")
        print(f"  {res.blocking_reasons[0] if res.blocking_reasons else 'no data'}")
        rule("!")
        print()
        print("  No synthetic substitute was used. To run this dataset for "
              "real, follow:")
        print()
        for line in (res.audit.acquisition_instructions.splitlines()
                     if res.audit else []):
            print(f"    {line}")
        rule("!")
        return 6

    out = new_run_dir(out_base, dataset, status)
    df, K, sensor = res.frame, res.n_categories, res.sensor
    from .demo_artefacts import representative_participant
    # RULE FIXED IN ADVANCE: absent an explicit --participant, feature the one
    # whose own rho* is CLOSEST to the cohort rho* -- representative, never
    # cherry-picked.
    pid = participant or representative_participant(res)
    selection_rule = (
        f"Participant {pid} chosen explicitly with --participant."
        if participant else
        f"Participant {pid} selected by a rule fixed in advance: the "
        "participant whose own rho* is CLOSEST to the cohort rho* - the "
        "representative case, never the most extreme.")
    g = df[df["pid"] == pid].sort_values("ts")

    # ---------------------------------------------------------- STAGE 1 ---
    head(1, "INGEST", "One person, one term.")
    a = res.audit
    print(f"  dataset                {a.dataset_name}   role = {a.role.value}")
    print(f"  DATA STATUS            {status.value}")
    print(f"  source                 {a.source_status}")
    print(f"  participants           {df['pid'].nunique()}")
    print(f"  observations           {len(df)}")
    print(f"  span                   {df['ts'].min().date()} .. "
          f"{df['ts'].max().date()}  "
          f"({a.longitudinal_span_days:.0f} days)" if a.longitudinal_span_days
          else "")
    print(f"  self-report            {', '.join(a.self_report_variables)}")
    print(f"  scale                  {a.self_report_scale}")
    print(f"  code -> severity       {dict(a.code_to_severity_mapping)}")
    print(f"  primary sensor         {sensor}")
    print(f"  preprocessing          retained {res.ledger.n_output}/"
          f"{res.ledger.n_input} rows ({100 * res.ledger.retention_rate:.1f}%)")
    for reason, n in (res.ledger.removed or {}).items():
        print(f"                         removed {n}: {reason}")
    if len(g):
        print(f"\n  demo participant       {pid}: {len(g)} observations, "
              f"{(g['ts'].max() - g['ts'].min()).days} days")
        print(f"  selection rule         {selection_rule}")

    # ---------------------------------------------------------- STAGE 2 ---
    head(2, "REPORTS", "Most days sit at one end of the scale - that "
                       "matters enormously later.")
    u = res.category_usage
    up = u[u["pid"] == pid]
    for _i, r in up.iterrows():
        counts = " ".join(f"{k}:{int(r[f'n_cat{k}']):>4}" for k in range(1, K + 1))
        print(f"  epoch {int(r['epoch']) + 1}   n={int(r['n']):>4}   "
              f"categories used={int(r['categories_used'])}   {counts}")
        print(f"            floor(cat 1) = {r['floor_rate']:.1%}   "
              f"ceiling(cat {K}) = {r['ceiling_rate']:.1%}")
    print(f"\n  cohort median floor rate   {u['floor_rate'].median():.1%}")
    print(f"  cohort median categories   {u['categories_used'].median():.0f} "
          f"of {K}   (assumption A5 needs >= 2 in BOTH epochs)")

    # ---------------------------------------------------------- STAGE 3 ---
    head(3, "SENSOR", "The passive signal, and where we cut the term in two.")
    for e in (0, 1):
        ge = g[g["epoch"] == e]
        if not len(ge):
            continue
        print(f"  epoch {e + 1}   n={len(ge):>4}   {sensor}: "
              f"mean={ge[sensor].mean():8.2f}  sd={ge[sensor].std(ddof=1):7.2f}  "
              f"[{ge[sensor].min():.1f}, {ge[sensor].max():.1f}]")
    ed = next((d for d in res.epochs if d.pid == pid), None)
    if ed:
        print(f"\n  epoch rule             {ed.rule}")
        print(f"  midpoint               {ed.midpoint}")
        print("  standardisation        WITHIN EACH EPOCH SEPARATELY "
              "(required, not optional)")
        print("                         no constant fitted on one epoch "
              "touches the other")

    # ---------------------------------------------------------- STAGE 4 ---
    head(4, "CONTEXT", "A hyperedge is several conditions holding at once. "
                       "We test whether it helps.")
    hg = res.hypergraphs.get(pid)
    print("  (a) continuous [FROZEN DEFAULT]   within-epoch standardised "
          "sensed level, every observation used")
    print("  (b) feature-vector bins           epoch-1 prototypes, "
          "compensatory distance")
    print("  (c) n-ary hyperedge               exact conjunction of "
          "discretised features")
    if hg:
        print(f"\n  participant {pid} hypergraph:")
        print(f"    factors                {hg.factors}")
        print(f"    vertices               {hg.n_vertices}")
        print(f"    hyperedges             {hg.n_edges} "
              f"(mean arity {hg.mean_arity():.1f})")
        print(f"    occupied both epochs   {len(hg.edges_occupied_both_epochs())}")
        ov = hg.occupancy_overlap()
        print(f"    epoch overlap          "
              f"{'n/a' if not np.isfinite(ov) else f'{ov:.2f}'}   "
              "(the twin reasons over this to decide trust)")
    print("\n  THE HYPERGRAPH IS NOT THE IDENTIFICATION MECHANISM. It is the")
    print("  contextual knowledge representation and an ablation arm.")

    # ---------------------------------------------------------- STAGE 5 ---
    head(5, "PERSONAL DIGITAL TWIN",
         "This models their measuring instrument, not their mood.")
    twin = res.twins.get(pid)
    if twin is not None:
        st = twin.state
        print(f"  pid                    {st.pid}")
        print(f"  data status            {st.data_status}")
        print(f"  current time           {st.current_time}")
        print(f"  observations seen      {st.n_observations_seen}")
        print(f"  category usage         {json.dumps(st.category_usage)}")
        print(f"  eligibility            {st.eligibility_status}")
        print(f"  audit flags            "
              f"{json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.audit_flags.items()})}")
        print(f"  knowledge nodes        {len(twin.knowledge.nodes)} "
              f"(append-only, provenance-carrying)")
        for kind in ("personalised_parameters", "state_history",
                     "context_relationship", "uncertainty_audit"):
            print(f"      {kind:<26} {len(twin.knowledge.history(kind))}")
        tp = out / "twins" / f"{pid}.json"
        twin.save(tp)
        print(f"\n  twin persisted to      {tp}")
    else:
        print("  (no twin built for this participant)")

    # ---------------------------------------------------------- STAGE 6 ---
    head(6, "ORDINAL MODEL  ** THE MONEY SHOT **",
         "If the curve got flatter, the same behaviour now earns a "
         "different number.")
    from .demo_artefacts import two_curve_figure
    from .estimators.slope_ratio import fit_person_epochs
    fits = fit_person_epochs(g, sensor, K, pid=pid, data_status=status)
    for e in (0, 1):
        f = fits[e]
        if f.converged:
            print(f"  epoch {e + 1}   beta = {f.beta:+.4f}   "
                  f"cutpoints = {[round(c, 3) for c in f.cutpoints]}   "
                  f"n = {f.n}   logL = {f.loglik:.1f}")
        else:
            print(f"  epoch {e + 1}   NOT FITTED: {f.reason}")
    fig_path = two_curve_figure(
        res, pid, out / "figures" / "fig01_two_curve_epoch1_vs_epoch2.png",
        selection_rule=selection_rule)
    if fig_path is not None:
        print(f"\n  rho*_{pid} = beta_2 / beta_1 = "
              f"{fits[1].beta / fits[0].beta:.3f}")
        print(f"  FIGURE                 {fig_path}")
    else:
        print("\n  NO FIGURE: this participant's epochs did not both converge. "
              "A missing figure is reported, never faked.")

    # ---------------------------------------------------------- STAGE 7 ---
    head(7, "AUDIT + PLACEBO",
         "Before believing it, we check whether we're allowed to.")
    asc = res.association
    print(f"  [9b] ASSOCIATION       median |beta| = {asc.median_abs_beta:.3f}   "
          f"IQR [{asc.iqr_low:.3f}, {asc.iqr_high:.3f}]   "
          f"determined {asc.n_determined}/{asc.n_screened}")
    print(f"                         {asc.recommendation}")
    n_ok = res.n_eligible
    print(f"\n  ELIGIBILITY            {n_ok}/{len(res.eligibility)} eligible")
    for r in res.eligibility:
        if not r.eligible:
            print(f"      EXCLUDED {r.pid:<8} {'; '.join(r.reasons)}")
    if n_ok == len(res.eligibility):
        print("      (no exclusions)")
    p = res.placebo
    print(f"\n  PLACEBO                n={p.n_participants}   "
          f"rho*={p.rho_star:.3f}   95% CI [{p.ci_low:.3f}, {p.ci_high:.3f}]")
    print(f"                         {p.verdict}")
    print("                         contiguous epoch-1 split-half: no "
          "response shift CAN exist between the halves")

    # ---------------------------------------------------------- STAGE 8 ---
    head(8, "RESULT", "If the estimate doesn't clear the band, we say so.")
    if res.primary is None:
        print("  PRIMARY NOT RUN - blocked by the gate above.")
        for b in res.blocking_reasons:
            print(f"      {b}")
    else:
        pr = res.primary
        un = pr.uncertainty
        print(f"  ESTIMAND               rho*  (the IDENTIFIED estimand)")
        print(f"  rho*                   {pr.rho_star:.4f}")
        print(f"  median per-person      {pr.median_rho_star:.4f}")
        if un:
            print(f"  95% CI                 [{un.ci_low:.4f}, {un.ci_high:.4f}]"
                  f"   ({un.n_resamples} participant-cluster bootstrap "
                  "resamples)")
            print(f"  CI excludes 1.0        "
                  f"{'YES' if un.excludes_null else 'NO'}")
        print(f"  participants used      {pr.n_participants_used}/"
              f"{pr.n_participants_screened}")
        for k, v in pr.exclusions.items():
            print(f"      NOT USED {k:<8} {v}")
        print()
        print(f"  1 - rho* = {pr.lower_bound_on_recalibration:+.4f} is a LOWER "
              "BOUND on the true multiplicative recalibration.")
        print("  rho ITSELF IS NOT POINT-IDENTIFIED.")
        print("  The additive component b2 - b1 is NOT IDENTIFIED and is NOT "
              "ESTIMATED.")

    # ---------------------------------------------------------- STAGE 9 ---
    head(9, "TWIN UPDATE",
         "The twin now remembers this epoch and how much to trust it.")
    if twin is not None and twin.state.history:
        h = twin.state.history[-1]
        print(f"  verdict                {h['verdict']}")
        print(f"  rho* recorded          {h['rho_star']:.4f}")
        print(f"  epoch span             {h['epoch_span']['start']} .. "
              f"{h['epoch_span']['end']}")
        print(f"  audit flags            "
              f"{json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in h['flags'].items()})}")
        if h["reasons"]:
            for r_ in h["reasons"]:
                print(f"      FLAGGED: {r_}")
        print(f"  history rows           {len(twin.state.history)} "
              "(append-only)")
    accepted = sum(1 for t in res.twins.values()
                   if t.state.history and t.state.history[-1]["verdict"] == "ACCEPTED")
    print(f"\n  twins updated          {len(res.twins)}   "
          f"ACCEPTED={accepted}   FLAGGED={len(res.twins) - accepted}")

    # ------------------------------------------------------------ artefacts
    from .demo_artefacts import write_demo_artefacts
    paths = write_demo_artefacts(res, out, pid=pid, n_resamples=n_resamples,
                                 cfg=cfg, selection_rule=selection_rule)

    meta = make_run_metadata(dataset=dataset, data_status=status, config=cfg,
                             started=t0, output_dir=out,
                             command=" ".join(sys.argv))
    write_metadata(meta, out, cfg)

    print()
    rule("=")
    print("  REPRODUCIBILITY")
    rule("=")
    print(f"  run id                 {meta.run_id}")
    print(f"  data status            {status.value}")
    print(f"  seed                   {meta.seed}")
    print(f"  config                 {meta.config_path}  "
          f"(digest {meta.config_digest})")
    print(f"  software version       aedt {meta.software_version}   "
          f"python {meta.python_version}")
    print(f"  git commit             {meta.git_commit or 'not a git checkout'}")
    print(f"  elapsed                {meta.elapsed_seconds:.1f}s")
    print(f"  output                 {out}")
    for label, pth in paths.items():
        print(f"      {label:<26} {Path(pth).relative_to(out)}")
    rule("=")
    print(f"  EVERY FIGURE AND TABLE ABOVE IS STAMPED {status.value}.")
    if status is DataStatus.SYNTHETIC:
        print("  THIS RUN IS SIMULATION. Nothing here is evidence about "
              "humans.")
        print("  NO REAL-DATA rho* RESULT EXISTS IN THIS PROJECT. One real")
        print("  dataset (RELAX, Zenodo 10.5281/zenodo.20701999) has been")
        print("  acquired and audited; it FAILS the frozen eligibility screen")
        print("  (0 of 31 participants), so no estimate was produced from it.")
        print("  See docs/dataset_audit.md.")
    rule("=")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aedt.demo",
        description="Review-2 demonstration of the Adaptive Emotional Digital "
                    "Twin pipeline.")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "studentlife", "relax", "wesad",
                             "pmdata"])
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run the PRE-SPECIFIED S1 sensitivity analysis "
                         "(MIN_REPORTS_PER_EPOCH=40) when the frozen screen "
                         "leaves nobody eligible")
    ap.add_argument("--root", default=None,
                    help="path to the real dataset archive (real datasets only)")
    ap.add_argument("--participant", default=None,
                    help="which participant to feature in stages 1-6 and 9")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bootstrap", type=int, default=999,
                    help="bootstrap resamples (the frozen primary uses 2000)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return run_demo(a.dataset, root=a.root, participant=a.participant,
                    out_base=a.out, n_resamples=a.bootstrap, quiet=a.quiet,
                    sensitivity=a.sensitivity)


if __name__ == "__main__":
    raise SystemExit(main())
