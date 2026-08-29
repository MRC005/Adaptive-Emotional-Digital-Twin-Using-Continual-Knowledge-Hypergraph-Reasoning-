"""Figure and table generation for one pipeline run.

Separated from ``demo.py`` so that ``scripts/generate_review2_outputs.py`` can
regenerate every artefact without re-printing the nine demo stages.

Every artefact is stamped with the run's DataStatus. Nothing is written without
one -- ``viz.save_figure`` and ``reporting.write_table`` both refuse.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DataStatus
from .pipeline import PipelineResult
from .reporting.tables import (contribution_table, dataset_audit_table,
                               eligibility_summary, estimator_table,
                               exclusions_table, participant_table,
                               placebo_table, sensitivity_table, status_board,
                               title_alignment_table, uncertainty_table,
                               write_table)
from .viz import (ablation_plot, audit_dashboard, architecture_diagram,
                  category_usage_plot, forest_plot, hypergraph_plot,
                  pipeline_diagram, placebo_plot)

log = logging.getLogger(__name__)

__all__ = ["write_demo_artefacts", "two_curve_figure",
           "representative_participant"]


def representative_participant(res: PipelineResult) -> str:
    """Pick the participant to feature in the two-curve figure.

    RULE, FIXED IN ADVANCE: the participant whose own rho* is CLOSEST to the
    cohort rho* -- the representative case, not the most extreme one.

    This matters. Featuring the participant with the largest apparent
    recalibration would flatter the method with a cherry-picked picture; the
    rule below cannot do that, because a participant is chosen for being
    TYPICAL. The figure states which rule selected them.
    """
    if res.primary is None or not res.primary.per_participant_pids:
        return str(res.frame["pid"].iloc[0]) if res.frame is not None else ""
    vals = np.asarray(res.primary.per_participant_rho_star, dtype=float)
    pids = list(res.primary.per_participant_pids)
    return pids[int(np.argmin(np.abs(np.log(vals)
                                     - res.primary.log_rho_star)))]


def two_curve_figure(res: PipelineResult, pid: str, path: Path, *,
                     selection_rule: str = "") -> Path | None:
    """The epoch-1 vs epoch-2 ordinal curves for one participant.

    Returns None when that participant's two epochs did not both converge --
    a missing figure is reported, never faked.
    """
    from .estimators.slope_ratio import (fit_person_epochs,
                                         standardise_within_epoch)
    from .viz.curves import two_curve_plot

    g = res.frame[res.frame["pid"] == pid].sort_values("ts")
    fits = fit_person_epochs(g, res.sensor, res.n_categories, pid=pid,
                             data_status=res.data_status)
    if not (fits[0].converged and fits[1].converged):
        log.warning("no two-curve figure for %s: epoch1=%s epoch2=%s", pid,
                    fits[0].reason or "ok", fits[1].reason or "ok")
        return None
    obs = []
    for e in (0, 1):
        ge = g[g["epoch"] == e]
        x, _m, _s = standardise_within_epoch(ge[res.sensor].to_numpy(float))
        obs.append(pd.DataFrame({"x": x, "report": ge["report"].to_numpy(),
                                 "epoch": e}))
    return two_curve_plot(
        fits, pid=pid, data_status=res.data_status, sensor_name=res.sensor,
        observations=pd.concat(obs), path=path,
        cohort_rho_star=res.primary.rho_star if res.primary else None,
        selection_rule=selection_rule)


def write_demo_artefacts(res: PipelineResult, out: Path, *, pid: str,
                         selection_rule: str = "",
                         n_resamples: int | None = 999, cfg=None,
                         run_ablation: bool = True,
                         run_envelope: bool = True,
                         run_baselines_too: bool = True) -> dict[str, Path]:
    """Write every figure and table for this run. Returns {label: path}."""
    paths: dict[str, Path] = {}
    figs, tabs = out / "figures", out / "tables"
    status, df, K, sensor = (res.data_status, res.frame, res.n_categories,
                             res.sensor)

    # ------------------------------------------------------------- figures
    paths["fig: architecture"] = architecture_diagram(
        data_status=status, path=figs / "fig00_architecture.png")
    paths["fig: pipeline"] = pipeline_diagram(
        data_status=status, path=figs / "fig00b_pipeline.png")

    # THE MONEY SHOT. Two curves on one axis is the entire project in one
    # picture, so it is generated here as well as by the demo narrative.
    two_curve = two_curve_figure(
        res, pid, figs / "fig01_two_curve_epoch1_vs_epoch2.png",
        selection_rule=selection_rule)
    if two_curve is not None:
        paths["fig: two-curve (MONEY SHOT)"] = two_curve

    if res.category_usage is not None and len(res.category_usage):
        paths["fig: category usage"] = category_usage_plot(
            res.category_usage, K, data_status=status,
            path=figs / "fig02_category_usage.png")

    if res.placebo is not None:
        paths["fig: placebo"] = placebo_plot(
            res.placebo, res.primary, path=figs / "fig03_placebo.png")

    envelope_band = None
    if run_envelope:
        from .audit.envelope import bias_envelope, measured_properties
        from .viz.diagnostics_viz import envelope_plot
        meas = measured_properties(df, sensor, K)
        # The envelope characterises the shape of the NULL distribution at
        # roughly this data's density; it is capped so the demo stays fast.
        # A full-density envelope is produced by scripts/run_experiments.py.
        env = bias_envelope(
            n_participants=min(int(cfg.get("envelope.n_participants", 48))
                               if cfg else 48, int(df["pid"].nunique())),
            n_per_epoch=int(min(
                int(cfg.get("envelope.n_per_epoch", 200)) if cfg else 200,
                max(60, np.median([len(g[g["epoch"] == 0])
                                   for _p, g in df.groupby("pid")])))),
            n_replications=int(cfg.get("envelope.n_replications", 6)) if cfg else 6,
            measured=meas, data_status=DataStatus.SYNTHETIC)
        envelope_band = (env.envelope_low, env.envelope_high)
        paths["fig: bias envelope"] = envelope_plot(
            env, res.primary, path=figs / "fig05_bias_envelope.png")
        paths["tab: sensitivity"] = write_table(
            sensitivity_table(env), tabs / "t07_sensitivity_bias_envelope",
            title="Bias envelope under the enumerated assumption violations")
        log.info("bias envelope [%.3f, %.3f] over %d scenarios",
                 env.envelope_low, env.envelope_high, len(env.scenarios))

    if res.primary is not None:
        paths["fig: forest"] = forest_plot(
            res.primary, envelope=envelope_band,
            path=figs / "fig04_forest_rho_star.png")

    hg = res.hypergraphs.get(pid)
    if hg is not None:
        paths["fig: hypergraph"] = hypergraph_plot(
            hg, data_status=status, path=figs / "fig06_hypergraph_context.png")

    if run_ablation:
        from .hypergraph.ablation import ablation_verdict, run_context_ablation
        ctx = [c for c in ((cfg.get("context.features", [sensor]) if cfg
                            else [sensor]) or [sensor]) if c in df.columns]
        true_rho = cfg.get("simulation.true_rho") if cfg else None
        tab = run_context_ablation(df, sensor, K, ctx_cols=ctx or [sensor],
                                   true_rho=true_rho, n_resamples=n_resamples,
                                   data_status=status)
        verdict = ablation_verdict(tab)
        paths["fig: ablation"] = ablation_plot(
            tab, data_status=status, verdict=verdict,
            path=figs / "fig07_hypergraph_ablation.png")
        tab_out = tab.copy()
        tab_out["data_status"] = status.value
        tab_out["verdict"] = verdict
        paths["tab: ablation"] = write_table(
            tab_out, tabs / "t09_hypergraph_ablation",
            title="Ablation 1 - context representation "
                  "(continuous vs feature-vector vs n-ary hyperedge)")

    if res.association is not None and res.placebo is not None:
        paths["fig: dashboard"] = audit_dashboard(
            eligibility=eligibility_summary(res.eligibility),
            association=res.association, placebo=res.placebo,
            primary=res.primary, dataset=res.dataset, data_status=status,
            envelope=envelope_band, path=figs / "fig08_audit_dashboard.png")

    # -------------------------------------------------------------- tables
    if res.audit is not None:
        paths["tab: dataset audit"] = write_table(
            dataset_audit_table([res.audit]), tabs / "t01_dataset_audit",
            title="Dataset audit (T4)")
    paths["tab: participants"] = write_table(
        participant_table(df, sensor, K, status), tabs / "t02_participants",
        title="Participant characteristics")
    paths["tab: eligibility"] = write_table(
        eligibility_summary(res.eligibility), tabs / "t03_eligibility",
        title="Eligibility screen - every exclusion with its reason")
    if res.primary is not None:
        paths["tab: primary"] = write_table(
            estimator_table(res.primary), tabs / "t04_primary_estimator",
            title="Primary estimator - rho* (the IDENTIFIED estimand)")
        paths["tab: exclusions"] = write_table(
            exclusions_table(res.primary), tabs / "t05_exclusions",
            title="Participants not used in the primary estimate")
        if res.primary.uncertainty is not None:
            paths["tab: uncertainty"] = write_table(
                uncertainty_table(res.primary.uncertainty),
                tabs / "t06_uncertainty",
                title="Uncertainty - participant-cluster bootstrap")
    if res.placebo is not None:
        paths["tab: placebo"] = write_table(
            placebo_table(res.placebo), tabs / "t05b_placebo",
            title="Placebo / negative control (gates the primary)")

    if run_baselines_too:
        from .baselines import run_baselines
        from .audit.eligibility import filter_eligible
        keep = filter_eligible(df, res.eligibility)
        bl = run_baselines(keep, sensor, K, seed=cfg.seed if cfg else 20260828,
                           n_resamples=n_resamples, data_status=status,
                           which=(cfg.get("baselines.run") if cfg else None))
        paths["tab: baselines"] = write_table(
            bl, tabs / "t08_baselines",
            title="Baseline comparison - judged on calibration, CI width, "
                  "stability and interpretability, NOT effect magnitude")

    paths["tab: status board"] = write_table(
        status_board(), tabs / "t10_status_board",
        title="Implementation status board (COMPLETED means implemented AND "
              "verified)")
    paths["tab: title alignment"] = write_table(
        title_alignment_table(), tabs / "t11_title_alignment",
        title="Title term -> actual module -> status -> role")
    paths["tab: contributions"] = write_table(
        contribution_table(), tabs / "t12_standard_vs_contribution",
        title="Standard technology vs our research contribution")

    return paths
