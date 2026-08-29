"""END-TO-END PIPELINE. Enforces the frozen EXPERIMENT EXECUTION ORDER.

    1  load (adapter)                 -> canonical LongFrame + DatasetAudit
    2  preprocess                     -> missingness ledger, no imputation
    3  temporal alignment / epochs    -> halves of each participant's own span
    4  features / context             -> continuous [+ vector, hyperedge]
    5  dataset audit                  -> the T4 table
    6  [9b] association strength      -> READ FIRST on real data
    7  eligibility screen             -> every exclusion by name
    8  PLACEBO                        -> GATES the primary
    9  primary rho*                   -> only if the placebo passed
   10  uncertainty                    -> participant-cluster bootstrap
   11  twin update                    -> ACCEPTED / FLAGGED-UNTRUSTWORTHY
   12  baselines, ablation, envelope  -> optional extras

A final primary result is NEVER labelled validated if a preceding check failed.
``PipelineResult.validated`` is False whenever the audit, the association
diagnostic, the eligibility screen or the placebo did not pass, and the reason
is carried in ``blocking_reasons``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .audit.diagnostics import AssociationStrength, association_strength
from .audit.eligibility import filter_eligible, screen_cohort
from .config import Config, load_config
from .constants import DataStatus, DatasetRole
from .contexts.continuous import continuous_context
from .errors import (NoEligibleParticipants, PlaceboFailed,
                     PlaceboUnrunnable, RealDataUnavailable)
from .estimators.slope_ratio import estimate_rho_star
from .hypergraph.structure import nary_context
from .inference.placebo import placebo_split_half
from .io import get_adapter
from .io.fixtures import is_fixture
from .preprocess.clean import MissingnessLedger, clean_long_frame
from .preprocess.epochs import assign_epochs, epoch_definitions
from .preprocess.reports import category_usage_table
from .schemas import (DatasetAudit, EligibilityResult, EstimatorResult,
                      PlaceboResult, Serialisable, validate_long_frame)
from .twin.update import run_longitudinal_update

log = logging.getLogger(__name__)

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass
class PipelineResult(Serialisable):
    """Everything one run produced, plus whether it may be called validated."""

    dataset: str
    data_status: DataStatus
    audit: DatasetAudit | None = None
    frame: pd.DataFrame | None = None
    n_categories: int = 5
    sensor: str = ""
    ledger: MissingnessLedger | None = None
    epochs: list = field(default_factory=list)
    category_usage: pd.DataFrame | None = None
    association: AssociationStrength | None = None
    eligibility: list[EligibilityResult] = field(default_factory=list)
    placebo: PlaceboResult | None = None
    primary: EstimatorResult | None = None
    twins: dict = field(default_factory=dict)
    hypergraphs: dict = field(default_factory=dict)
    blocking_reasons: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def validated(self) -> bool:
        """A primary result may be called VALIDATED only if nothing blocked it."""
        return (self.primary is not None
                and np.isfinite(self.primary.rho_star)
                and not self.blocking_reasons
                and self.data_status is DataStatus.REAL)

    @property
    def n_eligible(self) -> int:
        return sum(1 for r in self.eligibility if r.eligible)

    def to_dict(self) -> dict:                    # frames are not JSON-safe
        d = {"dataset": self.dataset,
             "data_status": self.data_status.value,
             "n_categories": self.n_categories,
             "sensor": self.sensor,
             "n_participants": (int(self.frame["pid"].nunique())
                                if self.frame is not None else 0),
             "n_observations": int(len(self.frame)) if self.frame is not None else 0,
             "n_eligible": self.n_eligible,
             "validated": self.validated,
             "blocking_reasons": list(self.blocking_reasons),
             "elapsed_seconds": self.elapsed_seconds}
        if self.audit is not None:
            d["audit"] = self.audit.to_dict()
        if self.ledger is not None:
            d["missingness_ledger"] = self.ledger.to_dict()
        if self.association is not None:
            d["association_strength"] = self.association.to_dict()
        if self.placebo is not None:
            d["placebo"] = self.placebo.to_dict()
        if self.primary is not None:
            d["primary"] = self.primary.to_dict()
        d["eligibility"] = [r.to_dict() for r in self.eligibility]
        return d


def _resolve_status(adapter, root, declared: DataStatus) -> DataStatus:
    """A fixture directory can NEVER produce a REAL stamp."""
    if declared is DataStatus.REAL and is_fixture(root):
        log.warning("root %s is a SYNTHETIC FIXTURE (marker file present); "
                    "results will be stamped SYNTHETIC, not REAL", root)
        return DataStatus.SYNTHETIC
    return declared


def run_pipeline(dataset: str, *, root: str | Path | None = None,
                 config: Config | None = None,
                 n_resamples: int | None = None,
                 build_twins: bool = True, twin_pids: list[str] | None = None,
                 build_hypergraphs: bool = True,
                 halt_on_placebo_failure: bool = True,
                 strict_real: bool = False) -> PipelineResult:
    """Run the full pipeline in the frozen order.

    ``strict_real`` makes a missing real archive a hard failure
    (``RealDataUnavailable``, exit 6) instead of a reported one -- used by the
    audit script when the caller explicitly asked for real data.
    """
    t0 = time.time()
    cfg = config or load_config(
        "simulation" if dataset == "synthetic" else dataset)
    adapter = type(get_adapter(dataset)).from_config(cfg)

    # ---- 1. LOAD --------------------------------------------------------
    log.info("STAGE 1  ingestion            adapter=%s role=%s",
             adapter.name, adapter.role.value)
    audit = adapter.audit(root)
    if not audit.local_files_available and adapter.role is not DatasetRole.SIMULATION:
        msg = audit.source_status
        log.error(msg)
        if strict_real:
            raise RealDataUnavailable(
                f"{msg}\n{adapter.acquisition_instructions}")
        res = PipelineResult(dataset=dataset, data_status=DataStatus.PLANNED,
                             audit=audit,
                             blocking_reasons=[msg],
                             elapsed_seconds=round(time.time() - t0, 3))
        return res

    loaded = adapter.load(root)
    status = _resolve_status(adapter, root, loaded.data_status)
    df = validate_long_frame(loaded.frame, loaded.sensor,
                             n_categories=loaded.n_categories)
    K, sensor = loaded.n_categories, loaded.sensor
    log.info("         participants=%d observations=%d status=%s",
             df["pid"].nunique(), len(df), status.value)

    blocking: list[str] = []
    if adapter.role is DatasetRole.BENCHMARK_PHYSIOLOGICAL:
        blocking.append(
            f"{dataset} is a PHYSIOLOGICAL BENCHMARK. It cannot support the "
            "primary longitudinal estimand rho*, and no primary result from it "
            "may be presented as longitudinal validation.")

    # ---- 2. PREPROCESS --------------------------------------------------
    log.info("STAGE 2  preprocessing")
    df, ledger = clean_long_frame(df, sensor)
    log.info("         retained %d/%d rows (%.1f%%); removed: %s",
             ledger.n_output, ledger.n_input, 100 * ledger.retention_rate,
             ledger.removed or "nothing")

    # ---- 3. EPOCHS ------------------------------------------------------
    log.info("STAGE 3  temporal alignment   rule=%s",
             cfg.get("epochs.rule", "own_span_halves"))
    df = assign_epochs(df, rule=cfg.get("epochs.rule", "own_span_halves"))
    df = validate_long_frame(df, sensor, require_epoch=True, n_categories=K)
    epochs = epoch_definitions(df)

    # ---- 4. FEATURES / CONTEXT ------------------------------------------
    log.info("STAGE 4  features and context")
    df = continuous_context(df, sensor)
    usage = category_usage_table(df, K)
    graphs: dict = {}
    if build_hypergraphs:
        ctx_cols = [c for c in cfg.get("context.features", [sensor])
                    if c in df.columns] or [sensor]
        df, graphs = nary_context(df, ctx_cols,
                                  n_bins=int(cfg.get("context.n_bins", 3)),
                                  max_factors=int(cfg.get("context.max_factors", 3)))
        log.info("         hypergraph: %d participants, mean %.1f hyperedges",
                 len(graphs),
                 float(np.mean([g.n_edges for g in graphs.values()]))
                 if graphs else 0.0)

    # ---- 5/6. AUDIT + [9b] ASSOCIATION ----------------------------------
    log.info("STAGE 5  dataset audit        status=%s", audit.data_status.value)
    assoc = association_strength(
        df, sensor, K, threshold=float(cfg.get("diagnostics.weak_association_beta",
                                               0.15)), data_status=status)
    log.info("STAGE 6  [9b] association     median|beta|=%.3f weak=%s",
             assoc.median_abs_beta, assoc.weak)
    log.info("         %s", assoc.recommendation)
    if assoc.weak:
        blocking.append(
            f"[9b] WEAK SENSOR-REPORT ASSOCIATION (median |beta| = "
            f"{assoc.median_abs_beta:.3f}). Any interval will be wide "
            "regardless of calibration; the pre-specified PC1 fallback "
            "covariate must be considered before interpretation.")

    # ---- 7. ELIGIBILITY --------------------------------------------------
    log.info("STAGE 7  eligibility screen   thresholds=%s",
             cfg.eligibility_thresholds())
    elig = screen_cohort(df, sensor, K, thresholds=cfg.eligibility_thresholds(),
                         data_status=status)
    n_ok = sum(r.eligible for r in elig)
    log.info("         eligible=%d excluded=%d", n_ok, len(elig) - n_ok)
    for r in elig:
        if not r.eligible:
            log.info("         EXCLUDED %-8s %s", r.pid, "; ".join(r.reasons))
    if n_ok == 0:
        blocking.append("No participant passed the eligibility screen.")
        res = PipelineResult(dataset=dataset, data_status=status, audit=audit,
                             frame=df, n_categories=K, sensor=sensor,
                             ledger=ledger, epochs=epochs, category_usage=usage,
                             association=assoc, eligibility=elig,
                             hypergraphs=graphs, blocking_reasons=blocking,
                             elapsed_seconds=round(time.time() - t0, 3))
        raise NoEligibleParticipants(
            "No eligible participants; the audit stops here. "
            f"Reasons: {sorted({x for r in elig for x in r.reasons})}")
    keep = filter_eligible(df, elig)

    # ---- 8. PLACEBO (GATES THE PRIMARY) ---------------------------------
    log.info("STAGE 8  PLACEBO              (runs BEFORE the primary)")
    placebo = placebo_split_half(keep, sensor, K, n_resamples=n_resamples,
                                 seed=cfg.seed, data_status=status)
    log.info("         n=%d rho*=%.3f CI=[%.3f, %.3f]  %s",
             placebo.n_participants, placebo.rho_star, placebo.ci_low,
             placebo.ci_high, placebo.verdict)

    result = PipelineResult(
        dataset=dataset, data_status=status, audit=audit, frame=df,
        n_categories=K, sensor=sensor, ledger=ledger, epochs=epochs,
        category_usage=usage, association=assoc, eligibility=elig,
        placebo=placebo, hypergraphs=graphs, blocking_reasons=blocking)

    if not placebo.runnable:
        result.blocking_reasons.append(placebo.verdict)
        result.elapsed_seconds = round(time.time() - t0, 3)
        if halt_on_placebo_failure:
            raise PlaceboUnrunnable(placebo.verdict)
        return result
    if placebo.rejected:
        result.blocking_reasons.append(
            "PLACEBO FAILED. The primary analysis is NOT run. "
            "This is the headline finding, not a bug to work around.")
        result.elapsed_seconds = round(time.time() - t0, 3)
        if halt_on_placebo_failure:
            raise PlaceboFailed(
                "The placebo rejected: the estimator fires where no "
                "recalibration can exist. The primary analysis is NOT run.")
        return result

    # ---- 9/10. PRIMARY + UNCERTAINTY ------------------------------------
    log.info("STAGE 9  primary rho*")
    primary = estimate_rho_star(
        keep, sensor, K, seed=cfg.seed, n_resamples=n_resamples,
        data_status=status,
        context_representation=cfg.get("context.representation", "continuous"),
        eligibility_status=f"{n_ok}/{len(elig)} eligible",
        min_abs_beta=cfg.eligibility_thresholds()["MIN_ABS_BETA"])
    u = primary.uncertainty
    log.info("         rho*=%.3f  95%% CI [%.3f, %.3f]  used %d/%d  %s",
             primary.rho_star, u.ci_low if u else float("nan"),
             u.ci_high if u else float("nan"), primary.n_participants_used,
             primary.n_participants_screened, primary.diagnostic_status)
    log.info("         rho is NOT directly identified; 1 - rho* = %.3f is a "
             "LOWER BOUND on the true recalibration",
             primary.lower_bound_on_recalibration)
    for pid, why in primary.exclusions.items():
        log.info("         NOT USED %-8s %s", pid, why)
    result.primary = primary

    # ---- 11. TWIN UPDATE -------------------------------------------------
    if build_twins:
        log.info("STAGE 11 longitudinal twin update")
        ctx_cols = [c for c in cfg.get("context.features", [sensor])
                    if c in df.columns] or [sensor]
        result.twins = run_longitudinal_update(
            keep, sensor, K, dataset=dataset, data_status=status,
            eligibility={r.pid: r for r in elig}, placebo=placebo,
            ctx_cols=ctx_cols, pids=twin_pids)
        verdicts = [t.state.history[-1]["verdict"] for t in result.twins.values()
                    if t.state.history]
        log.info("         twins updated: %d  ACCEPTED=%d FLAGGED=%d",
                 len(result.twins), verdicts.count("ACCEPTED"),
                 sum(1 for v in verdicts if v != "ACCEPTED"))

    result.elapsed_seconds = round(time.time() - t0, 3)
    log.info("PIPELINE COMPLETE  dataset=%s status=%s validated=%s (%.1fs)",
             dataset, status.value, result.validated, result.elapsed_seconds)
    if not result.validated and status is DataStatus.REAL:
        log.warning("RESULT IS NOT VALIDATED. Blocking reasons: %s",
                    result.blocking_reasons)
    return result
