"""MODULE 15 -- LONGITUDINAL TWIN UPDATE.

The update pipeline, in this exact order (project mandate):

    observe
      -> validate
      -> update features / context
      -> update continual knowledge
      -> update hypergraph / context state
      -> update ordinal model state where appropriate
      -> update uncertainty
      -> save provenance

NO FUTURE LEAKAGE. ``observe`` refuses an observation dated before the twin's
current time, and ``close_epoch`` fits each epoch independently.

ON EPOCH CLOSE (ROUND-17 §K): re-fit the ordinal model; run the eligibility
screen and the placebo; if BOTH pass, append {rho*, CI, flags, ACCEPTED}; if
either fails, append the diagnostics with FLAGGED-UNTRUSTWORTHY and LEAVE THE
CALIBRATION STATE UNCHANGED.

WHERE THE TWIN REASONS OVER THE HYPERGRAPH. ``close_epoch`` reads the
hyperedge occupancy overlap between the two epochs. Low overlap means the
epochs describe different situations, so a slope change may be a change in
circumstances rather than in reporting -- the scale-change vs relation-change
confound. That raises an audit flag which can move the verdict to
FLAGGED-UNTRUSTWORTHY. This is a TRUST decision over higher-order context, not
an identification mechanism.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..constants import SEED, DataStatus
from ..errors import DecisionRequired
from ..schemas import EligibilityResult, PlaceboResult, Serialisable
from .state import PersonalDigitalTwin, new_twin

log = logging.getLogger(__name__)

__all__ = ["TwinVerdict", "TwinUpdateOutcome", "observe", "close_epoch",
           "run_longitudinal_update", "MIN_HYPEREDGE_OVERLAP"]

# Below this epoch-to-epoch hyperedge overlap the two epochs are not describing
# comparable situations. Pre-specified, like every other threshold.
MIN_HYPEREDGE_OVERLAP = 0.20


class TwinVerdict(str, Enum):
    ACCEPTED = "ACCEPTED"
    FLAGGED_UNTRUSTWORTHY = "FLAGGED-UNTRUSTWORTHY"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class TwinUpdateOutcome(Serialisable):
    """What happened when one epoch closed."""

    pid: str
    verdict: TwinVerdict
    rho_star: float
    ci_low: float
    ci_high: float
    reasons: tuple[str, ...]
    audit_flags: dict
    calibration_state_changed: bool
    data_status: DataStatus


# ---------------------------------------------------------------- observe
def observe(twin: PersonalDigitalTwin, row: pd.Series, *,
            sensor: str, n_categories: int) -> PersonalDigitalTwin:
    """Ingest ONE observation. Steps 1-4 of the update pipeline."""
    ts = pd.Timestamp(row["ts"])

    # ---- validate -------------------------------------------------------
    rep = row.get("report")
    if rep is None or not np.isfinite(float(rep)):
        raise DecisionRequired(
            f"Twin {twin.pid}: observation at {ts} has no self-report. The "
            "outcome is never imputed.")
    rep = int(rep)
    if not (1 <= rep <= n_categories):
        raise DecisionRequired(
            f"Twin {twin.pid}: self-report {rep} outside 1..{n_categories} at "
            f"{ts}. The label remap is wrong for this dataset.")
    twin.advance_time(ts)                     # refuses the past => no leakage

    # ---- update features / context -------------------------------------
    value = float(row[sensor]) if sensor in row and pd.notna(row[sensor]) else None
    twin._set(
        n_observations_seen=twin.state.n_observations_seen + 1,
        feature_history=[*twin.state.feature_history,
                         {"ts": ts.isoformat(), "report": rep, sensor: value,
                          "epoch": int(row["epoch"]) if "epoch" in row else None}])

    usage = dict(twin.state.category_usage)
    ekey = str(int(row["epoch"])) if "epoch" in row and pd.notna(row["epoch"]) else "na"
    per_epoch = dict(usage.get(ekey, {}))
    per_epoch[str(rep)] = per_epoch.get(str(rep), 0) + 1
    usage[ekey] = per_epoch
    twin._set(category_usage=usage,
              context_state={"last_sensor_value": value, "last_ts": ts.isoformat()})

    # ---- update continual knowledge (2: state history) ------------------
    twin.knowledge.append(
        "state_history",
        {"ts": ts.isoformat(), "report": rep, "sensor": value,
         "epoch": ekey},
        valid_from=ts, provenance=f"observe:{twin.state.dataset}")
    return twin


# ------------------------------------------------------------ close_epoch
def close_epoch(twin: PersonalDigitalTwin, g: pd.DataFrame, *, sensor: str,
                n_categories: int, eligibility: EligibilityResult,
                placebo: PlaceboResult | None = None,
                ctx_cols: list[str] | None = None, seed: int = SEED,
                n_resamples: int | None = None) -> TwinUpdateOutcome:
    """Re-fit, audit, and append one recalibration record. Steps 5-8."""
    from ..audit.diagnostics import acf
    from ..estimators.slope_ratio import fit_person_epochs
    from ..hypergraph.structure import build_hypergraph

    reasons: list[str] = []
    status = twin.state.data_status

    # ---- update hypergraph / context state ------------------------------
    hg = build_hypergraph(g, ctx_cols or [sensor])
    overlap = hg.occupancy_overlap()
    twin._set(hyperedge_occupancy=hg.summary())
    twin.knowledge.append(
        "context_relationship", hg.summary(),
        valid_from=pd.Timestamp(g["ts"].max()),
        provenance="close_epoch:hypergraph")

    # ---- update ordinal model state -------------------------------------
    fits = fit_person_epochs(g, sensor, n_categories, pid=twin.pid,
                             data_status=status)
    ordinal_state = {str(e): f.to_dict() for e, f in fits.items()}
    twin._set(ordinal_state=ordinal_state)
    twin.knowledge.append(
        "personalised_parameters",
        {"epoch0": ordinal_state.get("0"), "epoch1": ordinal_state.get("1")},
        valid_from=pd.Timestamp(g["ts"].max()),
        provenance="close_epoch:ordinal_fit")

    # ---- audit flags -----------------------------------------------------
    R = g["report"].to_numpy(dtype=int)
    s0 = g[g["epoch"] == 0][sensor].to_numpy(dtype=float)
    s1 = g[g["epoch"] == 1][sensor].to_numpy(dtype=float)
    flags = {
        "boundary_rate": float(np.mean((R == 1) | (R == n_categories))),
        "lag1_autocorr_epoch0": acf(s0, 1),
        "var_ratio": (float(np.var(s1, ddof=1) / np.var(s0, ddof=1))
                      if len(s0) > 2 and len(s1) > 2 and np.var(s0, ddof=1) > 0
                      else float("nan")),
        "hyperedge_occupancy_overlap": overlap,
        "n_hyperedges": hg.n_edges,
    }
    twin._set(audit_flags=flags, eligibility_status=(
        "ELIGIBLE" if eligibility.eligible else "EXCLUDED"))

    # ---- verdict ---------------------------------------------------------
    if not eligibility.eligible:
        reasons.extend(eligibility.reasons)
    if placebo is not None and placebo.gates_primary:
        reasons.append(f"placebo gate: {placebo.verdict}")
    if np.isfinite(overlap) and overlap < MIN_HYPEREDGE_OVERLAP:
        reasons.append(
            f"hyperedge occupancy overlap {overlap:.2f} < "
            f"{MIN_HYPEREDGE_OVERLAP}: the two epochs describe different "
            "contexts, so a slope change may be a change in circumstances "
            "rather than in reporting")

    f0, f1 = fits.get(0), fits.get(1)
    rho = ci_lo = ci_hi = float("nan")
    if f0 is not None and f1 is not None and f0.converged and f1.converged \
            and np.sign(f0.beta) == np.sign(f1.beta):
        rho = float(f1.beta / f0.beta)
    elif not reasons:
        reasons.append("ordinal model did not produce an interpretable ratio")

    verdict = (TwinVerdict.ACCEPTED if not reasons and np.isfinite(rho)
               else TwinVerdict.FLAGGED_UNTRUSTWORTHY)

    # ---- update uncertainty ---------------------------------------------
    if verdict is TwinVerdict.ACCEPTED:
        # a single participant has no cluster to bootstrap over; the
        # participant-level interval belongs to the cohort estimator, and the
        # twin records the cohort interval when one is supplied
        twin._set(uncertainty_state={
            "note": ("per-person point estimate; the interval is a "
                     "participant-cluster bootstrap over the COHORT, computed "
                     "in aedt.inference.bootstrap"),
            "rho_star": rho})

    entry = {
        "closed_at_twin_time": twin.state.current_time,
        "epoch_span": {"start": str(g["ts"].min()), "end": str(g["ts"].max())},
        "rho_star": rho, "ci_low": ci_lo, "ci_high": ci_hi,
        "verdict": verdict.value, "reasons": reasons, "flags": flags,
        "data_status": status.value,
    }
    twin.append_history(entry)
    twin.knowledge.append("uncertainty_audit", entry,
                          valid_from=pd.Timestamp(g["ts"].max()),
                          provenance="close_epoch:verdict")
    twin.log_update("close_epoch", {"verdict": verdict.value,
                                    "n_reasons": len(reasons)})
    if verdict is TwinVerdict.FLAGGED_UNTRUSTWORTHY:
        log.info("twin %s FLAGGED-UNTRUSTWORTHY: %s", twin.pid,
                 "; ".join(reasons))
    return TwinUpdateOutcome(
        pid=twin.pid, verdict=verdict, rho_star=rho, ci_low=ci_lo,
        ci_high=ci_hi, reasons=tuple(reasons), audit_flags=flags,
        calibration_state_changed=verdict is TwinVerdict.ACCEPTED,
        data_status=status)


# --------------------------------------------------- full sequential update
def run_longitudinal_update(df: pd.DataFrame, sensor: str, n_categories: int, *,
                            dataset: str, data_status: DataStatus,
                            eligibility: dict[str, EligibilityResult],
                            placebo: PlaceboResult | None = None,
                            ctx_cols: list[str] | None = None,
                            pids: list[str] | None = None,
                            ) -> dict[str, PersonalDigitalTwin]:
    """Feed every observation to its twin IN TIME ORDER, then close the epoch."""
    twins: dict[str, PersonalDigitalTwin] = {}
    wanted = set(pids) if pids else None
    for pid, g in df.groupby("pid", sort=True):
        pid = str(pid)
        if wanted is not None and pid not in wanted:
            continue
        g = g.sort_values("ts")
        twin = new_twin(pid, dataset, data_status)
        for _i, row in g.iterrows():
            observe(twin, row, sensor=sensor, n_categories=n_categories)
        elig = eligibility.get(pid)
        if elig is None:
            raise DecisionRequired(
                f"No eligibility result for participant {pid}; the screen must "
                "run before the twin update.")
        close_epoch(twin, g, sensor=sensor, n_categories=n_categories,
                    eligibility=elig, placebo=placebo, ctx_cols=ctx_cols)
        twins[pid] = twin
    return twins
