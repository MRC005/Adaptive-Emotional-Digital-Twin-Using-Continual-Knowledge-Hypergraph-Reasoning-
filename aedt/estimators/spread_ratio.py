"""THE HYPERGRAPH-NATIVE ESTIMATOR FORM (ROUND-17 §M).

With a CATEGORICAL context (a hyperedge) instead of a continuous covariate,
the "slope" becomes the SPREAD OF CONTEXT EFFECTS across hyperedges, and rho*
is the ratio of the epoch-2 to epoch-1 spread. Same estimand, different
representation.

This exists so that Ablation 1 -- continuous covariate vs feature-vector bins
vs n-ary hyperedge -- is a genuine comparison of representations of the SAME
estimand, rather than a comparison of two different questions.

IMPORTANT. This is NOT the identification mechanism. The frozen primary method
(``estimators/slope_ratio.py``) uses a continuous covariate and every
observation. If the continuous covariate wins the ablation, that is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import (MIN_CATEGORIES_USED, MIN_REPORTS_PER_EPOCH, SEED,
                         DataStatus)
from ..schemas import EstimatorResult

__all__ = ["hyperedge_spread_ratio", "person_spread_log_ratio"]

MIN_EDGES_PER_EPOCH = 3
MIN_OBS_PER_EDGE = 4


def _epoch_spread(g: pd.DataFrame, edge_col: str) -> float:
    """SD across hyperedges of the mean standardised report within each edge.

    The report is standardised WITHIN THE EPOCH, mirroring the within-epoch
    standardisation of the continuous covariate, so the two representations are
    on comparable footing.
    """
    R = g["report"].to_numpy(dtype=float)
    if len(R) < MIN_REPORTS_PER_EPOCH or np.std(R) < 1e-12:
        return float("nan")
    z = (R - R.mean()) / R.std(ddof=1)
    tmp = pd.DataFrame({"edge": g[edge_col].to_numpy(), "z": z})
    counts = tmp.groupby("edge")["z"].count()
    keep = counts[counts >= MIN_OBS_PER_EDGE].index
    if len(keep) < MIN_EDGES_PER_EPOCH:
        return float("nan")
    means = tmp[tmp["edge"].isin(keep)].groupby("edge")["z"].mean().to_numpy()
    return float(np.std(means, ddof=1))


def person_spread_log_ratio(g: pd.DataFrame, edge_col: str
                            ) -> tuple[float, str]:
    """log(spread_2 / spread_1) for one participant."""
    g0 = g[g["epoch"] == 0]
    g1 = g[g["epoch"] == 1]
    if len(np.unique(g["report"])) < MIN_CATEGORIES_USED:
        return float("nan"), "fewer than 2 response categories used"
    s0 = _epoch_spread(g0, edge_col)
    s1 = _epoch_spread(g1, edge_col)
    if not np.isfinite(s0):
        return float("nan"), ("epoch 1: fewer than "
                              f"{MIN_EDGES_PER_EPOCH} occupied hyperedges "
                              f"with >= {MIN_OBS_PER_EDGE} observations")
    if not np.isfinite(s1):
        return float("nan"), ("epoch 2: fewer than "
                              f"{MIN_EDGES_PER_EPOCH} occupied hyperedges "
                              f"with >= {MIN_OBS_PER_EDGE} observations")
    if s0 < 1e-9 or s1 < 1e-9:
        return float("nan"), "zero context-effect spread in one epoch"
    return float(np.log(s1 / s0)), ""


def hyperedge_spread_ratio(df: pd.DataFrame, edge_col: str, *,
                           seed: int = SEED, n_resamples: int | None = None,
                           representation: str = "nary_hyperedge",
                           data_status: DataStatus = DataStatus.SYNTHETIC
                           ) -> EstimatorResult:
    """rho* as the ratio of context-effect spreads across epochs."""
    from ..inference.bootstrap import bootstrap_participants

    logs, pids, excl = [], [], {}
    for pid, g in df.groupby("pid", sort=True):
        v, why = person_spread_log_ratio(g, edge_col)
        if np.isfinite(v):
            logs.append(v)
            pids.append(str(pid))
        else:
            excl[str(pid)] = why
    finite = np.asarray(logs, dtype=float)
    n_screened = int(df["pid"].nunique())
    if len(finite) == 0:
        return EstimatorResult(
            estimand="rho_star", rho_star=float("nan"),
            log_rho_star=float("nan"), uncertainty=None,
            n_participants_used=0, n_participants_screened=n_screened,
            exclusions=excl, diagnostic_status="NO_USABLE_PARTICIPANTS",
            context_representation=representation, data_status=data_status)
    unc = bootstrap_participants(finite, n_resamples=n_resamples, seed=seed,
                                 data_status=data_status)
    return EstimatorResult(
        estimand="rho_star", rho_star=float(np.exp(finite.mean())),
        log_rho_star=float(finite.mean()), uncertainty=unc,
        n_participants_used=int(len(finite)),
        n_participants_screened=n_screened,
        per_participant_rho_star=tuple(float(v) for v in np.exp(finite)),
        per_participant_pids=tuple(pids), exclusions=excl,
        median_rho_star=float(np.exp(np.median(finite))),
        diagnostic_status="OK" if len(finite) >= 10 else
        "TOO_FEW_PARTICIPANTS_FOR_INFERENCE",
        context_representation=representation, data_status=data_status)
