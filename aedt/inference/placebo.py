"""MODULE 12 -- PLACEBO / NEGATIVE-CONTROL TEST.  ** OUR RESEARCH DESIGN **

Purpose  Detect apparent effects that cannot be recalibration.
Input    The eligible LongFrame, before the primary analysis is run.
Output   PlaceboResult, which GATES the primary.
Algorithm Split each participant's EPOCH 1 into two CONTIGUOUS halves and treat
         them as pseudo-epochs, then run the full estimator on them. No
         response shift can have occurred between two contiguous halves of the
         same epoch, so the estimator must not reject.

WHY CONTIGUOUS. Contiguous halves preserve serial dependence and the ordinal
structure. Naive shuffling would destroy both and would make the control pass
trivially.

WHY IT IS A REAL CONTROL. Validated on synthetic data in three regimes
(ROUND-17 §Q): it rejects 3.3% with no shift present, and -- critically --
5.0% and 6.7% when a genuine 30% and 15% recalibration IS present. It is
therefore not merely detecting temporal structure.

IF THE PLACEBO FAILS the primary result is NOT reported as validated. That is
the finding, not a bug to work around (exit code 5).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import MIN_REPORTS_PER_EPOCH, SEED, DataStatus
from ..schemas import PlaceboResult

log = logging.getLogger(__name__)

__all__ = ["placebo_split_half"]


def placebo_split_half(df: pd.DataFrame, sensor: str, K: int, *,
                       n_resamples: int | None = None, seed: int = SEED,
                       min_reports_per_epoch: int = MIN_REPORTS_PER_EPOCH,
                       data_status: DataStatus = DataStatus.SYNTHETIC
                       ) -> PlaceboResult:
    """Contiguous epoch-1 split-half. Runs BEFORE the primary analysis."""
    from ..estimators.slope_ratio import person_log_ratio
    from .bootstrap import MIN_PARTICIPANTS_FOR_CI, bootstrap_participants

    logs: list[float] = []
    for pid, g in df.groupby("pid", sort=True):
        g0 = g[g["epoch"] == 0].sort_values("ts")
        # a pseudo-epoch must itself clear the screen, so epoch 1 needs twice
        # the minimum before it can be halved
        if len(g0) < 2 * min_reports_per_epoch:
            continue
        h = len(g0) // 2
        a = g0.iloc[:h].copy()
        b = g0.iloc[h:].copy()
        a["epoch"] = 0
        b["epoch"] = 1
        v, _why, _f = person_log_ratio(pd.concat([a, b]), sensor, K,
                                       pid=str(pid), data_status=data_status)
        logs.append(v)

    usable = [v for v in logs if np.isfinite(v)]
    n_screened = int(df["pid"].nunique())
    n_with_enough = len(logs)
    if len(usable) < MIN_PARTICIPANTS_FOR_CI:
        # Be precise about WHICH count fell short: too few participants with
        # enough epoch-1 data, or too few whose split-half fit converged.
        if n_with_enough < MIN_PARTICIPANTS_FOR_CI:
            why = (f"only {n_with_enough} of {n_screened} participants have "
                   f"the {2 * min_reports_per_epoch} epoch-1 observations "
                   "needed to split epoch 1 in half")
        else:
            why = (f"{n_with_enough} participants had enough epoch-1 data but "
                   f"only {len(usable)} produced a convergent split-half fit")
        return PlaceboResult(
            n_participants=len(usable), rho_star=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), rejected=False,
            verdict=(f"NOT RUNNABLE: {why}; at least "
                     f"{MIN_PARTICIPANTS_FOR_CI} are required for a "
                     "participant-cluster interval. The primary analysis is "
                     "blocked."),
            runnable=False, data_status=data_status)

    unc = bootstrap_participants(usable, n_resamples=n_resamples, seed=seed,
                                 data_status=data_status)
    rejected = bool(unc.excludes_null)
    verdict = ("REJECTS -- the estimator fires where no recalibration can "
               "exist. The primary analysis is NOT run and this is the "
               "headline finding." if rejected else
               "does not reject -- the primary analysis may proceed")
    log.info("placebo n=%d rho*=%.3f CI=[%.3f, %.3f] %s",
             unc.n_participants, unc.point, unc.ci_low, unc.ci_high,
             "REJECTS" if rejected else "passes")
    return PlaceboResult(
        n_participants=unc.n_participants, rho_star=unc.point,
        ci_low=unc.ci_low, ci_high=unc.ci_high, rejected=rejected,
        verdict=verdict, runnable=True, data_status=data_status)
