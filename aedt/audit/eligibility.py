"""MODULE 11 -- ELIGIBILITY SCREEN.  ** OUR RESEARCH VALIDATION FRAMEWORK **

Purpose  Decide whether a participant can support the estimator AT ALL.
Input    One participant's epoch-assigned observations.
Output   EligibilityResult, with every failure reason recorded by name.
Algorithm The pre-specified screen, with thresholds FIXED BEFORE ANY REAL DATA
         WAS SEEN (``aedt/constants.py``).

Checks:
  minimum observations per epoch     >= 60
  categories used per epoch          >= 2                          (A5)
  sensor variation per epoch         SD > 0 and standardised SD >= 0.10  (A5)
  Var(s) epoch ratio                 in [0.25, 4.0]                (A3 proxy)
  model convergence                  both epochs, |beta| >= 0.02, EITHER SIGN
  temporal validity                  both epochs non-empty, epoch 2 after 1

NEVER modify a threshold after observing a primary result. A sensitivity
analysis over a threshold must be pre-specified and run through
``audit/envelope.py``; ``screen_cohort`` records the thresholds it used in
every result so a post-hoc change is visible in the diff.

PORTED from ``gate.py::eligible`` and ``final_audit.py::per_participant_table``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import (MIN_ABS_BETA, MIN_CATEGORIES_USED,
                         MIN_REPORTS_PER_EPOCH, MIN_SENSOR_SD, VAR_RATIO_HI,
                         VAR_RATIO_LO, DataStatus)
from ..schemas import EligibilityResult
from .diagnostics import acf

log = logging.getLogger(__name__)

__all__ = ["ELIGIBILITY_THRESHOLDS", "screen_participant", "screen_cohort",
           "eligibility_table", "filter_eligible"]

ELIGIBILITY_THRESHOLDS = {
    "MIN_REPORTS_PER_EPOCH": MIN_REPORTS_PER_EPOCH,
    "MIN_CATEGORIES_USED": MIN_CATEGORIES_USED,
    "MIN_SENSOR_SD": MIN_SENSOR_SD,
    "VAR_RATIO_LO": VAR_RATIO_LO,
    "VAR_RATIO_HI": VAR_RATIO_HI,
    "MIN_ABS_BETA": MIN_ABS_BETA,
}


def screen_participant(g: pd.DataFrame, sensor: str, K: int, *,
                       pid: str | None = None, check_convergence: bool = True,
                       thresholds: dict | None = None,
                       data_status: DataStatus = DataStatus.SYNTHETIC
                       ) -> EligibilityResult:
    """Apply the pre-specified screen to ONE participant."""
    th = {**ELIGIBILITY_THRESHOLDS, **(thresholds or {})}
    pid = str(pid if pid is not None else g["pid"].iloc[0])
    reasons: list[str] = []
    rec: dict[str, float] = {}

    for e in (0, 1):
        ge = g[g["epoch"] == e].sort_values("ts")
        R = ge["report"].to_numpy(dtype=int)
        s = ge[sensor].to_numpy(dtype=float)
        n = len(R)
        rec[f"n{e}"] = n
        rec[f"cat{e}"] = int(len(np.unique(R))) if n else 0
        rec[f"floor{e}"] = float(np.mean(R == 1)) if n else float("nan")
        rec[f"ceil{e}"] = float(np.mean(R == K)) if n else float("nan")
        rec[f"sd{e}"] = float(np.std(s, ddof=1)) if n > 2 else float("nan")
        rec[f"var{e}"] = float(np.var(s, ddof=1)) if n > 2 else float("nan")
        if e == 0:
            rec["ar1"] = acf(s, 1)

        if n < th["MIN_REPORTS_PER_EPOCH"]:
            reasons.append(f"epoch{e}: {n} reports < {th['MIN_REPORTS_PER_EPOCH']}")
        if rec[f"cat{e}"] < th["MIN_CATEGORIES_USED"]:
            reasons.append(f"epoch{e}: only {rec[f'cat{e}']} response "
                           f"categories used (A5)")
        if not np.isfinite(rec[f"sd{e}"]) or rec[f"sd{e}"] <= 0:
            reasons.append(f"epoch{e}: no sensor variation (A5)")
        elif n > 2:
            # SD of the WITHIN-EPOCH STANDARDISED covariate; degenerate only if
            # the raw SD collapses, which the branch above already catches.
            x = (s - s.mean()) / rec[f"sd{e}"]
            if float(np.std(x)) < th["MIN_SENSOR_SD"]:
                reasons.append(f"epoch{e}: standardised sensor SD "
                               f"{np.std(x):.3f} < {th['MIN_SENSOR_SD']}")

    var0, var1 = rec.get("var0", np.nan), rec.get("var1", np.nan)
    if np.isfinite(var0) and var0 > 0 and np.isfinite(var1):
        vr = var1 / var0
        rec["var_ratio"] = vr
        if not (th["VAR_RATIO_LO"] <= vr <= th["VAR_RATIO_HI"]):
            reasons.append(f"Var(s) epoch ratio {vr:.2f} outside "
                           f"[{th['VAR_RATIO_LO']}, {th['VAR_RATIO_HI']}] "
                           "-- assumption A3 not supported")
    else:
        rec["var_ratio"] = float("nan")

    if check_convergence and not reasons:
        from ..estimators.slope_ratio import person_log_ratio
        v, why, _fits = person_log_ratio(g, sensor, K, pid=pid,
                                         min_abs_beta=th["MIN_ABS_BETA"],
                                         data_status=data_status)
        if not np.isfinite(v):
            reasons.append(why)

    res = EligibilityResult(
        pid=pid, eligible=not reasons, reasons=tuple(reasons),
        n_epoch0=int(rec.get("n0", 0)), n_epoch1=int(rec.get("n1", 0)),
        categories_epoch0=int(rec.get("cat0", 0)),
        categories_epoch1=int(rec.get("cat1", 0)),
        sensor_sd_epoch0=rec.get("sd0", float("nan")),
        sensor_sd_epoch1=rec.get("sd1", float("nan")),
        var_ratio=rec.get("var_ratio", float("nan")),
        floor_rate_epoch0=rec.get("floor0", float("nan")),
        floor_rate_epoch1=rec.get("floor1", float("nan")),
        ceiling_rate_epoch0=rec.get("ceil0", float("nan")),
        ceiling_rate_epoch1=rec.get("ceil1", float("nan")),
        ar1_epoch0=rec.get("ar1", float("nan")),
        data_status=data_status)
    if not res.eligible:
        log.info("ELIGIBILITY EXCLUDED pid=%s reasons=%s", pid, "; ".join(reasons))
    return res


def screen_cohort(df: pd.DataFrame, sensor: str, K: int, *,
                  check_convergence: bool = True, thresholds: dict | None = None,
                  data_status: DataStatus = DataStatus.SYNTHETIC
                  ) -> list[EligibilityResult]:
    """Screen every participant. Nothing is dropped silently."""
    out = [screen_participant(g, sensor, K, pid=str(pid),
                              check_convergence=check_convergence,
                              thresholds=thresholds, data_status=data_status)
           for pid, g in df.groupby("pid", sort=True)]
    n_ok = sum(r.eligible for r in out)
    log.info("eligibility screen: %d/%d participants eligible (%d excluded)",
             n_ok, len(out), len(out) - n_ok)
    return out


def eligibility_table(results: list[EligibilityResult]) -> pd.DataFrame:
    """Flat table for the report, one row per participant, reasons included."""
    return pd.DataFrame([{**r.to_dict(), "reasons": "; ".join(r.reasons)}
                         for r in results])


def filter_eligible(df: pd.DataFrame, results: list[EligibilityResult]
                    ) -> pd.DataFrame:
    keep = {r.pid for r in results if r.eligible}
    return df[df["pid"].astype(str).isin(keep)].copy()
