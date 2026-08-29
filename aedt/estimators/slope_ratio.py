"""MODULE 10 -- SLOPE-RATIO ESTIMATOR.  ** OUR RESEARCH CONTRIBUTION **

Purpose  Estimate rho*, the identified attenuated recalibration ratio.
Input    A validated LongFrame with an 'epoch' column and a sensor feature.
Output   EstimatorResult: rho*, uncertainty, diagnostics, exclusions.
Algorithm  Per person, per epoch: standardise the sensor feature WITHIN THAT
         EPOCH, fit P(R<=k|x) = Phi(c_k - beta_e x) by ML, then

             rho*_p = beta_p2 / beta_p1 ,
             log rho* = (1/P) sum_p log rho*_p

         with a nonparametric bootstrap over PARTICIPANTS.

WHY THE RATIO IS CALIBRATED (ROUND-17 §G). Algebra gives

    beta_e = a_e lambda sigma_theta^2 / ( SD(s_e) sqrt(a_e^2 v + sigma_r^2) )

so with lambda, sigma_theta, sigma_p stable across epochs (A3),

    beta_2/beta_1 = rho sqrt( (v + sigma_r^2) / (rho^2 v + sigma_r^2) ) = rho*

  1. Under the null (rho = 1) the ratio is EXACTLY 1 -- no false positives.
  2. Under the alternative it is ATTENUATED TOWARD 1 -- conservative, never
     anti-conservative. Hence 1 - rho* is a LOWER BOUND on the recalibration.
  3. Errors-in-variables attenuation and any epoch-invariant link
     misspecification appear in BOTH slopes and cancel in the ratio.

WHAT IS NOT ESTIMATED. The additive component b_2 - b_1 is absorbed into the
threshold locations and is NOT identified under any specification examined
(Theorem T1(b)). It is never estimated, never reported, and ``EstimatorResult``
refuses to carry a value for it.

TEMPORAL INDEPENDENCE. Each epoch is fitted independently. The standardisers
are epoch-local by design: a constant fitted on epoch 1 must never touch epoch
2, because a shared standardiser would partly absorb the very drift under test.

PORTED from the validated ``final_audit.py::person_log_ratio`` /
``ordinal.py::method_ordinal``. Logic unchanged; packaging and typing added.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import (MIN_ABS_BETA, REQUIRE_MATCHING_SIGN, SEED, DataStatus)
from ..models.ordinal import ordinal_probit_fit
from ..schemas import (EstimatorResult, OrdinalFit, UncertaintyResult,
                       validate_long_frame)

log = logging.getLogger(__name__)

__all__ = ["standardise_within_epoch", "fit_person_epochs", "person_log_ratio",
           "estimate_rho_star"]


def standardise_within_epoch(s: np.ndarray) -> tuple[np.ndarray, float, float]:
    """x = (s - mean(s_e)) / SD(s_e), computed from THIS epoch only.

    Returns (x, mean, sd). Required, not optional (ROUND-17 §W).
    """
    s = np.asarray(s, dtype=float)
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if sd < 1e-12:
        return np.zeros_like(s), mean, sd
    return (s - mean) / sd, mean, sd


def fit_person_epochs(g: pd.DataFrame, sensor: str, K: int,
                      *, pid: str = "", min_abs_beta: float = MIN_ABS_BETA,
                      data_status: DataStatus = DataStatus.SYNTHETIC,
                      min_n: int = 5) -> dict[int, OrdinalFit]:
    """Fit both epochs INDEPENDENTLY for one participant."""
    fits: dict[int, OrdinalFit] = {}
    for e in (0, 1):
        ge = g[g["epoch"] == e].sort_values("ts")
        R = ge["report"].to_numpy(dtype=int)
        s = ge[sensor].to_numpy(dtype=float)
        if len(R) < min_n:
            fits[e] = OrdinalFit(pid=pid, epoch=e, beta=float("nan"),
                                 cutpoints=(), n=len(R), n_categories=K,
                                 converged=False,
                                 reason=f"only {len(R)} observations in epoch {e}",
                                 data_status=data_status)
            continue
        x, mean, sd = standardise_within_epoch(s)
        fits[e] = ordinal_probit_fit(x, R, K, pid=pid, epoch=e,
                                     min_abs_beta=min_abs_beta,
                                     standardiser=(mean, sd),
                                     data_status=data_status)
    return fits


def person_log_ratio(g: pd.DataFrame, sensor: str, K: int, *, pid: str = "",
                     min_abs_beta: float = MIN_ABS_BETA,
                     require_matching_sign: bool = REQUIRE_MATCHING_SIGN,
                     data_status: DataStatus = DataStatus.SYNTHETIC
                     ) -> tuple[float, str, dict[int, OrdinalFit]]:
    """log(beta_2 / beta_1) for one participant, or (nan, reason, fits)."""
    fits = fit_person_epochs(g, sensor, K, pid=pid, min_abs_beta=min_abs_beta,
                             data_status=data_status)
    f1, f2 = fits[0], fits[1]
    if not f1.converged:
        return float("nan"), f"epoch 1: {f1.reason}", fits
    if not f2.converged:
        return float("nan"), f"epoch 2: {f2.reason}", fits
    b1, b2 = f1.beta, f2.beta
    if require_matching_sign and np.sign(b1) != np.sign(b2):
        # A participant whose sensor-report association reverses between epochs
        # has a negative, uninterpretable ratio. Excluded by pre-stated rule.
        return (float("nan"),
                f"sensor-report slope flips sign between epochs "
                f"(b1={b1:+.3f}, b2={b2:+.3f}); the ratio is not interpretable",
                fits)
    return float(np.log(b2 / b1)), "", fits


def estimate_rho_star(df: pd.DataFrame, sensor: str, K: int, *,
                      bootstrap: bool = True, n_resamples: int | None = None,
                      seed: int = SEED,
                      data_status: DataStatus = DataStatus.SYNTHETIC,
                      context_representation: str = "continuous",
                      eligibility_status: str = "SCREENED",
                      min_abs_beta: float = MIN_ABS_BETA,
                      ) -> EstimatorResult:
    """The primary estimator. Returns rho* with participant-cluster inference.

    ``df`` must already have been through the eligibility screen; this function
    does not silently re-screen, but it does record every participant whose fit
    fails, with the reason.
    """
    from ..inference.bootstrap import bootstrap_participants

    validate_long_frame(df, sensor, require_epoch=True, n_categories=K)
    logs: list[float] = []
    pids: list[str] = []
    exclusions: dict[str, str] = {}
    for pid, g in df.groupby("pid", sort=True):
        v, why, _fits = person_log_ratio(g, sensor, K, pid=str(pid),
                                         min_abs_beta=min_abs_beta,
                                         data_status=data_status)
        if np.isfinite(v):
            logs.append(v)
            pids.append(str(pid))
        else:
            exclusions[str(pid)] = why
            log.info("excluded pid=%s reason=%s", pid, why)

    n_screened = int(df["pid"].nunique())
    finite = np.asarray(logs, dtype=float)
    if len(finite) == 0:
        return EstimatorResult(
            estimand="rho_star", rho_star=float("nan"),
            log_rho_star=float("nan"), uncertainty=None,
            n_participants_used=0, n_participants_screened=n_screened,
            exclusions=exclusions, diagnostic_status="NO_USABLE_PARTICIPANTS",
            eligibility_status=eligibility_status,
            context_representation=context_representation,
            data_status=data_status)

    log_point = float(finite.mean())
    unc: UncertaintyResult | None = None
    if bootstrap:
        unc = bootstrap_participants(finite, n_resamples=n_resamples, seed=seed,
                                     data_status=data_status)

    status = "OK"
    if len(finite) < 10:
        status = "TOO_FEW_PARTICIPANTS_FOR_INFERENCE"
    elif len(exclusions) > n_screened / 2:
        status = "MAJORITY_EXCLUDED"

    return EstimatorResult(
        estimand="rho_star",
        rho_star=float(np.exp(log_point)),
        log_rho_star=log_point,
        uncertainty=unc,
        n_participants_used=int(len(finite)),
        n_participants_screened=n_screened,
        per_participant_rho_star=tuple(float(v) for v in np.exp(finite)),
        per_participant_pids=tuple(pids),
        exclusions=exclusions,
        median_rho_star=float(np.exp(np.median(finite))),
        diagnostic_status=status,
        eligibility_status=eligibility_status,
        context_representation=context_representation,
        data_status=data_status)
