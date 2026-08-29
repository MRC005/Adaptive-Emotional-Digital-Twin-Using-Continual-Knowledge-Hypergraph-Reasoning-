"""MODULE 13 -- BIAS ENVELOPE.  ** OUR RESEARCH VALIDATION FRAMEWORK **

Purpose  Bound how far rho* can move under the ALLOWED assumption violations,
         when the truth is the NULL.
Input    The MEASURED properties of the analysed data -- floor rate, lag-1
         autocorrelation, Var(s) ratio, observation density, association
         strength -- plus the enumerated assumption scenarios.
Output   BiasEnvelopeResult: the range of rho* under those assumptions.
Algorithm Re-run the generator AT THE MEASURED VALUES under the TRUE NULL
         (rho = 1), across the pre-enumerated scenarios, and report the
         5th-95th percentile of the resulting rho*.

INTERPRETATION. A primary estimate that falls INSIDE the null envelope cannot
be distinguished from an artefact of the assumption violations that the data
itself exhibits. Only an estimate outside the envelope is evidence.

DISCIPLINE. The scenarios are enumerated in ``simulate/scenarios.py`` in
advance and are NOT chosen to optimise the result. Adding a scenario after
seeing the primary estimate would be a specification change and must be
recorded as such in ``docs/decision_required.md``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import SEED, DataStatus
from ..schemas import BiasEnvelopeResult

log = logging.getLogger(__name__)

__all__ = ["bias_envelope", "ENVELOPE_SCENARIOS", "measured_properties"]

# The assumption violations the envelope is allowed to explore. Frozen.
ENVELOPE_SCENARIOS = ("balanced", "skewed", "extreme_floor", "skewed_ar1",
                      "skewed_saturating", "extreme_floor_ar1",
                      "extreme_floor_saturating", "noisy_sensor",
                      "noisy_report")


def measured_properties(df: pd.DataFrame, sensor: str, K: int) -> dict:
    """The data properties the envelope is calibrated to."""
    from .diagnostics import acf
    floors, ars, ns, vrs = [], [], [], []
    for _pid, g in df.groupby("pid", sort=True):
        for e in (0, 1):
            ge = g[g["epoch"] == e].sort_values("ts")
            R = ge["report"].to_numpy(dtype=int)
            s = ge[sensor].to_numpy(dtype=float)
            if len(R) == 0:
                continue
            ns.append(len(R))
            floors.append(float(np.mean(R == 1)))
            if e == 0:
                ars.append(acf(s, 1))
        g0 = g[g["epoch"] == 0][sensor].to_numpy(float)
        g1 = g[g["epoch"] == 1][sensor].to_numpy(float)
        if len(g0) > 2 and len(g1) > 2 and np.var(g0, ddof=1) > 0:
            vrs.append(float(np.var(g1, ddof=1) / np.var(g0, ddof=1)))
    return {
        "median_floor_rate": float(np.nanmedian(floors)) if floors else float("nan"),
        "median_lag1_autocorr": float(np.nanmedian(ars)) if ars else float("nan"),
        "median_obs_per_epoch": float(np.median(ns)) if ns else float("nan"),
        "median_var_ratio": float(np.nanmedian(vrs)) if vrs else float("nan"),
        "n_participants": int(df["pid"].nunique()),
    }


def bias_envelope(*, n_participants: int = 48, n_per_epoch: int = 200,
                  scenarios: tuple[str, ...] = ENVELOPE_SCENARIOS,
                  n_replications: int = 20, seed: int = SEED,
                  measured: dict | None = None,
                  data_status: DataStatus = DataStatus.SYNTHETIC
                  ) -> BiasEnvelopeResult:
    """Range of rho* under the enumerated assumptions, with rho = 1 (the NULL).

    ``measured`` is the output of ``measured_properties`` and is recorded in
    the interpretation string so the envelope's calibration is auditable. It
    does not silently change the scenario definitions.
    """
    from ..simulate.scenarios import SCENARIOS, run_scenario

    by_scenario: dict[str, float] = {}
    all_vals: list[float] = []
    for i, name in enumerate(scenarios):
        if name not in SCENARIOS:
            raise KeyError(f"unknown envelope scenario {name!r}")
        vals = []
        for r in range(n_replications):
            # No bootstrap per replication: the envelope IS the spread of
            # point estimates under the null, so a per-replication interval
            # would be wasted work.
            out = run_scenario(name, 1.00, n_participants=n_participants,
                               n_per_epoch=n_per_epoch,
                               seed=seed + 1013 * i + 17 * r,
                               bootstrap=False, n_resamples=None)
            if np.isfinite(out["rho_star"]):
                vals.append(out["rho_star"])
        if vals:
            by_scenario[name] = float(np.median(vals))
            all_vals.extend(vals)
        else:
            by_scenario[name] = float("nan")
    if not all_vals:
        raise RuntimeError("Bias envelope produced no usable replications.")
    lo, hi = np.percentile(all_vals, [5, 95])
    meas = "" if not measured else (
        " Calibrated against measured properties: "
        + ", ".join(f"{k}={v:.3g}" for k, v in measured.items()) + ".")
    return BiasEnvelopeResult(
        scenarios=tuple(scenarios), rho_star_by_scenario=by_scenario,
        envelope_low=float(lo), envelope_high=float(hi),
        interpretation=(
            f"Under the TRUE NULL (rho = 1), rho* ranges over "
            f"[{lo:.3f}, {hi:.3f}] across {len(scenarios)} pre-enumerated "
            f"assumption violations at P={n_participants}, n={n_per_epoch} per "
            "epoch. A primary estimate inside this band cannot be distinguished "
            "from an artefact of those violations." + meas),
        n_replications=n_replications, data_status=data_status)
