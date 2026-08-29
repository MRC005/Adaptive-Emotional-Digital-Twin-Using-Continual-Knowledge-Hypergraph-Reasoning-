"""BASELINES -- only the ones the frozen documents specify (ROUND-14 §14).

No weak baseline is invented to make the method look good. Each one represents
something a competent researcher would actually do:

  raw_epoch_means     What the field does now: compare mean self-report between
                      epochs and call the difference change.
  epoch_normalised    The common ad-hoc fix: z-score the report within epoch,
                      which DESTROYS the very signal we are estimating -- it is
                      included to show that.
  koren_drift         Best prior art on within-person rating drift (Koren 2009):
                      a parametric within-person temporal drift term, no sensor.
  sensor_only         Ignore the self-report entirely; report the epoch change
                      in the sensor. Answers a different question, deliberately.
  ordinal_no_ratio    The ordinal model WITHOUT the slope-ratio contribution:
                      a pooled epoch-interaction term. This is the sharpest
                      baseline because it isolates OUR contribution.
  linear_anchor       Our own previous method, retained as the documented
                      FAILURE (asserted null bias -0.107 on 5-point scales).

JUDGED ON: placebo calibration, CI width, stability and interpretability --
NOT on effect magnitude (ROUND-16 §13).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import SEED, DataStatus
from ..models.ordinal import ordinal_probit_fit
from ..schemas import Serialisable

log = logging.getLogger(__name__)

__all__ = ["BaselineResult", "BASELINES", "run_baselines"]


@dataclass(frozen=True)
class BaselineResult(Serialisable):
    name: str
    represents: str
    statistic_name: str
    statistic: float
    ci_low: float
    ci_high: float
    n_participants: int
    estimates_rho_star: bool
    interpretation: str
    data_status: DataStatus = DataStatus.SYNTHETIC


def _boot(vals, seed, n_resamples):
    from ..inference.bootstrap import bootstrap_participants
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 10:
        return float(np.mean(v)) if len(v) else float("nan"), \
            float("nan"), float("nan"), len(v)
    u = bootstrap_participants(v, seed=seed, n_resamples=n_resamples)
    # bootstrap_participants works on the log scale; take logs back out
    return (float(v.mean()), float(np.log(u.ci_low)), float(np.log(u.ci_high)),
            len(v))


def raw_epoch_means(df, sensor, K, *, seed=SEED, n_resamples=None):
    vals = []
    for _pid, g in df.groupby("pid", sort=True):
        r0 = g.loc[g["epoch"] == 0, "report"]
        r1 = g.loc[g["epoch"] == 1, "report"]
        if len(r0) and len(r1):
            vals.append(float(r1.mean() - r0.mean()))
    stat, lo, hi, n = _boot(vals, seed, n_resamples)
    return BaselineResult(
        "raw_epoch_means", "What the field does now",
        "mean epoch-2 minus epoch-1 self-report", stat, lo, hi, n, False,
        "A difference in mean self-report. Cannot distinguish real change from "
        "recalibration -- which is the entire problem this project addresses.")


def epoch_normalised(df, sensor, K, *, seed=SEED, n_resamples=None):
    vals = []
    for _pid, g in df.groupby("pid", sort=True):
        parts = []
        for e in (0, 1):
            r = g.loc[g["epoch"] == e, "report"].to_numpy(float)
            if len(r) < 3 or r.std(ddof=1) < 1e-9:
                parts = []
                break
            parts.append((r - r.mean()) / r.std(ddof=1))
        if parts:
            vals.append(float(parts[1].mean() - parts[0].mean()))
    stat, lo, hi, n = _boot(vals, seed, n_resamples)
    return BaselineResult(
        "epoch_normalised", "The common ad-hoc fix",
        "mean difference after within-epoch z-scoring", stat, lo, hi, n, False,
        "Identically zero by construction. Within-epoch normalisation REMOVES "
        "exactly the between-epoch information the estimand is defined on; "
        "included to demonstrate that the popular fix answers nothing.")


def koren_drift(df, sensor, K, *, seed=SEED, n_resamples=None):
    """Koren-style within-person temporal drift, dev_u(t) = sign(t-t_u)|t-t_u|^b."""
    vals = []
    b = 0.4
    for _pid, g in df.groupby("pid", sort=True):
        g = g.sort_values("ts")
        t = (g["ts"] - g["ts"].min()).dt.total_seconds().to_numpy() / 86400.0
        if len(t) < 10 or t.max() <= 0:
            continue
        tm = t.mean()
        dev = np.sign(t - tm) * np.abs(t - tm) ** b
        y = g["report"].to_numpy(float)
        if np.std(dev) < 1e-9 or np.std(y) < 1e-9:
            continue
        vals.append(float(np.polyfit(dev, y, 1)[0]))
    stat, lo, hi, n = _boot(vals, seed, n_resamples)
    return BaselineResult(
        "koren_drift", "Best prior art on within-person rating drift (Koren 2009)",
        "per-person drift coefficient on dev_u(t)", stat, lo, hi, n, False,
        "Models drift as a parametric function of TIME with no external "
        "reference, so it cannot separate drift in the person from drift in "
        "the scale. This is precisely the gap the sensor reference fills.")


def sensor_only(df, sensor, K, *, seed=SEED, n_resamples=None):
    vals = []
    for _pid, g in df.groupby("pid", sort=True):
        s0 = g.loc[g["epoch"] == 0, sensor].to_numpy(float)
        s1 = g.loc[g["epoch"] == 1, sensor].to_numpy(float)
        if len(s0) < 3 or len(s1) < 3 or s0.std(ddof=1) < 1e-9:
            continue
        vals.append(float((s1.mean() - s0.mean()) / s0.std(ddof=1)))
    stat, lo, hi, n = _boot(vals, seed, n_resamples)
    return BaselineResult(
        "sensor_only", "Ignore the self-report entirely",
        "standardised epoch change in the sensor", stat, lo, hi, n, False,
        "Answers a different question: whether BEHAVIOUR changed, not whether "
        "REPORTING changed. Reported so the two are never conflated.")


def ordinal_no_ratio(df, sensor, K, *, seed=SEED, n_resamples=None):
    """The ordinal model WITHOUT the slope-ratio contribution.

    Fits an epoch-interaction slope on the POOLED, person-demeaned covariate:
    the standard thing to do with an ordinal outcome and two epochs. It shares
    the model class with our method and differs only in the ratio construction,
    so the comparison isolates the contribution rather than the machinery.
    """
    from ..estimators.slope_ratio import standardise_within_epoch
    rows = []
    for pid, g in df.groupby("pid", sort=True):
        for e in (0, 1):
            ge = g[g["epoch"] == e].sort_values("ts")
            if len(ge) < 5:
                continue
            x, _m, _s = standardise_within_epoch(ge[sensor].to_numpy(float))
            rows.append(pd.DataFrame({"pid": str(pid), "x": x, "epoch": e,
                                      "report": ge["report"].to_numpy(int)}))
    if not rows:
        return BaselineResult(
            "ordinal_no_ratio", "The ordinal model without our contribution",
            "pooled epoch-interaction slope", float("nan"), float("nan"),
            float("nan"), 0, False, "insufficient data")
    d = pd.concat(rows, ignore_index=True)
    fit_all = ordinal_probit_fit(d["x"].to_numpy(), d["report"].to_numpy(int), K)
    d1 = d[d["epoch"] == 1]
    d0 = d[d["epoch"] == 0]
    f0 = ordinal_probit_fit(d0["x"].to_numpy(), d0["report"].to_numpy(int), K)
    f1 = ordinal_probit_fit(d1["x"].to_numpy(), d1["report"].to_numpy(int), K)
    inter = (f1.beta - f0.beta) if (f0.converged and f1.converged) else float("nan")
    return BaselineResult(
        "ordinal_no_ratio", "The ordinal model WITHOUT the slope-ratio contribution",
        "pooled epoch-interaction slope (beta_2 - beta_1, pooled fit)",
        float(inter), float("nan"), float("nan"), int(d["pid"].nunique()), False,
        "Pools across participants, so a person with more observations "
        "dominates and person-specific thresholds are forced to be common. Our "
        "contribution is the per-person RATIO, which cancels the unknown sensor "
        "gain lambda; a pooled difference does not. "
        f"(Pooled single-slope fit for reference: beta = {fit_all.beta:.3f}.)")


def linear_anchor(df, sensor, K, *, seed=SEED, n_resamples=None):
    from ..estimators.linear_anchor import linear_anchor_ratio
    people = []
    for _pid, g in df.groupby("pid", sort=True):
        g0 = g[g["epoch"] == 0].sort_values("ts")
        g1 = g[g["epoch"] == 1].sort_values("ts")
        if len(g0) < 10 or len(g1) < 10:
            continue
        people.append((g0["report"].to_numpy(int), g1["report"].to_numpy(int),
                       g0[sensor].to_numpy(float), g1[sensor].to_numpy(float)))
    rho, conv = linear_anchor_ratio(people, K)
    return BaselineResult(
        "linear_anchor", "OUR OWN PREVIOUS METHOD -- a documented FAILURE",
        "affine anchor rho_hat", float(rho), float("nan"), float("nan"),
        len(people), False,
        "Models a BOUNDED ORDINAL response with an AFFINE equation. Fabricates "
        "roughly 10% apparent scale compression under the true null on 5-point "
        "scales (documented bias -0.107). Retained and asserted by a test so "
        "that a future 'fix' which breaks this known failure fails loudly.")


BASELINES = {
    "raw_epoch_means": raw_epoch_means,
    "epoch_normalised": epoch_normalised,
    "koren_drift": koren_drift,
    "sensor_only": sensor_only,
    "ordinal_no_ratio": ordinal_no_ratio,
    "linear_anchor": linear_anchor,
}


def run_baselines(df: pd.DataFrame, sensor: str, K: int, *,
                  which: list[str] | None = None, seed: int = SEED,
                  n_resamples: int | None = 399,
                  data_status: DataStatus = DataStatus.SYNTHETIC
                  ) -> pd.DataFrame:
    """Run the pre-specified baselines on the SAME eligible participants."""
    names = which or list(BASELINES)
    rows = []
    for name in names:
        fn = BASELINES.get(name)
        if fn is None:
            raise KeyError(f"Unknown baseline {name!r}; known: {sorted(BASELINES)}")
        try:
            r = fn(df, sensor, K, seed=seed, n_resamples=n_resamples)
        except Exception as exc:                  # a baseline failing is a result
            log.warning("baseline %s failed: %s", name, exc)
            rows.append({"name": name, "statistic": float("nan"),
                         "interpretation": f"failed: {exc}",
                         "data_status": data_status.value})
            continue
        d = r.to_dict()
        d["data_status"] = data_status.value
        rows.append(d)
    return pd.DataFrame(rows)
