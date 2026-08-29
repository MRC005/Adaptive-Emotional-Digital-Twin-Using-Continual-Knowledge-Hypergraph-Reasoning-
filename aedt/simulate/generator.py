"""SYNTHETIC data generator with a KNOWN rho.

PORTED from the validated research generators ``gate.py::person_series`` and
``skewtest.py::person`` (Rounds 14-15). The data-generating process is exactly
the frozen model of ROUND-17 §G:

    latent report   r* = a_e theta + b_e + eps        eps ~ N(0, sigma_r^2)
    observed        R  = k  iff  tau_{k-1} < r* <= tau_k    (tau FIXED per person)
    sensor          s  = lambda_p theta + kappa_p + eta     eta may be AR(1)

with a_1 = 1 and a_2 ~ N(rho, 0.08), and a genuine between-epoch change in the
construct itself (delta ~ N(-0.4, 0.30)) so that the estimator has to separate
recalibration from real change rather than being handed a clean null.

Thresholds are placed at controllable QUANTILES of the epoch-1 latent report,
which is how floor-heavy usage -- the regime real single-item stress items
actually live in -- is reproduced:

    balanced       [.20 .40 .60 .80]    all categories equally used
    skewed         [.45 .70 .86 .95]    ~45% of epoch-1 responses in category 1
    extreme floor  [.65 .84 .93 .975]   ~68% in category 1, a hard floor

EVERYTHING THIS MODULE PRODUCES IS SYNTHETIC and is stamped as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import SEED, DataStatus

__all__ = ["ar1", "simulate_person", "simulate_cohort", "cohort_to_long_frame",
           "THRESHOLD_PLACEMENTS", "SimulatedPerson"]

THRESHOLD_PLACEMENTS: dict[str, list[float]] = {
    "balanced": [.20, .40, .60, .80],
    "skewed": [.45, .70, .86, .95],
    "extreme_floor": [.65, .84, .93, .975],
}


def ar1(rng: np.random.Generator, n: int, sd: float, phi: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal SD ``sd``. Verbatim from Round 14."""
    e = rng.normal(0, sd, n)
    if phi == 0:
        return e
    out = np.empty(n)
    out[0] = e[0]
    k = np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + k * e[t]
    return out


@dataclass(frozen=True)
class SimulatedPerson:
    """One simulated participant with the truth retained for known-answer tests."""

    pid: str
    R1: np.ndarray
    R2: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    a1: np.ndarray
    a2_sensor: np.ndarray
    true_a2: float
    true_lambda: float
    true_delta: float
    n_categories: int


def simulate_person(rng: np.random.Generator, n_per_epoch: int, rho: float, *,
                    pid: str = "p00", placement: str = "skewed",
                    phi: float = 0.4, sigma_r: float = 0.5,
                    sigma_p: float = 0.8, link: str = "identity",
                    ) -> SimulatedPerson:
    """One person, two epochs, returned as ORDERED time series (not shuffled)."""
    if placement not in THRESHOLD_PLACEMENTS:
        raise ValueError(f"unknown threshold placement {placement!r}")
    cutq = THRESHOLD_PLACEMENTS[placement]
    K = len(cutq) + 1

    lam = rng.uniform(0.35, 0.90)
    kap = rng.normal(0.4, 0.3)
    a2 = rng.normal(rho, 0.08)
    delta = rng.normal(-0.4, 0.30)          # genuine change in the construct

    th1 = rng.normal(0.0, 1.0, n_per_epoch)
    th2 = rng.normal(delta, 1.0, n_per_epoch)

    def f(t: np.ndarray) -> np.ndarray:
        return np.tanh(t) if link == "tanh" else t

    r1s = 1.0 * f(th1) + rng.normal(0, sigma_r, n_per_epoch)
    r2s = a2 * f(th2) + 0.30 + rng.normal(0, sigma_r, n_per_epoch)

    # thresholds are FIXED for the person, placed on the epoch-1 latent scale
    cuts = np.quantile(r1s, cutq)
    R1 = np.searchsorted(cuts, r1s) + 1
    R2 = np.searchsorted(cuts, r2s) + 1

    s1 = lam * th1 + kap + ar1(rng, n_per_epoch, sigma_p, phi)
    s2 = lam * th2 + kap + ar1(rng, n_per_epoch, sigma_p, phi)

    # A SECOND observable sensor stream with a different loading on theta.
    # It is not used by the primary estimator -- it exists so that the
    # contextual layer has more than one factor to form CONJUNCTIONS over.
    # A one-factor "hyperedge" is just a bin, which would make the hypergraph
    # ablation meaningless.
    lam_a = rng.uniform(0.20, 0.60)
    a1 = lam_a * th1 + rng.normal(0.0, 0.3) + ar1(rng, n_per_epoch, 0.9, phi / 2)
    a2s = lam_a * th2 + rng.normal(0.0, 0.3) + ar1(rng, n_per_epoch, 0.9, phi / 2)
    return SimulatedPerson(pid=pid, R1=R1, R2=R2, s1=s1, s2=s2, a1=a1,
                           a2_sensor=a2s, true_a2=float(a2),
                           true_lambda=float(lam),
                           true_delta=float(delta), n_categories=K)


def simulate_cohort(rho: float, *, n_participants: int = 48,
                    n_per_epoch: int = 200, seed: int = SEED,
                    placement: str = "skewed", phi: float = 0.4,
                    sigma_r: float = 0.5, sigma_p: float = 0.8,
                    link: str = "identity") -> list[SimulatedPerson]:
    """A cohort with a known true rho. Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    return [simulate_person(rng, n_per_epoch, rho, pid=f"p{i:02d}",
                            placement=placement, phi=phi, sigma_r=sigma_r,
                            sigma_p=sigma_p, link=link)
            for i in range(n_participants)]


def cohort_to_long_frame(cohort: list[SimulatedPerson], *,
                         sensor: str = "conversation_minutes",
                         start: str = "2026-01-01",
                         observations_per_day: int = 5) -> pd.DataFrame:
    """Render a simulated cohort into the canonical LongFrame.

    The sensor is written on a plausible physical scale (conversation minutes,
    which FALL as stress rises) so that the sign handling exercised on real
    data is exercised here too. The epoch column is assigned from the
    generator's own two blocks, and the alignment module re-derives it from
    timestamps as a cross-check.
    """
    t0 = pd.Timestamp(start)
    rows = []
    for person in cohort:
        n = len(person.R1)
        for e, (R, s, a) in enumerate(((person.R1, person.s1, person.a1),
                                       (person.R2, person.s2, person.a2_sensor))):
            # ordered, evenly spaced occasions; epoch 2 follows epoch 1 in time
            offsets = (np.arange(n) + e * n) / observations_per_day
            ts = t0 + pd.to_timedelta(offsets, unit="D")
            minutes = np.clip(60.0 - 25.0 * s, 0.0, None)
            activity = np.clip(0.35 + 0.12 * a, 0.0, 1.0)
            for i in range(n):
                rows.append((person.pid, ts[i], int(R[i]), e,
                             float(minutes[i]), float(activity[i]),
                             float(s[i])))
    df = pd.DataFrame(rows, columns=["pid", "ts", "report", "epoch",
                                     sensor, "activity_level", "sensed_level"])
    df["day"] = df["ts"].dt.normalize()

    # A strictly CAUSAL contextual feature: the mean of the participant's own
    # PRIOR three observations of the primary sensor. shift(1) is what makes it
    # causal, and tests/unit/test_leakage.py asserts it never sees the present.
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    df["recent_sensor_trend"] = (
        df.groupby("pid")[sensor]
          .transform(lambda v: v.shift(1).rolling(3, min_periods=1).mean()))
    # the first observation of each participant has no past; it is left missing
    # rather than back-filled, and the hypergraph simply does not bin it
    df["hour_of_day"] = df["ts"].dt.hour.astype(float)
    df.attrs["data_status"] = DataStatus.SYNTHETIC.value
    df.attrs["n_categories"] = cohort[0].n_categories if cohort else 5
    return df
