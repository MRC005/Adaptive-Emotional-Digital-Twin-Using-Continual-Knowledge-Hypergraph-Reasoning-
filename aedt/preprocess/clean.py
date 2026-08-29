"""MODULE 2 -- PREPROCESSING.

Purpose  Clean and normalise raw records WITHOUT destroying auditability.
Input    Raw canonical records (LongFrame).
Output   Preprocessed records plus a missingness ledger.
Algorithm Explicit cleaning rules; missingness accounting; NO SILENT
         IMPUTATION -- and specifically no imputation of the OUTCOME.
Status   STANDARD.

Every row removed is counted, categorised and reported. A pipeline run that
drops 40% of its data must say so on the console and in the run log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schemas import Serialisable

log = logging.getLogger(__name__)

__all__ = ["MissingnessLedger", "clean_long_frame"]


@dataclass
class MissingnessLedger(Serialisable):
    """Every row that did not survive preprocessing, with its reason."""

    n_input: int = 0
    n_output: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    column_missing_rate: dict[str, float] = field(default_factory=dict)
    participants_lost: list[str] = field(default_factory=list)

    def record(self, reason: str, n: int) -> None:
        if n > 0:
            self.removed[reason] = self.removed.get(reason, 0) + int(n)
            log.info("preprocessing removed %d rows: %s", n, reason)

    @property
    def n_removed(self) -> int:
        return sum(self.removed.values())

    @property
    def retention_rate(self) -> float:
        return self.n_output / self.n_input if self.n_input else float("nan")


def clean_long_frame(df: pd.DataFrame, sensor: str, *,
                     drop_duplicate_occasions: bool = True,
                     sensor_outlier_z: float | None = 6.0,
                     ) -> tuple[pd.DataFrame, MissingnessLedger]:
    """Apply the explicit cleaning rules and return (clean frame, ledger).

    Rules, in order:
      1. Drop rows with a missing timestamp -- they cannot be ordered, and
         ordering is load-bearing for the epoch split and the placebo.
      2. Drop rows with a missing SELF-REPORT. The outcome is never imputed.
      3. Drop rows with a missing SENSOR value for the analysis feature. An
         imputed covariate would attenuate beta differently in the two epochs
         and bias the ratio.
      4. Drop exact duplicate (pid, ts) occasions, keeping the first.
      5. FLAG (do not drop) extreme sensor values beyond ``sensor_outlier_z``
         within-participant robust z. Flagging rather than dropping keeps the
         decision visible; the flag is carried in ``sensor_outlier_flag``.
    """
    led = MissingnessLedger(n_input=int(len(df)))
    pids_before = set(df["pid"].astype(str))
    out = df.copy()

    for col in ("ts", "report", sensor):
        if col in out.columns:
            led.column_missing_rate[col] = float(out[col].isna().mean())

    n = len(out)
    out = out[out["ts"].notna()]
    led.record("missing timestamp", n - len(out))

    n = len(out)
    out = out[out["report"].notna()]
    led.record("missing self-report (outcome NEVER imputed)", n - len(out))

    if sensor in out.columns:
        n = len(out)
        out = out[out[sensor].notna()]
        led.record(f"missing sensor feature '{sensor}' (covariate NEVER imputed)",
                   n - len(out))

    if drop_duplicate_occasions:
        n = len(out)
        out = out.sort_values(["pid", "ts"]).drop_duplicates(
            subset=["pid", "ts"], keep="first")
        led.record("duplicate (pid, ts) occasion", n - len(out))

    if sensor_outlier_z is not None and sensor in out.columns and len(out):
        flags = np.zeros(len(out), dtype=bool)
        vals = out[sensor].to_numpy(dtype=float)
        pos = 0
        for _pid, g in out.groupby("pid", sort=False):
            v = vals[pos:pos + len(g)]
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            scale = 1.4826 * mad
            if scale > 1e-12:
                flags[pos:pos + len(g)] = np.abs(v - med) / scale > sensor_outlier_z
            pos += len(g)
        out = out.assign(sensor_outlier_flag=flags)
        if flags.any():
            log.info("flagged (not dropped) %d extreme sensor values at |z|>%.1f",
                     int(flags.sum()), sensor_outlier_z)

    out = out.reset_index(drop=True)
    led.n_output = int(len(out))
    led.participants_lost = sorted(pids_before - set(out["pid"].astype(str)))
    if led.participants_lost:
        log.info("participants lost entirely in preprocessing: %s",
                 led.participants_lost)
    return out, led
