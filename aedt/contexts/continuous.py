"""MODULE 5 -- CONTEXT FORMATION (a): CONTINUOUS  [FROZEN DEFAULT].

Purpose  Build the contextual state used by the estimator.
Input    Aligned features.
Output   ContextState records, and the covariate column the model consumes.
Algorithm Within-epoch standardised sensed level. Uses EVERY observation.
Status   STANDARD / SYSTEM ENGINEERING.

This is the representation the FROZEN primary method uses. The other two are
ablation alternatives, not replacements.
"""
from __future__ import annotations

import pandas as pd

from ..constants import DataStatus
from ..estimators.slope_ratio import standardise_within_epoch
from ..schemas import ContextState

__all__ = ["continuous_context", "context_states"]


def continuous_context(df: pd.DataFrame, sensor: str, *,
                       out_col: str = "context_continuous") -> pd.DataFrame:
    """Add the within-epoch standardised sensed level as a column."""
    out = df.copy()
    vals = out[sensor].astype(float).copy()
    for (_pid, _e), idx in out.groupby(["pid", "epoch"]).groups.items():
        x, _m, _s = standardise_within_epoch(out.loc[idx, sensor].to_numpy(float))
        vals.loc[idx] = x
    out[out_col] = vals
    return out


def context_states(df: pd.DataFrame, col: str = "context_continuous",
                   status: DataStatus = DataStatus.SYNTHETIC
                   ) -> list[ContextState]:
    """Typed ContextState records, one per occasion."""
    return [ContextState(pid=str(r.pid), ts=r.ts, continuous=float(getattr(r, col)),
                         feature_names=(col,), data_status=status)
            for r in df.itertuples(index=False)]
