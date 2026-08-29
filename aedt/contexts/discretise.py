"""Discretisation with EPOCH-1 cut points only.

PORTED from ``amsr/contexts.py::discretise`` (Round 15).

Using pooled quantiles would leak epoch-2 information into the definition of
"the same situation" and would partly absorb the very drift we are testing for.
The cut points therefore come from epoch 1 and are applied UNCHANGED to
epoch 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["discretise_epoch1", "bin_labels"]

BIN_NAMES = {2: ("low", "high"), 3: ("low", "mid", "high"),
             4: ("q1", "q2", "q3", "q4")}


def bin_labels(n_bins: int) -> tuple[str, ...]:
    return BIN_NAMES.get(n_bins, tuple(f"b{i}" for i in range(n_bins)))


def discretise_epoch1(g: pd.DataFrame, ctx_cols: list[str], n_bins: int = 3
                      ) -> pd.DataFrame:
    """Per participant: cut points from epoch 1 only, applied to all epochs."""
    out = g.copy()
    e1 = g[g["epoch"] == g["epoch"].min()]
    for c in ctx_cols:
        v1 = e1[c].dropna().to_numpy(dtype=float)
        if len(v1) < n_bins * 2 or np.nanstd(v1) == 0:
            out["b_" + c] = np.nan
            continue
        qs = np.quantile(v1, np.linspace(0, 1, n_bins + 1)[1:-1])
        out["b_" + c] = np.digitize(g[c].to_numpy(dtype=float), qs).astype(float)
        out.loc[g[c].isna(), "b_" + c] = np.nan
    return out
