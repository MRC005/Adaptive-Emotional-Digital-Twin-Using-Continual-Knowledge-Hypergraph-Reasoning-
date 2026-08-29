"""MODULE 5 -- CONTEXT FORMATION (b): FEATURE-VECTOR BINS.

The non-graph way of doing exactly the same job, and the honest control for
the hypergraph claim. Proximity in the standardised context feature vector is
COMPENSATORY: a low value on one feature can be offset by a high value on
another. A hyperedge cannot be offset -- it is conjunctive and exact.

PORTED from ``amsr/contexts.py::anchors_vector`` (Round 15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import SEED

__all__ = ["vector_context"]


def vector_context(g: pd.DataFrame, ctx_cols: list[str], *, n_proto: int = 6,
                   radius: float = 1.6, seed: int = SEED,
                   out_col: str = "context_vector") -> pd.DataFrame:
    """Assign each occasion to its nearest epoch-1 prototype within ``radius``.

    Prototypes are drawn from EPOCH 1 only, and the standardiser is fitted on
    epoch 1 only, for the same anti-leakage reason as the discretiser.
    """
    out = g.copy()
    out[out_col] = np.nan
    sub = g.dropna(subset=ctx_cols)
    if sub.empty:
        return out
    e0 = (sub["epoch"] == 0).to_numpy()
    if e0.sum() < n_proto:
        return out
    X = sub[ctx_cols].to_numpy(dtype=float)
    mu = X[e0].mean(axis=0)
    sd = X[e0].std(axis=0, ddof=1)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    proto = Z[e0][rng.choice(int(e0.sum()), n_proto, replace=False)]
    d = np.linalg.norm(Z[:, None, :] - proto[None, :, :], axis=2)
    nearest = d.argmin(axis=1).astype(float)
    nearest[d.min(axis=1) > radius] = np.nan
    out.loc[sub.index, out_col] = nearest
    return out
