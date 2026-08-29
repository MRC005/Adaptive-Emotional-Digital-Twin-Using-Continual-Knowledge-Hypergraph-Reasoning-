"""MODULE 8 -- HYPERGRAPH / CONTEXT LAYER.

Purpose  Represent higher-order (conjunctive) contextual relationships among
         participant state, sensor features, behavioural context and time.
Input    Discretised context features for one participant.
Output   A ContextHypergraph: feature-value VERTICES joined by HYPEREDGES,
         each hyperedge carrying its per-epoch occupancy.
Algorithm Exact tuple of discretised feature bins = one hyperedge.
Status   The n-ary representation is STANDARD / EXISTING; its integration here
         is ENGINEERING INTEGRATION (ROUND-17 §J).

THE HONEST POSITION, WHICH MUST NOT BE SOFTENED (ROUND-17 §M):

  The hypergraph is NOT part of the identification mathematics. The frozen
  method regresses the ordinal response on a CONTINUOUS sensor covariate and
  uses every observation. rho* is identified because of the ratio construction
  in ``estimators/slope_ratio.py``, not because of anything in this file.

  Its role is the personal contextual KNOWLEDGE REPRESENTATION:
  "sleep poor AND activity low AND evening AND at home" is ONE hyperedge over
  feature-value vertices -- conjunctive and exact, where a feature-vector
  distance is compensatory.

  It is implemented as an ABLATION COMPONENT. The system supports
  WITH HYPERGRAPH vs WITHOUT HYPERGRAPH, and if the continuous covariate wins,
  that result is reported as it falls.

WHERE THE TWIN DOES "REASON" OVER IT. ``twin/update.py`` consults hyperedge
occupancy to decide whether an epoch update is trustworthy: an epoch whose
occupied hyperedges barely overlap epoch 1's is flagged, because the two epochs
are then not describing comparable situations. That is a real, implemented use
of the higher-order structure -- and it is a TRUST decision, not an
identification mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contexts.discretise import bin_labels, discretise_epoch1
from ..schemas import Hyperedge, Serialisable

__all__ = ["ContextHypergraph", "build_hypergraph", "nary_context",
           "MAX_FACTORS", "MIN_OBS_PER_EDGE"]

# With 5 factors x 3 bins there are 243 possible signatures, and a participant
# with ~20 reports per epoch can never populate 3 of them in BOTH epochs -- the
# hyperedges simply do not exist. The cap keeps the matcher usable; the
# feasibility limit itself is REPORTED as a finding, not hidden as a nuisance.
MAX_FACTORS = 3
MIN_OBS_PER_EDGE = 4


@dataclass
class ContextHypergraph(Serialisable):
    """Vertices are feature-value pairs; hyperedges are exact conjunctions."""

    pid: str
    vertices: list[str] = field(default_factory=list)
    edges: list[Hyperedge] = field(default_factory=list)
    factors: list[str] = field(default_factory=list)
    n_bins: int = 3

    # ------------------------------------------------------------ queries
    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def edges_occupied_both_epochs(self) -> list[Hyperedge]:
        return [e for e in self.edges if e.occupied_both_epochs]

    def occupancy_overlap(self) -> float:
        """Jaccard overlap between the epoch-1 and epoch-2 occupied edge sets.

        This is what the twin reasons over. A low overlap means the two epochs
        describe different situations, so a slope change may reflect changed
        circumstances rather than changed reporting -- exactly the
        scale-change vs relation-change confound named in the limitations.
        """
        a = {e.key for e in self.edges if e.n_epoch0 > 0}
        b = {e.key for e in self.edges if e.n_epoch1 > 0}
        if not a and not b:
            return float("nan")
        return len(a & b) / len(a | b)

    def mean_arity(self) -> float:
        return float(np.mean([e.arity for e in self.edges])) if self.edges else 0.0

    def summary(self) -> dict:
        both = self.edges_occupied_both_epochs()
        return {
            "pid": self.pid,
            "n_vertices": self.n_vertices,
            "n_hyperedges": self.n_edges,
            "n_edges_both_epochs": len(both),
            "mean_arity": self.mean_arity(),
            "occupancy_overlap": self.occupancy_overlap(),
            "factors": list(self.factors),
            "n_bins": self.n_bins,
        }


def build_hypergraph(g: pd.DataFrame, ctx_cols: list[str], *,
                     n_bins: int = 3, max_factors: int = MAX_FACTORS,
                     min_obs_per_edge: int = MIN_OBS_PER_EDGE
                     ) -> ContextHypergraph:
    """Build one participant's context hypergraph.

    Bins come from EPOCH 1 only (see ``contexts/discretise.py``).
    """
    pid = str(g["pid"].iloc[0]) if len(g) else ""
    use = list(ctx_cols)[:max_factors]
    hg = ContextHypergraph(pid=pid, factors=use, n_bins=n_bins)
    if not use:
        return hg

    gd = discretise_epoch1(g, use, n_bins=n_bins)
    bcols = ["b_" + c for c in use]
    sub = gd.dropna(subset=bcols)
    if sub.empty:
        return hg

    names = bin_labels(n_bins)
    hg.vertices = sorted({f"{c}={names[int(b)]}"
                          for c in use
                          for b in pd.unique(sub["b_" + c].dropna())
                          if 0 <= int(b) < len(names)})

    sig = sub[bcols].astype(int).agg(tuple, axis=1)
    for key, idx in sig.groupby(sig).groups.items():
        rows = sub.loc[idx]
        n0 = int((rows["epoch"] == 0).sum())
        n1 = int((rows["epoch"] == 1).sum())
        if n0 + n1 < min_obs_per_edge:
            continue
        verts = tuple(f"{c}={names[int(b)]}" for c, b in zip(use, key))
        r0 = rows.loc[rows["epoch"] == 0, "report"]
        r1 = rows.loc[rows["epoch"] == 1, "report"]
        hg.edges.append(Hyperedge(
            key="|".join(verts), vertices=verts, pid=pid,
            n_epoch0=n0, n_epoch1=n1,
            mean_report_epoch0=float(r0.mean()) if len(r0) else None,
            mean_report_epoch1=float(r1.mean()) if len(r1) else None))
    hg.edges.sort(key=lambda e: e.key)
    return hg


def nary_context(df: pd.DataFrame, ctx_cols: list[str], *, n_bins: int = 3,
                 max_factors: int = MAX_FACTORS,
                 out_col: str = "context_hyperedge"
                 ) -> tuple[pd.DataFrame, dict[str, ContextHypergraph]]:
    """Attach each occasion's hyperedge key, and return the per-person graphs."""
    out = df.copy()
    out[out_col] = pd.Series([None] * len(out), index=out.index, dtype=object)
    graphs: dict[str, ContextHypergraph] = {}
    use = list(ctx_cols)[:max_factors]
    names = bin_labels(n_bins)
    for pid, g in df.groupby("pid", sort=True):
        graphs[str(pid)] = build_hypergraph(g, ctx_cols, n_bins=n_bins,
                                            max_factors=max_factors)
        if not use:
            continue
        gd = discretise_epoch1(g, use, n_bins=n_bins)
        bcols = ["b_" + c for c in use]
        ok = gd[bcols].notna().all(axis=1)
        if not ok.any():
            continue
        keys = gd.loc[ok, bcols].astype(int).apply(
            lambda r: "|".join(f"{c}={names[int(b)]}"
                               for c, b in zip(use, r)), axis=1)
        out.loc[keys.index, out_col] = keys.to_numpy()
    return out, graphs
