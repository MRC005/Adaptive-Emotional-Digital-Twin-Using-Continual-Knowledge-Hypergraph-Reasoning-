"""Hypergraph / context visualisation.

Draws the feature-value VERTICES and the conjunctive HYPEREDGES that connect
them, with each edge's per-epoch occupancy. The point the picture must make is
the one from ROUND-17 §M: a hyperedge is SEVERAL CONDITIONS HOLDING AT ONCE,
which a compensatory feature-vector distance cannot express.

It also shows the epoch-occupancy overlap, which is what the twin actually
reasons over when deciding whether to trust an epoch update.
"""
from __future__ import annotations

import numpy as np

from ..constants import DataStatus
from ..hypergraph.structure import ContextHypergraph
from .style import apply_style, save_figure, stamp

__all__ = ["hypergraph_plot"]


def hypergraph_plot(hg: ContextHypergraph, *,
                    data_status: DataStatus = DataStatus.SYNTHETIC,
                    max_edges: int = 12, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6),
                            gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: vertices and hyperedges ------------------------------------
    ax = axes[0]
    ax.set_axis_off()
    verts = list(hg.vertices)
    if not verts:
        ax.text(0.5, 0.5, "No context hyperedges were formed.\n"
                          "At this observation density the conjunctive "
                          "contexts do not recur.\nThat is a FINDING, not a "
                          "nuisance.", ha="center", va="center", fontsize=11)
    else:
        n = len(verts)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pos = {v: (np.cos(a), np.sin(a)) for v, a in zip(verts, ang)}
        edges = sorted(hg.edges, key=lambda e: -(e.n_epoch0 + e.n_epoch1)
                       )[:max_edges]
        cmap = plt.get_cmap("tab20")
        for i, e in enumerate(edges):
            pts = [pos[v] for v in e.vertices if v in pos]
            if len(pts) < 2:
                continue
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            col = cmap(i % 20)
            both = e.occupied_both_epochs
            for (px, py) in pts:
                ax.plot([cx, px], [cy, py], color=col, lw=2.0 if both else 1.0,
                        alpha=0.75 if both else 0.32,
                        ls="-" if both else ":", zorder=1)
            ax.scatter([cx], [cy], s=44 + 5 * (e.n_epoch0 + e.n_epoch1) ** 0.5,
                       color=col, alpha=0.9, zorder=2,
                       marker="D" if both else "d")
        for v, (px, py) in pos.items():
            ax.add_patch(Circle((px, py), 0.055, color="#1f2937", zorder=3))
            ha = "left" if px > 0.05 else ("right" if px < -0.05 else "center")
            ax.text(px * 1.16, py * 1.16, v.replace("_", " "), ha=ha,
                    va="center", fontsize=8.2, zorder=4)
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Participant {hg.pid}: context hypergraph\n"
                 f"{hg.n_vertices} feature-value vertices, {hg.n_edges} "
                 f"hyperedges (mean arity {hg.mean_arity():.1f})",
                 loc="left", fontweight="bold", fontsize=11.5)

    # ---- right: per-epoch occupancy ---------------------------------------
    ax = axes[1]
    edges = sorted(hg.edges, key=lambda e: -(e.n_epoch0 + e.n_epoch1))[:max_edges]
    if edges:
        y = np.arange(len(edges))
        ax.barh(y - 0.2, [e.n_epoch0 for e in edges], 0.38, color="#1d4ed8",
                label="Epoch 1")
        ax.barh(y + 0.2, [e.n_epoch1 for e in edges], 0.38, color="#b91c1c",
                label="Epoch 2")
        ax.set_yticks(y)
        ax.set_yticklabels([e.key.replace("|", " AND ").replace("_", " ")
                            for e in edges], fontsize=7.4)
        ax.invert_yaxis()
        ax.set_xlabel("observations in this conjunctive context")
        ax.legend(fontsize=9)
    ov = hg.occupancy_overlap()
    ax.set_title("Hyperedge occupancy by epoch\n"
                 f"epoch-to-epoch overlap = "
                 f"{'n/a' if not np.isfinite(ov) else f'{ov:.2f}'}",
                 loc="left", fontweight="bold", fontsize=11.5)

    fig.text(0.5, -0.045,
             "A hyperedge is several conditions holding AT ONCE - conjunctive "
             "and exact, where a feature-vector distance is compensatory.\n"
             "The hypergraph is the twin's contextual knowledge "
             "representation and an ablation arm. It is NOT the identification "
             "mechanism for rho*.",
             ha="center", fontsize=9, color="#44403c")
    fig.tight_layout()
    stamp(fig, data_status)
    return save_figure(fig, path) if path else fig
