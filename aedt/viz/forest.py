"""Per-participant forest plot of rho*, with the cohort estimate and envelope.

Round 16's hostile review required this: the pooled point is a GEOMETRIC MEAN,
so a few extreme participants can move it. The forest plot makes that visible
rather than leaving it to a footnote.
"""
from __future__ import annotations

import numpy as np

from ..constants import DataStatus
from ..schemas import EstimatorResult
from .style import apply_style, save_figure, stamp

__all__ = ["forest_plot"]


def forest_plot(result: EstimatorResult, *, envelope: tuple[float, float] | None
                = None, path=None, max_shown: int = 60):
    import matplotlib.pyplot as plt

    apply_style()
    vals = np.asarray(result.per_participant_rho_star, dtype=float)
    pids = list(result.per_participant_pids)
    if len(vals) == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No participant produced a usable estimate.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        stamp(fig, result.data_status)
        return save_figure(fig, path) if path else fig

    order = np.argsort(vals)
    vals, pids = vals[order], [pids[i] for i in order]
    if len(vals) > max_shown:
        idx = np.linspace(0, len(vals) - 1, max_shown).astype(int)
        vals, pids = vals[idx], [pids[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.22 * len(vals) + 2.4)))
    y = np.arange(len(vals))
    ax.scatter(vals, y, s=26, color="#1f2937", zorder=3, label="participant")
    ax.axvline(1.0, color="#78716c", lw=1.4, ls="--",
               label="no recalibration ($\\rho^*=1$)")

    if envelope is not None and np.isfinite(envelope[0]):
        ax.axvspan(envelope[0], envelope[1], color="#fbbf24", alpha=0.16,
                   zorder=0,
                   label=f"null bias envelope [{envelope[0]:.3f}, "
                         f"{envelope[1]:.3f}]")

    unc = result.uncertainty
    if unc is not None and np.isfinite(unc.ci_low):
        ax.axvspan(unc.ci_low, unc.ci_high, color="#3b82f6", alpha=0.16,
                   zorder=1,
                   label=f"cohort 95% CI [{unc.ci_low:.3f}, {unc.ci_high:.3f}]")
    ax.axvline(result.rho_star, color="#1d4ed8", lw=2.2, zorder=2,
               label=f"cohort $\\rho^*$ = {result.rho_star:.3f} (geometric mean)")
    if np.isfinite(result.median_rho_star):
        ax.axvline(result.median_rho_star, color="#0f766e", lw=1.6, ls=":",
                   zorder=2,
                   label=f"median $\\rho^*$ = {result.median_rho_star:.3f}")

    ax.set_yticks(y)
    ax.set_yticklabels(pids, fontsize=7.5)
    ax.set_xlabel(r"$\rho^*$ (per participant)")
    ax.set_title(f"Per-participant recalibration ratio  -  "
                 f"{result.n_participants_used} of "
                 f"{result.n_participants_screened} participants used",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    stamp(fig, result.data_status,
          note=("The cohort point is a GEOMETRIC MEAN of per-person ratios, so "
                "extreme participants move it; the median is shown beside it."
                if result.data_status is DataStatus.REAL else ""))
    return save_figure(fig, path) if path else fig
