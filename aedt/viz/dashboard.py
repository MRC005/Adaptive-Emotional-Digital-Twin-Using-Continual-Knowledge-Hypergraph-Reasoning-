"""The audit dashboard: everything a reviewer needs to judge trust, on one page.

Deliberately shows the CHECKS BEFORE THE RESULT, in the order the pipeline runs
them, because that order is itself the argument: association strength, then
eligibility, then placebo, and only then rho*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DataStatus
from ..schemas import EstimatorResult, PlaceboResult
from .style import apply_style, save_figure, stamp

__all__ = ["audit_dashboard"]


def audit_dashboard(*, eligibility: pd.DataFrame, association,
                    placebo: PlaceboResult, primary: EstimatorResult | None,
                    dataset: str, data_status: DataStatus,
                    envelope: tuple[float, float] | None = None, path=None):
    import matplotlib.pyplot as plt

    apply_style()
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # ---- 1. eligibility ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    n_ok = int(eligibility["eligible"].sum()) if len(eligibility) else 0
    n_all = int(len(eligibility))
    ax.barh([0], [n_ok], height=0.5, color="#15803d",
            label=f"eligible ({n_ok})")
    ax.barh([0], [n_all - n_ok], left=[n_ok], height=0.5, color="#b91c1c",
            label=f"excluded ({n_all - n_ok})")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 0.6)
    ax.set_xlabel("participants")
    ax.set_title("1. Eligibility screen\n(thresholds fixed before any data)",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")

    # ---- 2. exclusion reasons --------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if n_all - n_ok > 0:
        reasons = []
        for r in eligibility.loc[~eligibility["eligible"], "reasons"]:
            for part in str(r).split(";"):
                p = part.strip()
                if p:
                    reasons.append(p.split("(")[0][:44])
        vc = pd.Series(reasons).value_counts().head(6)
        ax.barh(np.arange(len(vc)), vc.to_numpy(), color="#b45309")
        ax.set_yticks(np.arange(len(vc)))
        ax.set_yticklabels(vc.index, fontsize=7.2)
        ax.invert_yaxis()
        ax.set_xlabel("participants excluded")
    else:
        ax.text(0.5, 0.5, "No exclusions.", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("2. Every exclusion, with its reason\n(nothing dropped "
                 "silently)", loc="left", fontweight="bold", fontsize=10.5)

    # ---- 3. association strength [9b] ------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    med = getattr(association, "median_abs_beta", float("nan"))
    ax.barh([0], [med if np.isfinite(med) else 0], color="#0e7490", height=0.5)
    ax.axvline(0.15, color="#b91c1c", ls="--", lw=1.6,
               label="weak-association floor 0.15")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 0.6)          # keep the legend clear of the bar
    ax.set_xlabel(r"median $|\beta|$ in epoch 1")
    ax.set_xlim(0, max(0.25, (med if np.isfinite(med) else 0) * 1.3))
    ax.set_title("3. [9b] Sensor-report association\nREAD THIS FIRST",
                 loc="left", fontweight="bold", fontsize=10.5,
                 color="#b91c1c" if getattr(association, "weak", True) else "#0f766e")
    ax.legend(fontsize=8, loc="lower right")

    # ---- 4. placebo -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    col = "#b91c1c" if placebo.rejected else "#15803d"
    if np.isfinite(placebo.ci_low):
        ax.plot([placebo.ci_low, placebo.ci_high], [0, 0], lw=4, color=col,
                solid_capstyle="round")
    ax.scatter([placebo.rho_star], [0], s=90, color=col, zorder=3)
    ax.axvline(1.0, color="#78716c", ls="--", lw=1.3)
    ax.set_yticks([])
    ax.set_xlabel(r"placebo $\rho^*$")
    ax.set_title("4. Placebo (runs BEFORE the primary)\n"
                 + ("REJECTS - primary blocked" if placebo.rejected
                    else "does not reject - primary may run"),
                 loc="left", fontweight="bold", fontsize=10.5, color=col)

    # ---- 5. primary -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if primary is not None and np.isfinite(primary.rho_star):
        u = primary.uncertainty
        if envelope is not None and np.isfinite(envelope[0]):
            ax.axvspan(envelope[0], envelope[1], color="#fbbf24", alpha=0.2,
                       label="null envelope")
        if u is not None and np.isfinite(u.ci_low):
            ax.plot([u.ci_low, u.ci_high], [0, 0], lw=4, color="#1d4ed8",
                    solid_capstyle="round")
        ax.scatter([primary.rho_star], [0], s=100, color="#1d4ed8", zorder=3)
        ax.axvline(1.0, color="#78716c", ls="--", lw=1.3)
        ax.set_xlabel(r"primary $\rho^*$")
        ax.legend(fontsize=8, loc="lower right")
        sub = (f"{primary.rho_star:.3f}"
               + (f"  CI [{u.ci_low:.3f}, {u.ci_high:.3f}]" if u else ""))
    else:
        ax.text(0.5, 0.5, "NOT RUN", ha="center", va="center", fontsize=14,
                fontweight="bold", color="#b91c1c")
        ax.set_xticks([])
        sub = "blocked by the gate above"
    ax.set_yticks([])
    ax.set_title(f"5. Primary $\\rho^*$\n{sub}", loc="left",
                 fontweight="bold", fontsize=10.5)

    # ---- 6. the honest caption -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    lines = [
        f"dataset: {dataset}",
        f"data status: {data_status.value}",
        f"participants screened: {n_all}",
        f"participants used: "
        f"{primary.n_participants_used if primary else 0}",
        "",
        r"$\rho$ is NOT directly identified.",
        r"$1-\rho^*$ is a LOWER BOUND on the",
        "true multiplicative recalibration.",
        "The additive component is NOT",
        "identified and is not estimated.",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.6,
            family="monospace" if data_status else None)
    ax.set_title("6. What this does and does not say", loc="left",
                 fontweight="bold", fontsize=10.5)

    fig.suptitle(f"Audit dashboard - {dataset}  [{data_status.value}]\n"
                 "Checks run BEFORE the result, in the order shown",
                 fontsize=13, fontweight="bold", y=0.99)
    stamp(fig, data_status)
    return save_figure(fig, path) if path else fig
