"""THE MONEY SHOT: EPOCH 1 vs EPOCH 2 ORDINAL SENSOR -> REPORT CURVES.

"Two curves on one axis is the entire project in one picture." (ROUND-17 §P)

What the panel must be able to read off this figure:
  - the sensor -> report relationship in epoch 1,
  - the same relationship in epoch 2,
  - the ordinal category thresholds / fitted cumulative probabilities,
  - the estimated change, summarised by rho*,
  - the uncertainty,
  - whether the data are REAL or SYNTHETIC.

And the one sentence it supports:
  "If the curve got flatter, the same behaviour now earns a different number."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DataStatus
from ..models.ordinal import (predict_cumulative, predict_expected_category)
from ..schemas import OrdinalFit
from .style import EPOCH_COLOURS, apply_style, save_figure, stamp

__all__ = ["two_curve_plot"]


def two_curve_plot(fits: dict[int, OrdinalFit], *, pid: str,
                   rho_star: float | None = None,
                   ci: tuple[float, float] | None = None,
                   data_status: DataStatus = DataStatus.SYNTHETIC,
                   sensor_name: str = "sensor",
                   observations: pd.DataFrame | None = None,
                   path=None, cohort_rho_star: float | None = None,
                   selection_rule: str = ""):
    """Both epochs' fitted ordinal curves on one axis, with rho* annotated."""
    import matplotlib.pyplot as plt

    apply_style()
    f0, f1 = fits.get(0), fits.get(1)
    if f0 is None or f1 is None or not (f0.converged and f1.converged):
        raise ValueError(
            f"two_curve_plot needs two convergent fits for {pid}; got "
            f"epoch1={'ok' if f0 and f0.converged else 'FAILED'}, "
            f"epoch2={'ok' if f1 and f1.converged else 'FAILED'}.")

    K = f0.n_categories
    x = np.linspace(-2.5, 2.5, 240)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(bottom=0.28, top=0.80)

    # ---- left: E[report | sensor], the readable one-line-per-epoch view ----
    ax = axes[0]
    for e, f in ((0, f0), (1, f1)):
        y = predict_expected_category(f, x)
        ax.plot(x, y, color=EPOCH_COLOURS[e], lw=2.8,
                label=(f"Epoch {e + 1}   $\\beta$ = {f.beta:+.3f}   "
                       f"(n = {f.n})"))
    if observations is not None and len(observations):
        for e in (0, 1):
            g = observations[observations["epoch"] == e]
            if not len(g):
                continue
            jitter = (np.random.default_rng(20260828 + e)
                      .normal(0, 0.055, len(g)))
            ax.scatter(g["x"], g["report"] + jitter, s=7, alpha=0.16,
                       color=EPOCH_COLOURS[e], edgecolors="none", zorder=0)
    ax.set_xlabel(f"{sensor_name}\n(standardised WITHIN each epoch)")
    ax.set_ylabel("expected self-reported stress category")
    ax.set_ylim(0.7, K + 0.3)
    ax.set_yticks(range(1, K + 1))
    ax.set_title("The same signal, a different answer", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5)

    flatter = abs(f1.beta) < abs(f0.beta)
    ratio = f1.beta / f0.beta
    reading = ("The epoch-2 curve is FLATTER: the same behaviour now earns a "
               "different number."
               if flatter else
               "The epoch-2 curve is STEEPER: the sensor predicts the report "
               "MORE strongly in epoch 2.")
    if abs(ratio - 1.0) < 0.05:
        reading = ("The two curves nearly coincide: for this participant the "
                   "sensor-report relationship barely moved.")

    # ---- right: the cumulative ordinal curves and the thresholds ----------
    ax = axes[1]
    for e, f in ((0, f0), (1, f1)):
        cum = predict_cumulative(f, x)
        for k in range(cum.shape[1]):
            ax.plot(x, cum[:, k], color=EPOCH_COLOURS[e], lw=1.9,
                    alpha=0.95 - 0.1 * k,
                    ls="-" if e == 0 else "--",
                    label=(f"Epoch {e + 1}" if k == 0 else None))
    ax.axhline(0.5, color="#a8a29e", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(f"{sensor_name} (standardised within epoch)")
    ax.set_ylabel(r"$P(R \leq k \mid x) = \Phi(c_k - \beta_e x)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Ordinal cumulative curves, all {K - 1} thresholds",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5,
              bbox_to_anchor=(0.0, -0.13), ncol=2)

    # ---- the headline ------------------------------------------------------
    r = rho_star if rho_star is not None else f1.beta / f0.beta
    head = f"$\\rho^* = \\beta_2 / \\beta_1 = {r:.3f}$"
    if ci is not None and np.isfinite(ci[0]):
        head += f"    95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    if cohort_rho_star is not None:
        head += f"    (cohort $\\rho^*$ = {cohort_rho_star:.3f})"
    fig.suptitle(
        f"Participant {pid} - sensor to self-report relationship, epoch 1 vs "
        f"epoch 2\n{head}", fontsize=13.5, fontweight="bold", y=0.985)
    if selection_rule:
        fig.text(0.5, 0.885, selection_rule, ha="center", fontsize=9,
                 color="#57534e", style="italic")

    fig.text(0.5, 0.135, reading, ha="center", fontsize=10.5, color="#1f2937",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f4",
                       edgecolor="#d6d3d1"))
    fig.text(0.5, 0.055,
             r"$1-\rho^*$ is a LOWER BOUND on the true multiplicative "
             r"recalibration. $\rho$ itself is NOT point-identified; the "
             "additive component is NOT identified and is not estimated.",
             ha="center", fontsize=9, color="#44403c")
    stamp(fig, data_status)
    if path is not None:
        return save_figure(fig, path)
    return fig
