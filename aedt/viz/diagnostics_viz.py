"""Placebo, bias envelope and ablation figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DataStatus
from ..schemas import BiasEnvelopeResult, EstimatorResult, PlaceboResult
from .style import apply_style, save_figure, stamp

__all__ = ["placebo_plot", "envelope_plot", "ablation_plot"]


def placebo_plot(placebo: PlaceboResult, primary: EstimatorResult | None = None,
                 *, path=None):
    """The safety check, shown BEFORE the result it gates."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    rows = [("PLACEBO\ncontiguous epoch-1 split-half\n(no shift can exist)",
             placebo.rho_star, placebo.ci_low, placebo.ci_high,
             "#b91c1c" if placebo.rejected else "#15803d")]
    if primary is not None and primary.uncertainty is not None:
        rows.append((f"PRIMARY\nepoch 1 vs epoch 2\n"
                     f"({primary.n_participants_used} participants)",
                     primary.rho_star, primary.uncertainty.ci_low,
                     primary.uncertainty.ci_high, "#1d4ed8"))
    for i, (lab, pt, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=col, lw=3.4, solid_capstyle="round")
        ax.scatter([pt], [y], s=90, color=col, zorder=3)
        ax.text(pt, y + 0.19, f"{pt:.3f}", ha="center", fontsize=9.5, color=col)
    ax.axvline(1.0, color="#78716c", ls="--", lw=1.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlabel(r"$\rho^*$")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    verdict = ("PLACEBO REJECTS - the primary result is NOT reported as "
               "validated" if placebo.rejected else
               "Placebo does not reject - the primary analysis may proceed")
    ax.set_title(f"Negative control runs FIRST and gates the primary\n{verdict}",
                 loc="left", fontweight="bold",
                 color="#b91c1c" if placebo.rejected else "#15803d")
    fig.tight_layout()
    stamp(fig, placebo.data_status)
    return save_figure(fig, path) if path else fig


def envelope_plot(env: BiasEnvelopeResult, primary: EstimatorResult | None = None,
                  *, path=None):
    import matplotlib.pyplot as plt

    apply_style()
    names = list(env.rho_star_by_scenario)
    vals = [env.rho_star_by_scenario[n] for n in names]
    fig, ax = plt.subplots(figsize=(9.6, max(3.6, 0.42 * len(names) + 2.2)))
    y = np.arange(len(names))
    ax.axvspan(env.envelope_low, env.envelope_high, color="#fbbf24", alpha=0.2,
               label=f"null envelope [{env.envelope_low:.3f}, "
                     f"{env.envelope_high:.3f}]")
    ax.scatter(vals, y, s=48, color="#78350f", zorder=3)
    ax.axvline(1.0, color="#78716c", ls="--", lw=1.3, label=r"truth ($\rho=1$)")
    if primary is not None and np.isfinite(primary.rho_star):
        inside = env.envelope_low <= primary.rho_star <= env.envelope_high
        ax.axvline(primary.rho_star, color="#1d4ed8", lw=2.4,
                   label=(f"primary $\\rho^*$ = {primary.rho_star:.3f} "
                          f"({'INSIDE' if inside else 'outside'} the envelope)"))
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.set_xlabel(r"$\rho^*$ under the TRUE NULL ($\rho = 1$)")
    ax.set_title("Bias envelope: how far $\\rho^*$ moves under the allowed "
                 "assumption violations", loc="left", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    stamp(fig, env.data_status,
          note="An estimate INSIDE the band cannot be distinguished from an "
               "artefact of the assumption violations the data exhibits.")
    return save_figure(fig, path) if path else fig


def ablation_plot(table: pd.DataFrame, *,
                  data_status: DataStatus = DataStatus.SYNTHETIC,
                  verdict: str = "", path=None):
    """Ablation 1, with DISQUALIFIED arms marked as such.

    Ranking by CI width alone would reward a narrow interval around a wrong
    answer, so an arm that fails null calibration, fires its placebo, or points
    the wrong way is hatched and labelled rather than quietly winning.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    fig.subplots_adjust(bottom=0.36, top=0.86, wspace=0.42)

    def is_bad(row) -> list[str]:
        why = []
        if getattr(row, "placebo_rejects", None) is True:
            why.append("placebo fires")
        if getattr(row, "null_calibrated", None) is False:
            why.append("not null-calibrated")
        r = getattr(row, "effect_retention", float("nan"))
        if np.isfinite(r) and r < 0:
            why.append("wrong direction")
        return why

    labels = [r.replace("_", " ") for r in table["representation"]]
    bad = [is_bad(r) for r in table.itertuples(index=False)]
    cols = ["#7e22ce" if u else "#1d4ed8" for u in table["uses_hypergraph"]]
    y = np.arange(len(table))

    ax = axes[0]
    for i, row in enumerate(table.itertuples(index=False)):
        if np.isfinite(row.ci_low):
            ax.plot([row.ci_low, row.ci_high], [y[i], y[i]], lw=3.2,
                    color=cols[i], solid_capstyle="round",
                    alpha=0.35 if bad[i] else 1.0)
        ax.scatter([row.rho_star], [y[i]], s=85, color=cols[i], zorder=3,
                   alpha=0.35 if bad[i] else 1.0,
                   marker="X" if bad[i] else "o")
    ax.axvline(1.0, color="#78716c", ls="--", lw=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(r"$\rho^*$ with 95% CI")
    ax.set_title("Ablation 1: context representation", loc="left",
                 fontweight="bold")
    # place the label beside its own interval, never over the panel title
    for i, (row, why) in enumerate(zip(table.itertuples(index=False), bad)):
        if why and np.isfinite(row.ci_high):
            ax.annotate("DISQUALIFIED", (row.ci_high, y[i]), xytext=(7, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, fontweight="bold", color="#b91c1c")
    ax.margins(x=0.22)

    ax = axes[1]
    bars = ax.barh(y, table["ci_width"], color=cols, alpha=0.9)
    for i, (b, why) in enumerate(zip(bars, bad)):
        if why:
            b.set_hatch("///")
            b.set_alpha(0.35)
            b.set_edgecolor("#b91c1c")
            ax.text(b.get_width() * 0.5, y[i], "DISQUALIFIED\n" + ", ".join(why),
                    ha="center", va="center", fontsize=7.6, color="#7f1d1d",
                    fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("CI width (narrower is better)")
    ax.set_title("Precision is compared ONLY among usable arms\n"
                 "Calibration first, then CI width - never the reverse",
                 loc="left", fontweight="bold", fontsize=10.5)

    if verdict:
        fig.text(0.5, 0.16, verdict, ha="center", va="top", fontsize=9.2,
                 color="#44403c", wrap=True)
    fig.suptitle("Does the hypergraph help? We report it either way.",
                 fontsize=13, fontweight="bold", y=0.97)
    stamp(fig, data_status, note="")
    return save_figure(fig, path) if path else fig
