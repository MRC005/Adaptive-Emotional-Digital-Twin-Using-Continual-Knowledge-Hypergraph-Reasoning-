"""Shared figure style, and the NON-NEGOTIABLE data-status stamp.

EVERY figure this package produces carries a visible REAL / SYNTHETIC /
PLANNED badge. ``stamp`` is called by every plotting function, and
``tests/unit/test_viz_stamp.py`` asserts that a figure without one cannot be
saved through ``save_figure``.

The rule from ROUND-17 §AC: "Do not let a single number, plot or sentence imply
that any result came from real data." One blurred label costs more credibility
than every missing module combined.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..constants import DataStatus

__all__ = ["apply_style", "stamp", "save_figure", "STATUS_COLOURS",
           "EPOCH_COLOURS"]

STATUS_COLOURS = {
    DataStatus.REAL: "#1a7f37",        # green: audited files on disk
    DataStatus.SYNTHETIC: "#9a3412",   # burnt orange: simulation
    DataStatus.PLANNED: "#57534e",     # grey: not computed
}

EPOCH_COLOURS = {0: "#1d4ed8", 1: "#b91c1c"}   # epoch 1 blue, epoch 2 red

_STAMPED: set[int] = set()


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def stamp(fig, status: DataStatus, *, note: str = "") -> None:
    """Put the data-status badge on the figure. Called by every plot."""
    status = DataStatus(status)
    fig.text(0.995, 0.985, f" {status.value} ", ha="right", va="top",
             fontsize=11, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.35",
                       facecolor=STATUS_COLOURS[status], edgecolor="none"))
    if status is DataStatus.SYNTHETIC:
        note = note or ("Simulated data from the frozen generative model. "
                        "Not evidence about humans.")
    elif status is DataStatus.PLANNED:
        note = note or "Not computed. No data has been analysed for this panel."
    if note:
        fig.text(0.005, 0.005, note, ha="left", va="bottom", fontsize=7.5,
                 color="#57534e", style="italic")
    _STAMPED.add(id(fig))


def save_figure(fig, path: str | Path, *, status: DataStatus | None = None
                ) -> Path:
    """Save a figure. REFUSES to save one that has not been stamped."""
    if id(fig) not in _STAMPED:
        if status is None:
            raise ValueError(
                "Refusing to save an unstamped figure. Every figure must carry "
                "a visible REAL / SYNTHETIC / PLANNED badge; call viz.stamp() "
                "or pass status=.")
        stamp(fig, status)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p
