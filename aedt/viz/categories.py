"""Category-usage bar chart and floor rate -- demo stage 2.

"Most days sit at one end of the scale - that matters enormously later."

Floor-heavy usage is the regime real single-item stress measures actually live
in, and it is why the ordinal model was chosen over an affine one. This figure
is where a panel sees that with their own eyes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DataStatus
from .style import EPOCH_COLOURS, apply_style, save_figure, stamp

__all__ = ["category_usage_plot"]


def category_usage_plot(usage: pd.DataFrame, K: int, *,
                        data_status: DataStatus = DataStatus.SYNTHETIC,
                        pid: str | None = None, path=None):
    import matplotlib.pyplot as plt

    apply_style()
    if pid is not None:
        usage = usage[usage["pid"] == pid]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    width = 0.38
    xs = np.arange(1, K + 1)
    for e in (0, 1):
        sub = usage[usage["epoch"] == e]
        if not len(sub):
            continue
        counts = np.array([sub[f"n_cat{k}"].sum() for k in xs], dtype=float)
        total = counts.sum()
        share = counts / total if total else counts
        ax.bar(xs + (e - 0.5) * width, share, width, color=EPOCH_COLOURS[e],
               alpha=0.88, label=f"Epoch {e + 1}  (n = {int(total)})")
    ax.set_xticks(xs)
    ax.set_xlabel("self-reported stress category (1 = least, "
                  f"{K} = most)")
    ax.set_ylabel("share of responses")
    ax.set_title("Response-category usage", loc="left", fontweight="bold")
    ax.legend(fontsize=9.5)

    ax = axes[1]
    floors = usage.groupby("pid")["floor_rate"].mean().to_numpy()
    ceils = usage.groupby("pid")["ceiling_rate"].mean().to_numpy()
    ax.hist(100 * floors, bins=min(18, max(4, len(floors) // 2)),
            color="#b45309", alpha=0.85, label="floor (category 1)")
    ax.hist(100 * ceils, bins=min(18, max(4, len(ceils) // 2)),
            color="#0e7490", alpha=0.62, label=f"ceiling (category {K})")
    ax.axvline(100 * float(np.median(floors)), color="#7c2d12", lw=2,
               label=f"median floor {100 * np.median(floors):.0f}%")
    ax.set_xlabel("share of a participant's responses at the boundary (%)")
    ax.set_ylabel("participants")
    ax.set_title("Boundary (floor / ceiling) rates", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Most days sit at one end of the scale - which is exactly why "
                 "an ordinal model, not a linear one", fontsize=12.5,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    stamp(fig, data_status)
    return save_figure(fig, path) if path else fig
