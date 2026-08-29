"""System architecture and data-pipeline diagrams, drawn from the FROZEN §I.

The four CONTRIBUTION blocks are coloured distinctly with a legend, exactly as
ROUND-17 §T slide 9 requires. Everything else is stamped STANDARD or
ENGINEERING INTEGRATION.
"""
from __future__ import annotations

from ..constants import DataStatus
from .style import apply_style, save_figure, stamp

__all__ = ["architecture_diagram", "pipeline_diagram", "ARCHITECTURE_STAGES"]

# (label, classification) -- classification drives the colour
ARCHITECTURE_STAGES = [
    ("DATA SOURCES\nStudentLife - RELAX - WESAD - PMData - synthetic", "STANDARD"),
    ("DATA INGESTION\nadapters -> canonical LongFrame; halt on surprise", "STANDARD"),
    ("PREPROCESSING\nmissingness ledger; no imputation of the outcome", "STANDARD"),
    ("TEMPORAL ALIGNMENT\ncausal windows; epochs = halves of each own span", "STANDARD"),
    ("SENSOR FEATURE EXTRACTION\nconversation minutes (primary); HR; activity", "STANDARD"),
    ("CONTEXT FORMATION\n(a) continuous [default] (b) vector bins (c) n-ary", "STANDARD"),
    ("PERSONAL DIGITAL TWIN\npersistent TwinState: thresholds, slopes, history", "ENGINEERING"),
    ("CONTINUAL KNOWLEDGE\nappend-only store; provenance; temporal inspection", "ENGINEERING"),
    ("HYPERGRAPH LAYER\nhigher-order context [ABLATION, not identification]", "ENGINEERING"),
    ("ORDINAL SELF-REPORT MODEL\nP(R<=k|x) = Phi(c_k - beta_e x), ML per person", "STANDARD"),
    ("SLOPE-RATIO ESTIMATOR\nrho* = beta_2/beta_1; additive NOT IDENTIFIED", "CONTRIBUTION"),
    ("ELIGIBILITY / QUALITY AUDIT\nfixed thresholds; every exclusion printed", "ENGINEERING"),
    ("PLACEBO VALIDATION\ncontiguous epoch-1 split-half; GATES the primary", "CONTRIBUTION"),
    ("BIAS ENVELOPE\nrho* range under the allowed assumption violations", "CONTRIBUTION"),
    ("UNCERTAINTY\nparticipant-cluster bootstrap, 2000 resamples", "STANDARD"),
    ("LONGITUDINAL TWIN UPDATE\nappend {rho*, CI, verdict}: ACCEPTED / FLAGGED", "ENGINEERING"),
    ("VISUALISATION & OUTPUT\ntwo-curve plot; forest; dashboard; every one stamped", "ENGINEERING"),
]

COLOURS = {
    "STANDARD": ("#e7e5e4", "#57534e"),
    "ENGINEERING": ("#dbeafe", "#1e40af"),
    "CONTRIBUTION": ("#fde68a", "#92400e"),
}
LEGEND = {
    "STANDARD": "STANDARD / EXISTING technology",
    "ENGINEERING": "ENGINEERING INTEGRATION (ours, but not novel)",
    "CONTRIBUTION": "OUR RESEARCH CONTRIBUTION",
}


def architecture_diagram(*, data_status: DataStatus = DataStatus.SYNTHETIC,
                         path=None, title: str | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    apply_style()
    n = len(ARCHITECTURE_STAGES)
    # 1.35 units of blank space is reserved at the bottom for the legend so it
    # can never overlap the data-status note.
    fig, ax = plt.subplots(figsize=(9.6, 0.86 * n + 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.86 * n + 1.35)
    ax.set_axis_off()

    for i, (label, kind) in enumerate(ARCHITECTURE_STAGES):
        y = 0.86 * (n - 1 - i) + 1.35
        face, edge = COLOURS[kind]
        ax.add_patch(FancyBboxPatch(
            (0.7, y), 8.6, 0.66, boxstyle="round,pad=0.035,rounding_size=0.06",
            facecolor=face, edgecolor=edge, linewidth=1.5))
        head, _, sub = label.partition("\n")
        ax.text(5.0, y + 0.44, head, ha="center", va="center", fontsize=9.4,
                fontweight="bold", color=edge)
        if sub:
            ax.text(5.0, y + 0.17, sub, ha="center", va="center", fontsize=7.6,
                    color="#44403c")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, y - 0.02), (5.0, y - 0.19), arrowstyle="-|>",
                mutation_scale=11, color="#78716c", linewidth=1.1))

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[k][0],
                             edgecolor=COLOURS[k][1], linewidth=1.4)
               for k in LEGEND]
    ax.legend(handles, list(LEGEND.values()), loc="lower center",
              bbox_to_anchor=(0.5, 0.012), ncol=1, fontsize=9.5,
              borderaxespad=0.0)
    # the badge sits top-right, so the title is left-aligned and kept clear of it
    ax.set_title(title or ("Adaptive Emotional Digital Twin Using Continual "
                           "Knowledge Hypergraph Reasoning\n"
                           "System architecture (FROZEN)"),
                 fontsize=12.5, fontweight="bold", loc="left", pad=16,
                 linespacing=1.7)
    fig.tight_layout()
    stamp(fig, data_status,
          note="Architecture diagram. Four blocks are the research "
               "contribution; the rest is standard technology or integration.")
    return save_figure(fig, path) if path else fig


def pipeline_diagram(*, data_status: DataStatus = DataStatus.SYNTHETIC,
                     path=None):
    """The compact data-flow diagram used on the demo's first slide."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    apply_style()
    stages = ["DATA", "PREPROCESS", "FEATURES", "CONTEXT", "DIGITAL\nTWIN",
              "ORDINAL\nMODEL", "SLOPE\nRATIO", "PLACEBO\n/ AUDIT", "RESULT"]
    hot = {"SLOPE\nRATIO", "PLACEBO\n/ AUDIT"}
    fig, ax = plt.subplots(figsize=(13.2, 2.5))
    ax.set_xlim(0, len(stages) * 1.42)
    ax.set_ylim(0, 2.0)
    ax.set_axis_off()
    for i, s in enumerate(stages):
        x = i * 1.42 + 0.12
        face, edge = (COLOURS["CONTRIBUTION"] if s in hot
                      else COLOURS["STANDARD"])
        ax.add_patch(FancyBboxPatch(
            (x, 0.62), 1.16, 0.78,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=1.5))
        ax.text(x + 0.58, 1.01, s, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=edge)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.19, 1.01), (x + 1.39, 1.01), arrowstyle="-|>",
                mutation_scale=12, color="#78716c", linewidth=1.2))
    ax.set_title("End-to-end pipeline", fontsize=12, fontweight="bold",
                 loc="left")
    fig.tight_layout()
    stamp(fig, data_status)
    return save_figure(fig, path) if path else fig
