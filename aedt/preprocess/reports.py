"""SELF-REPORT PROCESSING: label TEXT -> severity, and category usage.

Safety-critical (ROUND-17 §J: "ENGINEERING INTEGRATION, but safety-critical").

The single most dangerous silent failure in this project is a stress scale
whose stored integer codes are NOT in severity order. StudentLife's
``EMA_definition.json`` stores the response options in file order, and that
order is not guaranteed to be the severity order. Mapping by POSITION would
silently reverse the scale and invert every conclusion.

Therefore: the map is built from LABEL TEXT, and an unrecognised label raises
``DecisionRequired`` rather than guessing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import STRESS_LABEL_TO_SEVERITY, normalise_label
from ..errors import DecisionRequired

log = logging.getLogger(__name__)

__all__ = ["remap_report_labels", "build_code_to_severity",
           "category_usage_table", "detect_reversed_coding"]


def build_code_to_severity(options: list[str], *,
                           label_map: dict[str, int] | None = None,
                           first_code: int = 1) -> dict[int, int]:
    """Map STORED CODE -> SEVERITY using the option TEXT, never its position.

    ``options`` is the response-option list in the order the file stores it;
    the stored code is assumed to be ``first_code + index``, which is the only
    positional assumption made and is verified against the data separately.
    """
    lm = STRESS_LABEL_TO_SEVERITY if label_map is None else label_map
    code_to_sev: dict[int, int] = {}
    unknown: list[tuple[int, str]] = []
    for i, lab in enumerate(options, start=first_code):
        key = normalise_label(lab)
        if key in lm:
            code_to_sev[i] = lm[key]
        else:
            unknown.append((i, lab))
    if unknown:
        raise DecisionRequired(
            "Dataset stress labels differ from expected mapping. "
            f"Unrecognised options: {unknown}. Full list as stored: {options}. "
            "Do NOT guess the severity order -- confirm the wording against "
            "the dataset documentation, then update "
            "aedt.constants.STRESS_LABEL_TO_SEVERITY.")
    if len(set(code_to_sev.values())) != len(code_to_sev):
        raise DecisionRequired(
            f"The label->severity map is not injective: {code_to_sev}. "
            "Two stored options map to the same severity.")
    return code_to_sev


def detect_reversed_coding(code_to_severity: dict[int, int]) -> bool:
    """True when the stored codes are NOT already in severity order.

    This is exactly the trap the specification flags. It is reported loudly,
    never corrected silently and never ignored.
    """
    codes = sorted(code_to_severity)
    sevs = [code_to_severity[c] for c in codes]
    return sevs != sorted(sevs)


def remap_report_labels(codes, code_to_severity: dict[int, int], *,
                        strict: bool = True) -> np.ndarray:
    """Apply the code -> severity map. Unknown codes halt in strict mode."""
    arr = np.asarray(codes)
    out = np.full(len(arr), np.nan)
    unknown = set()
    for i, c in enumerate(arr):
        try:
            key = int(c)
        except (TypeError, ValueError):
            unknown.add(c)
            continue
        if key in code_to_severity:
            out[i] = code_to_severity[key]
        else:
            unknown.add(key)
    if unknown and strict:
        raise DecisionRequired(
            f"Stored self-report codes {sorted(unknown, key=str)} are absent "
            f"from the code->severity map {code_to_severity}. The file "
            "contains a response option the specification does not describe.")
    return out


def category_usage_table(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """Per participant per epoch: how often each category was used.

    Assumption A5 requires >= 2 categories used in BOTH epochs, and the floor
    rate drives the whole floor-effect discussion, so this table is produced on
    every run and shown as demo stage 2.
    """
    rows = []
    has_epoch = "epoch" in df.columns
    for pid, g in df.groupby("pid", sort=True):
        epochs = sorted(pd.unique(g["epoch"])) if has_epoch else [None]
        for e in epochs:
            ge = g if e is None else g[g["epoch"] == e]
            R = ge["report"].to_numpy(dtype=int)
            if len(R) == 0:
                continue
            counts = np.bincount(R, minlength=K + 1)[1:K + 1]
            rec = {"pid": str(pid), "epoch": e, "n": int(len(R)),
                   "categories_used": int((counts > 0).sum()),
                   "floor_rate": float(np.mean(R == 1)),
                   "ceiling_rate": float(np.mean(R == K))}
            for k in range(1, K + 1):
                rec[f"n_cat{k}"] = int(counts[k - 1])
            rows.append(rec)
    return pd.DataFrame(rows)
