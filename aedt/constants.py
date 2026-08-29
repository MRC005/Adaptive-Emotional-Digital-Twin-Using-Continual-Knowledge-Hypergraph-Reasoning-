"""FROZEN constants. Fixed before any real data was seen. Do not edit.

Provenance for every value in this file:

  SEED                       ROUND-17 §V ("seed 20260828 everywhere")
  MIN_REPORTS_PER_EPOCH      ROUND-17 §W, ROUND-16 §4-9
  MIN_CATEGORIES_USED        ROUND-17 §W  (also assumption A5)
  MIN_SENSOR_SD              ROUND-17 §W
  VAR_RATIO_LO / _HI         ROUND-17 §W  (observable proxy for A3)
  MIN_ABS_BETA               ROUND-16 §3, self-correction 26 -- EITHER SIGN
  REQUIRE_MATCHING_SIGN      ROUND-16 §3, self-correction 26
  BOOTSTRAP_B                ROUND-17 §W ("2000 resamples")
  PLACEBO_MAX_REJECT         final_audit.py (Round 16)
  WEAK_ASSOCIATION_BETA      ROUND-16 §4-9, diagnostic [9b]
  STRESS_LABEL_TO_SEVERITY   ROUND-17 §W ("remapped BY LABEL TEXT")

Changing any of these after observing a primary result is forbidden. A
pre-specified sensitivity analysis over a threshold must be declared in
``docs/frozen_scientific_specification.md`` before it is run, and must be
executed through ``aedt.audit.envelope``, never by editing this file.
"""
from __future__ import annotations

from enum import Enum

# ----------------------------------------------------------- reproducibility
SEED = 20260828

# ------------------------------------------------- eligibility screen (§W)
MIN_REPORTS_PER_EPOCH = 60
MIN_CATEGORIES_USED = 2
MIN_SENSOR_SD = 0.10
VAR_RATIO_LO = 0.25
VAR_RATIO_HI = 4.0

# Self-correction 26 (Round 16): the slope must be WELL DETERMINED, not
# positive. Conversation minutes fall as stress rises; requiring beta > 0 cut
# the usable fixture sample from 48 to 2 -- a silent selection on the outcome.
MIN_ABS_BETA = 0.02
REQUIRE_MATCHING_SIGN = True

# ------------------------------------------------------------- inference
BOOTSTRAP_B = 2000
CI_ALPHA = 0.05
PLACEBO_MAX_REJECT = 0.075

# -------------------------------------------------------- diagnostics (9b)
WEAK_ASSOCIATION_BETA = 0.15   # median |beta| below this => weak association

# ---------------------------------------------- self-report label mapping
# Keys are lowercased and whitespace-collapsed by ``normalise_label``.
# NEVER infer severity from the stored integer code or from list position.
STRESS_LABEL_TO_SEVERITY: dict[str, int] = {
    "feeling great": 1,
    "feeling good": 2,
    "a little stressed": 3,
    "definitely stressed": 4,
    "stressed out": 5,
}

# Known-answer test tolerances (ROUND-17 §W "Mandatory known-answer tests").
TOL_SLOPE_RECOVERY = 0.05      # ordinal_probit recovers a known slope within 5%
TOL_NULL_RATIO = 0.03          # slope_ratio returns 1.00 +/- 0.03 under the null
LINEAR_ANCHOR_KNOWN_NULL_BIAS = -0.107   # the documented failure, asserted


class DataStatus(str, Enum):
    """Provenance stamp carried by every record, result, figure and table.

    There is no third state for "probably real". A result is REAL only if it
    was computed from audited files on disk; everything else is SYNTHETIC, and
    anything not yet computed is PLANNED.
    """

    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    PLANNED = "PLANNED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class DatasetRole(str, Enum):
    """The scientific role a dataset may play. Roles are NOT interchangeable.

    PRIMARY_LONGITUDINAL   supports estimation of rho*
    BENCHMARK_PHYSIOLOGICAL validates feature extraction only; it must never be
                           presented as longitudinal validation of rho*
    """

    PRIMARY_LONGITUDINAL = "PRIMARY_LONGITUDINAL"
    LONGITUDINAL_ALTERNATIVE = "LONGITUDINAL_ALTERNATIVE"
    BENCHMARK_PHYSIOLOGICAL = "BENCHMARK_PHYSIOLOGICAL"
    ROBUSTNESS_OPTIONAL = "ROBUSTNESS_OPTIONAL"
    CONDITIONAL_SECONDARY = "CONDITIONAL_SECONDARY"
    SIMULATION = "SIMULATION"


def normalise_label(s: object) -> str:
    """Lowercase, collapse whitespace, treat underscores as spaces."""
    return " ".join(str(s).lower().replace("_", " ").split())
