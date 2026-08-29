"""Adaptive Emotional Digital Twin Using Continual Knowledge Hypergraph Reasoning.

The scientific specification implemented here is FROZEN. See
``docs/frozen_scientific_specification.md``; the authoritative source is
``ROUND-17-FINAL-FREEZE-PACKAGE.md`` §F–§H of the historical project record.

Primary estimand: rho_star, the identified attenuated recalibration ratio.
The additive component b_2 - b_1 is NOT identified and is never estimated.
"""
from __future__ import annotations

__version__ = "0.4.0"
__title__ = ("Adaptive Emotional Digital Twin Using "
             "Continual Knowledge Hypergraph Reasoning")

from .constants import SEED, DataStatus
from .errors import DecisionRequired, ScientificError

__all__ = ["SEED", "DataStatus", "DecisionRequired", "ScientificError",
           "__version__", "__title__"]
