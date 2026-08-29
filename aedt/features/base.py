"""MODULE 4 -- SENSOR FEATURE EXTRACTION (common interface).

Purpose  Turn raw sensor streams into interpretable time-window features.
Input    Aligned sensor streams.
Output   Feature vectors on the canonical LongFrame.
Algorithm Dataset-specific extraction behind ONE common interface, so that a
         new dataset needs a new extractor, not a new pipeline.
Status   STANDARD.

Every extractor declares its ``FeatureSpec`` -- name, unit, expected sign of
the association with stress, and whether it is eligible to be the PRIMARY
covariate. The expected sign is DOCUMENTATION, never an assumption the
estimator relies on: the frozen eligibility rule accepts either sign and only
requires the sign to MATCH ACROSS EPOCHS (self-correction 26).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

__all__ = ["FeatureSpec", "FeatureExtractor", "registry", "register"]


@dataclass(frozen=True)
class FeatureSpec:
    """Declaration of one extracted feature."""

    name: str
    unit: str
    description: str
    expected_sign_vs_stress: str = "unknown"   # "positive" | "negative" | "unknown"
    primary_eligible: bool = False
    modality: str = "unspecified"


class FeatureExtractor(ABC):
    """Common interface. One implementation per (dataset, modality) pair."""

    spec: FeatureSpec

    @abstractmethod
    def extract(self, raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return a frame with columns (pid, ts, <spec.name>)."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} {self.spec.name}>"


registry: dict[str, FeatureExtractor] = {}


def register(extractor: FeatureExtractor) -> FeatureExtractor:
    """Add an extractor to the global registry, keyed by feature name."""
    registry[extractor.spec.name] = extractor
    return extractor
