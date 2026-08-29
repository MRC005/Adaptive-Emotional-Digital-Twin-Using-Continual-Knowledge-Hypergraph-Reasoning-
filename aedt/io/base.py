"""MODULE 1 -- DATA SOURCES AND INGESTION (adapter interface).

Purpose  Load actual dataset files through dataset-specific adapters.
Input    Raw dataset files on disk.
Output   Canonical participant-aware records (the LongFrame) + a DatasetAudit.
Algorithm Dataset-specific parser -> canonical schema; halt on any surprise.
Status   STANDARD.

TWO OPERATIONS, DELIBERATELY SEPARATE:

  ``audit(root)``  ALWAYS safe to call. Reports what is and is not present.
                   Never raises for absent files; it reports their absence.
  ``load(root)``   Requires the files. Raises ``RealDataUnavailable`` when they
                   are missing and ``DecisionRequired`` when they are present
                   but disagree with the specification.

A dataset's SCIENTIFIC ROLE is declared on the adapter and is not negotiable at
call time. A benchmark dataset cannot be promoted to primary longitudinal
validation by passing a flag; ``supports_primary_analysis`` has to return True,
and it only does so when the audit establishes repeated per-participant
observations with usable repeated self-report.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import RealDataUnavailable
from ..schemas import DatasetAudit

log = logging.getLogger(__name__)

__all__ = ["DatasetAdapter", "LoadResult", "ADAPTERS", "register_adapter",
           "get_adapter"]


@dataclass
class LoadResult:
    """What a successful load returns."""

    frame: pd.DataFrame
    n_categories: int
    sensor: str
    data_status: DataStatus
    provenance: dict = field(default_factory=dict)
    audit: DatasetAudit | None = None


class DatasetAdapter(ABC):
    """One adapter per dataset. Subclasses declare their scientific role."""

    name: str = "unnamed"
    role: DatasetRole = DatasetRole.ROBUSTNESS_OPTIONAL
    primary_sensor: str = ""
    report_variable: str = ""
    acquisition_instructions: str = ""
    #: Datasets whose structure cannot support a within-person epoch ratio
    #: must leave this False. It is a property of the DATA, not a preference.
    can_support_longitudinal_estimand: bool = False

    # ------------------------------------------------------------- helpers
    @classmethod
    def from_config(cls, cfg) -> "DatasetAdapter":
        """Build an adapter configured by ``cfg``.

        The default ignores the config because most adapters are parameterised
        entirely by the files they read. Adapters with knobs (the simulator,
        RELAX's column names) override this so that every knob lives in a
        config file rather than in code.
        """
        return cls()

    def resolve_root(self, root: str | Path | None) -> Path | None:
        if root is None:
            return None
        p = Path(root).expanduser()
        return p if p.exists() else None

    def files_present(self, root: str | Path | None) -> bool:
        """Cheap existence check used by the demo's fail-safe path."""
        p = self.resolve_root(root)
        return p is not None and bool(self.locate(p))

    # -------------------------------------------------------------- contract
    @abstractmethod
    def locate(self, root: Path) -> dict[str, list[Path]]:
        """Return the files this adapter recognises under ``root``.

        Must not raise for a missing directory -- return an empty mapping.
        """

    @abstractmethod
    def audit(self, root: str | Path | None) -> DatasetAudit:
        """Produce the structured audit report. ALWAYS safe to call."""

    @abstractmethod
    def load(self, root: str | Path | None) -> LoadResult:
        """Parse into the canonical LongFrame, or raise."""

    # ------------------------------------------------------------ defaults
    def require_files(self, root: str | Path | None) -> Path:
        """Raise ``RealDataUnavailable`` unless the files are actually there."""
        p = self.resolve_root(root)
        if p is None:
            raise RealDataUnavailable(
                f"REAL DATA UNAVAILABLE - {self.name.upper()}: no directory at "
                f"{root!r}.\n{self.acquisition_instructions}")
        found = self.locate(p)
        if not any(found.values()):
            raise RealDataUnavailable(
                f"REAL DATA UNAVAILABLE - {self.name.upper()}: {p} exists but "
                f"contains none of the expected files.\n"
                f"{self.acquisition_instructions}")
        return p

    def unavailable_audit(self, root: str | Path | None, reason: str
                          ) -> DatasetAudit:
        """The audit emitted when the files are not present. Never guesses."""
        return DatasetAudit(
            dataset_name=self.name, role=self.role,
            data_status=DataStatus.PLANNED,
            source_status=reason, local_files_available=False,
            root_path=str(root) if root else None,
            self_report_variables=(self.report_variable,)
            if self.report_variable else (),
            eligible_for_primary_analysis=None,
            eligible_for_benchmark_analysis=None,
            acquisition_instructions=self.acquisition_instructions,
            exclusion_reasons=(reason,),
            notes=("No file was opened. Every field left None is UNKNOWN, not "
                   "zero and not assumed.",))


ADAPTERS: dict[str, DatasetAdapter] = {}


def register_adapter(adapter: DatasetAdapter) -> DatasetAdapter:
    ADAPTERS[adapter.name] = adapter
    return adapter


def get_adapter(name: str) -> DatasetAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
