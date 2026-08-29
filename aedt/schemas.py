"""Canonical typed schemas for every critical scientific object.

Rule (project mandate): no anonymous unstructured dictionaries for critical
scientific objects. Every object below validates itself on construction and
carries provenance -- at minimum a ``DataStatus`` and the dataset it came from.

The frame-level contract is ``LongFrame``: one row per (participant, occasion)
with columns::

    pid   str        participant identifier
    ts    datetime64 occasion timestamp (tz-naive local, or tz-aware)
    day   datetime64 normalised local day
    report int       ordinal self-report, 1..K, SEVERITY-ORDERED
    epoch int        0 or 1, assigned from the participant's OWN span
    <sensor columns> float

``validate_long_frame`` is the single place that contract is enforced.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .constants import DataStatus, DatasetRole
from .errors import DecisionRequired

REQUIRED_LONGFRAME_COLUMNS = ("pid", "ts", "report")


# ---------------------------------------------------------------- helpers
def _jsonable(o: Any) -> Any:
    """Recursively convert to something ``json.dump`` accepts."""
    if isinstance(o, DataStatus) or isinstance(o, DatasetRole):
        return o.value
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    return o


class Serialisable:
    """Mixin: dataclass -> plain JSON-safe dict, and back to JSON text."""

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))  # type: ignore[arg-type]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# ---------------------------------------------------------------- records
@dataclass(frozen=True)
class Participant(Serialisable):
    """One study participant, with the coverage facts the screen needs."""

    pid: str
    dataset: str
    data_status: DataStatus
    n_observations: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    span_days: float | None = None

    def __post_init__(self) -> None:
        if not str(self.pid).strip():
            raise DecisionRequired("Participant constructed with an empty pid.")
        if self.n_observations < 0:
            raise ValueError("n_observations must be non-negative")


@dataclass(frozen=True)
class SelfReport(Serialisable):
    """One ordinal self-report occasion.

    ``severity`` is the remapped 1..K value where larger = more stress.
    ``raw_code`` is the integer as stored in the source file, retained so the
    remap is auditable and reversible. ``raw_label`` is the option TEXT.
    """

    pid: str
    ts: pd.Timestamp
    severity: int
    n_categories: int
    raw_code: int | None = None
    raw_label: str | None = None
    instrument: str = "unspecified"
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if not (1 <= self.severity <= self.n_categories):
            raise DecisionRequired(
                f"Self-report severity {self.severity} outside 1..{self.n_categories} "
                f"for participant {self.pid}. The remap is wrong or the scale "
                f"differs from the specification.")


@dataclass(frozen=True)
class SensorWindow(Serialisable):
    """Features extracted from one causal window ending at ``ts``.

    ``window_start``/``window_end`` are retained so the leakage test can assert
    that ``window_end <= ts`` for every window that feeds a report at ``ts``.
    """

    pid: str
    ts: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    features: Mapping[str, float]
    n_raw_samples: int = 0
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if self.window_end > self.ts:
            raise DecisionRequired(
                f"Future leakage: sensor window for {self.pid} ends at "
                f"{self.window_end} but is aligned to a report at {self.ts}.")
        if self.window_start > self.window_end:
            raise ValueError("window_start after window_end")


@dataclass(frozen=True)
class Observation(Serialisable):
    """A self-report joined to its causally aligned sensor features."""

    pid: str
    ts: pd.Timestamp
    report: int
    n_categories: int
    features: Mapping[str, float]
    epoch: int | None = None
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if self.epoch is not None and self.epoch not in (0, 1):
            raise DecisionRequired(
                f"Epoch must be 0 or 1 (halves of the participant's own span); "
                f"got {self.epoch} for {self.pid}.")


@dataclass(frozen=True)
class EpochDefinition(Serialisable):
    """How one participant's timeline was cut in two.

    Frozen rule (§W): epochs are halves of each participant's OWN span. A
    calendar split or a pooled split would make the estimand incomparable
    across participants with different enrolment windows.
    """

    pid: str
    rule: str
    start: pd.Timestamp
    midpoint: pd.Timestamp
    end: pd.Timestamp
    n_epoch0: int
    n_epoch1: int

    def __post_init__(self) -> None:
        if not (self.start <= self.midpoint <= self.end):
            raise DecisionRequired(
                f"Epoch midpoint for {self.pid} is outside the observed span.")


@dataclass(frozen=True)
class ContextState(Serialisable):
    """The contextual state at one occasion, in all three representations."""

    pid: str
    ts: pd.Timestamp
    continuous: float
    binned: tuple[int, ...] = ()
    hyperedge_key: str | None = None
    feature_names: tuple[str, ...] = ()
    data_status: DataStatus = DataStatus.SYNTHETIC


@dataclass(frozen=True)
class KnowledgeNode(Serialisable):
    """One append-only entry in the continual knowledge store."""

    node_id: str
    pid: str
    kind: str
    created_at: str
    valid_from: pd.Timestamp
    payload: Mapping[str, Any]
    provenance: str
    superseded_by: str | None = None


@dataclass(frozen=True)
class KnowledgeRelation(Serialisable):
    """A binary relation between two knowledge nodes (provenance edges)."""

    relation_id: str
    src: str
    dst: str
    kind: str
    created_at: str


@dataclass(frozen=True)
class Hyperedge(Serialisable):
    """A conjunctive context: several feature-value vertices holding at once.

    ``vertices`` are ``"<feature>=<bin>"`` strings. The edge is exact and
    conjunctive, which is what distinguishes it from a compensatory distance
    in feature-vector space.
    """

    key: str
    vertices: tuple[str, ...]
    pid: str
    n_epoch0: int = 0
    n_epoch1: int = 0
    mean_report_epoch0: float | None = None
    mean_report_epoch1: float | None = None

    @property
    def arity(self) -> int:
        return len(self.vertices)

    @property
    def occupied_both_epochs(self) -> bool:
        return self.n_epoch0 > 0 and self.n_epoch1 > 0


@dataclass(frozen=True)
class EligibilityResult(Serialisable):
    """Outcome of the pre-specified screen for one participant.

    Every exclusion carries its reason. Nothing is dropped silently.
    """

    pid: str
    eligible: bool
    reasons: tuple[str, ...] = ()
    n_epoch0: int = 0
    n_epoch1: int = 0
    categories_epoch0: int = 0
    categories_epoch1: int = 0
    sensor_sd_epoch0: float = float("nan")
    sensor_sd_epoch1: float = float("nan")
    var_ratio: float = float("nan")
    floor_rate_epoch0: float = float("nan")
    floor_rate_epoch1: float = float("nan")
    ceiling_rate_epoch0: float = float("nan")
    ceiling_rate_epoch1: float = float("nan")
    ar1_epoch0: float = float("nan")
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if not self.eligible and not self.reasons:
            raise ValueError("An ineligible participant must carry a reason.")


@dataclass(frozen=True)
class OrdinalFit(Serialisable):
    """One per-person per-epoch ordinal probit fit.

    P(R <= k | x) = Phi(c_k - beta * x), x standardised WITHIN THIS EPOCH.
    """

    pid: str
    epoch: int
    beta: float
    cutpoints: tuple[float, ...]
    n: int
    n_categories: int
    converged: bool
    loglik: float = float("nan")
    reason: str = ""
    standardiser_mean: float = float("nan")
    standardiser_sd: float = float("nan")
    data_status: DataStatus = DataStatus.SYNTHETIC


@dataclass(frozen=True)
class UncertaintyResult(Serialisable):
    """Participant-cluster bootstrap interval. Never resamples observations."""

    method: str
    n_participants: int
    n_resamples: int
    point: float
    ci_low: float
    ci_high: float
    resampling_unit: str = "participant"
    seed: int | None = None
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if self.resampling_unit != "participant":
            raise DecisionRequired(
                "Uncertainty must resample participants. Resampling "
                "observations treats repeated measures as independent "
                "participants and is a bug, not an option.")

    @property
    def excludes_null(self) -> bool:
        """Does the 95% CI exclude rho* = 1 (equivalently log rho* = 0)?"""
        if not np.isfinite(self.ci_low) or not np.isfinite(self.ci_high):
            return False
        return not (self.ci_low <= 1.0 <= self.ci_high)


@dataclass(frozen=True)
class EstimatorResult(Serialisable):
    """The primary scientific output: rho*, the IDENTIFIED estimand.

    ``rho_star`` estimates rho * sqrt((v + sigma_r^2) / (rho^2 v + sigma_r^2)),
    which is >= rho for rho < 1. Therefore ``1 - rho_star`` is a LOWER BOUND on
    the true multiplicative recalibration, and rho itself is NOT identified.
    ``additive_component`` is fixed at None by construction and is never
    populated: b_2 - b_1 is absorbed into the threshold locations (T1(b)).
    """

    estimand: str
    rho_star: float
    log_rho_star: float
    uncertainty: UncertaintyResult | None
    n_participants_used: int
    n_participants_screened: int
    per_participant_rho_star: tuple[float, ...] = ()
    per_participant_pids: tuple[str, ...] = ()
    exclusions: Mapping[str, str] = field(default_factory=dict)
    median_rho_star: float = float("nan")
    diagnostic_status: str = "OK"
    eligibility_status: str = "UNKNOWN"
    context_representation: str = "continuous"
    additive_component: None = None
    additive_component_status: str = "NOT IDENTIFIED"
    data_status: DataStatus = DataStatus.SYNTHETIC

    def __post_init__(self) -> None:
        if self.estimand != "rho_star":
            raise DecisionRequired(
                f"Primary estimand must be 'rho_star', not {self.estimand!r}. "
                "rho itself is not point-identified (Theorem T1(a)).")
        if self.additive_component is not None:
            raise DecisionRequired(
                "The additive component b_2 - b_1 is NOT identified and must "
                "never be reported as estimated (Theorem T1(b)).")

    @property
    def lower_bound_on_recalibration(self) -> float:
        """1 - rho_star: a LOWER bound on the true recalibration magnitude."""
        return 1.0 - self.rho_star


@dataclass(frozen=True)
class PlaceboResult(Serialisable):
    """Contiguous epoch-1 split-half negative control. GATES the primary.

    No response shift can have occurred between two contiguous halves of the
    same epoch, so a rejection means the estimator is detecting something other
    than recalibration. Contiguous halves preserve serial dependence and the
    ordinal structure; shuffling would destroy both.
    """

    n_participants: int
    rho_star: float
    ci_low: float
    ci_high: float
    rejected: bool
    verdict: str
    runnable: bool = True
    data_status: DataStatus = DataStatus.SYNTHETIC

    @property
    def gates_primary(self) -> bool:
        """True when the primary analysis must NOT run."""
        return (not self.runnable) or self.rejected


@dataclass(frozen=True)
class BiasEnvelopeResult(Serialisable):
    """Range of rho* under the allowed assumption violations, under the NULL.

    Assumptions are enumerated in advance; none is chosen to optimise the
    result. The envelope is the 5th-95th percentile of rho* when the truth is
    rho = 1, at the measured data properties.
    """

    scenarios: tuple[str, ...]
    rho_star_by_scenario: Mapping[str, float]
    envelope_low: float
    envelope_high: float
    interpretation: str
    n_replications: int
    data_status: DataStatus = DataStatus.SYNTHETIC


@dataclass(frozen=True)
class TwinState(Serialisable):
    """The Personal Digital Twin: a persistent, per-person model of that
    person's MEASURING INSTRUMENT -- not of their mood.

    Persistence is load-bearing, not decorative: the estimand is a ratio across
    epochs, so without a persisted epoch-1 state there is nothing to compare
    epoch 2 against.

    Every field evolves using information available up to ``current_time``
    only. ``history`` is append-only.
    """

    pid: str
    dataset: str
    data_status: DataStatus
    current_time: str | None = None
    n_observations_seen: int = 0
    feature_history: list[dict] = field(default_factory=list)
    context_state: dict = field(default_factory=dict)
    baseline_state: dict = field(default_factory=dict)
    category_usage: dict = field(default_factory=dict)
    ordinal_state: dict = field(default_factory=dict)
    epoch_info: dict = field(default_factory=dict)
    knowledge_state: dict = field(default_factory=dict)
    hyperedge_occupancy: dict = field(default_factory=dict)
    uncertainty_state: dict = field(default_factory=dict)
    eligibility_status: str = "UNKNOWN"
    audit_flags: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    update_log: list[dict] = field(default_factory=list)
    schema_version: str = "1.0"


@dataclass(frozen=True)
class DatasetAudit(Serialisable):
    """The structured audit report emitted for every dataset, real or not.

    Every field the project mandate enumerates is present. Fields that cannot
    be determined because the files are absent are None, never guessed.
    """

    dataset_name: str
    role: DatasetRole
    data_status: DataStatus
    source_status: str
    local_files_available: bool
    root_path: str | None = None
    files_found: tuple[str, ...] = ()
    participant_count: int | None = None
    observation_count: int | None = None
    sensor_modalities: tuple[str, ...] = ()
    self_report_variables: tuple[str, ...] = ()
    self_report_scale: str | None = None
    stress_labels: tuple[str, ...] = ()
    raw_stored_codes: tuple[int, ...] = ()
    code_to_label_mapping: Mapping[int, str] = field(default_factory=dict)
    code_to_severity_mapping: Mapping[int, int] = field(default_factory=dict)
    timestamps_present: bool | None = None
    timestamp_format: str | None = None
    timezone: str | None = None
    longitudinal_span_days: float | None = None
    observations_per_participant: Mapping[str, int] = field(default_factory=dict)
    median_observations_per_participant: float | None = None
    missingness: Mapping[str, float] = field(default_factory=dict)
    participant_level_coverage: Mapping[str, float] = field(default_factory=dict)
    sensor_report_alignment: str | None = None
    conversation_context_available: bool | None = None
    eligible_for_primary_analysis: bool | None = None
    eligible_for_benchmark_analysis: bool | None = None
    exclusion_reasons: tuple[str, ...] = ()
    acquisition_instructions: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.data_status is DataStatus.REAL and not self.local_files_available:
            raise DecisionRequired(
                f"{self.dataset_name}: audit claims REAL data status but no "
                "local files were found. A REAL stamp requires audited files "
                "on disk.")


@dataclass(frozen=True)
class RunMetadata(Serialisable):
    """Reproducibility record written beside every result folder."""

    run_id: str
    started_at: str
    finished_at: str | None
    dataset: str
    data_status: DataStatus
    config_path: str | None
    config_digest: str
    seed: int
    software_version: str
    python_version: str
    package_versions: Mapping[str, str]
    git_commit: str | None
    command: str
    elapsed_seconds: float | None = None
    output_dir: str | None = None


# ------------------------------------------------------- frame validation
def validate_long_frame(df: pd.DataFrame, sensor: str | None = None,
                        *, require_epoch: bool = False,
                        n_categories: int | None = None) -> pd.DataFrame:
    """Enforce the LongFrame contract. Raises DecisionRequired on any breach.

    This is deliberately the ONLY place the contract is checked, so that a
    loader cannot quietly emit a frame that later code has to guess about.
    """
    if not isinstance(df, pd.DataFrame):
        raise DecisionRequired("LongFrame is not a DataFrame.")
    missing = [c for c in REQUIRED_LONGFRAME_COLUMNS if c not in df.columns]
    if missing:
        raise DecisionRequired(
            f"LongFrame is missing required columns {missing}. "
            f"Columns present: {list(df.columns)}")
    if df.empty:
        raise DecisionRequired("LongFrame is empty; no observations to analyse.")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        raise DecisionRequired(
            f"LongFrame column 'ts' has dtype {df['ts'].dtype}, not datetime. "
            "Timestamps must be parsed in the adapter, not downstream.")
    if df["ts"].isna().any():
        raise DecisionRequired(
            f"{int(df['ts'].isna().sum())} rows have unparseable timestamps. "
            "Inspect the source timestamp field rather than dropping them here.")
    rep = pd.to_numeric(df["report"], errors="coerce")
    if rep.isna().any():
        raise DecisionRequired(
            f"{int(rep.isna().sum())} self-report values are non-numeric after "
            "the remap. The label mapping is wrong or the file differs.")
    if (rep != rep.round()).any():
        raise DecisionRequired(
            "Self-report values are not integers. The ordinal model requires "
            "integer category indices 1..K.")
    if rep.min() < 1:
        raise DecisionRequired(
            f"Self-report minimum is {rep.min()}; categories must be 1-based "
            "severity codes after the remap.")
    if n_categories is not None and rep.max() > n_categories:
        raise DecisionRequired(
            f"Self-report maximum {int(rep.max())} exceeds K={n_categories}.")
    if require_epoch:
        if "epoch" not in df.columns:
            raise DecisionRequired("LongFrame has no 'epoch' column.")
        bad = set(pd.unique(df["epoch"])) - {0, 1}
        if bad:
            raise DecisionRequired(f"Epoch values outside {{0,1}}: {sorted(bad)}")
    if sensor is not None:
        if sensor not in df.columns:
            raise DecisionRequired(
                f"Sensor feature {sensor!r} absent from the LongFrame. "
                f"Columns present: {list(df.columns)}")
        if not pd.api.types.is_numeric_dtype(df[sensor]):
            raise DecisionRequired(f"Sensor feature {sensor!r} is not numeric.")
    return df


def frame_to_participants(df: pd.DataFrame, dataset: str,
                          status: DataStatus) -> list[Participant]:
    """Summarise a validated LongFrame into typed Participant records."""
    out: list[Participant] = []
    for pid, g in df.groupby("pid", sort=True):
        lo, hi = g["ts"].min(), g["ts"].max()
        out.append(Participant(
            pid=str(pid), dataset=dataset, data_status=status,
            n_observations=int(len(g)),
            first_ts=lo.isoformat(), last_ts=hi.isoformat(),
            span_days=float((hi - lo).total_seconds() / 86400.0)))
    return out


__all__ = [n for n in dir() if n[0].isupper()] + [
    "validate_long_frame", "frame_to_participants",
    "REQUIRED_LONGFRAME_COLUMNS"]
