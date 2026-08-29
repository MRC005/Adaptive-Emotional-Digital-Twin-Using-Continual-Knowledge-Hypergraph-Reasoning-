"""MODULE 6 -- PERSONAL DIGITAL TWIN (persistent state).

Purpose  A participant-specific, persistent model of that person's MEASURING
         INSTRUMENT -- explicitly not a model of their mood.
Input    Observations up to the current time.
Output   A serialisable TwinState that survives between runs.
Status   ENGINEERING INTEGRATION. The term "digital twin" is DESCRIPTIVE here,
         not a novelty claim (ROUND-17 §K).

WHY PERSISTENCE IS LOAD-BEARING, NOT DECORATIVE. The estimand is a RATIO ACROSS
EPOCHS. Without persisted per-person state there is no epoch 1 to compare
epoch 2 against, so the object is doing real work.

WHAT IS STORED (ROUND-17 §K): response-category usage per epoch; fitted
thresholds c_k; sensor->report slope beta_e; epoch standardisers; self-report
and sensor history in canonical form; context / hyperedge occupancy; the
recalibration history {rho*_t, CI_t}; audit flags (boundary rate, lag-1
autocorrelation, Var(s) ratio); and the update log.

TEMPORAL DISCIPLINE. ``TwinState.current_time`` only ever moves forward, and
``observe`` refuses an observation dated before it. A twin therefore cannot be
fed the future, by construction rather than by convention.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..constants import DataStatus
from ..errors import DecisionRequired
from ..knowledge.store import ContinualKnowledgeStore
from ..schemas import TwinState

log = logging.getLogger(__name__)

__all__ = ["PersonalDigitalTwin", "new_twin", "load_twin"]


class PersonalDigitalTwin:
    """A TwinState plus its continual-knowledge store and the update rules."""

    def __init__(self, state: TwinState,
                 knowledge: ContinualKnowledgeStore | None = None):
        self.state = state
        self.knowledge = knowledge or ContinualKnowledgeStore(pid=state.pid)

    # --------------------------------------------------------------- basics
    @property
    def pid(self) -> str:
        return self.state.pid

    @property
    def current_time(self) -> pd.Timestamp | None:
        return (pd.Timestamp(self.state.current_time)
                if self.state.current_time else None)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (f"<PersonalDigitalTwin {self.pid} "
                f"n={self.state.n_observations_seen} "
                f"t={self.state.current_time} "
                f"status={self.state.data_status}>")

    # ------------------------------------------------------------- mutation
    def _set(self, **kw) -> None:
        self.state = replace(self.state, **kw)

    def advance_time(self, ts: pd.Timestamp) -> None:
        """Move the twin's clock forward. NEVER backward."""
        ts = pd.Timestamp(ts)
        if self.current_time is not None and ts < self.current_time:
            raise DecisionRequired(
                f"Temporal leakage: twin {self.pid} is at "
                f"{self.state.current_time} and was asked to process an "
                f"observation dated {ts}. TwinState may evolve only using "
                "information available up to the current time.")
        self._set(current_time=ts.isoformat())

    def log_update(self, action: str, detail: dict) -> None:
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "twin_time": self.state.current_time,
                 "action": action, **detail}
        self._set(update_log=[*self.state.update_log, entry])

    def append_history(self, entry: dict) -> None:
        """Append one recalibration record. History is APPEND-ONLY."""
        self._set(history=[*self.state.history, entry])

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "twin_state": self.state.to_dict(),
            "knowledge": self.knowledge.to_dict(),
        }, indent=2))
        log.info("twin %s saved to %s", self.pid, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PersonalDigitalTwin":
        d = json.loads(Path(path).read_text())
        s = d["twin_state"]
        state = TwinState(
            pid=s["pid"], dataset=s["dataset"],
            data_status=DataStatus(s["data_status"]),
            current_time=s.get("current_time"),
            n_observations_seen=s.get("n_observations_seen", 0),
            feature_history=s.get("feature_history", []),
            context_state=s.get("context_state", {}),
            baseline_state=s.get("baseline_state", {}),
            category_usage=s.get("category_usage", {}),
            ordinal_state=s.get("ordinal_state", {}),
            epoch_info=s.get("epoch_info", {}),
            knowledge_state=s.get("knowledge_state", {}),
            hyperedge_occupancy=s.get("hyperedge_occupancy", {}),
            uncertainty_state=s.get("uncertainty_state", {}),
            eligibility_status=s.get("eligibility_status", "UNKNOWN"),
            audit_flags=s.get("audit_flags", {}),
            history=s.get("history", []),
            update_log=s.get("update_log", []),
            schema_version=s.get("schema_version", "1.0"))
        store = ContinualKnowledgeStore(pid=state.pid)
        k = d.get("knowledge", {})
        from ..schemas import KnowledgeNode, KnowledgeRelation
        store.nodes = [KnowledgeNode(
            node_id=n["node_id"], pid=n["pid"], kind=n["kind"],
            created_at=n["created_at"], valid_from=pd.Timestamp(n["valid_from"]),
            payload=n["payload"], provenance=n["provenance"],
            superseded_by=n.get("superseded_by")) for n in k.get("nodes", [])]
        store.relations = [KnowledgeRelation(**r) for r in k.get("relations", [])]
        store.current_time = (pd.Timestamp(k["current_time"])
                              if k.get("current_time") else None)
        return cls(state, store)


def new_twin(pid: str, dataset: str, data_status: DataStatus
             ) -> PersonalDigitalTwin:
    """A twin that knows nothing yet."""
    return PersonalDigitalTwin(TwinState(pid=str(pid), dataset=dataset,
                                         data_status=data_status))


def load_twin(path: str | Path) -> PersonalDigitalTwin:
    return PersonalDigitalTwin.load(path)
