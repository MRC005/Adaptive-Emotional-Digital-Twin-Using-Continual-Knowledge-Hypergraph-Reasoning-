"""MODULE 7 -- CONTINUAL KNOWLEDGE REPRESENTATION.

Purpose  Preserve and update participant-specific knowledge longitudinally.
Input    Observations, fits, context states, audit verdicts.
Output   An APPEND-ONLY, provenance-carrying, temporally inspectable store.
Algorithm Append-only node log with supersession links. Nothing is overwritten.
Status   ENGINEERING INTEGRATION -- a core engineering component, and
         explicitly NOT a novelty claim.

WHAT UPDATES -- exactly four things, and nothing else (ROUND-17 §L):
  1. personalised parameters (thresholds, slope)
  2. longitudinal state history (reports, fitted latent positions)
  3. context relationships (hyperedge occupancy and effects)
  4. uncertainty and audit state

THIS FILE IS DATA MEMORY, NOT CONTINUAL LEARNING. No model parameter is
touched here: appending knowledge moves nothing. That distinction is the point,
and conflating the two is the specific overclaim this project guards against.

Continual learning of model PARAMETERS does now exist, in ``aedt/continual/ewc.py``
(Layer 1, offline research pipeline), where an EWC penalty protects earlier
tasks while later ones are learned. It is deliberately separate from this store:
one records what happened, the other changes what a model believes. Ablation 7
still tests whether the rolling update here buys anything over a single
two-epoch fit, and if it does not, we say so.

CAUSALITY. ``append`` refuses a node whose ``valid_from`` precedes the store's
current time. Knowledge may only be added going forward, so a later fit can
never be back-dated into a window it did not inform.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DecisionRequired
from ..schemas import KnowledgeNode, KnowledgeRelation, Serialisable

log = logging.getLogger(__name__)

__all__ = ["ContinualKnowledgeStore", "KNOWLEDGE_KINDS"]

KNOWLEDGE_KINDS = (
    "personalised_parameters",   # (1) thresholds, slope
    "state_history",             # (2) reports, fitted latent positions
    "context_relationship",      # (3) hyperedge occupancy and effects
    "uncertainty_audit",         # (4) uncertainty and audit state
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ContinualKnowledgeStore(Serialisable):
    """Append-only per-participant knowledge, with provenance and time travel."""

    pid: str
    nodes: list[KnowledgeNode] = field(default_factory=list)
    relations: list[KnowledgeRelation] = field(default_factory=list)
    current_time: pd.Timestamp | None = None

    # ------------------------------------------------------------- writing
    def append(self, kind: str, payload: dict[str, Any], *,
               valid_from: pd.Timestamp, provenance: str,
               supersedes: str | None = None) -> KnowledgeNode:
        """Add one knowledge node. NEVER overwrites; may only supersede."""
        if kind not in KNOWLEDGE_KINDS:
            raise DecisionRequired(
                f"Unknown knowledge kind {kind!r}. Exactly four things update "
                f"(ROUND-17 §L): {KNOWLEDGE_KINDS}. Adding a fifth is a scope "
                "change and must be recorded as a decision.")
        valid_from = pd.Timestamp(valid_from)
        if self.current_time is not None and valid_from < self.current_time:
            raise DecisionRequired(
                f"Causal violation in the knowledge store for {self.pid}: a "
                f"node valid from {valid_from} would be inserted behind the "
                f"store's current time {self.current_time}. Knowledge may only "
                "be appended going forward.")
        digest = hashlib.sha256(
            json.dumps({"pid": self.pid, "kind": kind, "n": len(self.nodes),
                        "payload": str(payload)},
                       sort_keys=True).encode()).hexdigest()[:12]
        node = KnowledgeNode(
            node_id=f"{self.pid}:{kind}:{len(self.nodes):04d}:{digest}",
            pid=self.pid, kind=kind, created_at=_now(), valid_from=valid_from,
            payload=payload, provenance=provenance)
        self.nodes.append(node)
        self.current_time = valid_from
        if supersedes is not None:
            if not any(n.node_id == supersedes for n in self.nodes[:-1]):
                raise DecisionRequired(
                    f"Cannot supersede unknown node {supersedes!r}.")
            self.relations.append(KnowledgeRelation(
                relation_id=f"rel:{len(self.relations):04d}",
                src=node.node_id, dst=supersedes, kind="supersedes",
                created_at=_now()))
            # supersession is recorded as an EDGE; the superseded node's
            # content is left intact so history stays inspectable
            idx = next(i for i, n in enumerate(self.nodes)
                       if n.node_id == supersedes)
            old = self.nodes[idx]
            self.nodes[idx] = KnowledgeNode(
                node_id=old.node_id, pid=old.pid, kind=old.kind,
                created_at=old.created_at, valid_from=old.valid_from,
                payload=old.payload, provenance=old.provenance,
                superseded_by=node.node_id)
        log.debug("knowledge append pid=%s kind=%s node=%s", self.pid, kind,
                  node.node_id)
        return node

    # ------------------------------------------------------------- reading
    def as_of(self, when: pd.Timestamp) -> list[KnowledgeNode]:
        """Temporal inspection: what did the store know at time ``when``?"""
        when = pd.Timestamp(when)
        return [n for n in self.nodes if n.valid_from <= when]

    def latest(self, kind: str, *, when: pd.Timestamp | None = None
               ) -> KnowledgeNode | None:
        """Most recent non-superseded node of ``kind``, optionally as of a time."""
        pool = self.nodes if when is None else self.as_of(when)
        live = [n for n in pool if n.kind == kind and n.superseded_by is None]
        return live[-1] if live else None

    def history(self, kind: str | None = None) -> list[KnowledgeNode]:
        return [n for n in self.nodes if kind is None or n.kind == kind]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "node_id": n.node_id, "kind": n.kind,
            "valid_from": n.valid_from, "created_at": n.created_at,
            "provenance": n.provenance, "superseded_by": n.superseded_by,
            "payload": json.dumps(n.payload, default=str)[:400],
        } for n in self.nodes])

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ContinualKnowledgeStore":
        d = json.loads(Path(path).read_text())
        store = cls(pid=d["pid"])
        store.nodes = [KnowledgeNode(
            node_id=n["node_id"], pid=n["pid"], kind=n["kind"],
            created_at=n["created_at"], valid_from=pd.Timestamp(n["valid_from"]),
            payload=n["payload"], provenance=n["provenance"],
            superseded_by=n.get("superseded_by")) for n in d["nodes"]]
        store.relations = [KnowledgeRelation(**r) for r in d["relations"]]
        store.current_time = (pd.Timestamp(d["current_time"])
                              if d.get("current_time") else None)
        return store
