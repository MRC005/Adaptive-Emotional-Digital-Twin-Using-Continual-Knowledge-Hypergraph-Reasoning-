"""LAYER 1 / MODULE 4 -- the event Knowledge Hypergraph.

Purpose  Represent one context-rich emotional event as ONE higher-order
         relation over typed entities, and accumulate those relations into a
         personal knowledge structure that can be queried.
Input    ``EmotionalEvent`` objects.
Output   ``EventHypergraph``: typed vertices, hyperedges, an incidence matrix.
Status   ENGINEERING. The n-ary representation is standard; the contribution
         is that it is built from real stored events and is inspectable.

WHY THIS IS A HYPERGRAPH AND NOT A GRAPH

    e1 = {User_001, hospital appointment, poor sleep, tomorrow, anxiety}

is ONE relation of arity 5. Decomposing it into pairwise edges

    User_001 -- anxiety
    anxiety  -- hospital appointment
    anxiety  -- poor sleep

loses the fact that these co-occurred *in the same episode*. The pairwise
version cannot distinguish "poor sleep and an appointment happened together
and anxiety followed" from "poor sleep happened in March, an appointment in
July, anxiety in both". The conjunction is the object of interest, and a
hyperedge is what represents it exactly.

THIS IS NOT A DRAWING. The structure here is built from stored
``EmotionalEvent`` records, carries an incidence matrix, and is what the HGNN
in ``aedt/models/hgnn.py`` consumes. The interface renders it from this object;
it does not lay out a decorative diagram beside it.

RELATION TO ``structure.py``. That module builds hyperedges over *discretised
sensor bins* for the Review 2 ablation, on cohort data. This one builds them
over *event entities* for one person. Same mathematics, different vertex
universe, and they are deliberately not merged: one is a population analysis
object and the other is a personal knowledge object.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..emotion.events import EmotionalEvent

__all__ = ["EntityType", "Vertex", "EventHyperedge", "EventHypergraph",
           "build_event_hypergraph"]

#: field name -> vertex type. Every context field becomes a typed entity, so
#: "poor" as sleep is a different vertex from "poor" as anything else.
FIELD_TYPES: dict[str, str] = {
    "emotion": "Emotion", "event": "Event", "time_context": "Time",
    "sleep": "Sleep", "activity": "Activity", "social": "Social",
    "workload": "Workload", "location": "Location",
}


class EntityType:
    PERSON = "Person"
    EMOTION = "Emotion"
    EVENT = "Event"
    TIME = "Time"
    SLEEP = "Sleep"
    ACTIVITY = "Activity"
    SOCIAL = "Social"
    WORKLOAD = "Workload"
    LOCATION = "Location"


@dataclass(frozen=True)
class Vertex:
    """A typed entity. ``key`` is what the incidence matrix is indexed by."""

    key: str            # "Sleep=poor"
    type: str           # "Sleep"
    value: str          # "poor"

    @property
    def label(self) -> str:
        return f"{self.type}: {self.value}"


@dataclass(frozen=True)
class EventHyperedge:
    """One episode as one n-ary relation."""

    edge_id: str
    vertices: tuple[str, ...]      # vertex keys
    timestamp: str
    person_id: str
    raw_text: str = ""
    data_status: str = "USER"
    sources: dict = field(default_factory=dict)   # field -> provenance source

    @property
    def arity(self) -> int:
        return len(self.vertices)

    def to_dict(self) -> dict:
        return {"edge_id": self.edge_id, "vertices": list(self.vertices),
                "arity": self.arity, "timestamp": self.timestamp,
                "person_id": self.person_id, "raw_text": self.raw_text,
                "data_status": self.data_status, "sources": self.sources}


@dataclass
class EventHypergraph:
    """Typed vertices + hyperedges + incidence, for one person."""

    person_id: str
    vertices: dict[str, Vertex] = field(default_factory=dict)
    edges: list[EventHyperedge] = field(default_factory=list)

    # ------------------------------------------------------------ building
    def add_event(self, ev: EmotionalEvent) -> EventHyperedge:
        """Add one event as one hyperedge. UNKNOWN fields contribute nothing.

        A missing field must not become a vertex: "sleep unknown" is not a
        state of the world, and letting it join edges would make two people
        who both failed to mention sleep look similar because of it.
        """
        keys: list[str] = []
        person_key = f"{EntityType.PERSON}={ev.person_id}"
        self.vertices.setdefault(person_key,
                                 Vertex(person_key, EntityType.PERSON, ev.person_id))
        keys.append(person_key)

        sources = {}
        for fname, vtype in FIELD_TYPES.items():
            p = ev.get(fname)
            if not p.known:
                continue
            key = f"{vtype}={p.value}"
            self.vertices.setdefault(key, Vertex(key, vtype, str(p.value)))
            keys.append(key)
            sources[fname] = p.source.value

        edge = EventHyperedge(
            edge_id=ev.event_id, vertices=tuple(keys), timestamp=ev.timestamp,
            person_id=ev.person_id, raw_text=ev.raw_text,
            data_status=ev.data_status, sources=sources)
        self.edges.append(edge)
        self.edges.sort(key=lambda e: e.timestamp)
        return edge

    # ------------------------------------------------------------ structure
    @property
    def vertex_keys(self) -> list[str]:
        return sorted(self.vertices)

    def incidence(self) -> np.ndarray:
        """H with H[v, e] = 1 when vertex v is in edge e. |V| x |E|."""
        vk = self.vertex_keys
        idx = {k: i for i, k in enumerate(vk)}
        H = np.zeros((len(vk), len(self.edges)), dtype=float)
        for j, e in enumerate(self.edges):
            for k in e.vertices:
                if k in idx:
                    H[idx[k], j] = 1.0
        return H

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """(vertex degree, edge degree) -- the two diagonals HGNN needs."""
        H = self.incidence()
        return H.sum(axis=1), H.sum(axis=0)

    # -------------------------------------------------------------- queries
    def neighbours(self, vertex_key: str) -> list[str]:
        """Vertices sharing at least one hyperedge with this one."""
        out: set[str] = set()
        for e in self.edges:
            if vertex_key in e.vertices:
                out.update(e.vertices)
        out.discard(vertex_key)
        return sorted(out)

    def edges_containing(self, *vertex_keys: str) -> list[EventHyperedge]:
        """Edges containing ALL the given vertices -- a conjunctive query."""
        want = set(vertex_keys)
        return [e for e in self.edges if want.issubset(set(e.vertices))]

    def co_occurrence(self, min_count: int = 2) -> list[tuple[str, str, int]]:
        """Vertex pairs that recur together, commonest first."""
        c: Counter = Counter()
        for e in self.edges:
            vs = sorted(v for v in e.vertices if not v.startswith("Person="))
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    c[(vs[i], vs[j])] += 1
        return [(a, b, n) for (a, b), n in c.most_common() if n >= min_count]

    def vertices_by_type(self, vtype: str) -> list[Vertex]:
        return [v for v in self.vertices.values() if v.type == vtype]

    def summary(self) -> dict:
        vd, ed = self.degrees()
        by_type: Counter = Counter(v.type for v in self.vertices.values())
        return {
            "person_id": self.person_id,
            "n_vertices": len(self.vertices), "n_edges": len(self.edges),
            "mean_arity": float(ed.mean()) if len(ed) else 0.0,
            "max_arity": int(ed.max()) if len(ed) else 0,
            "vertices_by_type": dict(by_type),
            "recurring_pairs": len(self.co_occurrence(min_count=2)),
            "synthetic_edges": sum(1 for e in self.edges
                                   if e.data_status != "USER"),
        }

    def to_dict(self) -> dict:
        return {"person_id": self.person_id,
                "vertices": [{"key": v.key, "type": v.type, "value": v.value}
                             for v in self.vertices.values()],
                "edges": [e.to_dict() for e in self.edges],
                "summary": self.summary()}


def build_event_hypergraph(person_id: str,
                           events: Iterable[EmotionalEvent]) -> EventHypergraph:
    g = EventHypergraph(person_id=person_id)
    for ev in sorted(events, key=lambda e: e.timestamp):
        g.add_event(ev)
    return g
