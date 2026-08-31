"""LAYER 1 / MODULE 3 -- the structured emotional event.

Purpose  Turn one interaction into an inspectable record with per-field
         provenance, so nothing downstream has to guess where a value came from.
Input    Raw text, optional explicit check-in fields, model output.
Output   ``EmotionalEvent``.
Status   ENGINEERING. The schema is the contribution here, not an algorithm.

WHY PER-FIELD PROVENANCE IS THE POINT

The same field can arrive four different ways, and conflating them is how a
demonstration starts lying. `sleep` might be:

  EXTRACTED     the text said "I slept four hours"
  USER_REPORTED the person set the sleep field on the check-in form
  INFERRED      nothing said it, but this person's pattern suggests it
  UNKNOWN       nobody knows

Only the first two are observations. The third is a guess and must be visibly
a guess; the fourth must never be silently filled. Every field therefore
carries its own ``FieldSource``, and the interface renders it.

NOTHING HERE INFERS ANYTHING. This module records what it is told. The
inference path exists so that a later component can add a field and have it
marked INFERRED, not so that this module can invent one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = ["FieldSource", "Provenanced", "EmotionalEvent", "new_event_id"]


class FieldSource(str, Enum):
    """Where one field's value came from. There is no 'probably' state."""

    EXTRACTED = "extracted"          # identified in the text
    USER_REPORTED = "user_reported"  # supplied explicitly through a field
    MODEL = "model"                  # produced by a trained model
    INFERRED = "inferred"            # estimated from this person's history
    CORRECTED = "corrected"          # the user overrode an earlier value
    UNKNOWN = "unknown"              # not known; NEVER a filled-in default


@dataclass(frozen=True)
class Provenanced:
    """One value plus how it was obtained and how sure we are of it."""

    value: Any
    source: FieldSource = FieldSource.UNKNOWN
    confidence: float | None = None
    evidence: str = ""               # the span or field that produced it

    @property
    def known(self) -> bool:
        return self.source is not FieldSource.UNKNOWN and self.value is not None

    def to_dict(self) -> dict:
        return {"value": self.value, "source": self.source.value,
                "confidence": self.confidence, "evidence": self.evidence}

    @classmethod
    def unknown(cls) -> "Provenanced":
        return cls(value=None, source=FieldSource.UNKNOWN)


def new_event_id(pid: str, ts: str, text: str) -> str:
    """Stable id from content, so re-ingesting the same event cannot duplicate it."""
    h = hashlib.sha256(f"{pid}|{ts}|{text}".encode()).hexdigest()
    return f"ev_{h[:16]}"


@dataclass(frozen=True)
class EmotionalEvent:
    """One context-rich emotional event.

    The fields a hyperedge is later built from are the ``Provenanced`` ones.
    ``raw_text`` is kept so a reviewer can always see what was actually said.
    """

    event_id: str
    person_id: str
    timestamp: str                       # ISO-8601
    raw_text: str = ""

    emotion: Provenanced = field(default_factory=Provenanced.unknown)
    event: Provenanced = field(default_factory=Provenanced.unknown)
    time_context: Provenanced = field(default_factory=Provenanced.unknown)
    sleep: Provenanced = field(default_factory=Provenanced.unknown)
    activity: Provenanced = field(default_factory=Provenanced.unknown)
    social: Provenanced = field(default_factory=Provenanced.unknown)
    workload: Provenanced = field(default_factory=Provenanced.unknown)
    location: Provenanced = field(default_factory=Provenanced.unknown)

    model_version: str = ""
    data_status: str = "USER"            # USER | SYNTHETIC_DEMO
    corrections: tuple[str, ...] = ()    # names of fields the user overrode

    #: Fields that participate in the hypergraph and in similarity.
    CONTEXT_FIELDS = ("emotion", "event", "time_context", "sleep",
                      "activity", "social", "workload", "location")

    # ------------------------------------------------------------- access
    def get(self, name: str) -> Provenanced:
        v = getattr(self, name, None)
        return v if isinstance(v, Provenanced) else Provenanced.unknown()

    def known_fields(self) -> dict[str, Provenanced]:
        return {f: self.get(f) for f in self.CONTEXT_FIELDS if self.get(f).known}

    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(f for f in self.CONTEXT_FIELDS if not self.get(f).known)

    # ------------------------------------------------------------ mutation
    def with_correction(self, field_name: str, value: Any,
                        evidence: str = "user correction") -> "EmotionalEvent":
        """Return a copy with one field overridden BY THE USER.

        The override is marked CORRECTED rather than overwriting the original
        source, and the field name is recorded, so a reviewer can always see
        that a human disagreed with the extractor and where.
        """
        if field_name not in self.CONTEXT_FIELDS:
            raise KeyError(f"{field_name!r} is not a correctable field; "
                           f"correctable: {self.CONTEXT_FIELDS}")
        return replace(
            self,
            **{field_name: Provenanced(value=value, source=FieldSource.CORRECTED,
                                       confidence=1.0, evidence=evidence)},
            corrections=tuple(sorted({*self.corrections, field_name})),
        )

    # --------------------------------------------------------------- io
    def to_dict(self) -> dict:
        d = {"event_id": self.event_id, "person_id": self.person_id,
             "timestamp": self.timestamp, "raw_text": self.raw_text,
             "model_version": self.model_version,
             "data_status": self.data_status,
             "corrections": list(self.corrections)}
        for f in self.CONTEXT_FIELDS:
            d[f] = self.get(f).to_dict()
        d["source"] = {f: self.get(f).source.value for f in self.CONTEXT_FIELDS}
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalEvent":
        kw: dict[str, Any] = {}
        for f in cls.CONTEXT_FIELDS:
            raw = d.get(f)
            if isinstance(raw, dict):
                kw[f] = Provenanced(
                    value=raw.get("value"),
                    source=FieldSource(raw.get("source", "unknown")),
                    confidence=raw.get("confidence"),
                    evidence=raw.get("evidence", ""))
        return cls(event_id=d["event_id"], person_id=d["person_id"],
                   timestamp=d["timestamp"], raw_text=d.get("raw_text", ""),
                   model_version=d.get("model_version", ""),
                   data_status=d.get("data_status", "USER"),
                   corrections=tuple(d.get("corrections", ())), **kw)
