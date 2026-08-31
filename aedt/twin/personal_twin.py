"""LAYER 1 / MODULE 7 -- the personal Adaptive Emotional Digital Twin.

Purpose  Hold one person's emotional history, retrieve comparable past
         episodes for a new one, and state a pattern only when the history
         actually supports it.
Input    ``EmotionalEvent`` objects, in time order.
Output   A profile, retrieved episodes with an explanation, and either a
         pattern statement or an explicit refusal to make one.
Status   ENGINEERING. Retrieval is exact and explainable by design.

WHY RETRIEVAL IS A WEIGHTED FIELD MATCH AND NOT AN EMBEDDING

The user must be told *why* two episodes were called similar. A field-level
match produces "same sleep, same event category, both weekday evenings", which
a person can check and disagree with. A cosine distance in an embedding space
produces a number nobody can audit. The higher-order structure is preserved by
requiring the conjunction to match, not a sum of independent similarities:
matching on sleep AND event scores far above matching on either alone.

THE EVIDENCE FLOOR, AND WHY IT IS NOT NEGOTIABLE

``MIN_EPISODES_FOR_PATTERN`` episodes must share the context before anything
is said about a tendency, and the statement always carries the counts. Below
that the twin says it is still learning. A demonstration is not a reason to
lower it: a twin that guesses confidently from two episodes is the failure
mode this whole project is written against.

WHAT IS NEVER CLAIMED. Causation. The wording throughout is "accompanied",
"was followed by", "in N of M similar episodes". Not "caused", not "will".
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..emotion.events import EmotionalEvent, FieldSource

__all__ = ["PersonalEmotionalTwin", "SimilarEpisode", "PatternInsight",
           "MIN_EPISODES_FOR_PATTERN", "FIELD_WEIGHTS"]

#: Episodes sharing the context before a tendency may be stated at all.
MIN_EPISODES_FOR_PATTERN = 3

#: How much each matching field contributes to similarity. Event category and
#: sleep dominate because they are the fields users actually act on; time of
#: day is weak because "evening" matches almost everything.
FIELD_WEIGHTS: dict[str, float] = {
    "event": 3.0, "sleep": 2.5, "workload": 2.0, "social": 1.5,
    "activity": 1.0, "location": 1.0, "time_context": 0.5,
}


@dataclass(frozen=True)
class SimilarEpisode:
    """One retrieved past episode, with the reason it was retrieved."""

    event_id: str
    timestamp: str
    emotion: str | None
    score: float
    matched_fields: tuple[tuple[str, str], ...]   # (field, shared value)
    raw_text: str = ""

    @property
    def explanation(self) -> str:
        if not self.matched_fields:
            return "no fields in common"
        return "same " + ", ".join(f"{f} ({v})" for f, v in self.matched_fields)

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "timestamp": self.timestamp,
                "emotion": self.emotion, "score": round(self.score, 3),
                "matched_fields": [list(m) for m in self.matched_fields],
                "explanation": self.explanation, "raw_text": self.raw_text}


@dataclass(frozen=True)
class PatternInsight:
    """What the history supports -- possibly nothing."""

    sufficient: bool
    statement: str
    n_similar: int = 0
    n_supporting: int = 0
    dominant_emotion: str | None = None
    matched_context: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"sufficient": self.sufficient, "statement": self.statement,
                "n_similar": self.n_similar, "n_supporting": self.n_supporting,
                "dominant_emotion": self.dominant_emotion,
                "matched_context": list(self.matched_context),
                "caveats": list(self.caveats)}


class PersonalEmotionalTwin:
    """One person's emotional history and what may honestly be said from it."""

    def __init__(self, person_id: str, *, data_status: str = "USER"):
        self.person_id = person_id
        self.data_status = data_status          # USER | SYNTHETIC_DEMO
        self.events: list[EmotionalEvent] = []
        self.update_log: list[dict] = []

    # ------------------------------------------------------------- history
    def add_event(self, ev: EmotionalEvent) -> None:
        """Append one event. History is append-only and time-ordered."""
        if ev.person_id != self.person_id:
            raise ValueError(f"event belongs to {ev.person_id}, not {self.person_id}")
        if any(e.event_id == ev.event_id for e in self.events):
            return                                # idempotent re-ingest
        self.events.append(ev)
        self.events.sort(key=lambda e: e.timestamp)
        self.update_log.append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": "event_added", "event_id": ev.event_id,
            "emotion": ev.get("emotion").value,
            "data_status": ev.data_status})

    def clear(self) -> None:
        self.events.clear()
        self.update_log.append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": "history_cleared"})

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def is_synthetic(self) -> bool:
        return any(e.data_status != "USER" for e in self.events)

    # ----------------------------------------------------------- retrieval
    def similar_episodes(self, ev: EmotionalEvent, *, top_k: int = 5,
                         min_score: float = 1.5) -> list[SimilarEpisode]:
        """Past episodes sharing context with ``ev``, best first.

        The query event itself is excluded. ``min_score`` keeps out episodes
        that share only a weak field: matching on "evening" alone is not
        similarity, and returning it would make the explanation dishonest.
        """
        out: list[SimilarEpisode] = []
        for past in self.events:
            if past.event_id == ev.event_id or past.timestamp > ev.timestamp:
                continue
            matched, score = [], 0.0
            for f, w in FIELD_WEIGHTS.items():
                a, b = ev.get(f), past.get(f)
                if a.known and b.known and a.value == b.value:
                    matched.append((f, str(a.value)))
                    score += w
            # conjunction bonus: two fields matching together is stronger
            # evidence than the same two matching in different episodes
            if len(matched) >= 2:
                score *= 1.0 + 0.15 * (len(matched) - 1)
            if score >= min_score:
                out.append(SimilarEpisode(
                    event_id=past.event_id, timestamp=past.timestamp,
                    emotion=past.get("emotion").value, score=score,
                    matched_fields=tuple(matched), raw_text=past.raw_text))
        out.sort(key=lambda s: -s.score)
        return out[:top_k]

    # ------------------------------------------------------------- pattern
    def pattern_insight(self, ev: EmotionalEvent, *,
                        min_episodes: int = MIN_EPISODES_FOR_PATTERN
                        ) -> PatternInsight:
        """State a tendency, or say plainly that the history cannot support one."""
        similar = self.similar_episodes(ev, top_k=25, min_score=1.5)
        if len(similar) < min_episodes:
            return PatternInsight(
                sufficient=False,
                n_similar=len(similar),
                statement=(
                    f"Your Digital Twin is still learning. It found "
                    f"{len(similar)} comparable past "
                    f"{'episode' if len(similar) == 1 else 'episodes'}, and needs "
                    f"at least {min_episodes} before describing a pattern."),
                caveats=("A pattern stated from one or two episodes would be "
                         "noise, not a pattern.",))

        emotions = [s.emotion for s in similar if s.emotion]
        if not emotions:
            return PatternInsight(
                sufficient=False, n_similar=len(similar),
                statement=("Comparable episodes were found, but none carries a "
                           "recorded emotion, so no tendency can be described."))

        counts = Counter(emotions)
        dominant, n_dom = counts.most_common(1)[0]
        shared = Counter()
        for s in similar:
            for f, v in s.matched_fields:
                shared[f"{f}: {v}"] += 1
        context = tuple(k for k, n in shared.most_common(3) if n >= min_episodes)

        return PatternInsight(
            sufficient=True, n_similar=len(similar), n_supporting=n_dom,
            dominant_emotion=dominant, matched_context=context,
            statement=(
                f"Across {len(similar)} comparable past episodes"
                + (f" ({', '.join(context)})" if context else "")
                + f", the most frequently recorded feeling was "
                  f"{dominant}, in {n_dom} of them."),
            caveats=(
                "This is an association in your own history, not a cause.",
                "It describes what was recorded before, not what will happen.",
                f"Based on {len(similar)} episodes, which is a small sample.",
            ))

    # -------------------------------------------------------------- profile
    def profile(self) -> dict:
        """What the twin currently knows. Every count is derived, not stored."""
        if not self.events:
            return {"person_id": self.person_id, "n_events": 0,
                    "status": "no history yet",
                    "data_status": self.data_status}

        emotions = [e.get("emotion").value for e in self.events
                    if e.get("emotion").known]
        recent = self.events[-1]
        field_values: dict[str, Counter] = defaultdict(Counter)
        for e in self.events:
            for f, p in e.known_fields().items():
                if f != "emotion":
                    field_values[f][str(p.value)] += 1

        recurring = {f: [{"value": v, "n": n} for v, n in c.most_common(3) if n >= 2]
                     for f, c in field_values.items()}
        recurring = {f: v for f, v in recurring.items() if v}

        # co-occurrences that have happened often enough to be worth naming
        pairs: Counter = Counter()
        for e in self.events:
            kf = sorted((f, str(p.value)) for f, p in e.known_fields().items()
                        if f != "emotion")
            for i in range(len(kf)):
                for j in range(i + 1, len(kf)):
                    pairs[(f"{kf[i][0]}: {kf[i][1]}", f"{kf[j][0]}: {kf[j][1]}")] += 1

        return {
            "person_id": self.person_id,
            "data_status": self.data_status,
            "is_synthetic": self.is_synthetic,
            "n_events": len(self.events),
            "first_event": self.events[0].timestamp,
            "last_event": recent.timestamp,
            "recent_emotion": recent.get("emotion").value,
            "recent_text": recent.raw_text,
            "emotion_counts": dict(Counter(emotions).most_common()),
            "recurring_context": recurring,
            "frequent_combinations": [
                {"a": a, "b": b, "n": n} for (a, b), n in pairs.most_common(5) if n >= 2],
            "ready_for_patterns": len(self.events) >= MIN_EPISODES_FOR_PATTERN,
        }

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"person_id": self.person_id, "data_status": self.data_status,
                "events": [e.to_dict() for e in self.events],
                "update_log": self.update_log}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalEmotionalTwin":
        t = cls(d["person_id"], data_status=d.get("data_status", "USER"))
        t.events = [EmotionalEvent.from_dict(e) for e in d.get("events", [])]
        t.events.sort(key=lambda e: e.timestamp)
        t.update_log = list(d.get("update_log", []))
        return t

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PersonalEmotionalTwin":
        return cls.from_dict(json.loads(Path(path).read_text()))
