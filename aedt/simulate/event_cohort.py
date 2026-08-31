"""SYNTHETIC event cohort for the Layer 1 model experiments.

Purpose  Generate many people's emotional-event histories from a KNOWN rule,
         so a model comparison has ground truth to be right or wrong about.
Input    A seed and cohort size.
Output   An ``EventHypergraph`` plus the rule that generated it.
Status   SYNTHETIC. Never presented as data about people.

WHY THE RULE IS HIGHER-ORDER ON PURPOSE

The generating rule is deliberately CONJUNCTIVE: the emotion depends on
combinations, not on any field alone.

    poor sleep AND high workload            -> stress
    poor sleep AND hospital appointment     -> anxiety
    good sleep AND social event             -> joy
    isolated  AND low activity              -> sadness

Each of those conditions is individually uninformative: "poor sleep" appears in
both the stress rule and the anxiety rule, "good sleep" in joy and in the
default. A model that can only weigh fields independently is therefore
structurally limited on this data, and one that can represent the conjunction
is not.

This is a fair test of the hypothesis "higher-order structure helps", and it is
also a friendly one — the data was built so that the hypothesis COULD be true.
That is stated in the experiment's protocol block, because a favourable
generator is exactly the kind of detail that turns a real result into a
misleading one when it goes unmentioned.

LABEL NOISE. ``noise`` of the labels are replaced by a uniformly random
emotion, so no model can reach 1.0 and the ceiling is visible.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from ..emotion.events import EmotionalEvent, FieldSource, Provenanced, new_event_id
from ..hypergraph.event_graph import EventHypergraph

__all__ = ["build_cohort_events", "build_cohort_hypergraph", "RULE"]

SLEEP = ("poor", "good")
WORKLOAD = ("high", "low")
SOCIAL = ("with others", "isolated")
ACTIVITY = ("high", "low")
EVENTS = ("examination", "deadline", "hospital appointment", "social event",
          "work", "study", "family")

#: The generating rule, in evaluation order. Conjunctive by construction.
RULE = (
    (("sleep", "poor"), ("workload", "high"), "stress"),
    (("sleep", "poor"), ("event", "hospital appointment"), "anxiety"),
    (("sleep", "good"), ("event", "social event"), "joy"),
    (("social", "isolated"), ("activity", "low"), "sadness"),
)
DEFAULT_EMOTION = "neutral"


def _emotion_for(fields: dict, rng, noise: float) -> str:
    for (fa, va), (fb, vb), emo in RULE:
        if fields.get(fa) == va and fields.get(fb) == vb:
            label = emo
            break
    else:
        label = DEFAULT_EMOTION
    if rng.random() < noise:
        pool = ["stress", "anxiety", "joy", "sadness", "neutral"]
        label = pool[rng.integers(len(pool))]
    return label


def build_cohort_events(*, n_people: int = 60, n_events: int = 40,
                        seed: int = 20260828, noise: float = 0.15
                        ) -> list[EmotionalEvent]:
    """One synthetic cohort of provenance-stamped events."""
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out: list[EmotionalEvent] = []

    for p in range(n_people):
        pid = f"Sim_{p:03d}"
        for i in range(n_events):
            fields = {
                "sleep": SLEEP[rng.integers(2)],
                "workload": WORKLOAD[rng.integers(2)],
                "social": SOCIAL[rng.integers(2)],
                "activity": ACTIVITY[rng.integers(2)],
                "event": EVENTS[rng.integers(len(EVENTS))],
            }
            emo = _emotion_for(fields, rng, noise)
            ts = (base + timedelta(days=int(i))).isoformat(timespec="seconds")
            prov = {k: Provenanced(value=v, source=FieldSource.USER_REPORTED,
                                   confidence=1.0, evidence="simulated")
                    for k, v in fields.items()}
            out.append(EmotionalEvent(
                event_id=new_event_id(pid, ts, f"sim{i}"),
                person_id=pid, timestamp=ts, raw_text="",
                emotion=Provenanced(value=emo, source=FieldSource.USER_REPORTED,
                                    confidence=1.0, evidence="simulated"),
                data_status="SYNTHETIC_DEMO", model_version="simulator",
                **prov))
    return out


def build_cohort_hypergraph(*, n_people: int = 60, n_events: int = 40,
                            seed: int = 20260828, noise: float = 0.15
                            ) -> tuple[EventHypergraph, dict]:
    """The cohort as ONE hypergraph, plus the rule that made it."""
    events = build_cohort_events(n_people=n_people, n_events=n_events,
                                 seed=seed, noise=noise)
    g = EventHypergraph(person_id=f"cohort_{n_people}")
    for ev in events:
        g.add_event(ev)
    return g, {"rule": [(a, b, e) for a, b, e in RULE],
               "default": DEFAULT_EMOTION, "noise": noise,
               "n_people": n_people, "n_events_per_person": n_events}
