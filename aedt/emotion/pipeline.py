"""LAYER 1 -- text in, structured emotional event out.

One function, so the ordering of the perception steps lives in one place:
detect emotion, extract context, then let explicit user fields override both.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .context import extract_context
from .detect import EmotionDetector, default_detector, self_reported_emotion
from .events import EmotionalEvent, FieldSource, Provenanced, new_event_id

__all__ = ["build_event"]


def build_event(text: str, *, person_id: str, timestamp: str | None = None,
                user_fields: dict | None = None,
                detector: EmotionDetector | None = None,
                data_status: str = "USER") -> EmotionalEvent:
    """Build one provenance-carrying event from an interaction."""
    ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
    det = detector or default_detector()

    pred = det.predict(text)
    ctx = extract_context(text, user_fields=user_fields)

    # The MODEL is the emotion. An explicit statement is reported BESIDE it,
    # never instead of it.
    #
    # The old precedence let a regex silently overwrite the Transformer, and
    # its failures were not hypothetical: "I am not sure I am good enough" was
    # reported as joy, and "...which I am grateful for, but I am anxious" as
    # gratitude. A keyword cannot outrank a classifier without at minimum
    # negation, scope and temporal handling -- and even with those it should
    # inform the reader, not replace the model.
    #
    # A field the person filled in explicitly still wins: that is a genuine
    # self-report through a structured control, not a substring match.
    uf = user_fields or {}
    stated = self_reported_emotion(text)

    if uf.get("emotion"):
        emotion = Provenanced(value=uf["emotion"], source=FieldSource.USER_REPORTED,
                              confidence=1.0, evidence="check-in field")
    else:
        emotion = Provenanced(value=pred.label, source=FieldSource.MODEL,
                              confidence=float(pred.score),
                              evidence=f"{pred.backend}:{pred.raw_label}")

    stated_emotion = (
        Provenanced(value=stated[0], source=FieldSource.EXTRACTED, confidence=0.9,
                    evidence=f'stated in the text: "{stated[1]}"')
        if stated else Provenanced.unknown())

    return EmotionalEvent(
        event_id=new_event_id(person_id, ts, text),
        person_id=person_id, timestamp=ts, raw_text=text,
        emotion=emotion, stated_emotion=stated_emotion,
        event=ctx.get("event", Provenanced.unknown()),
        time_context=ctx.get("time_context", Provenanced.unknown()),
        sleep=ctx.get("sleep", Provenanced.unknown()),
        activity=ctx.get("activity", Provenanced.unknown()),
        social=ctx.get("social", Provenanced.unknown()),
        workload=ctx.get("workload", Provenanced.unknown()),
        location=ctx.get("location", Provenanced.unknown()),
        model_version=pred.model, data_status=data_status)
