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

    # Precedence, strongest evidence first:
    #   1. a field the person filled in            -> USER_REPORTED
    #   2. an explicit "I am ..." in the text       -> EXTRACTED (a self-report)
    #   3. the Transformer's inference              -> MODEL
    # Step 2 exists because GoEmotions has no "stress" label at all, so the
    # model cannot express the construct even when the sentence states it.
    uf = user_fields or {}
    stated = self_reported_emotion(text)
    if uf.get("emotion"):
        emotion = Provenanced(value=uf["emotion"], source=FieldSource.USER_REPORTED,
                              confidence=1.0, evidence="check-in field")
    elif stated:
        label, span = stated
        emotion = Provenanced(value=label, source=FieldSource.EXTRACTED,
                              confidence=0.95,
                              evidence=f'stated in the text: "{span}"')
    else:
        emotion = Provenanced(value=pred.label, source=FieldSource.MODEL,
                              confidence=float(pred.score),
                              evidence=f"{pred.backend}:{pred.raw_label}")

    return EmotionalEvent(
        event_id=new_event_id(person_id, ts, text),
        person_id=person_id, timestamp=ts, raw_text=text,
        emotion=emotion,
        event=ctx.get("event", Provenanced.unknown()),
        time_context=ctx.get("time_context", Provenanced.unknown()),
        sleep=ctx.get("sleep", Provenanced.unknown()),
        activity=ctx.get("activity", Provenanced.unknown()),
        social=ctx.get("social", Provenanced.unknown()),
        workload=ctx.get("workload", Provenanced.unknown()),
        location=ctx.get("location", Provenanced.unknown()),
        model_version=pred.model, data_status=data_status)
