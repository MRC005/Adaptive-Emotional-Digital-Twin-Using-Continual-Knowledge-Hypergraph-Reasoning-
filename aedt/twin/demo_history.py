"""LAYER 1 -- the synthetic demonstration history.

Purpose  Give a reviewer a twin with enough past to actually demonstrate
         retrieval and pattern statements, without using anyone's real data.
Input    A seed.
Output   ``EmotionalEvent`` objects stamped ``SYNTHETIC_DEMO``.
Status   SYNTHETIC. Labelled at every level: on each event, on the twin, on
         each hyperedge, and in the interface.

WHY THIS EXISTS AT ALL

A new twin has no history, so on a fresh install every honest answer is "still
learning". That is correct behaviour and terrible for a demonstration. The fix
is a fictional person, marked as fictional everywhere, not a lowered evidence
floor.

WHAT IS BUILT IN, DELIBERATELY

Three recurring situations, so retrieval has something true to find:

  1. exam/deadline + poor sleep + high workload -> stress, repeatedly
  2. hospital appointment + poor sleep          -> anxiety, repeatedly
  3. social evening + good sleep                -> joy, repeatedly

plus unrelated filler episodes so the retrieval has to discriminate rather
than return everything. The pattern is real *in this fictional history* and
the twin discovers it from the events; nothing is precomputed and no insight
is written into the data.

THE TEXTS ARE HANDWRITTEN, NOT MODEL-GENERATED, so the emotion detector is
genuinely being asked to classify sentences it has never seen, rather than
replaying labels that were baked in.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..emotion.events import EmotionalEvent, FieldSource, Provenanced, new_event_id

__all__ = ["build_demo_history", "DEMO_PERSON_ID", "DEMO_SCRIPT"]

DEMO_PERSON_ID = "Demo_User"

#: (days_ago, text, explicit check-in fields). Fields left out are recovered
#: from the text by the normal extractor, so the demo exercises the real path.
DEMO_SCRIPT: tuple[tuple[int, str, dict], ...] = (
    (86, "First week back. Quiet so far, nothing much on.", {}),
    (82, "Went for a long run this morning and slept well last night. Feeling good.", {}),
    (78, "I am stressed. Deadline for the group assignment tomorrow and I barely slept.",
     {"workload": "high"}),
    (74, "Had dinner with friends, really enjoyed it. Slept well afterwards.", {}),
    (71, "I couldn't sleep again. Hospital appointment tomorrow for my scan.", {}),
    (70, "The appointment went fine in the end. Relieved.", {}),
    (65, "I am exhausted. Two deadlines this week and I slept about four hours.",
     {"workload": "high"}),
    (61, "Quiet day, stayed home and read.", {}),
    (58, "I am anxious about the exam tomorrow. Couldn't sleep at all.",
     {"workload": "high"}),
    (54, "Went to the gym, slept well, feeling much better today.", {}),
    (49, "Another hospital appointment tomorrow. Didn't sleep well thinking about it.", {}),
    (45, "Saw my family at the weekend, it was lovely.", {}),
    (40, "I am stressed and overwhelmed. Presentation tomorrow and I hardly slept.",
     {"workload": "high"}),
    (36, "Nothing much on today. Watched a film.", {}),
    (31, "I feel low. Argued with my flatmate and slept badly.", {}),
    (27, "Long walk with a friend, slept well. Feeling calm.", {}),
    (22, "I am stressed. Coursework due tomorrow, worked until 2am, four hours sleep.",
     {"workload": "high"}),
    (18, "Hospital appointment tomorrow again. I couldn't sleep.", {}),
    (14, "Results came back clear. I feel grateful.", {}),
    (9, "Busy week at work but I slept well and it was manageable.", {}),
    (5, "I am tired but okay. Quiet weekend, saw nobody.", {}),
    (2, "Went out for a birthday dinner with friends. Slept well. Really enjoyed it.", {}),
)


def build_demo_history(person_id: str = DEMO_PERSON_ID, *,
                       now: datetime | None = None,
                       detector=None) -> list[EmotionalEvent]:
    """Build the fictional history by running the REAL perception pipeline.

    The emotion and context of each entry are produced by the same detector
    and extractor a live check-in uses. Nothing is hard-coded, so if the model
    changes, this history changes with it -- which is the honest behaviour.
    """
    from ..emotion.pipeline import build_event

    base = now or datetime.now(timezone.utc)
    out: list[EmotionalEvent] = []
    for days_ago, text, fields in DEMO_SCRIPT:
        ts = (base - timedelta(days=days_ago)).replace(
            hour=20, minute=0, second=0, microsecond=0)
        ev = build_event(text, person_id=person_id,
                         timestamp=ts.isoformat(timespec="seconds"),
                         user_fields=fields or None, detector=detector,
                         data_status="SYNTHETIC_DEMO")
        out.append(ev)
    return out
