"""LAYER 1 / MODULE 2 -- context extraction.

Purpose  Recover the situation around a reported feeling: what happened, when,
         how the person slept, how busy they were, whether other people were
         involved.
Input    Free text, plus any fields the person filled in explicitly.
Output   ``Provenanced`` values with per-field evidence spans.
Status   ENGINEERING. Deterministic, lexicon- and pattern-based.

WHY THIS IS RULES AND NOT A NEURAL EXTRACTOR

A neural slot-filler for these categories would need labelled check-in data
that does not exist for this project. Writing one on unlabelled text and
reporting its output as "extracted" would produce confident nonsense. The
rules here fire only on explicit surface evidence and return UNKNOWN
otherwise, which is the honest failure mode: a missing field is visible in the
interface and can be filled in by the user, whereas a wrong field is not.

Every value carries the exact substring that produced it, so a reviewer can
check any extraction against the sentence in one glance.

WHAT THIS DELIBERATELY DOES NOT DO. It never infers a field from another
field. "I could not sleep" sets `sleep`; it does not also set `workload` on
the theory that tired people are busy. Inference is a separate, labelled step
that belongs to the twin, not to the extractor.
"""
from __future__ import annotations

import re
from typing import Iterable

from .events import FieldSource, Provenanced

__all__ = ["extract_context", "CONTEXT_LEXICON", "EVENT_LEXICON"]

# --------------------------------------------------------------------------
# Lexicons. Each entry maps a canonical value to the surface forms that
# license it. Patterns are matched on word boundaries against lowercased text.
# --------------------------------------------------------------------------

#: event / situation category -> trigger phrases
EVENT_LEXICON: dict[str, tuple[str, ...]] = {
    "hospital appointment": ("hospital", "doctor", "appointment", "clinic",
                             "gp ", "surgery", "scan", "x-ray", "checkup",
                             "check-up", "medical"),
    "examination": ("exam", "examination", "test tomorrow", "finals",
                    "midterm", "viva", "assessment"),
    "deadline": ("deadline", "due tomorrow", "submission", "assignment due",
                 "hand in", "hand-in"),
    # "interview" was listed here and caused the extractor to display
    # "presentation" while its own evidence span read "interview". Distinct
    # events must stay distinct; a category whose evidence contradicts it is
    # worse than no category.
    "presentation": ("presentation", "present to", "defence", "defense",
                     "viva voce"),
    "interview": ("interview", "technical round", "hiring", "recruiter"),
    "work": ("work", "shift", "office", "meeting", "project"),
    "study": ("study", "studying", "revision", "revising", "coursework",
              "lecture", "class"),
    "travel": ("flight", "train", "travel", "journey", "commute", "trip"),
    "family": ("family", "mum", "mom", "dad", "parents", "sister", "brother",
               "partner", "wife", "husband"),
    "social event": ("party", "wedding", "dinner with", "night out",
                     "birthday", "gathering"),
    "conflict": ("argument", "argued", "fight", "fell out", "conflict",
                 "disagreement"),
    "bereavement": ("funeral", "passed away", "died", "bereavement"),
    "finances": ("rent", "bills", "money", "debt", "loan", "payment due"),
}

#: field -> {canonical value: trigger phrases}
CONTEXT_LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "sleep": {
        "poor": ("couldn't sleep", "could not sleep", "cant sleep",
                 "can't sleep", "no sleep", "barely slept", "hardly slept",
                 "didn't sleep", "did not sleep", "insomnia", "awake all night",
                 "restless night", "bad night", "slept badly", "poor sleep",
                 "tossing and turning", "up all night"),
        "good": ("slept well", "good sleep", "rested", "well rested",
                 "well-rested", "slept great", "solid sleep"),
    },
    "workload": {
        "high": ("busy", "swamped", "overloaded", "so much work", "workload",
                 "lots to do", "piled up", "back to back", "back-to-back",
                 "overwhelmed with work", "deadline", "too much on"),
        "low": ("quiet day", "nothing much on", "light day", "free day",
                "not much to do"),
    },
    "social": {
        "isolated": ("alone", "lonely", "by myself", "on my own", "no one to",
                     "nobody to", "isolated"),
        # "with my" was too loose: it fired on "argued with my flatmate",
        # which is contact but not the companionable sense the field means.
        # Phrases here must imply time spent together, not mere mention.
        "with others": ("with friends", "we went", "met up", "spent time with",
                        "together with", "dinner with", "out with", "saw my family",
                        "with a friend"),
    },
    "activity": {
        "low": ("didn't leave", "did not leave", "stayed in", "stayed home",
                "in bed all", "no exercise", "sedentary"),
        "high": ("went for a run", "ran ", "gym", "workout", "exercised",
                 "walked a lot", "cycled", "swim"),
    },
    "location": {
        "home": ("at home", "stayed home", "from home", "in my room"),
        "work": ("at work", "in the office", "at the office"),
        "campus": ("on campus", "at uni", "at university", "in the library",
                   "at college"),
        "hospital": ("at the hospital", "in hospital", "at the clinic"),
    },
}

#: time expressions -> canonical value. Order matters: longer forms first.
_TIME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bday after tomorrow\b", "day after tomorrow"),
    (r"\btomorrow morning\b", "tomorrow"),
    (r"\btomorrow\b", "tomorrow"),
    (r"\byesterday\b", "yesterday"),
    (r"\blast night\b", "last night"),
    (r"\btonight\b", "tonight"),
    (r"\bthis morning\b", "this morning"),
    (r"\bthis afternoon\b", "this afternoon"),
    (r"\bthis evening\b", "this evening"),
    (r"\bnext week\b", "next week"),
    (r"\blast week\b", "last week"),
    (r"\btoday\b", "today"),
    (r"\bin (\d+) days?\b", "in {0} days"),
    (r"\bon (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "{0}"),
)

#: "I slept four hours" / "only 4 hours" -> poor when below this many hours
_SLEEP_HOURS = re.compile(
    r"\b(?:slept|sleep|got)\s+(?:only\s+)?(?:about\s+)?"
    r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:hours|hrs|h)\b")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
POOR_SLEEP_HOURS = 6.0


def _find(text: str, phrases: Iterable[str]) -> str | None:
    """Return the matched surface form, or None. Longest match wins."""
    best = None
    for p in phrases:
        if p in text and (best is None or len(p) > len(best)):
            best = p
    return best


def _extract_time(text: str) -> Provenanced:
    for pattern, canon in _TIME_PATTERNS:
        m = re.search(pattern, text)
        if m:
            value = canon.format(*m.groups()) if m.groups() else canon
            return Provenanced(value=value, source=FieldSource.EXTRACTED,
                               confidence=0.95, evidence=m.group(0))
    return Provenanced.unknown()


def _extract_sleep_hours(text: str) -> Provenanced | None:
    """A stated duration beats a vague phrase, so this is tried first."""
    m = _SLEEP_HOURS.search(text)
    if not m:
        return None
    raw = m.group(1)
    hours = _WORDNUM.get(raw, None)
    if hours is None:
        try:
            hours = float(raw)
        except ValueError:
            return None
    value = "poor" if hours < POOR_SLEEP_HOURS else "good"
    return Provenanced(value=value, source=FieldSource.EXTRACTED,
                       confidence=0.9,
                       evidence=f"{m.group(0)} ({hours:g}h, "
                                f"threshold {POOR_SLEEP_HOURS:g}h)")


def extract_context(text: str, *, user_fields: dict | None = None
                    ) -> dict[str, Provenanced]:
    """Extract the context fields from ``text``, then apply ``user_fields``.

    A value the person supplied explicitly always wins over one recovered from
    text: they know their own sleep better than a regular expression does.
    Fields with no evidence are returned as UNKNOWN rather than omitted, so the
    caller can show what is missing.
    """
    low = f" {(text or '').lower().strip()} "
    out: dict[str, Provenanced] = {}

    # sleep: a stated number of hours first, then vague phrases
    hours = _extract_sleep_hours(low)
    out["sleep"] = hours if hours else Provenanced.unknown()

    for field, values in CONTEXT_LEXICON.items():
        if field == "sleep" and out["sleep"].known:
            continue
        hit_value, hit_phrase = None, None
        for canon, phrases in values.items():
            m = _find(low, phrases)
            if m and (hit_phrase is None or len(m) > len(hit_phrase)):
                hit_value, hit_phrase = canon, m
        out[field] = (Provenanced(value=hit_value, source=FieldSource.EXTRACTED,
                                  confidence=0.8, evidence=hit_phrase.strip())
                      if hit_value else Provenanced.unknown())

    # event category
    ev_value, ev_phrase = None, None
    for canon, phrases in EVENT_LEXICON.items():
        m = _find(low, phrases)
        if m and (ev_phrase is None or len(m) > len(ev_phrase)):
            ev_value, ev_phrase = canon, m
    out["event"] = (Provenanced(value=ev_value, source=FieldSource.EXTRACTED,
                                confidence=0.75, evidence=ev_phrase.strip())
                    if ev_value else Provenanced.unknown())

    out["time_context"] = _extract_time(low)

    # explicit check-in fields override extraction
    for k, v in (user_fields or {}).items():
        if v in (None, "", "unknown"):
            continue
        out[k] = Provenanced(value=v, source=FieldSource.USER_REPORTED,
                             confidence=1.0, evidence="check-in field")
    return out
