"""Layer 1: emotion detection, context extraction, events, corrections.

The assertions here are about CONTRACTS and HONESTY properties, not about
model accuracy -- accuracy is measured separately by
``scripts/eval_emotion_model.py`` on a held-out split.

What these guard:
  * an unstated field stays UNKNOWN and is never filled with a default;
  * every extracted value carries the phrase that produced it;
  * a user correction is recorded as a correction, not silently merged;
  * a self-report is reported BESIDE the model and never overrides it;
  * negated and past-framed statements are refused rather than guessed;
  * the lexicon fallback is always labelled a lexicon, so a demo can never
    present it as the Transformer.
"""
from __future__ import annotations

import pytest

from aedt.emotion.context import POOR_SLEEP_HOURS, extract_context
from aedt.emotion.detect import (CHECKIN_EMOTIONS, EmotionDetector,
                                 goemotions_to_checkin, self_reported_emotion)
from aedt.emotion.events import EmotionalEvent, FieldSource, Provenanced
from aedt.emotion.pipeline import build_event


@pytest.fixture(scope="module")
def lex():
    """The lexicon path, so these tests never depend on downloading weights."""
    return EmotionDetector(force_lexicon=True)


# ------------------------------------------------------------ context rules
def test_unstated_fields_are_unknown_not_defaulted():
    ctx = extract_context("Nothing much happened.")
    for field in ("sleep", "social", "activity", "location"):
        assert ctx[field].source is FieldSource.UNKNOWN
        assert ctx[field].value is None
        assert not ctx[field].known


def test_every_extracted_value_carries_its_evidence():
    ctx = extract_context("I couldn't sleep, hospital appointment tomorrow.")
    for name, p in ctx.items():
        if p.known:
            assert p.evidence, f"{name} was extracted with no evidence span"


def test_stated_sleep_hours_beat_vague_phrases():
    ctx = extract_context("I slept only four hours.")
    assert ctx["sleep"].value == "poor"
    assert "4" in ctx["sleep"].evidence
    assert str(int(POOR_SLEEP_HOURS)) in ctx["sleep"].evidence   # threshold shown
    assert extract_context("I slept eight hours.")["sleep"].value == "good"


def test_user_fields_override_extraction():
    ctx = extract_context("I couldn't sleep.", user_fields={"sleep": "good"})
    assert ctx["sleep"].value == "good"
    assert ctx["sleep"].source is FieldSource.USER_REPORTED


def test_extractor_does_not_infer_one_field_from_another():
    """"Could not sleep" says nothing about workload, and must not claim to."""
    ctx = extract_context("I could not sleep at all last night.")
    assert ctx["sleep"].known
    assert not ctx["workload"].known


def test_contact_is_not_confused_with_companionship():
    """A regression: "argued with my flatmate" used to read as time spent together."""
    assert not extract_context("Argued with my flatmate.")["social"].known
    assert extract_context("Had dinner with friends.")["social"].value == "with others"


# ------------------------------------------------------------- self-report
@pytest.mark.parametrize("text,label", [
    ("I am stressed and exhausted.", "stress"),
    ("I'm so anxious about tomorrow.", "anxiety"),
    ("I feel really grateful today.", "gratitude"),
    ("I am furious.", "anger"),
])
def test_explicit_statements_are_detected(text, label):
    got = self_reported_emotion(text)
    assert got is not None and got[0] == label
    assert got[1] in text.lower()          # the span really is in the sentence


def test_no_self_report_when_nothing_is_stated():
    assert self_reported_emotion("The weather is bad today.") is None


def test_a_statement_is_reported_beside_the_model_not_instead_of_it(lex):
    """UPDATED: this test previously asserted the regex OVERRODE the model.

    That behaviour was the root cause of a documented failure -- "I am not sure
    I am good enough" was reported as joy -- so the precedence was removed. The
    statement is now carried in its own field. GoEmotions still has no "stress"
    class, so `stated_emotion` remains how that construct reaches the user; it
    simply no longer overwrites the classifier.
    """
    ev = build_event("I am stressed and exhausted.", person_id="U1", detector=lex)
    assert ev.emotion.source is FieldSource.MODEL
    assert ev.stated_emotion.value == "stress"
    assert ev.stated_emotion.source is FieldSource.EXTRACTED
    assert "stated in the text" in ev.stated_emotion.evidence


def test_user_reported_emotion_outranks_everything(lex):
    ev = build_event("I am stressed.", person_id="U1", detector=lex,
                     user_fields={"emotion": "calm"})
    assert ev.emotion.value == "calm"
    assert ev.emotion.source is FieldSource.USER_REPORTED


# ------------------------------------------------------------- the model
def test_lexicon_is_always_labelled_a_lexicon(lex):
    p = lex.predict("I feel awful and hopeless.")
    assert p.backend == "lexicon"
    assert not p.is_model            # nothing may present this as the Transformer


def test_mapped_labels_are_valid_and_unmapped_ones_fall_back_explicitly():
    """UPDATED: full 28-label coverage is no longer required, deliberately.

    The previous version asserted every GoEmotions label had a target, which is
    what motivated `realization -> confusion` -- a mapping with no semantic
    justification that produced a real failure. Coverage is not a reason. A
    label with no defensible target is left unmapped and surfaces as neutral,
    with the raw label shown beside it.
    """
    from aedt.emotion.detect import _GOEMOTIONS_TO_CHECKIN
    for label in _GOEMOTIONS_TO_CHECKIN.values():
        assert label in CHECKIN_EMOTIONS
    assert "realization" not in _GOEMOTIONS_TO_CHECKIN
    assert goemotions_to_checkin("not_a_real_label") == "neutral"
    assert goemotions_to_checkin("realization") == "neutral"


def test_empty_input_is_not_given_a_confident_answer(lex):
    p = lex.predict("   ")
    assert p.label == "neutral" and p.score == 0.0


# ---------------------------------------------------------------- events
def test_event_records_a_source_for_every_field(lex):
    ev = build_event("I couldn't sleep, exam tomorrow.", person_id="U1", detector=lex)
    d = ev.to_dict()
    for f in EmotionalEvent.CONTEXT_FIELDS:
        assert f in d["source"]
        assert d["source"][f] in {s.value for s in FieldSource}


def test_event_id_is_stable_for_the_same_input(lex):
    a = build_event("same text", person_id="U1", timestamp="2026-01-01T00:00:00", detector=lex)
    b = build_event("same text", person_id="U1", timestamp="2026-01-01T00:00:00", detector=lex)
    assert a.event_id == b.event_id


def test_correction_is_recorded_as_a_correction(lex):
    ev = build_event("I couldn't sleep.", person_id="U1", detector=lex)
    fixed = ev.with_correction("sleep", "good")
    assert fixed.sleep.value == "good"
    assert fixed.sleep.source is FieldSource.CORRECTED
    assert "sleep" in fixed.corrections
    assert ev.sleep.value == "poor"          # the original is untouched


def test_correcting_an_unknown_field_name_is_refused(lex):
    ev = build_event("hello", person_id="U1", detector=lex)
    with pytest.raises(KeyError):
        ev.with_correction("not_a_field", "x")


def test_event_round_trips_through_json(lex):
    ev = build_event("I am anxious, appointment tomorrow, slept four hours.",
                     person_id="U1", detector=lex)
    back = EmotionalEvent.from_dict(ev.to_dict())
    assert back.event_id == ev.event_id
    for f in EmotionalEvent.CONTEXT_FIELDS:
        assert back.get(f).value == ev.get(f).value
        assert back.get(f).source is ev.get(f).source


# ---------------------------------------------- repaired failure modes (Part K)
@pytest.mark.parametrize("text", [
    "I am not sure I am good enough.",      # was reported as JOY
    "I am not happy about this.",
    "I don't feel calm at all.",
])
def test_negated_statements_are_not_reported_as_feelings(text):
    assert self_reported_emotion(text) is None


def test_past_framing_is_not_reported_as_a_current_feeling():
    assert self_reported_emotion("Yesterday I was anxious, but now I feel relieved.") is None


def test_the_last_surviving_statement_wins_not_the_first():
    """'grateful ... but anxious' must resolve to the current state, not the aside."""
    got = self_reported_emotion("I am grateful, but I am anxious.")
    assert got is not None and got[0] == "anxiety"


def test_a_statement_never_overrides_the_model(lex):
    """The regex may inform, never replace. This was the root cause of Case 1."""
    ev = build_event("I am stressed and exhausted.", person_id="U1", detector=lex)
    assert ev.emotion.source is FieldSource.MODEL
    assert ev.stated_emotion.value == "stress"
    assert ev.stated_emotion.source is FieldSource.EXTRACTED
    assert "stated in the text" in ev.stated_emotion.evidence


def test_realization_is_no_longer_mapped_to_confusion():
    """Coverage is not a semantic justification."""
    assert goemotions_to_checkin("realization") != "confusion"


def test_interview_is_not_reported_as_a_presentation():
    """The category must never contradict its own evidence span."""
    c = extract_context("I have an interview tomorrow.")
    assert c["event"].value == "interview"
    assert "interview" in c["event"].evidence
    assert extract_context("My presentation is tomorrow.")["event"].value == "presentation"


def test_a_correction_never_destroys_the_model_prediction(lex):
    """The disagreement is the error signal; losing it makes evaluation impossible."""
    ev = build_event("I am fine.", person_id="U1", detector=lex)
    original = ev.emotion.value
    fixed = ev.with_correction("emotion", "sadness")
    assert fixed.emotion.value == "sadness"
    assert fixed.emotion.source is FieldSource.CORRECTED
    assert ev.emotion.value == original          # the original object is intact
    assert "emotion" in fixed.corrections
