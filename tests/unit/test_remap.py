"""MANDATORY LABEL-REVERSAL TESTS (test B).

Deliberately reverse / scramble the ordinal labels. The system must detect or
flag the mapping inconsistency where it is detectable, and must NEVER silently
continue.

Mapping by POSITION would silently invert the stress scale and reverse every
conclusion. This is the single most dangerous silent failure in the project.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import STRESS_LABEL_TO_SEVERITY, normalise_label
from aedt.errors import DecisionRequired
from aedt.preprocess.reports import (build_code_to_severity,
                                     detect_reversed_coding,
                                     remap_report_labels)

CANONICAL = ["Feeling great", "Feeling good", "A little stressed",
             "Definitely stressed", "Stressed out"]


def test_canonical_order_maps_to_identity():
    m = build_code_to_severity(CANONICAL)
    assert m == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert not detect_reversed_coding(m)


def test_scrambled_option_order_still_produces_the_correct_severity_map():
    """MANDATORY (ROUND-17 §W): a scrambled option order must still produce the
    correct severity map, because the map is keyed on LABEL TEXT."""
    scrambled = ["A little stressed", "Definitely stressed", "Stressed out",
                 "Feeling good", "Feeling great"]
    m = build_code_to_severity(scrambled)
    assert m == {1: 3, 2: 4, 3: 5, 4: 2, 5: 1}
    assert detect_reversed_coding(m), "the reversal must be DETECTED and reported"


def test_fully_reversed_order_is_detected():
    m = build_code_to_severity(list(reversed(CANONICAL)))
    assert m == {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    assert detect_reversed_coding(m)
    # and the remap actually inverts the stored codes
    assert list(remap_report_labels([1, 2, 3, 4, 5], m)) == [5, 4, 3, 2, 1]


def test_unrecognised_label_halts_with_decision_required():
    with pytest.raises(DecisionRequired) as e:
        build_code_to_severity(["Feeling great", "Somewhat tense",
                                "A little stressed"])
    msg = str(e.value)
    assert "DECISION REQUIRED" in msg
    assert "differ from expected mapping" in msg
    assert "Somewhat tense" in msg
    assert "Do NOT guess" in msg


def test_case_and_whitespace_and_underscores_are_normalised():
    m = build_code_to_severity(["  FEELING   GREAT ", "feeling_good",
                               "A Little Stressed", "DEFINITELY stressed",
                               "stressed  out"])
    assert m == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert normalise_label("  Feeling_GREAT  ") == "feeling great"


def test_duplicate_labels_are_rejected_not_collapsed():
    with pytest.raises(DecisionRequired, match="not injective"):
        build_code_to_severity(["Feeling great", "Feeling great",
                                "A little stressed"])


def test_unknown_stored_code_halts_rather_than_dropping_the_row():
    m = build_code_to_severity(CANONICAL)
    with pytest.raises(DecisionRequired, match="absent from the code->severity"):
        remap_report_labels([1, 2, 9], m)


def test_non_strict_mode_marks_unknown_codes_missing_never_guesses():
    m = build_code_to_severity(CANONICAL)
    out = remap_report_labels([1, 9, 5], m, strict=False)
    assert out[0] == 1 and out[2] == 5
    assert np.isnan(out[1]), "an unknown code must be missing, never guessed"


def test_studentlife_fixture_scrambles_on_purpose(studentlife_fixture):
    """The fixture stores options OUT of severity order specifically so that
    this trap is exercised on every adapter run."""
    from aedt.io import get_adapter
    a = get_adapter("studentlife").audit(studentlife_fixture)
    assert detect_reversed_coding(dict(a.code_to_severity_mapping))
    assert any("NOT in severity order" in n for n in a.notes)


def test_reversing_the_labels_reverses_the_measured_slope(small_frame):
    """The end-to-end consequence: an inverted map inverts the sensor->report
    slope. This is why the map is never inferred."""
    from aedt.estimators.slope_ratio import fit_person_epochs
    pid = small_frame["pid"].iloc[0]
    g = small_frame[small_frame["pid"] == pid].copy()
    normal = fit_person_epochs(g, "conversation_minutes", 5, pid=pid)
    g["report"] = 6 - g["report"]                 # reverse the severity coding
    flipped = fit_person_epochs(g, "conversation_minutes", 5, pid=pid)
    assert np.sign(normal[0].beta) != np.sign(flipped[0].beta)
    assert flipped[0].beta == pytest.approx(-normal[0].beta, rel=0.05)
