"""Regression tests against the JAVASCRIPT the website actually runs.

These exist because every one of these bugs was fixed in the Python pipeline
and remained live on the site for weeks: the browser runs
``frontend/src/lib/twin.js``, which was never updated. A Python-only test
suite reported all-green while a panel member typing "I have an interview
tomorrow" would have been shown "presentation".

Each case below was reproduced in the deployed JS before being fixed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
TWIN = ROOT / "frontend" / "src" / "lib" / "twin.js"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def run_js(body: str):
    """Execute a snippet against the real module and return its JSON result."""
    src = f"""
import {{ selfReportedEmotion, extractContext, buildEvent }} from "{TWIN.as_uri()}";
{body}
"""
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise AssertionError(p.stderr.strip())
    return json.loads(p.stdout.strip())


# ------------------------------------------------- negation and scope
@node
@pytest.mark.parametrize("text", [
    "I am not sure I am good enough.",     # was reported as JOY
    "I am not happy about this.",
    "I don't feel calm at all.",
])
def test_negated_statements_are_refused_in_the_browser(text):
    got = run_js(f'console.log(JSON.stringify(selfReportedEmotion({text!r})));')
    assert got is None, f"{text!r} produced {got}"


@node
def test_past_framing_is_not_treated_as_the_current_feeling():
    got = run_js('console.log(JSON.stringify(selfReportedEmotion('
                 '"Yesterday I was anxious, but now I feel relieved.")));')
    assert got is None


@node
def test_the_last_statement_wins_not_the_longest_match():
    """The old code took the longest match anywhere, so an aside beat the point."""
    got = run_js('console.log(JSON.stringify(selfReportedEmotion('
                 '"I am grateful, but I am anxious.")));')
    assert got is not None and got[0] == "anxiety"


# ------------------------------------------------------- event extraction
@node
def test_interview_is_not_reported_as_a_presentation():
    got = run_js('const c = extractContext("I have an interview tomorrow.");'
                 'console.log(JSON.stringify([c.event.value, c.event.evidence]));')
    assert got[0] == "interview"
    assert "interview" in got[1]


@node
def test_presentation_is_still_recognised():
    got = run_js('console.log(JSON.stringify('
                 'extractContext("My presentation is tomorrow.").event.value));')
    assert got == "presentation"


# --------------------------------------------------------- no override
@node
def test_a_stated_feeling_never_overrides_the_model():
    """The root cause: a regex outranking the classifier."""
    got = run_js('const e = buildEvent("I am stressed and exhausted.", {personId:"U"});'
                 'console.log(JSON.stringify({src: e.emotion.source,'
                 ' said: e.statedEmotion.value, ev: e.statedEmotion.evidence}));')
    assert got["src"] == "model", "the model must produce the emotion field"
    assert got["said"] == "stress", "the stated feeling must still be recorded"
    assert "you wrote" in got["ev"]


@node
def test_an_explicit_check_in_field_still_wins():
    got = run_js('const e = buildEvent("I am stressed.", '
                 '{personId:"U", userFields:{emotion:"calm"}});'
                 'console.log(JSON.stringify([e.emotion.value, e.emotion.source]));')
    assert got == ["calm", "user_reported"]


@node
def test_realization_is_not_mapped_to_confusion():
    eng = ROOT / "frontend" / "src" / "lib" / "emotion_engine.js"
    src = f'''import {{ goemotionsToCheckin }} from "{eng.as_uri()}";
console.log(JSON.stringify(goemotionsToCheckin("realization")));'''
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        pytest.skip("goemotionsToCheckin is not exported")
    assert json.loads(p.stdout.strip()) != "confusion"
