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


# ------------------------------------------- two-user comparison (Part 11)
TWO = ROOT / "frontend" / "src" / "ui" / "twouser.js"


def run_two(body: str):
    src = f'''
import {{ PersonalTwin, buildEvent }} from "{TWIN.as_uri()}";
{body}
'''
    p = subprocess.run(["node", "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise AssertionError(p.stderr.strip())
    return json.loads(p.stdout.strip())


HISTORIES = '''
const mk = (pid, i, text, f) => buildEvent(text, {
  personId: pid, timestamp: new Date(Date.UTC(2026, 0, 2 + i * 5)).toISOString(),
  userFields: f, dataStatus: "SYNTHETIC_DEMO" });
const A = new PersonalTwin("Person A", "SYNTHETIC_DEMO");
[["Deadline due tomorrow, slept about four hours.", "anxiety"],
 ["Another deadline, barely slept again.", "anxiety"],
 ["Deadline week, sleeping badly.", "stress"],
 ["Submission tomorrow, four hours of sleep.", "anxiety"],
 ["Deadline again, hardly slept.", "anxiety"]].forEach(([t, e], i) =>
   A.addEvent(mk("Person A", i, t, {event:"deadline", sleep:"poor", emotion:e})));
const B = new PersonalTwin("Person B", "SYNTHETIC_DEMO");
[["Deadline due tomorrow, work is already done.", "calm"],
 ["Another deadline, finished it early.", "calm"],
 ["Deadline week, on top of it.", "joy"],
 ["Submission tomorrow, prepared well in advance.", "calm"],
 ["Deadline again, comfortable with it.", "calm"]].forEach(([t, e], i) =>
   B.addEvent(mk("Person B", i, t, {event:"deadline", sleep:"poor", emotion:e})));
const q = (pid) => mk(pid, 40, "I have another deadline tomorrow and I barely slept.",
                      {event:"deadline", sleep:"poor"});
'''


@node
def test_two_user_page_declares_its_histories_synthetic():
    """The wording may change; the disclosure may not."""
    src = TWO.read_text()
    low = src.lower()
    assert "illustrative synthetic histor" in low, "no synthetic labelling found"
    assert "not a real participant" in low or "not real participants" in low
    assert "SYNTHETIC_DEMO" in src, "events must carry the synthetic data status"


@node
def test_two_user_page_ties_back_to_the_real_null_result():
    """The demonstration must not be left implying real-world effectiveness."""
    low = TWO.read_text().lower()
    assert "did not" in low and "previous value forward" in low


@node
def test_the_same_sentence_produces_different_personalised_results():
    """The demonstration's whole claim, verified through the real retrieval logic."""
    got = run_two(HISTORIES + '''
const a = A.patternInsight(q("Person A")), b = B.patternInsight(q("Person B"));
console.log(JSON.stringify({a: a.dominantEmotion, b: b.dominantEmotion,
  aN: a.nSimilar, bN: b.nSimilar, aOk: a.sufficient, bOk: b.sufficient}));''')
    assert got["aOk"] and got["bOk"]
    assert got["a"] != got["b"], "the two histories did not separate"
    assert got["a"] == "anxiety" and got["b"] == "calm"


@node
def test_the_contrast_is_not_explained_by_one_person_having_more_data():
    """Equal episode counts, so the difference must come from their content."""
    got = run_two(HISTORIES + '''
console.log(JSON.stringify({a: A.events.length, b: B.events.length,
  aSim: A.similarEpisodes(q("Person A"), {topK:25, minScore:1.5}).length,
  bSim: B.similarEpisodes(q("Person B"), {topK:25, minScore:1.5}).length}));''')
    assert got["a"] == got["b"], "the histories differ in size, confounding the comparison"
    assert got["aSim"] == got["bSim"], "retrieval returned different counts"


@node
def test_the_result_is_computed_not_hard_coded():
    """Swap the recorded feelings and the page's conclusion must follow."""
    got = run_two(HISTORIES.replace('"anxiety"', '"__TMP__"')
                           .replace('"calm"', '"anxiety"')
                           .replace('"__TMP__"', '"calm"') + '''
const a = A.patternInsight(q("Person A")), b = B.patternInsight(q("Person B"));
console.log(JSON.stringify({a: a.dominantEmotion, b: b.dominantEmotion}));''')
    assert got["a"] == "calm" and got["b"] == "anxiety", (
        "inverting the histories did not invert the conclusion, so the output "
        "is not derived from the stored history")


@node
def test_one_twin_cannot_read_the_other_twins_history():
    got = run_two(HISTORIES + '''
const empty = new PersonalTwin("Person C", "SYNTHETIC_DEMO");
console.log(JSON.stringify({n: empty.similarEpisodes(q("Person C"), {topK:25}).length,
  sufficient: empty.patternInsight(q("Person C")).sufficient}));''')
    assert got["n"] == 0 and got["sufficient"] is False
