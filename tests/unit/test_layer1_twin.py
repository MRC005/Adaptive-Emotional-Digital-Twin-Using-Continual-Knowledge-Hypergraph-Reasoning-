"""Layer 1: event hypergraph, personal twin, retrieval, pattern floor.

The properties under test are the ones a demonstration is most tempted to
break: that an unknown field never becomes a graph vertex, that the evidence
floor holds even when it makes the demo less impressive, that synthetic
history is labelled everywhere, and that retrieval can always explain itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aedt.emotion.detect import EmotionDetector
from aedt.emotion.events import EmotionalEvent, FieldSource, Provenanced
from aedt.emotion.pipeline import build_event
from aedt.hypergraph.event_graph import build_event_hypergraph
from aedt.twin.demo_history import DEMO_PERSON_ID, build_demo_history
from aedt.twin.personal_twin import (MIN_EPISODES_FOR_PATTERN,
                                     PersonalEmotionalTwin)


@pytest.fixture(scope="module")
def lex():
    return EmotionDetector(force_lexicon=True)


def _ev(pid, day, text, fields, lex, status="USER"):
    ts = (datetime(2026, 1, 1, tzinfo=timezone.utc)
          + timedelta(days=day)).isoformat(timespec="seconds")
    return build_event(text, person_id=pid, timestamp=ts, user_fields=fields,
                       detector=lex, data_status=status)


@pytest.fixture(scope="module")
def stocked(lex):
    """A twin with a real repeated pattern: poor sleep + exam -> stress."""
    t = PersonalEmotionalTwin("U1")
    rows = [
        (1, "I am stressed, exam tomorrow, slept badly.",
         {"sleep": "poor", "event": "examination", "emotion": "stress"}),
        (5, "I am stressed, another exam tomorrow, no sleep.",
         {"sleep": "poor", "event": "examination", "emotion": "stress"}),
        (9, "I am anxious, exam tomorrow, slept badly.",
         {"sleep": "poor", "event": "examination", "emotion": "anxiety"}),
        (12, "I am stressed, exam tomorrow again, barely slept.",
         {"sleep": "poor", "event": "examination", "emotion": "stress"}),
        (15, "Lovely day with friends.",
         {"sleep": "good", "event": "social event", "emotion": "joy"}),
    ]
    for d, text, f in rows:
        t.add_event(_ev("U1", d, text, f, lex))
    return t


# ------------------------------------------------------------- hypergraph
def test_one_event_is_one_hyperedge(lex):
    ev = _ev("U1", 0, "I am stressed, exam tomorrow, slept badly.",
             {"sleep": "poor", "event": "examination"}, lex)
    g = build_event_hypergraph("U1", [ev])
    assert len(g.edges) == 1
    e = g.edges[0]
    assert e.arity >= 4                      # person + emotion + sleep + event
    assert any(v.startswith("Sleep=") for v in e.vertices)
    assert any(v.startswith("Event=") for v in e.vertices)


def test_unknown_fields_never_become_vertices(lex):
    """Two people who both failed to mention sleep must not match on it."""
    ev = _ev("U1", 0, "Something happened.", {}, lex)
    g = build_event_hypergraph("U1", [ev])
    for key in g.vertices:
        assert "None" not in key
        assert "unknown" not in key.lower()


def test_incidence_matrix_matches_the_edges(stocked):
    g = build_event_hypergraph("U1", stocked.events)
    H = g.incidence()
    assert H.shape == (len(g.vertices), len(g.edges))
    assert set(H.flatten().tolist()) <= {0.0, 1.0}
    vd, ed = g.degrees()
    assert ed.tolist() == [float(e.arity) for e in g.edges]


def test_conjunctive_query_returns_only_edges_with_all_vertices(stocked):
    g = build_event_hypergraph("U1", stocked.events)
    hits = g.edges_containing("Sleep=poor", "Event=examination")
    assert hits
    for e in hits:
        assert "Sleep=poor" in e.vertices and "Event=examination" in e.vertices


def test_recurring_pairs_require_repetition(stocked):
    g = build_event_hypergraph("U1", stocked.events)
    for a, b, n in g.co_occurrence(min_count=2):
        assert n >= 2


# --------------------------------------------------------------- the twin
def test_history_is_append_only_and_time_ordered(stocked):
    ts = [e.timestamp for e in stocked.events]
    assert ts == sorted(ts)


def test_reingesting_the_same_event_is_idempotent(lex):
    t = PersonalEmotionalTwin("U1")
    ev = _ev("U1", 0, "hello", {}, lex)
    t.add_event(ev); t.add_event(ev)
    assert t.n_events == 1


def test_an_event_from_another_person_is_refused(lex):
    t = PersonalEmotionalTwin("U1")
    with pytest.raises(ValueError):
        t.add_event(_ev("SOMEONE_ELSE", 0, "hello", {}, lex))


def test_retrieval_never_returns_the_future(stocked, lex):
    query = _ev("U1", 3, "I am stressed, exam tomorrow, slept badly.",
                {"sleep": "poor", "event": "examination"}, lex)
    for s in stocked.similar_episodes(query):
        assert s.timestamp <= query.timestamp


def test_every_retrieved_episode_can_explain_itself(stocked, lex):
    query = _ev("U1", 20, "I am stressed, exam tomorrow, slept badly.",
                {"sleep": "poor", "event": "examination"}, lex)
    sims = stocked.similar_episodes(query)
    assert sims
    for s in sims:
        assert s.matched_fields
        assert s.explanation.startswith("same ")


def test_conjunction_scores_above_a_single_field(stocked, lex):
    both = _ev("U1", 20, "x", {"sleep": "poor", "event": "examination"}, lex)
    one = _ev("U1", 20, "y", {"sleep": "poor"}, lex)
    top_both = stocked.similar_episodes(both)[0].score
    top_one = stocked.similar_episodes(one)[0].score
    assert top_both > top_one


# --------------------------------------------------------- evidence floor
def test_no_pattern_below_the_floor(lex):
    t = PersonalEmotionalTwin("U1")
    t.add_event(_ev("U1", 0, "I am stressed, exam tomorrow, slept badly.",
                    {"sleep": "poor", "event": "examination"}, lex))
    q = _ev("U1", 5, "I am stressed, exam tomorrow, slept badly.",
            {"sleep": "poor", "event": "examination"}, lex)
    ins = t.pattern_insight(q)
    assert ins.sufficient is False
    assert "still learning" in ins.statement.lower()
    assert str(MIN_EPISODES_FOR_PATTERN) in ins.statement


def test_pattern_appears_once_the_floor_is_met(stocked, lex):
    q = _ev("U1", 20, "I am stressed, exam tomorrow, slept badly.",
            {"sleep": "poor", "event": "examination"}, lex)
    ins = stocked.pattern_insight(q)
    assert ins.sufficient is True
    assert ins.n_similar >= MIN_EPISODES_FOR_PATTERN
    assert ins.dominant_emotion == "stress"
    # the claim must carry its counts
    assert str(ins.n_supporting) in ins.statement
    assert str(ins.n_similar) in ins.statement


def test_pattern_statement_never_claims_causation(stocked, lex):
    q = _ev("U1", 20, "I am stressed, exam tomorrow, slept badly.",
            {"sleep": "poor", "event": "examination"}, lex)
    ins = stocked.pattern_insight(q)

    # The STATEMENT describes what was recorded, and makes no forward or causal
    # claim. Checked on the statement alone: the caveats legitimately contain
    # these words inside negations ("not what will happen"), and a naive
    # substring check over both would flag the disclaimer as the offence.
    statement = ins.statement.lower()
    for word in ("causes", "caused", "because of", "will happen", "diagnos",
                 "guarantee", "predicts that"):
        assert word not in statement, f"statement makes a {word!r} claim"
    assert "recorded" in statement or "was" in statement

    # and the caveats must actually carry the disclaimers
    caveats = " ".join(ins.caveats).lower()
    assert "not a cause" in caveats
    assert "not what will happen" in caveats


# ------------------------------------------------------------- synthetic
def test_demo_history_is_labelled_synthetic_everywhere(lex):
    events = build_demo_history(detector=lex)
    assert len(events) >= 10
    assert all(e.data_status == "SYNTHETIC_DEMO" for e in events)

    t = PersonalEmotionalTwin(DEMO_PERSON_ID, data_status="SYNTHETIC_DEMO")
    for e in events:
        t.add_event(e)
    assert t.is_synthetic
    assert t.profile()["is_synthetic"] is True

    g = build_event_hypergraph(DEMO_PERSON_ID, events)
    assert g.summary()["synthetic_edges"] == len(events)


def test_demo_history_supports_a_pattern(lex):
    """The demo must actually demonstrate something, without lowering the floor."""
    events = build_demo_history(detector=lex)
    t = PersonalEmotionalTwin(DEMO_PERSON_ID, data_status="SYNTHETIC_DEMO")
    for e in events:
        t.add_event(e)
    q = build_event("I am stressed. Exam tomorrow and I barely slept.",
                    person_id=DEMO_PERSON_ID, detector=lex,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    ins = t.pattern_insight(q)
    assert ins.sufficient is True


def test_clearing_history_empties_the_twin_and_is_logged():
    d = EmotionDetector(force_lexicon=True)
    t = PersonalEmotionalTwin("U1")
    t.add_event(_ev("U1", 0, "hello", {}, d))
    t.clear()
    assert t.n_events == 0
    assert any(x["action"] == "history_cleared" for x in t.update_log)


def test_twin_round_trips_through_json(stocked):
    back = PersonalEmotionalTwin.from_dict(stocked.to_dict())
    assert back.n_events == stocked.n_events
    assert [e.event_id for e in back.events] == [e.event_id for e in stocked.events]


# ------------------------------------------------------- personalisation
def test_same_situation_different_history_gives_different_insight(lex):
    """The core Digital Twin claim: the history, not the sentence, decides.

    Two fictional people meet the same event. One has repeatedly reported
    anxiety around exams after poor sleep; the other has repeatedly reported
    calm around exams when prepared and with friends. The same query must
    produce different personalised statements, or the twin is just a
    sentence classifier with extra steps.
    """
    a = PersonalEmotionalTwin("User_A", data_status="SYNTHETIC_DEMO")
    for day, emo in ((1, "anxiety"), (6, "anxiety"), (11, "anxiety"), (16, "anxiety")):
        a.add_event(_ev("User_A", day, "Exam tomorrow and I slept badly.",
                        {"event": "examination", "sleep": "poor", "emotion": emo},
                        lex, status="SYNTHETIC_DEMO"))

    b = PersonalEmotionalTwin("User_B", data_status="SYNTHETIC_DEMO")
    for day, emo in ((1, "calm"), (6, "calm"), (11, "joy"), (16, "calm")):
        b.add_event(_ev("User_B", day, "Exam tomorrow, well prepared, revised with friends.",
                        {"event": "examination", "sleep": "good",
                         "social": "with others", "emotion": emo},
                        lex, status="SYNTHETIC_DEMO"))

    qa = _ev("User_A", 25, "I have an exam tomorrow.",
             {"event": "examination", "sleep": "poor"}, lex)
    qb = _ev("User_B", 25, "I have an exam tomorrow.",
             {"event": "examination", "sleep": "good", "social": "with others"}, lex)

    ia, ib = a.pattern_insight(qa), b.pattern_insight(qb)
    assert ia.sufficient and ib.sufficient
    assert ia.dominant_emotion == "anxiety"
    assert ib.dominant_emotion == "calm"
    assert ia.dominant_emotion != ib.dominant_emotion, (
        "the same situation produced the same insight for two different "
        "histories, so the twin is not personalised")
    # and each statement carries its own counts, not a generic template
    assert str(ia.n_supporting) in ia.statement
    assert str(ib.n_supporting) in ib.statement


def test_one_twin_never_sees_another_twins_history(lex):
    a = PersonalEmotionalTwin("User_A")
    b = PersonalEmotionalTwin("User_B")
    a.add_event(_ev("User_A", 1, "Exam tomorrow, slept badly.",
                    {"event": "examination", "sleep": "poor"}, lex))
    q = _ev("User_B", 5, "I have an exam tomorrow.",
            {"event": "examination", "sleep": "poor"}, lex)
    assert b.similar_episodes(q) == []
    assert b.pattern_insight(q).sufficient is False
