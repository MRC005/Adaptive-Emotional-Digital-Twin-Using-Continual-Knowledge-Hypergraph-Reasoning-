"""PIN THE REAL-DATA FINDING: PMData fails the frozen eligibility screen.

Runs only when the real PMData files are present (``data/raw/pmdata``);
skipped otherwise, so the suite stays green on a clean checkout.

Extract with:
    unzip -o data/raw/pmdata.zip 'participant-overview.xlsx' \
        'p*/pmsys/wellness.csv' 'p*/fitbit/resting_heart_rate.json' \
        -d data/raw/pmdata
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aedt.constants import DataStatus
from aedt.errors import DecisionRequired, NoEligibleParticipants
from aedt.io import get_adapter
from aedt.io.pmdata import PMDataAdapter

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "pmdata"
HAVE_REAL = any(ROOT.glob("p*/pmsys/wellness.csv"))

pytestmark = pytest.mark.skipif(
    not HAVE_REAL, reason="real PMData files absent; extract from the archive")


@pytest.fixture(scope="module")
def audit():
    return get_adapter("pmdata").audit(str(ROOT))


def test_audit_runs_on_real_files_and_is_stamped_real(audit):
    assert audit.data_status is DataStatus.REAL
    assert audit.local_files_available
    assert "simula.no/pmdata" in audit.source_status


def test_two_participants_have_no_primary_sensor_at_all(audit):
    """p12 and p13 ship no resting_heart_rate.json. Silently analysing 14 of
    16 while reporting 'PMData (16 participants)' would misstate the sample."""
    assert audit.participant_count == 14
    assert any("no resting_heart_rate.json" in r
               for r in audit.exclusion_reasons)


def test_scale_direction_is_not_documented_and_blocks_the_primary(audit):
    """PMSys `stress` is a bare integer with no label text anywhere in the
    release, so severity direction cannot be verified the way StudentLife and
    RELAX allow."""
    assert audit.stress_labels == ()
    assert dict(audit.code_to_severity_mapping) == {}
    assert "NOT DOCUMENTED AND NOT CONFIRMED" in (audit.self_report_scale or "")
    assert any("DIRECTION is not documented" in r
               for r in audit.exclusion_reasons)
    assert audit.eligible_for_primary_analysis is False


def test_blank_submissions_are_dropped_but_ambiguous_zeros_are_flagged(audit):
    m = dict(audit.missingness)
    assert m["blank_submissions_dropped"] > 0
    assert m["stress_out_of_documented_range"] > 0
    assert any("blank submissions" in n for n in audit.notes)


def test_halt_mode_refuses_the_ambiguous_zeros():
    """The conservative setting must stop rather than silently drop rows whose
    meaning the release does not define."""
    strict = PMDataAdapter(zero_stress_handling="halt")
    with pytest.raises(DecisionRequired) as e:
        strict.load(str(ROOT))
    msg = str(e.value)
    assert "outside the documented" in msg
    assert "Do NOT guess" in msg


def test_pmdata_fails_the_frozen_eligibility_screen(audit):
    """THE FINDING. Even the densest participants fail -- on report count, on
    Var(s) stability (A3), or on a sign flip."""
    assert audit.eligible_for_primary_analysis is False
    assert any("Only 4 of 14" in r for r in audit.exclusion_reasons)


def test_pipeline_refuses_to_produce_a_primary_result():
    from aedt.config import load_config
    from aedt.pipeline import run_pipeline
    with pytest.raises(NoEligibleParticipants):
        run_pipeline("pmdata", root=str(ROOT), config=load_config("pmdata"),
                     n_resamples=99, build_twins=False,
                     build_hypergraphs=False)


def test_mixed_timezone_join_is_handled():
    """wellness is tz-aware UTC, Fitbit resting HR is naive. Merging them
    directly raises; both must be pinned to one clock."""
    res = get_adapter("pmdata").load(str(ROOT))
    assert len(res.frame) > 1000, "the join produced almost nothing"
    assert res.frame["ts"].dt.tz is not None, "report timestamps lost their tz"
    # the join key must be a single unambiguous clock
    assert res.frame["day"].dt.tz is None, "join key must be a single clock"
    assert str(res.frame["day"].dtype).startswith("datetime64")
    audit = get_adapter("pmdata").audit(str(ROOT))
    assert "UTC calendar day" in (audit.sensor_report_alignment or "")
