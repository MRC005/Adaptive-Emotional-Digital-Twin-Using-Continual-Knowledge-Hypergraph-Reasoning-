"""PIN THE FINDING: the StudentLife RDS repackaging is DEFECTIVE.

Runs only when the converted interim CSVs are present; skipped otherwise.

    python scripts/convert_studentlife_rds.py

This failure is qualitatively different from RELAX's and PMData's. Those two
datasets are intact and simply cannot support the method. This archive is a
*damaged conversion* of a dataset that almost certainly can — so the tests
below pin the diagnosis, not a verdict on StudentLife itself.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aedt.constants import DataStatus
from aedt.errors import DecisionRequired
from aedt.io import get_adapter

ROOT = Path(__file__).resolve().parents[2] / "data" / "interim" / "studentlife"
HAVE = (ROOT / "stress_ema.csv").exists() and (ROOT / "conversation.csv").exists()

pytestmark = pytest.mark.skipif(
    not HAVE, reason="converted StudentLife CSVs absent; "
                     "run scripts/convert_studentlife_rds.py")


@pytest.fixture(scope="module")
def audit():
    return get_adapter("studentlife").audit(str(ROOT))


def test_the_stress_response_column_is_unnamed_and_mostly_empty():
    """The response arrived in a column literally named 'null'. Note the
    converter must NOT let pandas read that name back as NaN."""
    d = pd.read_csv(ROOT / "stress_ema.csv",
                    converters={"response_column_name": str})
    assert str(d["response_column_name"].iloc[0]) == "null"
    assert d["response"].isna().mean() > 0.8


def test_only_a_tiny_fraction_of_responses_survived(audit):
    """~735 EMA responses per student are expected; this archive has a
    maximum of 6."""
    assert audit.observation_count == 122
    assert audit.median_observations_per_participant <= 5
    per = dict(audit.observations_per_participant)
    assert max(per.values()) < 10, (
        "if this archive now yields a realistic number of responses, the "
        "repackaging has been fixed and the documented finding must be "
        "revisited rather than this test deleted")


def test_the_sensor_side_is_intact(audit):
    """The diagnosis depends on this: conversation data is perfect, so the
    failure is specific to the EMA conversion."""
    assert any("SENSOR SIDE IS INTACT" in n for n in audit.notes)
    assert dict(audit.missingness)["conversation_episodes_invalid"] == 0.0
    assert audit.conversation_context_available is True


def test_the_codebook_is_absent_so_severity_cannot_be_verified(audit):
    """EMA_definition.json is what the frozen specification keys its remap on."""
    assert audit.stress_labels == ()
    assert dict(audit.code_to_severity_mapping) == {}
    assert "UNVERIFIABLE" in (audit.self_report_scale or "")
    assert any("EMA_definition.json is absent" in r
               for r in audit.exclusion_reasons)


def test_reports_and_sensing_do_not_overlap_in_time(audit):
    assert "zero temporal overlap" in (audit.sensor_report_alignment or "")
    assert any("NOT ONE stress response" in r for r in audit.exclusion_reasons)


def test_audit_is_produced_even_though_loading_fails(audit):
    """A dataset that fails still deserves a complete, machine-readable audit
    record saying precisely why."""
    assert audit.data_status is DataStatus.REAL
    assert audit.local_files_available is True
    assert audit.eligible_for_primary_analysis is False
    assert audit.participant_count == 46
    assert len(audit.exclusion_reasons) >= 4


def test_the_diagnosis_blames_the_conversion_not_studentlife(audit):
    """This matters for the recommendation: the fix is a different download,
    not a change to the method."""
    joined = " ".join(audit.notes)
    assert "defect of the repackaging" in joined
    assert "NOT a property of StudentLife" in joined
    assert "ORIGINAL Dartmouth release" in joined


def test_loading_refuses_rather_than_returning_junk():
    with pytest.raises(DecisionRequired) as e:
        get_adapter("studentlife").load(str(ROOT))
    assert "ORIGINAL Dartmouth release" in str(e.value)


def test_pipeline_refuses_to_produce_a_primary_result():
    from aedt.config import load_config
    from aedt.pipeline import run_pipeline
    with pytest.raises(DecisionRequired):
        run_pipeline("studentlife", root=str(ROOT),
                     config=load_config("studentlife"), n_resamples=99,
                     build_twins=False, build_hypergraphs=False)
