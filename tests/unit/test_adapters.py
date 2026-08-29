"""Dataset adapter tests using tiny SYNTHETIC fixtures shaped like each release."""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import DataStatus, DatasetRole
from aedt.errors import DecisionRequired, RealDataUnavailable, ScientificError
from aedt.io import ADAPTERS, get_adapter
from aedt.io.fixtures import FIXTURE_MARKER, is_fixture
from aedt.schemas import validate_long_frame


def test_every_adapter_is_registered_with_a_declared_role():
    assert set(ADAPTERS) == {"synthetic", "studentlife", "pmdata", "relax",
                             "wesad"}
    for name, a in ADAPTERS.items():
        assert isinstance(a.role, DatasetRole)
        assert a.name == name
        if a.role is not DatasetRole.SIMULATION:
            assert a.acquisition_instructions, f"{name} has no instructions"


def test_dataset_roles_match_the_frozen_hierarchy():
    assert get_adapter("studentlife").role is DatasetRole.PRIMARY_LONGITUDINAL
    assert get_adapter("relax").role is DatasetRole.LONGITUDINAL_ALTERNATIVE
    assert get_adapter("wesad").role is DatasetRole.BENCHMARK_PHYSIOLOGICAL
    assert get_adapter("pmdata").role is DatasetRole.CONDITIONAL_SECONDARY


# ------------------------------------------------------------- StudentLife
def test_studentlife_fixture_loads_into_the_canonical_frame(studentlife_fixture):
    res = get_adapter("studentlife").load(studentlife_fixture)
    validate_long_frame(res.frame, res.sensor, n_categories=res.n_categories)
    assert res.n_categories == 5
    assert res.sensor == "conversation_minutes"
    assert res.frame["pid"].nunique() == 24
    assert res.provenance["timestamp_format"].startswith("unix epoch")
    # the fixture stores the options OUT of severity order on purpose
    assert res.provenance["stored_codes_in_severity_order"] is False


def test_studentlife_audit_reports_every_mandated_field(studentlife_fixture):
    a = get_adapter("studentlife").audit(studentlife_fixture)
    assert a.local_files_available
    assert a.participant_count == 24
    assert a.observation_count > 0
    assert len(a.stress_labels) == 5
    assert dict(a.code_to_severity_mapping)
    assert a.timestamps_present is True
    assert a.longitudinal_span_days > 0
    assert a.conversation_context_available is True
    assert a.eligible_for_benchmark_analysis is False
    assert a.median_observations_per_participant > 0


def test_studentlife_halts_on_an_unknown_label(tmp_path):
    import json
    from aedt.io.fixtures import make_studentlife_fixture
    root = make_studentlife_fixture(tmp_path / "bad", n_participants=2, days=20)
    p = root / "EMA" / "EMA_definition.json"
    d = json.loads(p.read_text())
    d["Stress"]["responses"][0] = "Mildly perturbed"      # not in the spec
    p.write_text(json.dumps(d))
    with pytest.raises(DecisionRequired, match="differ from expected mapping"):
        get_adapter("studentlife").audit(root)


def test_studentlife_halts_when_conversation_files_are_missing(tmp_path):
    import shutil
    from aedt.io.fixtures import make_studentlife_fixture
    root = make_studentlife_fixture(tmp_path / "noconv", n_participants=2,
                                    days=20)
    shutil.rmtree(root / "sensing")
    with pytest.raises(DecisionRequired, match="No conversation CSVs"):
        get_adapter("studentlife").load(root)


# ------------------------------------------------------------------ PMData
def test_pmdata_fixture_loads(pmdata_fixture):
    res = get_adapter("pmdata").load(pmdata_fixture)
    validate_long_frame(res.frame, res.sensor, n_categories=res.n_categories)
    assert res.sensor == "resting_hr"
    assert res.frame["pid"].nunique() == 14


def test_pmdata_audit_records_that_labels_are_numeric_only(pmdata_fixture):
    a = get_adapter("pmdata").audit(pmdata_fixture)
    assert a.stress_labels == ()
    assert any("no label text" in n for n in a.notes)
    assert any("confirm the scale direction" in n.lower() for n in a.notes)


def test_pmdata_reports_missing_variables_rather_than_guessing(tmp_path):
    from aedt.io.fixtures import make_pmdata_fixture
    root = make_pmdata_fixture(tmp_path / "pm2", n_participants=3, days=100)
    for f in root.glob("p*/pmsys/wellness.csv"):
        f.unlink()
    a = get_adapter("pmdata").audit(root)
    assert not a.local_files_available
    assert "DECISION REQUIRED" in a.source_status
    assert "required variables unavailable" in a.source_status


# ------------------------------------------------------------------- RELAX
def test_relax_fixture_loads_through_the_causal_window(relax_fixture):
    res = get_adapter("relax").load(relax_fixture)
    validate_long_frame(res.frame, res.sensor, n_categories=res.n_categories)
    assert res.sensor == "heart_rate_bpm"
    assert res.n_categories == 7
    assert res.frame["pid"].nunique() == 14
    # alignment metadata proves causality was checked, on a tz-aware clock
    assert f"{res.sensor}_window_end" in res.frame.columns
    assert (res.frame[f"{res.sensor}_window_end"] <= res.frame["ts"]).all()
    # heart rate derived from IBI must be physiologically plausible
    assert 30 < res.frame[res.sensor].min() < res.frame[res.sensor].max() < 200


def test_relax_severity_is_reversed_from_the_anchor_text(relax_fixture):
    """ifb-2 is anchored 'excited'..'calm', so 7 = calm = LEAST stressed.
    Mapping by stored value would invert the scale."""
    res = get_adapter("relax").load(relax_fixture)
    df = res.frame
    assert (df.loc[df["raw_response"] == 7, "report"] == 1).all()
    assert (df.loc[df["raw_response"] == 1, "report"] == 7).all()
    assert res.provenance["severity_reversed"] is True


def test_relax_halts_when_the_released_anchors_differ(tmp_path):
    """If the release is re-anchored, the DIRECTION of severity can no longer
    be trusted, so the adapter must stop rather than guess."""
    import pandas as pd
    from aedt.io.fixtures import make_relax_fixture
    root = make_relax_fixture(tmp_path / "rx2", n_participants=3,
                              reports_per_participant=40)
    f = root / "metadata" / "questionnaires.xlsx"
    d = pd.read_excel(f, sheet_name="ifb")
    d.loc[d["question_id"] == "ifb-2", "answer_labels_en"] = "['calm', 'excited']"
    with pd.ExcelWriter(f) as w:
        d.to_excel(w, sheet_name="ifb", index=False)
    with pytest.raises(DecisionRequired) as e:
        get_adapter("relax").load(root)
    msg = str(e.value)
    assert "differ from expected mapping" in msg
    assert "cannot be trusted" in msg
    assert "Do NOT guess" in msg


def test_relax_timestamp_crosscheck_is_performed(relax_fixture):
    """manual_date (epoch ms) and readable_date must describe the same instant;
    otherwise the timezone of readable_date is unknown."""
    res = get_adapter("relax").load(relax_fixture)
    assert res.provenance["timestamp_crosscheck_max_delta_s"] < 1.0
    assert "epoch ms" in res.provenance["timestamp_source"]


def test_relax_drops_flagged_and_implausible_ibi_never_repairs(relax_fixture):
    res = get_adapter("relax").load(relax_fixture)
    q = res.provenance["sensor_quality"]
    assert q, "no per-participant sensor quality was recorded"
    one = next(iter(q.values()))
    assert one["n_dropped_device_flagged"] > 0, (
        "the fixture plants device-flagged samples; they must be dropped")
    assert one["n_kept"] < one["n_raw"]


def test_relax_rejects_an_unknown_item():
    from aedt.io.relax import RelaxAdapter
    with pytest.raises(DecisionRequired, match="Unknown RELAX item"):
        RelaxAdapter(item="not-an-item")


# ------------------------------------------------------------------- WESAD
def test_wesad_cannot_support_the_longitudinal_estimand():
    a = get_adapter("wesad")
    assert a.can_support_longitudinal_estimand is False
    from aedt.io.wesad import BENCHMARK_ONLY_REASON, assert_benchmark_only
    assert "cannot identify" in BENCHMARK_ONLY_REASON
    with pytest.raises(ScientificError, match="CANNOT SUPPORT THE PRIMARY"):
        assert_benchmark_only("wesad")


def test_only_studentlife_relax_and_pmdata_claim_longitudinal_support():
    longitudinal = {n for n, a in ADAPTERS.items()
                    if a.can_support_longitudinal_estimand}
    assert longitudinal == {"synthetic", "studentlife", "pmdata", "relax"}


# --------------------------------------------------------------- fixtures
def test_a_fixture_directory_is_marked_and_detectable(studentlife_fixture):
    assert (studentlife_fixture / FIXTURE_MARKER).exists()
    assert is_fixture(studentlife_fixture)
    assert not is_fixture("/tmp")
    assert not is_fixture(None)


def test_fixture_marker_says_it_is_not_real(studentlife_fixture):
    txt = (studentlife_fixture / FIXTURE_MARKER).read_text()
    assert "SYNTHETIC FIXTURE" in txt
    assert "NO real participant data" in txt
