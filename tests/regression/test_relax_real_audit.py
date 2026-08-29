"""PIN THE REAL-DATA FINDING: RELAX fails the frozen eligibility screen.

These tests run ONLY when the real RELAX files are present
(``data/raw/relax``). They are skipped otherwise, so the suite stays green on a
clean checkout -- but where the data IS present they guarantee that the honest
negative result cannot be quietly lost by a later change.

Acquire with:  python scripts/fetch_relax.py --root data/raw/relax
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aedt.constants import DataStatus, MIN_REPORTS_PER_EPOCH
from aedt.errors import NoEligibleParticipants
from aedt.io import get_adapter

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "relax"
HAVE_REAL = (ROOT / "questionnaire_responses.xlsx").exists() and any(
    ROOT.glob("data/*/ibi_data.parquet"))

pytestmark = pytest.mark.skipif(
    not HAVE_REAL,
    reason="real RELAX files absent; run scripts/fetch_relax.py to enable")


@pytest.fixture(scope="module")
def audit():
    return get_adapter("relax").audit(str(ROOT))


def test_audit_runs_on_real_files_and_is_stamped_real(audit):
    assert audit.data_status is DataStatus.REAL
    assert audit.local_files_available
    assert "zenodo.20701999" in audit.source_status


def test_severity_direction_is_verified_not_assumed(audit):
    """ifb-2 is anchored 'excited'..'calm', so the scale is REVERSED. If this
    ever silently flips, every RELAX conclusion inverts."""
    assert audit.stress_labels == ("excited", "calm")
    m = dict(audit.code_to_severity_mapping)
    assert m[1] == 7 and m[7] == 1, f"severity map is not reversed: {m}"
    assert any("VERIFIED against the released anchor text" in n
               for n in audit.notes)


def test_timestamps_are_utc_and_cross_checked(audit):
    assert audit.timestamps_present is True
    assert "epoch ms" in (audit.timestamp_format or "")
    assert "UTC" in (audit.timezone or "")


def test_relax_fails_the_frozen_eligibility_screen(audit):
    """THE FINDING. RELAX is longitudinal, ordinal and well documented, and it
    still cannot support rho* because the self-report density is about half
    what the frozen screen requires."""
    assert audit.eligible_for_primary_analysis is False
    assert audit.participant_count == 31
    assert audit.median_observations_per_participant < 2 * MIN_REPORTS_PER_EPOCH
    reason = " ".join(audit.exclusion_reasons)
    assert "0 of 31" in reason
    assert "cannot support the frozen PRIMARY endpoint" in reason


def test_no_participant_reaches_the_required_density(audit):
    per = dict(audit.observations_per_participant)
    assert per, "no per-participant counts recorded"
    assert max(per.values()) < 2 * MIN_REPORTS_PER_EPOCH, (
        f"the densest participant now has {max(per.values())} reports; if this "
        "exceeds 120 the documented finding has changed and the docs must be "
        "updated rather than this test deleted")


def test_missingness_is_reported_not_imputed(audit):
    m = dict(audit.missingness)
    assert m["reports_without_sensor_coverage"] > 0
    assert m["ibi_samples_discarded"] > 0
    assert any("DROPPED, never repaired or imputed" in n for n in audit.notes)


def test_pipeline_refuses_to_produce_a_primary_result():
    """The whole point: the software will not manufacture an estimate from a
    dataset that fails its own screen."""
    from aedt.config import load_config
    from aedt.pipeline import run_pipeline
    with pytest.raises(NoEligibleParticipants):
        run_pipeline("relax", root=str(ROOT), config=load_config("relax"),
                     n_resamples=99, build_twins=False,
                     build_hypergraphs=False)
