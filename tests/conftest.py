"""Shared fixtures. All synthetic; deterministic under the frozen seed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedt.constants import SEED
from aedt.simulate.generator import cohort_to_long_frame, simulate_cohort


@pytest.fixture(scope="session")
def null_cohort():
    """P=50, 300/epoch, TRUE rho = 1.00 -- the calibration case."""
    return simulate_cohort(1.00, n_participants=50, n_per_epoch=300, seed=SEED)


@pytest.fixture(scope="session")
def null_frame(null_cohort):
    return cohort_to_long_frame(null_cohort)


@pytest.fixture(scope="session")
def shift_cohort():
    """P=50, 300/epoch, TRUE rho = 0.85 -- a real 15% recalibration."""
    return simulate_cohort(0.85, n_participants=50, n_per_epoch=300, seed=SEED)


@pytest.fixture(scope="session")
def shift_frame(shift_cohort):
    return cohort_to_long_frame(shift_cohort)


@pytest.fixture(scope="session")
def small_frame():
    """A cheap frame for structural tests that do not need statistical power."""
    return cohort_to_long_frame(
        simulate_cohort(0.85, n_participants=6, n_per_epoch=120, seed=SEED))


@pytest.fixture(scope="session")
def studentlife_fixture(tmp_path_factory):
    from aedt.io.fixtures import make_studentlife_fixture
    return make_studentlife_fixture(
        tmp_path_factory.mktemp("sl"), n_participants=24, days=60)


@pytest.fixture(scope="session")
def pmdata_fixture(tmp_path_factory):
    from aedt.io.fixtures import make_pmdata_fixture
    return make_pmdata_fixture(tmp_path_factory.mktemp("pm"),
                               n_participants=14, days=200)


@pytest.fixture(scope="session")
def relax_fixture(tmp_path_factory):
    from aedt.io.fixtures import make_relax_fixture
    return make_relax_fixture(tmp_path_factory.mktemp("rx"),
                              n_participants=14,
                              reports_per_participant=130)
