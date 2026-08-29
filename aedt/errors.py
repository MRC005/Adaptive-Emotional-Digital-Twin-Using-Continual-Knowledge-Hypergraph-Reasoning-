"""Fatal scientific errors. None of these is ever swallowed or defaulted.

Exit codes are distinct so that a wrapper script cannot accidentally proceed
past a failed gate. They match the codes frozen in ``final_audit.py``
(Round 16) and are asserted by the regression tests.
"""
from __future__ import annotations


class ScientificError(Exception):
    """Base class. Every subclass halts the pipeline with a non-zero status."""

    exit_code = 1


class DecisionRequired(ScientificError):
    """A scientific decision is unspecified and must NOT be guessed.

    Raised when the data disagrees with the frozen specification: an
    unrecognised self-report label, an unparseable timestamp field, a missing
    required column. The message must name the exact discrepancy.
    """

    exit_code = 2

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"DECISION REQUIRED: {super().__str__()}"


class RealDataUnavailable(ScientificError):
    """Real mode was requested but the dataset files are not present.

    Never substitute synthetic data for real data. This exception exists so
    that the substitution is impossible rather than merely discouraged.
    """

    exit_code = 6


class NoEligibleParticipants(ScientificError):
    """The eligibility screen excluded everybody."""

    exit_code = 3


class PlaceboUnrunnable(ScientificError):
    """Not enough epoch-1 data to run the placebo, which gates the primary."""

    exit_code = 4


class PlaceboFailed(ScientificError):
    """The placebo rejected. The primary analysis is NOT run.

    This is a finding, not a bug to work around.
    """

    exit_code = 5


class ConvergenceError(ScientificError):
    """The ordinal model did not converge where convergence was required."""

    exit_code = 7
