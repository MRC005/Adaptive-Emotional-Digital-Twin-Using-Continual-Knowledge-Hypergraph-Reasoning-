"""Every figure carries a visible REAL / SYNTHETIC / PLANNED badge.

"Do not let a single number, plot or sentence imply that any result came from
real data." One blurred label costs more than every missing module combined.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from aedt.constants import DataStatus
from aedt.reporting.tables import write_table
from aedt.viz.style import STATUS_COLOURS, save_figure, stamp


def test_saving_an_unstamped_figure_is_refused(tmp_path):
    fig, _ax = plt.subplots()
    with pytest.raises(ValueError, match="unstamped figure"):
        save_figure(fig, tmp_path / "x.png")
    plt.close(fig)


def test_stamp_text_appears_on_the_figure():
    for status in (DataStatus.REAL, DataStatus.SYNTHETIC, DataStatus.PLANNED):
        fig, _ax = plt.subplots()
        stamp(fig, status)
        texts = [t.get_text() for t in fig.texts]
        assert any(status.value in t for t in texts)
        plt.close(fig)


def test_each_status_has_a_distinct_colour():
    assert len(set(STATUS_COLOURS.values())) == 3


def test_synthetic_figures_carry_the_not_evidence_note():
    fig, _ax = plt.subplots()
    stamp(fig, DataStatus.SYNTHETIC)
    assert any("Not evidence about humans" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_writing_an_unstamped_table_is_refused(tmp_path):
    with pytest.raises(ValueError, match="without a data_status"):
        write_table(pd.DataFrame({"a": [1]}), tmp_path / "t")


def test_written_table_declares_its_status_in_the_markdown(tmp_path):
    p = write_table(pd.DataFrame({"a": [1]}), tmp_path / "t",
                    status=DataStatus.SYNTHETIC, title="Demo")
    md = p.with_suffix(".md").read_text()
    assert "**DATA STATUS: SYNTHETIC**" in md
    assert p.exists() and p.with_suffix(".md").exists()


def test_two_curve_plot_is_stamped(tmp_path, small_frame):
    from aedt.estimators.slope_ratio import fit_person_epochs
    from aedt.viz.curves import two_curve_plot
    g = small_frame[small_frame["pid"] == "p00"]
    fits = fit_person_epochs(g, "conversation_minutes", 5, pid="p00")
    if not (fits[0].converged and fits[1].converged):
        pytest.skip("fixture participant did not converge")
    out = two_curve_plot(fits, pid="p00", data_status=DataStatus.SYNTHETIC,
                         path=tmp_path / "c.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_two_curve_plot_refuses_a_nonconvergent_fit(small_frame):
    from aedt.schemas import OrdinalFit
    from aedt.viz.curves import two_curve_plot
    bad = OrdinalFit(pid="p", epoch=1, beta=float("nan"), cutpoints=(), n=0,
                     n_categories=5, converged=False, reason="degenerate")
    with pytest.raises(ValueError, match="two convergent fits"):
        two_curve_plot({0: bad, 1: bad}, pid="p")
