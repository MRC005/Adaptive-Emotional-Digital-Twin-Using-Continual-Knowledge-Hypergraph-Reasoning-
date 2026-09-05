"""The Explorer must show the study, not a flattering version of it.

The pages it feeds are the ones a panel will look at, so the risks are not
cosmetic: a participant record that disagrees with the aggregate, an invented
threshold, a fabricated confidence interval, or a source identifier leaking out
of an "anonymised" file. Each of those is asserted against here.

The strongest guard is the first one: averaging the per-participant records has
to reproduce the frozen experiment's published aggregate exactly. If it ever
does not, the Explorer is showing something the study did not find.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "frontend" / "src" / "data" / "participants.json"
FINDINGS = ROOT / "frontend" / "src" / "data" / "findings.json"
EXPORTER = ROOT / "scripts" / "export_participant_explorer.py"
EXPLORER = ROOT / "frontend" / "src" / "ui" / "explorer.js"
LONGITUDINAL = ROOT / "frontend" / "src" / "lib" / "longitudinal.js"
PARTICIPANTS_LIB = ROOT / "frontend" / "src" / "lib" / "participants.js"

pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="participants.json absent; run scripts/export_participant_explorer.py")


@pytest.fixture(scope="module")
def data():
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def findings():
    return json.loads(FINDINGS.read_text(encoding="utf-8"))


# ------------------------------------------------------- agreement with study
@pytest.mark.parametrize("K", ["20", "80"])
def test_the_participant_records_average_to_the_published_aggregate(data, K):
    """The whole basis for showing per-participant numbers at all."""
    scored = [p for p in data["participants"] if p["by_K"][K].get("scored")]
    assert scored, f"no participants scored at K={K}"
    twin = sum(p["by_K"][K]["twin"]["macro_f1"] for p in scored) / len(scored)
    pers = sum(p["by_K"][K]["persistence"]["macro_f1"] for p in scored) / len(scored)

    exp = ROOT / "results" / "twin" / "twin_experiment.json"
    if not exp.exists():
        pytest.skip("results/twin/twin_experiment.json absent — the aggregate "
                    "CANNOT be cross-checked. Re-run the frozen experiment.")
    agg = {str(x["K"]): x for x in json.loads(
        exp.read_text(encoding="utf-8"))["results"]}[K]["models"]
    assert twin == pytest.approx(agg["T_twin"]["macro_f1_mean"], abs=1e-6), (
        "the per-participant twin scores no longer average to the published "
        "figure; the Explorer would be showing a different experiment")
    assert pers == pytest.approx(agg["B1_persistence"]["macro_f1_mean"], abs=1e-6)


def test_persistence_still_wins_overall(findings):
    """The conclusion the interface must never quietly reverse."""
    m = findings["headline"]["models"]
    assert m["B1_persistence"]["macro_f1"] > m["T_twin"]["macro_f1"]
    assert round(m["T_twin"]["macro_f1"], 4) == 0.2793
    assert round(m["B1_persistence"]["macro_f1"], 4) == 0.3320


def test_the_individual_picture_differs_from_the_aggregate(data):
    """A real finding, and the reason the Explorer exists."""
    K = str(data["K_primary"])
    scored = [p for p in data["participants"] if p["by_K"][K].get("scored")]
    better = [p for p in scored if p["by_K"][K]["twin_better"]]
    assert 0 < len(better) < len(scored), (
        "either nobody or everybody was better served by the twin; the "
        "per-participant view would have nothing to show")
    assert data["n_twin_better_primary"] == len(better)


# ----------------------------------------------------------------- integrity
def test_the_full_observation_series_are_not_redistributed(data):
    """The archive says NOT redistributed; anonymisation does not override that.

    All participants keep their summary statistics -- derived values of the kind
    a paper prints in a table. Only a small, stated illustrative subset ships
    its raw observation series, and the complete set stays under results/,
    which is gitignored.
    """
    withs = [p for p in data["participants"] if p.get("series")]
    total = len(data["participants"])
    assert data["series_mode"] == "illustrative", (
        f"series_mode is {data['series_mode']!r}; 'all' would redistribute "
        "every participant's observations")
    assert 0 < len(withs) <= 5, (
        f"{len(withs)} of {total} participants ship observation series; the "
        "shipped subset must stay small and illustrative")
    assert len(withs) == data["n_series_published"]
    assert {p["id"] for p in withs} == set(data["illustrative_ids"])
    assert "not redistributed" in data["_redistribution"].lower()
    assert data["illustrative_rule"], "the subset rule must be stated in the file"

    # every participant keeps the statistics the Explorer needs
    for p in data["participants"]:
        assert "autocorrelation" in p and "n_observations" in p
        assert "by_K" in p
        if not p.get("series"):
            assert p.get("series_published") is False


def test_the_illustrative_subset_is_chosen_by_predictability_not_performance(data):
    """A subset picked by how well the twin did would flatter the result."""
    rule = data["illustrative_rule"].lower()
    assert "autocorrelation" in rule
    for word in ("macro", "f1", "twin", "persistence", "advantage"):
        assert word not in rule, (
            f"the subset rule mentions {word!r}; it must not depend on outcome")


def test_the_local_full_record_is_gitignored_and_not_shipped():
    full = ROOT / "results" / "explorer" / "participants_full.json"
    if not full.exists():
        pytest.skip("results/explorer/participants_full.json absent; run the "
                    "exporter to regenerate the local record")
    import subprocess
    r = subprocess.run(["git", "check-ignore", str(full)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "the full observation record is NOT gitignored; it would be committed")
    assert full.stat().st_size > DATA.stat().st_size, (
        "the local record should be larger than the shipped one")


def test_no_source_identifier_or_calendar_date_is_published():
    blob = DATA.read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{32}\b", blob), "a source id leaked"
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", blob), "a calendar date leaked"
    ids = {p["id"] for p in json.loads(blob)["participants"]}
    assert all(re.fullmatch(r"P\d+", i) for i in ids), f"unexpected labels: {ids}"


def test_the_only_categorical_rule_comes_from_existing_methodology(data):
    """No invented HIGH/MEDIUM/LOW. The thresholds must be the published ones."""
    from aedt.audit.ceiling import (NEAR_UNPREDICTABLE_BELOW,
                                    WELL_PREDICTABLE_ABOVE)
    rule = data["_definitions"]["predictability_band"]
    assert str(NEAR_UNPREDICTABLE_BELOW) in rule
    assert str(WELL_PREDICTABLE_ABOVE) in rule
    for p in data["participants"]:
        r, b = p["autocorrelation"], p["predictability_band"]
        if r is None:
            assert b is None
        elif r < NEAR_UNPREDICTABLE_BELOW:
            assert b == "near-unpredictable", p["id"]
        elif r > WELL_PREDICTABLE_ABOVE:
            assert b == "well-predictable", p["id"]
        else:
            assert b == "typical", p["id"]


def test_no_per_participant_uncertainty_is_fabricated(data):
    """The bootstrap is over the cohort; a single person has no interval."""
    blob = json.dumps(data).lower()
    for word in ("ci_low", "ci_high", "confidence_interval", '"ci"'):
        assert word not in blob, f"{word} appears in a per-participant record"
    assert "no per-participant confidence intervals" in data["_uncertainty"].lower()


def test_undefined_metrics_are_null_never_zero(data):
    """qwk is undefined for single-class predictions; null says so, 0.0 lies."""
    seen_null = False
    for p in data["participants"]:
        for K, blk in p["by_K"].items():
            if not blk.get("scored"):
                continue
            for model in ("twin", "persistence"):
                if blk[model].get("qwk") is None:
                    seen_null = True
    assert seen_null, ("no undefined kappa found at all — either the data "
                       "changed or NaN is being written as a number again")


# -------------------------------------------------------------- the frontend
def test_the_frontend_hardcodes_no_research_numbers():
    """Every figure must come from a data file, not from a component."""
    forbidden = ("0.2793", "0.3320", "0.2847", "0.0104", "0.0185")
    for path in (EXPLORER, PARTICIPANTS_LIB, LONGITUDINAL):
        src = path.read_text(encoding="utf-8")
        found = [f for f in forbidden if f in src]
        assert not found, f"{path.name} hardcodes research values: {found}"


def test_the_chart_refuses_to_interpolate_across_gaps():
    src = LONGITUDINAL.read_text(encoding="utf-8")
    assert "MAX_JOIN_DAYS" in src
    assert "segments" in src, "no gap-splitting routine found"
    low = src.lower()
    assert "confidence band" in low or "no confidence bands" in low, (
        "the file should state that no uncertainty band is drawn")


def test_the_exporter_fits_nothing_and_defines_no_new_metric():
    src = EXPORTER.read_text(encoding="utf-8")
    for banned in ("LogisticRegression", "StandardScaler", "GridSearch",
                   "predict_proba", "cross_val"):
        assert banned not in src, f"{banned} appears in the exporter"
    assert "run_twin_experiment import" in src, (
        "the exporter must reuse the frozen experiment rather than reimplement it")
    assert "write_experiment_record" in src, "no reproducibility record written"


# ------------------------------------------------------- cross-page buttons
UI_DIR = ROOT / "frontend" / "src" / "ui"


def test_every_page_that_renders_a_cross_page_button_also_wires_one():
    """A button that does nothing is worse than no button.

    Two of these shipped dead: `discovered.js` rendered a `data-go2` button and
    attached no listener at all, and `twouser.js` wired only the FIRST match
    with `querySelector`, so its second button stopped working as soon as the
    staged demo rendered another one. Both are caught by requiring the plural
    form wherever such a button is rendered.
    """
    offenders = []
    for path in sorted(UI_DIR.glob("*.js")):
        src = path.read_text(encoding="utf-8")
        renders = 'data-go2="' in src
        if not renders:
            continue
        wires_all = 'querySelectorAll("[data-go2]")' in src
        wires_one = 'querySelector("[data-go2]")' in src and not wires_all
        if not wires_all:
            offenders.append(
                f"{path.name}: renders a data-go2 button but "
                + ("wires only the first with querySelector"
                   if wires_one else "attaches no listener"))
    assert not offenders, "dead cross-page buttons:\n  " + "\n  ".join(offenders)


# ------------------------------------------------------- number formatting
#: Result fields that are floats. Interpolating one of these straight into
#: markup prints its full double precision.
FLOAT_FIELDS = (
    "within_person_autocorrelation", "variance_explained", "icc_between_person",
    "strongest_behaviour_r", "behaviour_variance_explained",
    "per_person_r_median", "early_late_r", "macro_f1", "accuracy", "mae",
    "mean_diff", "ci_low", "ci_high", "twin_minus_persistence_macro_f1",
    "autocorrelation", "variability_sd", "mean_stress",
)


def test_no_page_interpolates_a_raw_float_into_markup():
    """0.3395326907808691 reached the live page this way.

    The hand-typed ceiling values were already short, so bare interpolation
    looked fine until the figures were regenerated at full precision. Every
    float must go through a formatter.
    """
    import re
    offenders = []
    pattern = re.compile(
        r"\$\{\s*[A-Za-z_$][\w$]*\.(?:[\w$]+\.)*(" + "|".join(FLOAT_FIELDS)
        + r")(?:\[\d\])?\s*\}")
    for path in sorted(list(UI_DIR.glob("*.js"))
                       + [ROOT / "frontend" / "src" / "main.js"]):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                offenders.append(f"{path.name}:{n} {m.group(0)}")
    assert not offenders, (
        "raw float interpolations (wrap in f2/f3/f4 or toFixed):\n  "
        + "\n  ".join(offenders))
