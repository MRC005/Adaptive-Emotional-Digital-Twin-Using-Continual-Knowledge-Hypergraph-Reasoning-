"""The participant pages must ship SYNTHETIC demonstration data, and say so.

No participant-level research record is distributed with this application.
The Explorer, the two-person comparison and Twin Evolution all draw simulated
people written by
``scripts/make_synthetic_demo_data.py``. These tests check that the shipped
files are exactly what that script produces, that every one of them declares
itself synthetic, that the script can read nothing but its own constants, and
that every page drawing them shows the synthetic-data label.

Nothing here runs a research model or reads a result file.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "frontend" / "src" / "data"
DATA = DATA_DIR / "participants.json"
TRAJECTORIES = DATA_DIR / "trajectories.json"
COHORT = DATA_DIR / "cohort_summary.json"
GENERATOR = ROOT / "scripts" / "make_synthetic_demo_data.py"
EXPLORER = ROOT / "frontend" / "src" / "ui" / "explorer.js"
LONGITUDINAL = ROOT / "frontend" / "src" / "lib" / "longitudinal.js"
PARTICIPANTS_LIB = ROOT / "frontend" / "src" / "lib" / "participants.js"
UI = ROOT / "frontend" / "src" / "ui"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return _load(DATA)


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("make_synthetic_demo_data", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ it is synthetic
@pytest.mark.parametrize("path", [DATA, TRAJECTORIES])
def test_every_participant_file_declares_itself_synthetic(path):
    doc = _load(path)
    assert doc["_synthetic"] is True
    assert doc["_provenance"].startswith("SYNTHETIC DEMONSTRATION DATA")
    assert "REAL" not in doc["_provenance"]


def test_the_shipped_files_are_exactly_what_the_generator_writes(generator):
    """Same seed, same order of draws as ``main``: the files must match."""
    rng = random.Random(generator.SEED)
    assert _load(TRAJECTORIES) == generator.make_trajectories(rng)
    assert _load(DATA) == generator.make_participants(rng)


def test_the_generator_can_read_nothing_but_its_own_constants():
    src = GENERATOR.read_text(encoding="utf-8")
    imports = set(re.findall(r"^import\s+([\w.]+)", src, flags=re.M)) | set(
        re.findall(r"^from\s+([\w.]+)\s+import", src, flags=re.M))
    assert imports <= {"json", "math", "random", "pathlib"}, imports
    code = src.replace("research_results", "")
    for banned in ("data/raw", "read_csv", "pandas", "numpy", "import aedt", "from aedt",
                   "general_ema", ".npz", "pickle", "predictions_", '"results"'):
        assert banned not in code, banned
    # the one file it opens is the page's own data file, to replace one section
    assert src.count("read_text") == 1 and "results.read_text" in src


def test_no_source_identifier_or_calendar_date_is_published():
    for path in (DATA, TRAJECTORIES):
        blob = path.read_text(encoding="utf-8")
        assert not re.search(r"\b[0-9a-f]{32}\b", blob), "an identifier-like token"
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b", blob), "a calendar date"
    ids = {p["id"] for p in _load(DATA)["participants"]}
    assert all(re.fullmatch(r"S\d\d", i) for i in ids), ids


# ------------------------------------------------------------------- schema
def test_the_records_have_the_shape_the_pages_read(data):
    assert data["split"] == "synthetic" and data["K_primary"] in data["K_grid"]
    assert data["n_test_participants"] == len(data["participants"])
    kp = str(data["K_primary"])
    scored = [p for p in data["participants"] if p["by_K"][kp]["scored"]]
    assert data["n_scored_primary"] == len(scored) > 0
    assert data["n_twin_better_primary"] == sum(p["by_K"][kp]["twin_better"] for p in scored)
    assert set(data["illustrative_ids"]) <= {p["id"] for p in data["participants"]}
    for p in data["participants"]:
        assert p["predictability_band"] in ("well-predictable", "typical", "near-unpredictable")
        assert len(p["series"]) == p["n_observations"] and p["series"][0]["day"] == 0
        for r in p["series"]:
            assert 1 <= r["observed"] <= 5 and isinstance(r["warmup"], bool)
        for blk in p["by_K"].values():
            if blk["scored"]:
                for model in ("twin", "persistence"):
                    assert 0 <= blk[model]["macro_f1"] <= 1 and blk[model]["qwk"] is None
            else:
                assert blk["reason"]


def test_the_walkthrough_rows_carry_every_field_the_page_prints():
    """evolution.js prints each row's persistence forecast beside the stand-in
    forecast. A row without the field is rendered as the word "undefined"."""
    doc = _load(TRAJECTORIES)
    needed = {"day", "observed", "warmup", "predicted", "baseline", "actual_next"}
    for person in doc["participants"]:
        rows = person["series"]
        assert rows, "a simulated person has no rows"
        for i, row in enumerate(rows):
            assert needed <= set(row), sorted(needed - set(row))
            assert row["baseline"] == row["observed"]
            assert row["actual_next"] == (rows[i + 1]["observed"] if i + 1 < len(rows) else None)


def test_no_interval_is_invented_for_a_simulated_person(data):
    blob = json.dumps(data).lower()
    for word in ("ci_low", "ci_high", "confidence_interval", '"ci"'):
        assert word not in blob


def test_the_real_aggregate_counts_live_in_their_own_file():
    c = _load(COHORT)
    assert set(c) >= {"K", "n_test_participants", "n_scored", "n_twin_better", "_sources"}
    assert "participants" not in c
    assert all(isinstance(c[k], int) for k in ("K", "n_test_participants", "n_scored",
                                                "n_twin_better"))


# ------------------------------------------------------------------- labels
@pytest.mark.parametrize("page", ["explorer.js", "twouser.js", "evolution.js"])
def test_every_page_drawing_simulated_people_shows_the_label(page):
    src = (UI / page).read_text(encoding="utf-8")
    assert "syntheticDataBanner(" in src, f"{page} draws demo people without the label"


def test_no_page_calls_the_simulated_people_real():
    for path in sorted(UI.glob("*.js")):
        src = path.read_text(encoding="utf-8")
        for phrase in ("Real anonymised longitudinal data", "two REAL people",
                       "nothing here is synthetic", "REAL held-out participant",
                       "opened once"):
            assert phrase not in src, f"{path.name}: {phrase!r}"


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
