"""StudentLife, ORIGINAL Dartmouth release: why it cannot support this method.

An earlier audit ran against a third-party RDS repackaging and concluded it was
defective. With the original archive in place that conclusion can be checked
properly, and the reason is now precise: the repackaging kept the ~10% of
records whose answer sits under a literal "null" key and dropped the ~90%
keyed "level".

The rejection itself does not rest on that. Two independent properties block
this archive, and these tests pin both:

  1. the only item with usable density is not ordered by severity;
  2. every properly ordered item is far too sparse, and the shortfall is large
     enough that no defensible threshold rescues it.
"""
from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest

from aedt.constants import DataStatus, MIN_REPORTS_PER_EPOCH
from aedt.inference.bootstrap import MIN_PARTICIPANTS_FOR_CI
from aedt.io import StudentLifeAdapter
from aedt.io.studentlife import normalise_ema_definition, parse_bracketed_options
from aedt.preprocess.reports import build_code_to_severity, detect_reversed_coding

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "raw" / "studentlife"
STRESS = DATA / "EMA" / "response" / "Stress"
have_data = pytest.mark.skipif(
    not STRESS.is_dir(),
    reason="StudentLife original release not present at data/raw/studentlife")


def _stress_records():
    for f in sorted(glob.glob(str(STRESS / "*.json"))):
        uid = os.path.basename(f).replace("Stress_", "").replace(".json", "")
        try:
            recs = json.load(open(f))
        except Exception:
            continue
        if isinstance(recs, list):
            yield uid, recs


# ------------------------------------------------------- container handling
@have_data
def test_ema_definition_is_a_list_in_the_original_release():
    raw = json.load(open(DATA / "EMA" / "EMA_definition.json"))
    assert isinstance(raw, list), "the original release ships a list, not a mapping"
    norm = normalise_ema_definition(raw)
    assert "Stress" in norm


def test_normalise_accepts_both_containers():
    assert normalise_ema_definition({"Stress": {"a": 1}}) == {"Stress": {"a": 1}}
    assert "Stress" in normalise_ema_definition([{"name": "Stress", "questions": []}])


def test_bracketed_options_are_parsed_and_their_codes_checked():
    opts = parse_bracketed_options("[1]Low, [2]Mid, [3]High, ")
    assert opts == ["Low", "Mid", "High"]
    from aedt.errors import DecisionRequired
    with pytest.raises(DecisionRequired):          # codes must be a 1..n run
        parse_bracketed_options("[1]Low, [3]High, ")


# --------------------------------------------------- the non-monotone scale
@have_data
def test_stress_options_are_not_ordered_by_severity():
    """The core reason this archive cannot be used as stored."""
    raw = normalise_ema_definition(json.load(open(DATA / "EMA" / "EMA_definition.json")))
    opts = parse_bracketed_options(raw["Stress"]["questions"][0]["options"])
    assert opts == ["A little stressed", "Definitely stressed", "Stressed out",
                    "Feeling good", "Feeling great"]
    code_to_sev = build_code_to_severity(opts)
    assert detect_reversed_coding(code_to_sev) is True, (
        "stored codes must be flagged as NOT in severity order")
    # 3 is the most stressed, 5 the least: the numbering is not monotone.
    assert code_to_sev[3] > code_to_sev[2] > code_to_sev[1] > code_to_sev[4] > code_to_sev[5]


@have_data
def test_adapter_remaps_by_label_rather_than_trusting_the_code():
    a = StudentLifeAdapter().audit(DATA)
    assert dict(a.code_to_severity_mapping) == {1: 3, 2: 4, 3: 5, 4: 2, 5: 1}


# --------------------------------------------------- the malformed records
@have_data
def test_about_a_tenth_of_records_use_a_null_key_and_it_also_holds_locations():
    keys = collections.Counter()
    null_kinds = collections.Counter()
    for _, recs in _stress_records():
        for r in recs:
            if not isinstance(r, dict):
                continue
            keys["level" if "level" in r else "null" if "null" in r else "other"] += 1
            if "null" in r:
                v = str(r["null"]).strip()
                null_kinds["digit" if v.isdigit() else "coordinate" if "," in v else "other"] += 1
    assert keys["level"] > keys["null"] * 5, "the well-formed key must be the majority"
    assert null_kinds["digit"] > 0 and null_kinds["coordinate"] > 0, (
        "the null key holds both recoverable answers and GPS strings, which is why "
        "only digits inside the code range may be accepted from it")


@have_data
def test_loader_recovers_the_well_formed_majority():
    """The repackaging kept the wrong key; the adapter must keep the right one."""
    res = StudentLifeAdapter().load(DATA)
    rec = res.provenance.get("response_recovery", {})
    assert rec.get("recovered_via_level", 0) > 1500, rec
    assert rec.get("recovered_via_level", 0) > 5 * rec.get("recovered_via_null-key", 1), rec


# ------------------------------------------------------------- the density
@have_data
def test_too_few_participants_have_enough_repeated_measurement():
    a = StudentLifeAdapter().audit(DATA)
    assert a.data_status is DataStatus.REAL          # the files ARE real and were read
    assert a.eligible_for_primary_analysis is False
    assert a.participant_count and a.participant_count > 40
    assert a.median_observations_per_participant < 2 * MIN_REPORTS_PER_EPOCH


@have_data
def test_rejection_does_not_depend_on_the_number_60():
    """Sweep the per-window minimum. If StudentLife only qualifies far below the
    pre-specified rule, the verdict is about the data, not about the threshold."""
    RECODE = {5: 1, 4: 2, 1: 3, 2: 4, 3: 5}
    per = collections.defaultdict(list)
    for uid, recs in _stress_records():
        for r in recs:
            if not isinstance(r, dict) or "resp_time" not in r:
                continue
            raw = r.get("level", r.get("null"))
            v = str(raw).strip() if raw is not None else ""
            if v.isdigit() and 1 <= int(v) <= 5:
                per[uid].append(int(r["resp_time"]))

    def qualifying(thr):
        n = 0
        for ts in per.values():
            t = np.array(sorted(ts))
            if len(t) < 2:
                continue
            mid = t.min() + (t.max() - t.min()) / 2
            if (t <= mid).sum() >= thr and (t > mid).sum() >= thr:
                n += 1
        return n

    assert qualifying(MIN_REPORTS_PER_EPOCH) < MIN_PARTICIPANTS_FOR_CI
    # and it stays short well below the pre-specified rule
    assert qualifying(40) < MIN_PARTICIPANTS_FOR_CI, (
        "if 40 per window sufficed, the rejection would be threshold-sensitive")


@have_data
def test_every_monotone_alternative_item_is_sparser_than_stress():
    """The obvious escape hatch -- use a properly ordered item -- does not work."""
    def total(kind, key):
        n = 0
        for f in glob.glob(str(DATA / "EMA" / "response" / kind / "*.json")):
            try:
                recs = json.load(open(f))
            except Exception:
                continue
            n += sum(1 for r in recs if isinstance(r, dict)
                     and str(r.get(key, "")).strip().isdigit())
        return n

    stress = total("Stress", "level")
    for kind, key in [("Behavior", "anxious"), ("Sleep", "rate"), ("Social", "number")]:
        assert total(kind, key) < stress, f"{kind}/{key} unexpectedly denser than Stress"


def test_audit_is_safe_without_files(tmp_path):
    a = StudentLifeAdapter().audit(tmp_path / "absent")
    assert a.local_files_available is False
    assert a.data_status is DataStatus.PLANNED
