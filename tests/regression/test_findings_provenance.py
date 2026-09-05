"""Every number on the website must be traceable to an artefact, not to a file.

This test exists because the rule was already stated and was already broken:
``scripts/export_findings.py`` carried twelve ceiling statistics and four
cohort descriptors as literals, while the file it wrote claimed they came from
the experiment's result files. Nothing caught it, because nothing looked.

Three guards, in increasing strength:
  1. the exporter must not contain the reported values as literals;
  2. the keys the frontend reads must be the keys the computation produces;
  3. when the generated artefact exists, findings.json must equal it exactly.

Guard 3 skips while the archive is unavailable, and says so loudly rather than
passing quietly — a skipped check is not a satisfied one.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from aedt.audit.ceiling import ceiling_statistics

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts" / "export_findings.py"
FINDINGS = ROOT / "frontend" / "src" / "data" / "findings.json"
CEILING_RESULT = ROOT / "results" / "twin" / "ceiling.json"

#: The values the website reports. If any of these reappears as a literal in
#: the exporter, someone has typed a research number by hand again.
FORBIDDEN_LITERALS = {
    0.339, 0.115, 0.246, 0.0907, 0.008, 0.346, 0.239, 0.466, -0.243, 0.687,
    0.13, 0.17, 0.355, 194, 218, 4.8, 35348, 25966,
}

#: Blocks in findings.json that must each declare where they came from.
DATA_BLOCKS = ("cohort", "headline", "learning_curve", "ablation", "ceiling")


def _numeric_literals(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            out.add(node.value)
        # unary minus, e.g. -0.243
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)):
            out.add(-node.operand.value)
    return out


def test_the_exporter_contains_no_hand_typed_research_numbers():
    found = _numeric_literals(EXPORTER) & FORBIDDEN_LITERALS
    assert not found, (
        f"scripts/export_findings.py contains research values as literals: "
        f"{sorted(found)}. Every reported number must be read from a result "
        "file. This is the exact defect this test was written to end.")


def test_the_exporter_reads_the_generated_ceiling_artefact():
    src = EXPORTER.read_text(encoding="utf-8")
    assert "results/twin/ceiling.json" in src, (
        "the exporter no longer reads the computed ceiling artefact")
    assert "run_ceiling_analysis.py" in src, (
        "the exporter should name the script that produces its missing input")


def test_the_exporter_refuses_rather_than_writing_a_partial_file():
    """A missing input must stop the write, not silently drop a block."""
    src = EXPORTER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    # the refusal path must return before any write_text call
    returns_before_write = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) \
                and node.value.value == 6:
            returns_before_write = True
    assert returns_before_write, (
        "export_findings.main() must return exit code 6 when an input is "
        "missing, so a partial findings.json is never written")


def test_every_ceiling_key_the_site_reads_is_actually_computed():
    """A rename in the computation must fail here, not render 'undefined'."""
    published = set(json.loads(FINDINGS.read_text(encoding="utf-8"))["ceiling"])

    # a tiny synthetic frame is enough to enumerate the produced key set
    rows = []
    for p in range(5):
        for d in range(60):
            rows.append((f"p{p}", d, float((d * 7 + p) % 5 + 1)))
    frame = pd.DataFrame(rows, columns=["participant_id", "day_ord", "stress"])
    computed = set(ceiling_statistics(frame, value_col="stress",
                                      day_col="day_ord").to_dict())
    # these two come from the sensing screen in run_ceiling_analysis.py
    computed |= {"strongest_behaviour_r", "behaviour_variance_explained"}

    missing = published - computed
    assert not missing, (
        f"findings.json publishes ceiling keys nothing computes: "
        f"{sorted(missing)}")


def test_findings_declares_provenance_for_every_block():
    d = json.loads(FINDINGS.read_text(encoding="utf-8"))
    prov = d.get("_provenance")
    assert prov, "findings.json carries no _provenance block"
    for block in DATA_BLOCKS:
        assert block in d, f"{block} missing from findings.json"
        assert block in prov, f"{block} has no declared provenance"
    assert "_source" not in d, (
        "the old _source field claimed the ceiling came from the experiment "
        "result files; it did not, and the field must not come back")


def test_published_ceiling_matches_the_generated_artefact():
    """The real guard. Skips only while the archive is genuinely unavailable."""
    if not CEILING_RESULT.exists():
        pytest.skip(
            "results/twin/ceiling.json is absent, so the published ceiling "
            "CANNOT be verified. The College Experience archive is required; "
            "run scripts/run_ceiling_analysis.py once it is restored. Until "
            "then the values in findings.json are unverified and are marked "
            "as such in its _provenance block.")
    generated = json.loads(CEILING_RESULT.read_text(encoding="utf-8"))
    published = json.loads(FINDINGS.read_text(encoding="utf-8"))
    for key, value in published["ceiling"].items():
        assert generated["ceiling"][key] == value, (
            f"ceiling.{key}: site shows {value!r}, the artefact says "
            f"{generated['ceiling'][key]!r}. Re-run scripts/export_findings.py.")
    for key in ("participants", "years", "reports", "prediction_pairs"):
        assert generated["cohort"][key] == published["cohort"][key], (
            f"cohort.{key} disagrees with the generated artefact")
