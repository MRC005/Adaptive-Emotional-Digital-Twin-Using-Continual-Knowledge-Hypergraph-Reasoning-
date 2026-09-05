"""The browser must compute the SAME estimator as the Python reference.

The deployed application runs the ordinal probit slope-ratio estimator in
JavaScript so that it genuinely computes without a backend. That creates a real
risk: two implementations of the same science can silently diverge.

This test removes the risk. It generates fixed cases with numpy, fits them in
Python, fits the identical arrays in Node, and requires the slopes and
cutpoints to agree. If someone edits either implementation and they drift
apart, this fails.

Skipped when Node is unavailable (Python-only environments still run the rest
of the suite).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aedt.models.ordinal import ordinal_probit_fit

ROOT = Path(__file__).resolve().parents[2]
ESTIMATOR = ROOT / "frontend" / "src" / "lib" / "estimator.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not ESTIMATOR.exists(),
    reason="node or the JS estimator is unavailable")

# slope agreement required between the two implementations
TOL_BETA = 1e-3
TOL_CUTS = 1e-3

CASES = [(1, 0.80, 5, 600), (2, -0.65, 5, 400), (3, 1.20, 5, 900),
         (4, 0.35, 7, 700), (5, 0.50, 5, 250)]


def _make(seed: int, beta: float, K: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    latent = beta * x + rng.normal(0, 1, n)
    cuts = np.quantile(latent, np.linspace(0, 1, K + 1)[1:K])
    y = (np.searchsorted(cuts, latent) + 1).astype(int)
    return x, y


@pytest.fixture(scope="module")
def js_results():
    payload = []
    for seed, beta, K, n in CASES:
        x, y = _make(seed, beta, K, n)
        payload.append({"x": x.tolist(), "y": y.tolist(), "K": K})
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.json"
        inp.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(td) / "run.mjs"
        # Node's ESM loader takes a URL, not a filesystem path. On Windows a
        # bare "C:\\..." is read as an unknown URL scheme and the import fails
        # with ERR_UNSUPPORTED_ESM_URL_SCHEME, which silently cost this test on
        # every Windows machine. as_uri() is correct on POSIX too.
        script.write_text(f"""
import {{ ordinalProbitFit }} from {json.dumps(ESTIMATOR.as_uri())};
import {{ readFileSync }} from "node:fs";
const cases = JSON.parse(readFileSync({json.dumps(str(inp))}, "utf8"));
console.log(JSON.stringify(cases.map(c => {{
  const f = ordinalProbitFit(c.x, c.y, c.K);
  return {{ converged: f.converged, beta: f.beta, cutpoints: f.cutpoints ?? [] }};
}})));
""", encoding="utf-8")
        r = subprocess.run(["node", str(script)], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_slope_agrees_with_python(js_results, idx):
    seed, beta, K, n = CASES[idx]
    x, y = _make(seed, beta, K, n)
    py = ordinal_probit_fit(x, y, K)
    js = js_results[idx]
    assert py.converged and js["converged"], (py.reason, js)
    assert abs(py.beta - js["beta"]) < TOL_BETA, (
        f"case {seed}: python beta {py.beta:.6f} vs JS {js['beta']:.6f} — the "
        "browser is no longer computing the same estimator as the reference")


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_cutpoints_agree_with_python(js_results, idx):
    seed, beta, K, n = CASES[idx]
    x, y = _make(seed, beta, K, n)
    py = ordinal_probit_fit(x, y, K)
    js = js_results[idx]
    assert len(js["cutpoints"]) == len(py.cutpoints) == K - 1
    worst = max(abs(a - b) for a, b in zip(py.cutpoints, js["cutpoints"]))
    assert worst < TOL_CUTS, f"case {seed}: cutpoints differ by {worst:.2e}"


def test_both_recover_the_known_slope(js_results):
    """Agreement is necessary but not sufficient — both must also be RIGHT.

    Checked only on the larger cases. At n = 250 a single sample carries ~30%
    sampling error on the slope, and the PYTHON reference shows the same error
    on the same data — so a tight tolerance there would be testing luck, not
    correctness. Agreement between the two implementations is asserted for
    every case above; unbiasedness of the estimator itself is established by
    tests/synthetic/test_known_answer.py, which averages over cohorts.
    """
    checked = 0
    for idx, (seed, beta, K, n) in enumerate(CASES):
        if n < 600:
            continue
        js = js_results[idx]
        rel = abs(js["beta"] - beta) / abs(beta)
        assert rel < 0.20, (
            f"case {seed} (n={n}): JS recovered {js['beta']:.4f} for a true "
            f"slope of {beta} — {rel:.1%} error")
        checked += 1
    assert checked >= 2, "no large-n case was actually checked"
