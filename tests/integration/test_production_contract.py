"""The production API contract the deployed frontend depends on.

These exist because the deployment broke in ways unit tests could not catch:
the root route was missing, the emotion route was absent from the deployed
build, and /health returned "ok" while saying nothing about whether the model
was actually loaded. Each of those is now asserted here.

The live-deployment checks are opt-in:

    AEDT_LIVE_URL=https://aedt-api.onrender.com python3 -m pytest \\
        tests/integration/test_production_contract.py

Without that variable they skip, because a test suite that fails when someone
is offline is a test suite people learn to ignore.
"""
from __future__ import annotations

import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from backend.app import app                        # noqa: E402

LIVE = os.environ.get("AEDT_LIVE_URL", "").rstrip("/")
live_only = pytest.mark.skipif(not LIVE, reason="set AEDT_LIVE_URL to test the deployment")

#: Exactly what the frontend calls. If this list and the frontend disagree, the
#: browser gets a 404 and silently falls back -- the original failure.
REQUIRED_ROUTES = {
    ("GET", "/"), ("GET", "/health"), ("POST", "/api/emotion"),
}


@pytest.fixture(scope="module")
def client():
    # TestClient as a context manager runs lifespan, so the model is warmed
    # exactly as it is in production.
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------- the contract
def test_every_route_the_frontend_calls_exists(client):
    have = {(m, r.path) for r in app.routes if hasattr(r, "methods")
            for m in r.methods}
    for method, path in REQUIRED_ROUTES:
        assert (method, path) in have, f"{method} {path} is missing"


def test_root_is_informative_rather_than_404(client):
    """A bare URL returning {"detail":"Not Found"} is what made this look dead."""
    r = client.get("/")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "/health" in d["endpoints"] and "/api/emotion" in d["endpoints"]


def test_health_distinguishes_alive_from_model_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "model" in d, "health must report the model, not just the process"
    assert d["model"]["status"] in {"loaded", "unavailable"}
    assert "roberta" in d["model"]["name"].lower()
    if d["model"]["status"] == "unavailable":
        assert "reason" in d["model"], "an unavailable model must say why"


def test_health_leaks_no_paths_or_environment(client):
    blob = str(client.get("/health").json())
    for leak in ("/Users/", "/home/", "Traceback", "site-packages", "HF_TOKEN"):
        assert leak not in blob


# ------------------------------------------------------------ the endpoint
def test_emotion_returns_the_shape_the_frontend_parses(client):
    r = client.post("/api/emotion", json={
        "text": "My presentation is tomorrow and I slept only four hours."})
    if r.status_code == 503:
        pytest.skip("model not loaded in this environment")
    assert r.status_code == 200
    d = r.json()
    p = d["prediction"]
    for key in ("emotion", "confidence", "backend", "model", "raw_label",
                "distribution", "raw_top"):
        assert key in p, f"frontend reads prediction.{key}"
    assert p["backend"] == "transformer"
    assert 0.0 <= p["confidence"] <= 1.0
    assert "context" in d and "inference_ms" in d


def test_multi_label_scores_are_independent_not_a_distribution(client):
    """Sigmoid outputs must NOT be normalised: the UI explains them as independent."""
    r = client.post("/api/emotion", json={
        "text": "I am stressed, exhausted, and nervous about tomorrow."})
    if r.status_code == 503:
        pytest.skip("model not loaded in this environment")
    raw = r.json()["prediction"]["raw_top"]
    assert len(raw) >= 2
    total = sum(s for _, s in raw)
    # a softmax over the top-6 would land near 1.0; independent sigmoids do not
    assert not (0.97 <= total <= 1.03), (
        "the scores look normalised, which would make the multi-label "
        "explanation in the interface false")


def test_unavailable_model_returns_503_and_never_a_disguised_fallback(monkeypatch, client):
    """The endpoint must refuse rather than answer with a lexicon labelled 'model'."""
    from aedt.emotion import onnx_detect

    class Dead:
        def predict(self, text):
            return None

    monkeypatch.setattr(onnx_detect, "get_detector", lambda: Dead())
    r = client.post("/api/emotion", json={"text": "hello"})
    assert r.status_code == 503
    assert "not loaded" in r.json()["detail"].lower()


def test_input_validation(client):
    assert client.post("/api/emotion", json={"text": ""}).status_code == 422
    assert client.post("/api/emotion", json={"text": "   "}).status_code == 400
    assert client.post("/api/emotion", json={"text": "x" * 5000}).status_code == 422


def test_no_identifier_is_accepted_so_there_is_no_history_to_leak(client):
    r = client.post("/api/emotion",
                    json={"text": "hello", "person_id": "U1", "user_id": "abc"})
    if r.status_code == 503:
        pytest.skip("model not loaded in this environment")
    assert "U1" not in str(r.json()) and "abc" not in str(r.json())


def test_cors_allows_a_vercel_origin(client):
    """Vercel renames deployments constantly; the regex is what keeps them working."""
    r = client.options("/api/emotion", headers={
        "Origin": "https://aedt-frontend-abc123.vercel.app",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == \
        "https://aedt-frontend-abc123.vercel.app"


def test_cors_refuses_an_unrelated_origin(client):
    r = client.options("/api/emotion", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.headers.get("access-control-allow-origin") != "https://example.com"


# --------------------------------------------------------- the deployment
@live_only
def test_live_root_and_health():
    import urllib.request, json
    with urllib.request.urlopen(f"{LIVE}/", timeout=90) as f:
        assert json.load(f)["status"] == "ok"
    with urllib.request.urlopen(f"{LIVE}/health", timeout=90) as f:
        d = json.load(f)
    assert d["model"]["status"] == "loaded", (
        f"the deployed model is {d['model'].get('status')}: "
        f"{d['model'].get('reason', 'no reason given')}")


@live_only
def test_live_emotion_endpoint_runs_the_real_model():
    import urllib.request, json
    body = json.dumps({"text": "I am nervous because I have an important "
                               "presentation tomorrow."}).encode()
    req = urllib.request.Request(f"{LIVE}/api/emotion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as f:
        d = json.load(f)
    assert d["prediction"]["backend"] == "transformer"
    assert "roberta" in d["prediction"]["model"].lower()
