"""The /api/emotion endpoint: contract, statelessness, and no identifiers."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from backend.app import app                        # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_returns_a_prediction_context_and_its_backend(client):
    r = client.post("/api/emotion", json={
        "text": "I am stressed and exhausted, I slept four hours before my exam tomorrow."})
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"]["backend"] in {"transformer", "lexicon"}
    assert 0.0 <= d["prediction"]["confidence"] <= 1.0
    # the context layer ran, and found what the sentence actually says
    ctx = {k: v["value"] for k, v in d["context"].items() if v["value"]}
    assert ctx.get("sleep") == "poor"
    assert ctx.get("event") == "examination"
    assert ctx.get("time_context") == "tomorrow"
    # and the explicit self-report was detected
    assert d["self_reported"]["label"] == "stress"


def test_accepts_no_identifier_so_there_is_no_history_to_leak(client):
    r = client.post("/api/emotion", json={
        "text": "hello", "person_id": "U1", "user_id": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert "U1" not in str(body) and "abc" not in str(body)


def test_empty_text_is_refused(client):
    assert client.post("/api/emotion", json={"text": ""}).status_code == 422
    assert client.post("/api/emotion", json={"text": "   "}).status_code == 400


def test_overlong_text_is_refused(client):
    assert client.post("/api/emotion",
                       json={"text": "x" * 5000}).status_code == 422
