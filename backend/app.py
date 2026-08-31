"""AEDT public demonstration API.

A deliberately small, read-mostly service that exposes ONLY:

  * the project's status and audit verdicts (aggregate counts, no participants)
  * precomputed, validated SYNTHETIC demonstration results
  * a genuine but bounded synthetic computation (`POST /api/demo/run`)

WHAT IT NEVER EXPOSES, BY CONSTRUCTION -- there is no code path to it:
  * participant-level rows from RELAX, PMData or StudentLife
  * identifiers, raw timestamps, or any file from data/raw or data/interim
  * filesystem paths, environment variables, or credentials

Real-data processing stays local. The datasets audited by this project are
licensed for research use, not for public redistribution, and nothing
participant-level is bundled or served.

Run locally:   uvicorn backend.app:app --reload --port 8000
Health:        GET /health
"""
from __future__ import annotations

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aedt import __version__
from aedt.emotion.onnx_detect import ONNX_FILE, ONNX_REPO
from aedt.constants import (BOOTSTRAP_B, MIN_CATEGORIES_USED,
                            MIN_REPORTS_PER_EPOCH, SEED, VAR_RATIO_HI,
                            VAR_RATIO_LO, DataStatus)

# ---------------------------------------------------------------- config
ALLOWED = [o.strip() for o in os.environ.get(
    "AEDT_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:4173,"
    "http://127.0.0.1:4173").split(",") if o.strip()]

#: Vercel gives every deployment its own hostname (production, previews, and a
#: new one per commit), so an explicit allow-list cannot keep up and would fail
#: silently in the browser as a CORS error. A regex scoped to vercel.app is the
#: narrowest rule that actually works. It is NOT "*": credentials stay off and
#: only GET/POST with Content-Type are admitted. Set AEDT_ALLOWED_ORIGIN_REGEX
#: to tighten it to one project once its hostname is stable.
ALLOWED_REGEX = os.environ.get(
    "AEDT_ALLOWED_ORIGIN_REGEX",
    r"https://.*\.vercel\.app")
MAX_PARTICIPANTS = 60
MAX_PER_EPOCH = 400
MAX_BOOTSTRAP = 999

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start serving IMMEDIATELY; load the model in the background.

    The model must not be loaded inline here. Measured from a cold cache the
    load takes ~250 s (a 125 MB download plus session build), and a lifespan
    that blocks for that long means the port never opens, Render's health
    check times out, and the deploy is marked failed. The service would look
    broken for the one reason that is entirely avoidable.

    So the port opens at once and /health tells the truth in three states:

        loading      the background task is still working
        loaded       ready
        unavailable  it failed, with the reason

    The browser already knows how to wait on "loading" -- that is the WAKING
    state -- so a cold start reads as "waking up", not as a failure.

    The model is still loaded exactly ONCE per process, never per request.
    """
    _app.state.model_ready = False
    _app.state.model_loading = True
    _app.state.model_error = None
    _app.state.model_load_seconds = None

    async def _warm() -> None:
        from aedt.emotion.onnx_detect import get_detector
        det = get_detector()
        t0 = time.time()
        # to_thread: the load is blocking CPU/IO work and must not stall the
        # event loop, or /health would be unanswerable while it runs
        ok = await asyncio.to_thread(det.load)
        if ok:
            await asyncio.to_thread(det.predict, "warm up")
        _app.state.model_ready = ok
        _app.state.model_error = det.load_error
        _app.state.model_load_seconds = round(time.time() - t0, 3)
        _app.state.model_loading = False

    task = asyncio.create_task(_warm())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="AEDT demonstration API",
    version=__version__,
    description=("Aggregate status and synthetic demonstration results for the "
                 "Adaptive Emotional Digital Twin project. Serves no "
                 "participant-level data."),
    docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,          # never "*": the browser is the only client
    allow_origin_regex=ALLOWED_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600)


@app.middleware("http")
async def _harden(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


# ------------------------------------------------------------ root/health
@app.get("/")
def root() -> dict:
    """A root route, so hitting the bare URL is informative rather than a 404.

    Its absence is what made the deployment look dead: the service was healthy
    the whole time, but the first thing anyone tries returned
    {"detail":"Not Found"}.
    """
    return {"status": "ok",
            "service": "Adaptive Emotional Digital Twin backend",
            "version": __version__,
            "docs": "/api/docs", "health": "/health",
            "endpoints": ["/health", "/api/emotion", "/api/demo/scenarios",
                          "/api/demo/run", "/api/project-status",
                          "/api/results/summary"]}


@app.get("/health")
def health() -> dict:
    """Alive AND ready are different states, and this reports both.

    A health check that returns ok while the model is missing is how a browser
    ends up silently using a word list under a label that says RoBERTa.
    """
    ready = bool(getattr(app.state, "model_ready", False))
    loading = bool(getattr(app.state, "model_loading", False))
    model = {
        "name": ONNX_REPO,
        "artifact": ONNX_FILE,
        "status": "loaded" if ready else "loading" if loading else "unavailable",
        "load_seconds": getattr(app.state, "model_load_seconds", None),
    }
    if not ready and not loading and getattr(app.state, "model_error", None):
        # the class of failure, not a stack trace or a filesystem path
        model["reason"] = str(app.state.model_error).split(":")[0]
    return {"status": "ok", "service": "AEDT", "version": __version__,
            "model": model}


# ---------------------------------------------------------- project status
@app.get("/api/project-status")
def project_status() -> dict:
    """Aggregate status only. Counts and verdicts, never rows."""
    return {
        "title": ("Adaptive Emotional Digital Twin Using Continual Knowledge "
                  "Hypergraph Reasoning"),
        "estimand": "rho_star",
        "estimand_note": ("rho itself is NOT point-identified; 1 - rho* is a "
                          "LOWER BOUND on the true recalibration. The additive "
                          "component is provably not identifiable."),
        "real_data_result_exists": False,
        "frozen_thresholds": {
            "min_reports_per_epoch": MIN_REPORTS_PER_EPOCH,
            "min_categories_used": MIN_CATEGORIES_USED,
            "var_ratio_window": [VAR_RATIO_LO, VAR_RATIO_HI],
            "bootstrap_resamples": BOOTSTRAP_B,
            "seed": SEED},
        "datasets": [
            {"name": "synthetic", "status": "SYNTHETIC", "participants": 48,
             "eligible": 48, "verdict": "estimator runs; known true rho = 0.85"},
            {"name": "RELAX", "status": "REAL", "participants": 31,
             "eligible": 0,
             "verdict": "fails eligibility: median ~50 aligned reports vs 120 required"},
            {"name": "PMData", "status": "REAL", "participants": 14,
             "eligible": 0,
             "verdict": ("fails eligibility, assumption A3 (Var ratios to 13.6), "
                         "and an undocumented scale direction")},
            {"name": "StudentLife (RDS repackaging)", "status": "REAL",
             "participants": 46, "eligible": 0,
             "verdict": ("defective conversion: response column named 'null', "
                         "88% NA, no temporal overlap with sensing")}],
        "tests_passing": 256,
    }


# -------------------------------------------------------------- scenarios
SCENARIOS = {
    "balanced": {"label": "Balanced scale use",
                 "description": "All five response categories used evenly — the best case.",
                 "placement": "balanced", "phi": 0.0},
    "skewed": {"label": "Floor-heavy scale use",
               "description": "About 45% of responses sit in category 1 — the realistic case.",
               "placement": "skewed", "phi": 0.4},
    "extreme_floor": {"label": "Severe floor effect",
                      "description": "About 68% in category 1 — a hard floor.",
                      "placement": "extreme_floor", "phi": 0.4},
    "serial_dependence": {"label": "Autocorrelated sensor",
                          "description": "AR(1) sensor noise at phi = 0.6.",
                          "placement": "skewed", "phi": 0.6},
}


@app.get("/api/demo/scenarios")
def scenarios() -> dict:
    return {"scenarios": [{"id": k, **{x: v[x] for x in ("label", "description")}}
                          for k, v in SCENARIOS.items()],
            "true_rho_options": [1.00, 0.85, 0.70],
            "note": "All scenarios are SYNTHETIC simulations from the frozen model."}


class RunRequest(BaseModel):
    scenario: Literal["balanced", "skewed", "extreme_floor", "serial_dependence"] = "skewed"
    true_rho: float = Field(0.85, ge=0.5, le=1.0)
    n_participants: int = Field(30, ge=10, le=MAX_PARTICIPANTS)
    n_per_epoch: int = Field(150, ge=60, le=MAX_PER_EPOCH)
    bootstrap: int = Field(299, ge=99, le=MAX_BOOTSTRAP)


@app.post("/api/demo/run")
def run_demo(req: RunRequest) -> dict:
    """A GENUINE synthetic computation, bounded so it cannot be abused.

    Simulates a cohort with a known true rho and runs the real estimator,
    eligibility screen and placebo. No user data, no filesystem access, no
    arbitrary parameters -- every field is validated and range-capped above.
    """
    from aedt.audit.eligibility import filter_eligible, screen_cohort
    from aedt.estimators.slope_ratio import estimate_rho_star
    from aedt.inference.placebo import placebo_split_half
    from aedt.simulate.generator import cohort_to_long_frame, simulate_cohort

    sc = SCENARIOS[req.scenario]
    t0 = time.time()
    try:
        cohort = simulate_cohort(
            req.true_rho, n_participants=req.n_participants,
            n_per_epoch=req.n_per_epoch, seed=SEED,
            placement=sc["placement"], phi=sc["phi"])
        df = cohort_to_long_frame(cohort)
        screened = screen_cohort(df, "conversation_minutes", 5)
        keep = filter_eligible(df, screened)
        n_ok = sum(r.eligible for r in screened)
        if n_ok < 10:
            return {"data_status": DataStatus.SYNTHETIC.value,
                    "computation": "live",
                    "eligible": n_ok, "screened": len(screened),
                    "blocked": True,
                    "message": ("Fewer than 10 participants passed the "
                                "eligibility screen, so no interval can be "
                                "formed. The estimator was not run."),
                    "elapsed_seconds": round(time.time() - t0, 2)}
        pl = placebo_split_half(keep, "conversation_minutes", 5,
                                n_resamples=req.bootstrap)
        out = {"data_status": DataStatus.SYNTHETIC.value,
               "computation": "live",
               "scenario": req.scenario, "true_rho": req.true_rho,
               "screened": len(screened), "eligible": n_ok,
               "exclusions": [{"pid": r.pid, "reasons": list(r.reasons)}
                              for r in screened if not r.eligible][:20],
               "placebo": {"rho_star": pl.rho_star, "ci_low": pl.ci_low,
                           "ci_high": pl.ci_high, "rejected": pl.rejected,
                           "runnable": pl.runnable, "verdict": pl.verdict}}
        if pl.gates_primary:
            out["blocked"] = True
            out["message"] = ("The placebo gates the primary analysis, so no "
                              "estimate was produced. That is the intended "
                              "behaviour.")
            out["elapsed_seconds"] = round(time.time() - t0, 2)
            return out
        res = estimate_rho_star(keep, "conversation_minutes", 5,
                                n_resamples=req.bootstrap)
        u = res.uncertainty
        out.update(blocked=False, primary={
            "rho_star": res.rho_star, "median_rho_star": res.median_rho_star,
            "ci_low": u.ci_low if u else None, "ci_high": u.ci_high if u else None,
            "excludes_null": bool(u.excludes_null) if u else None,
            "lower_bound_on_recalibration": res.lower_bound_on_recalibration,
            "n_used": res.n_participants_used,
            "per_participant_rho_star": [round(v, 4) for v in
                                         res.per_participant_rho_star]},
            elapsed_seconds=round(time.time() - t0, 2))
        return out
    except Exception as exc:                      # never leak a stack trace
        raise HTTPException(
            status_code=500,
            detail={"error": "computation_failed",
                    "message": "The simulation could not be completed.",
                    "kind": type(exc).__name__}) from None


# ---------------------------------------------------------------- results
@app.get("/api/results/summary")
def results_summary() -> dict:
    """Precomputed, validated SYNTHETIC results from the committed run."""
    return {
        "data_status": DataStatus.SYNTHETIC.value,
        "computation": "precomputed",
        "note": ("Regenerated deterministically by "
                 "scripts/generate_review2_outputs.py at seed 20260828."),
        "primary": {"rho_star": 0.9144, "ci_low": 0.8697, "ci_high": 0.9633,
                    "median_rho_star": 0.9614, "n_used": 48, "n_screened": 48,
                    "excludes_null": True,
                    "lower_bound_on_recalibration": 0.0856,
                    "true_rho_in_simulation": 0.85},
        "placebo": {"rho_star": 0.9841, "ci_low": 0.9269, "ci_high": 1.0414,
                    "rejected": False},
        "ablation": [
            {"representation": "continuous", "rho_star": 0.914,
             "ci_width": 0.094, "effect_retention": 0.571,
             "null_calibrated": True, "verdict": "retained"},
            {"representation": "feature_vector", "rho_star": 0.961,
             "ci_width": 0.131, "effect_retention": 0.262,
             "null_calibrated": True, "verdict": "usable, weaker"},
            {"representation": "nary_hyperedge", "rho_star": 1.061,
             "ci_width": 0.078, "effect_retention": -0.405,
             "null_calibrated": False, "verdict": "DISQUALIFIED"}],
        "known_failure": {"method": "affine anchor (difference-in-differences)",
                          "null_bias_5pt": -0.107, "null_bias_7pt": -0.107,
                          "note": ("fabricated scale compression when nothing "
                                   "changed; asserted by a regression test")},
        "bias_envelope": {"low": 0.973, "high": 1.046, "n_scenarios": 9},
    }


# --------------------------------------------------------------- Layer 1
class CheckIn(BaseModel):
    """One interaction. Text only: no identifier is accepted or stored."""

    text: str = Field(..., min_length=1, max_length=1000)


@app.post("/api/emotion")
def detect_emotion(req: CheckIn) -> dict:
    """RoBERTa emotion detection + rule-based context extraction.

    Stateless by construction: the text is classified and discarded, no
    identifier is accepted, and nothing is written to disk. There is therefore
    no check-in history on the server that could leak. The browser keeps the
    history locally.

    If the model is unavailable this returns 503 rather than a lexicon answer.
    The caller must be able to tell the difference, because the interface
    promises which engine ran.
    """
    from aedt.emotion.context import extract_context
    from aedt.emotion.detect import self_reported_emotion
    from aedt.emotion.onnx_detect import get_detector

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    t0 = time.time()
    pred = get_detector().predict(text)
    if pred is None:
        loading = bool(getattr(app.state, "model_loading", False))
        raise HTTPException(
            status_code=503,
            detail=("The emotion model is still loading on this instance. "
                    "Retry shortly." if loading else
                    "The emotion model is not loaded on this instance. No "
                    "prediction was produced; do not present a fallback as a "
                    "model result."),
            headers={"Retry-After": "10"} if loading else None)

    ctx = extract_context(text)
    stated = self_reported_emotion(text)
    return {
        "data_status": DataStatus.SYNTHETIC.value,   # not participant data
        "prediction": pred.to_dict(),
        "self_reported": ({"label": stated[0], "span": stated[1]}
                          if stated else None),
        "context": {k: v.to_dict() for k, v in ctx.items()},
        "model_available": True,
        "inference_ms": round(1000 * (time.time() - t0), 1),
        "note": "Stateless. The text is not stored and no identifier is accepted.",
    }
