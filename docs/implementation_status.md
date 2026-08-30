# Implementation status — strict

> **COMPLETED** means implemented **and verified** (a passing test, or a
> reproduced historical result with evidence), or an existing historical result
> with evidence.
>
> **Code that merely exists is not COMPLETED.** The empty cells in this table
> are the most credibility-building object in the project; discovered
> overstatement costs far more than admitted incompleteness.

Regenerate as a table with `python scripts/generate_review2_outputs.py`
→ `tables/t10_status_board.md`.

## Status board

| Item | Completed | In progress | Planned | Evidence |
|---|:--:|:--:|:--:|---|
| Repository structure and packaging | ✅ | | | `aedt/` package, `pyproject.toml`, `configs/`, `scripts/`, `tests/` |
| Configuration system | ✅ | | | `aedt/config.py` + 6 YAML configs; frozen-spec deviations detected and written to every run |
| Canonical schemas | ✅ | | | `aedt/schemas.py` — 15 typed dataclasses, self-validating |
| Synthetic dataset ingestion | ✅ | | | `aedt/io/synthetic.py`, `aedt/simulate/` |
| Preprocessing + missingness ledger | ✅ | | | `aedt/preprocess/clean.py`; outcome never imputed |
| Temporal alignment (causal) | ✅ | | | `aedt/alignment/align.py`; leakage assertion + planted-future test |
| Feature extraction interface | ✅ | | | `aedt/features/` — 7 extractors, one interface |
| Context representation (3 forms) | ✅ | | | `aedt/contexts/` + `aedt/hypergraph/structure.py` |
| TwinState + persistence | ✅ | | | `aedt/twin/state.py`; JSON round-trip test |
| Continual knowledge store | ✅ | | | `aedt/knowledge/store.py`; append-only, causality-checked |
| Hypergraph / context layer | ✅ | | | `aedt/hypergraph/` — built, visualised, ablated |
| Ordinal probit model | ✅ | | | `aedt/models/ordinal.py` — ported; known-answer test |
| Slope-ratio estimator (ρ\*) | ✅ | | | `aedt/estimators/slope_ratio.py` — ported; known-answer tests |
| Eligibility screen | ✅ | | | `aedt/audit/eligibility.py`; frozen thresholds asserted |
| Placebo framework | ✅ | | | `aedt/inference/placebo.py`; gates the primary; exit code 5 |
| Uncertainty (participant bootstrap) | ✅ | | | `aedt/inference/bootstrap.py`; resampling unit asserted |
| Bias envelope | ✅ | | | `aedt/audit/envelope.py`; 9 pre-enumerated scenarios |
| Dataset audit framework | ✅ | | | `aedt/io/base.py` + per-adapter `audit()`; every mandated field |
| **StudentLife adapter** | ⚠️ | | | Implemented; **fixture-tested only. NEVER RUN ON THE REAL ARCHIVE.** |
| **PMData adapter** | ✅ | | | **REAL FILES OPENED AND AUDITED.** 16 participants; **0 of 14 eligible**; two real bugs fixed |
| **RELAX adapter** | ✅ | | | **REAL FILES OPENED AND AUDITED.** Schema verified; anchors verified; 31 participants; **fails the eligibility screen** |
| **WESAD adapter** | ⚠️ | | | Implemented; **benchmark only**, refuses the primary by construction |
| Epoch 1 vs Epoch 2 visualisation | ✅ | | | `aedt/viz/curves.py` — the two-curve plot |
| Figure / table generation pipeline | ✅ | | | 10 figures, 13 tables, all stamped |
| End-to-end synthetic demo | ✅ | | | `scripts/run_demo.py` — 9 stages, subprocess-tested |
| Baselines | ✅ | | | 6 of the 8 listed; 2 deferred with reasons (`decision_required.md`) |
| Hypergraph ablation (Ablation 1) | ✅ | | | Run on synthetic data; result reported whichever way it falls |
| Test suite | ✅ | | | 225 tests: unit, integration, synthetic, regression |
| Controlled synthetic simulations | ✅ | | | Historical: 13 misspecification scenarios (Rounds 14–15) |
| Realistic synthetic simulation gate | ✅ | | | Historical: power, coverage, placebo at realistic density |
| **REAL-DATA ACQUISITION (RELAX)** | ✅ | | | Acquired from Zenodo 10.5281/zenodo.20701999 (CC-BY-4.0); 453 MB of 16.5 GB; provenance with SHA-256 recorded |
| **REAL-DATA AUDIT (T4)** | | | ⛔ | `scripts/audit_dataset.py` written and fixture-tested; blocked only on files |
| **REAL-DATA primary ρ\*** | | | ⛔ | Gated behind audit → [9b] → eligibility → placebo |
| Cross-dataset pooling | | | ⛔ | Requires two audited real cohorts |
| Ablations 2–7 | | | ⛔ | Only Ablation 1 implemented |
| Manuscript | | 🔄 | | Theory and simulation supported; **empirical section has no evidence** |
| Patent | | | ⛔ | Position RED. No concrete novel mechanism found; no drafting undertaken |

## Per-module detail

| Module | Code | Unit tested | Integration tested | Demonstrable | Scientific validation |
|---|:--:|:--:|:--:|:--:|:--:|
| Ordinal probit | ✅ | ✅ | ✅ | ✅ | ✅ 13 scenarios (historical) |
| Slope-ratio estimator | ✅ | ✅ | ✅ | ✅ | ✅ |
| Participant bootstrap | ✅ | ✅ | ✅ | ✅ | ✅ |
| Placebo test | ✅ | ✅ | ✅ | ✅ | ✅ 3 regimes |
| Eligibility screen | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bias envelope | ✅ | ✅ | ✅ | ✅ | ✅ synthetic |
| Affine-DiD failure analysis | ✅ | ✅ | ✅ | ✅ | ✅ −0.107 reproduced |
| Simulator + 13 scenarios | ✅ | ✅ | ✅ | ✅ | ✅ |
| TwinState / persistence | ✅ | ✅ | ✅ | ✅ | n/a (engineering) |
| Continual knowledge | ✅ | ✅ | ✅ | ✅ | n/a (engineering) |
| Hypergraph context | ✅ | ✅ | ✅ | ✅ | ✅ Ablation 1, synthetic |
| Baselines (6) | ✅ | ✅ | ✅ | ✅ | partial |
| Visualisation | ✅ | ✅ | ✅ | ✅ | n/a |
| Demo script | ✅ | — | ✅ | ✅ | n/a |
| StudentLife loader | ✅ | ✅ fixture | ✅ fixture | ✅ | ❌ **never run on real files** (host unreachable) |
| **PMData loader** | ✅ | ✅ | ✅ | ✅ | ✅ **RUN ON REAL FILES; dataset fails eligibility** |
| **RELAX loader** | ✅ | ✅ | ✅ | ✅ | ✅ **RUN ON REAL FILES; dataset fails eligibility** |
| WESAD loader | ✅ | ✅ | ✅ | ✅ | ❌ **never run on real files** |

## Verified behaviour, reproduced from the historical record

All **SYNTHETIC**.

| Property | Value | Where |
|---|---|---|
| Null calibration, floor-heavy 5-point | ρ\* ≈ 1.00 (\|error\| < 0.03) | `tests/synthetic/test_known_answer.py` |
| Null calibration under AR(1) φ = 0.6 | ρ\* ≈ 1.00 | same |
| Attenuation at true ρ = 0.85 | ρ\* ∈ (0.85, 1.00), ≈ 0.93–0.95 | same |
| Attenuation at true ρ = 0.70 | ρ\* ≈ 0.86 | same |
| Placebo, no shift | does not reject | `tests/unit/test_placebo.py` |
| Placebo, **real 30% shift present** | **does not reject** | same |
| Affine estimator null bias, K = 5 | **−0.107** | `tests/regression/test_known_failures.py` |
| Affine estimator null bias, K = 7 | **−0.107** | same |
| **Ablation 1: continuous arm, null calibration** | holds its size at ρ = 1 | `tests/unit/test_hypergraph.py` |
| **Ablation 1: n-ary hyperedge arm, null calibration** | **FAILS — falsely rejects at ρ = 1** | same |
| Withdrawn per-anchor artefact | −0.186 | same |
| Continuous reference bias | −0.008 | same |

## The real-data result, stated plainly

**TWO datasets were acquired, audited, and rejected by the project's own
screen. No ρ\* estimate exists from either.**

### PMData (Thambawita et al. 2020) — 0 of 14 eligible

| | measured |
|---|---|
| participants with wellness **and** resting HR | **14 of 16** (p12, p13 have no HR file) |
| matched reports | 1 348 of 1 747 wellness rows |
| median matched/participant | 95.5 (max 147) |
| participants ≥120 (60/epoch) | 4 of 14 |
| **eligible** | **0 of 14** |

Exclusions: 9 too few reports · **3 A3 violations** (Var(s) ratios to **13.6**)
· 2 sign flips · 1 \|β\| below floor. **Even the four densest fail** — on A3 or
a sign flip, not on count, so more data would not help. Fitbit resting HR is an
*algorithmic daily estimate*, and its variance moves for device reasons.

Separately blocking: the PMSys `stress` **scale direction is undocumented** in
the release (no README, no codebook), so it is not a verified severity scale at
all.

### RELAX (Halmich et al. 2026) — 0 of 31 eligible

| | measured on the real files |
|---|---|
| participants | 31 |
| aligned reports | 1 519 |
| median reports/participant | **50** (item `ifb-2`, after causal alignment) |
| maximum reports/participant | **93** |
| frozen requirement | **≥120** (60 per epoch) |
| **eligible at the frozen threshold** | **0 / 31** |
| eligible at the pre-specified S1 relaxation (40/epoch) | **2 / 31** |
| S1 placebo | **NOT RUNNABLE** (needs ≥10 participants) |
| reports with no sensor coverage | 28.0% (left missing, never imputed) |
| IBI samples discarded (device-flagged or implausible) | 36.3% |

**No ρ\* estimate was produced, and none should have been.** Both the primary
and the pre-specified sensitivity analysis terminate with exit code 3.

## What is honestly weak

1. **No real-data empirical result exists.** One real dataset has now been
   opened and audited, which is progress over "no file has been opened" — but
   it *failed*, so the empirical claim still rests on nothing. The obstacle is
   no longer a download; it is that the method needs ~120 repeated ordinal
   self-reports per person and current open longitudinal datasets do not
   provide that density.
2. **The hypergraph-native estimator form FAILS on synthetic data, and we
   report it.** Ablation 1 disqualifies the n-ary spread-ratio arm for two
   independent reasons: it does not hold its size when the truth is ρ = 1
   (falsely rejects, ρ\* ≈ 1.05 with a CI excluding 1), and it reports the
   effect in the **wrong direction** at true ρ = 0.85 (ρ\* ≈ 1.06, effect
   retention −0.41).

   A non-obvious detail worth stating: **the split-half placebo does not catch
   this**, because both pseudo-epochs come from within epoch 1 and so contain
   no genuine construct change. The spread-ratio form confounds genuine change
   with scale change, which only a null-cohort check reveals. That is why
   Ablation 1 runs one.

   The continuous covariate is retained as the frozen primary. The hypergraph
   earns its architectural place as the twin's contextual knowledge
   representation and its trust signal — **not** as the method.
3. **"Reasoning" is the weakest title term.** One rule-based trust decision, not
   a reasoning engine. Stated in `tables/t11_title_alignment.md`.
4. **Only one of seven ablations** is implemented and run.
5. **Six of eight baselines**; the two omitted are documented with reasons.
6. **Underpowered at 48 participants** for ρ = 0.85 (68.3% vs a pre-committed
   70%). Powered for ρ ≤ 0.80.
7. **The scale-change vs relation-change confound is unresolved** and is the
   sharpest referee objection. It belongs in the abstract.
