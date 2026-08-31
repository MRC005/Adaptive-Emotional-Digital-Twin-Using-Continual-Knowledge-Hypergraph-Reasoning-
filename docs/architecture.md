# Architecture

**Adaptive Emotional Digital Twin Using Continual Knowledge Hypergraph Reasoning**

The architecture is **FROZEN** (`ROUND-17-FINAL-FREEZE-PACKAGE.md` §I). Each
module below states its purpose, inputs, outputs, algorithm and — the part that
matters most for credibility — whether it is **STANDARD** technology,
**ENGINEERING INTEGRATION**, or **OUR RESEARCH CONTRIBUTION**.

```
DATA SOURCES ── StudentLife · RELAX · WESAD · PMData · synthetic
     ↓
DATA INGESTION ── adapters → canonical LongFrame; halt on surprise
     ↓
PREPROCESSING ── missingness ledger; outlier flags; NO imputation of the outcome
     ↓
TEMPORAL ALIGNMENT ── causal windows; epochs = halves of each participant's OWN span
     ↓
SENSOR FEATURE EXTRACTION ── conversation minutes (primary); HR; activity; entropy
     ↓
CONTEXT FORMATION ── (a) continuous [DEFAULT] (b) feature-vector bins (c) n-ary hyperedge
     ↓
PERSONAL DIGITAL TWIN ── persistent TwinState: thresholds, slopes, history, uncertainty
     ↓
CONTINUAL KNOWLEDGE ── append-only store; provenance; temporal inspection
     ↓
HYPERGRAPH LAYER ── higher-order context [ABLATION, not identification]
     ↓
ORDINAL SELF-REPORT MODEL ── P(R≤k|x) = Φ(c_k − β_e x), ML per person per epoch
     ↓
SLOPE-RATIO ESTIMATOR ── ρ* = β₂/β₁; additive component reported NOT IDENTIFIED
     ↓
ELIGIBILITY / QUALITY AUDIT ── fixed thresholds; every exclusion printed with reason
     ↓
PLACEBO VALIDATION ── contiguous epoch-1 split-half; GATES the primary; exit code 5
     ↓
UNCERTAINTY ESTIMATION ── participant-cluster bootstrap, 2000 resamples
     ↓
LONGITUDINAL TWIN UPDATE ── append {ρ*, CI, verdict}; ACCEPTED or FLAGGED
     ↓
VISUALISATION ── two-curve plot; forest; category usage; audit dashboard
```

Generated as a figure by `aedt/viz/architecture.py` →
`figures/fig00_architecture.png`, with the contribution blocks coloured.

---

## Module 1 — Data sources and ingestion

- **Purpose** Load actual dataset files through dataset-specific adapters.
- **Input** Raw dataset files on disk.
- **Output** Canonical participant-aware `LongFrame` + a `DatasetAudit`.
- **Algorithm** Dataset-specific parser → canonical schema; halt on any surprise.
- **Status** **STANDARD**
- **Code** `aedt/io/` — `base.py`, `synthetic.py`, `studentlife.py`, `pmdata.py`, `relax.py`, `wesad.py`, `fixtures.py`

Two operations are deliberately separate. `audit(root)` is **always safe** and
reports what is and is not present. `load(root)` requires the files and raises
`RealDataUnavailable` when they are missing, `DecisionRequired` when they are
present but disagree with the specification.

## Module 2 — Preprocessing

- **Purpose** Clean and normalise without destroying auditability.
- **Input** Raw canonical records.
- **Output** Preprocessed records + `MissingnessLedger`.
- **Algorithm** Explicit rules; missingness accounting; **no silent imputation**.
- **Status** **STANDARD**
- **Code** `aedt/preprocess/clean.py`

Rules in order: drop missing timestamps; drop missing **self-report** (the
outcome is never imputed); drop missing **sensor** value (an imputed covariate
would attenuate β differently in the two epochs and bias the ratio); drop exact
duplicate occasions; **flag — never drop** extreme sensor values.

## Module 3 — Temporal alignment

- **Purpose** Attach sensor observations to self-report occasions.
- **Input** Timestamped sensor samples and reports.
- **Output** Aligned participant-time records with window metadata.
- **Algorithm** Configurable **causal** window `(ts − lookback, ts − lag]`.
- **Status** **STANDARD**
- **Code** `aedt/alignment/align.py`, `aedt/preprocess/epochs.py`

**No future information may be used.** `assert_no_leakage` is a hard assertion
run on every alignment, and `tests/unit/test_leakage.py` plants a future sample
and requires rejection.

## Module 4 — Sensor feature extraction

- **Purpose** Turn raw streams into interpretable time-window features.
- **Input** Aligned sensor streams.
- **Output** Feature vectors on the LongFrame.
- **Algorithm** Dataset-specific extraction behind one common interface.
- **Status** **STANDARD**
- **Code** `aedt/features/` — 7 extractors

Each declares a `FeatureSpec` with its expected sign versus stress. **That
expectation is documentation only** — the estimator accepts either sign and
requires only that the sign match across epochs (self-correction 26).

## Module 5 — Context formation

- **Purpose** Construct contextual state from behavioural information.
- **Input** Features.
- **Output** `ContextState` and the covariate the model consumes.
- **Algorithm** Three representations: continuous (frozen default), epoch-1
  prototype feature-vector bins, exact n-ary conjunctions.
- **Status** **STANDARD / SYSTEM ENGINEERING**
- **Code** `aedt/contexts/`

All discretisation uses **epoch-1 cut points only**. Pooled quantiles would
leak epoch-2 information into the definition of "the same situation".

## Module 6 — Personal Digital Twin

- **Purpose** A persistent, per-person model of that person's **measuring
  instrument** — not of their mood.
- **Input** Observations up to the current time.
- **Output** Serialisable `TwinState`.
- **Status** **ENGINEERING INTEGRATION** — the term is descriptive, not a
  novelty claim.
- **Code** `aedt/twin/state.py`

**Persistence is load-bearing.** The estimand is a ratio *across epochs*;
without persisted per-person state there is no epoch 1 to compare against.
`TwinState.current_time` only moves forward, and `observe` refuses an
observation dated before it — leakage is impossible by construction.

## Module 7 — Continual knowledge representation

- **Purpose** Preserve and update participant knowledge longitudinally.
- **Output** Append-only, provenance-carrying, temporally inspectable store.
- **Status** **ENGINEERING INTEGRATION**
- **Code** `aedt/knowledge/store.py`

**Exactly four things update:** personalised parameters; state history; context
relationships; uncertainty and audit state. **No continual-learning algorithm
runs in THIS module** — appending knowledge moves no model parameter.
EWC continual learning of parameters is implemented in
`aedt/continual/ewc.py` (Layer 1, offline research pipeline). Keeping the
two apart is a deliberate
scope decision. Supersession is recorded as an *edge*; the superseded node's
content is left intact so history stays inspectable.

## Module 8 — Hypergraph / context layer

- **Purpose** Represent higher-order (conjunctive) contextual relationships.
- **Output** `ContextHypergraph`: feature-value vertices joined by hyperedges
  with per-epoch occupancy.
- **Status** **STANDARD** representation; **ENGINEERING INTEGRATION** here.
- **Code** `aedt/hypergraph/structure.py`, `aedt/hypergraph/ablation.py`

> **The hypergraph is NOT the identification mechanism.** The frozen method
> regresses the ordinal response on a **continuous** sensor covariate and uses
> every observation. ρ\* is identified by the ratio construction in
> `estimators/slope_ratio.py`, not by anything in this module.
> `tests/unit/test_hypergraph.py::test_the_primary_estimator_does_not_depend_on_the_hypergraph`
> proves it: deleting every context column leaves the primary estimate
> bit-identical.

Its role is the personal contextual **knowledge representation**: *sleep poor ∧
activity low ∧ evening ∧ at home* is one hyperedge over feature-value vertices
— conjunctive and exact, where a feature-vector distance is compensatory. It is
implemented as an **ablation component** with a WITH/WITHOUT comparison.

**Where the twin does reason over it.** `twin/update.py::close_epoch` reads the
epoch-to-epoch hyperedge occupancy overlap; below 0.20 the update is flagged
**FLAGGED-UNTRUSTWORTHY**, because the two epochs are not describing comparable
situations. That is a real, implemented use of higher-order structure — and it
is a **trust** decision, not an identification mechanism.

## Module 9 — Ordinal self-report model

- **Algorithm** ML on `P(R ≤ k | x) = Φ(c_k − β_e x)`, cutpoints parameterised
  as `c₀` plus log-increments so ordering holds by construction.
- **Status** **STANDARD MODELING COMPONENT USED IN OUR CONTRIBUTION** (McCullagh 1980)
- **Code** `aedt/models/ordinal.py`

`statsmodels` is deliberately not used (ROUND-17 §S.13).

## Module 10 — Slope-ratio estimator

- **Status** ⭐ **OUR RESEARCH CONTRIBUTION**
- **Code** `aedt/estimators/slope_ratio.py`

Outputs `rho_star`, uncertainty, CI, diagnostic status and eligibility status.
`EstimatorResult` **refuses** any estimand other than `rho_star` and **refuses**
to carry an additive component.

## Module 11 — Eligibility screen

- **Status** **ENGINEERING INTEGRATION** / OUR VALIDATION FRAMEWORK
- **Code** `aedt/audit/eligibility.py`

Thresholds are defined in `aedt/constants.py` **before** any analysis. A config
override is detected by `Config.deviations_from_frozen()` and written to the
run folder — a threshold cannot be changed silently.

## Module 12 — Placebo test

- **Status** ⭐ **OUR RESEARCH CONTRIBUTION** (the design; the technique is standard)
- **Code** `aedt/inference/placebo.py`

Contiguous epoch-1 split-half. Validated in three regimes: it does **not** fire
even when a genuine 30% recalibration is present in the full series. **Runs
before the primary and gates it** (exit code 5).

## Module 13 — Bias envelope

- **Status** ⭐ **OUR RESEARCH CONTRIBUTION**
- **Code** `aedt/audit/envelope.py`

Nine **pre-enumerated** assumption violations, re-run under the true null. An
estimate inside the band cannot be distinguished from an artefact. The
scenarios are declared in advance and are not chosen to optimise the result.

## Module 14 — Uncertainty estimation

- **Status** **STANDARD** (clustered on participants — load-bearing)
- **Code** `aedt/inference/bootstrap.py`

`UncertaintyResult` **refuses** any resampling unit other than `participant`.

## Module 15 — Longitudinal twin update

- **Code** `aedt/twin/update.py`

`observe → validate → features/context → continual knowledge → hypergraph →
ordinal state → uncertainty → provenance`. On epoch close: re-fit, screen,
placebo; if both pass append `{ρ*, CI, flags, ACCEPTED}`; if either fails append
the diagnostics as **FLAGGED-UNTRUSTWORTHY** and **leave the calibration state
unchanged**.

## Module 16 — Visualisation and output

- **Code** `aedt/viz/`, `aedt/reporting/`

Every figure carries a **REAL / SYNTHETIC / PLANNED** badge; `save_figure`
refuses an unstamped figure and `write_table` refuses a table with no
`data_status` column. The primary visualisation is
`viz/curves.py::two_curve_plot`.

---

## Standard vs contribution

Generated as `tables/t12_standard_vs_contribution.{csv,md}`. **Four
contributions; everything else is standard technology or integration:**

1. **Identifiability analysis** — ρ\* identified up to an explicit conservative
   factor; the additive component provably not identifiable.
2. **The ordinal slope-ratio estimator** and the algebra showing why
   epoch-invariant attenuation and link misspecification cancel.
3. **Failure analysis of the natural affine approach** — quantified bias with
   direction (−0.107 on 5-point scales) and five failed remedies.
4. **A real-world validity protocol** — eligibility screen, contiguous
   split-half placebo, and empirically calibrated bias envelope.

## Data flow

```
raw files → adapter → LongFrame → clean → epochs → features → context
   → per-epoch ordinal fit → slope ratio → {eligibility, placebo}
   → participant bootstrap → TwinState → figures + tables + run metadata
```

## Error handling and exit codes

| Code | Meaning |
|---|---|
| 0 | completed |
| 2 | `DECISION REQUIRED` — a scientific decision is unspecified; nothing guessed |
| 3 | no eligible participants |
| 4 | placebo not runnable — the primary is blocked |
| 5 | **placebo FAILED** — the primary is NOT run; that is the finding |
| 6 | `REAL DATA UNAVAILABLE` — no synthetic substitute was used |
| 7 | ordinal model non-convergence where convergence was required |

Distinct codes exist so a wrapper script cannot accidentally proceed past a
failed gate.
