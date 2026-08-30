# Adaptive Emotional Digital Twin Using Continual Knowledge Hypergraph Reasoning

> **Longitudinal self-report is the yardstick of digital mental health. This
> project asks whether the yardstick itself moves — and builds the estimator
> that measures it.**

---

## ⚠️ Scientific status — read this first

| | Status |
|---|---|
| **Implemented & verified** | Full pipeline, 16 modules, 230+ tests passing |
| **Validated on SYNTHETIC data** | Estimator calibration, placebo, bias envelope, ablation |
| **Validated on REAL data** | ❌ **NONE.** No primary ρ\* result exists from any real dataset |
| **Real data acquired & audited** | ✅ **RELAX** (31 pp), ✅ **PMData** (16 pp), ⚠️ **StudentLife** (RDS repackaging) — **all three blocked** |
| **Pending** | The **original** Dartmouth StudentLife release |

**There is no real-data empirical result in this repository, and the software
refuses to manufacture one.**

**Two** real datasets have been downloaded, schema-verified and put through
the strict audit. **Both fail it**, for different reasons:

| dataset | participants | eligible | why it fails |
|---|---|---|---|
| **RELAX** (Halmich et al. 2026, Zenodo [10.5281/zenodo.20701999](https://doi.org/10.5281/zenodo.20701999), CC-BY-4.0) | 31 | **0 / 31** | self-report density — median ~50 aligned reports vs the 120 required |
| **PMData** (Thambawita et al. 2020, [Simula](https://datasets.simula.no/pmdata/), CC BY 4.0) | 16 (14 usable) | **0 / 14** | density **+** assumption **A3** (Var(s) epoch ratios to 13.6) **+** an undocumented scale direction |
| **StudentLife** — *RDS repackaging* (Wang et al. 2014) | 46 | **0 / 46** | **defective conversion**: the stress response column is named `null` and is 88% NA; 122 of ~35 000 responses survive, and none overlap the sensing period |

The pipeline exits non-zero on each and reports the failure. **No threshold was
relaxed to make any of them fit.**

**The StudentLife failure is different in kind, and that matters.** RELAX and
PMData are intact datasets that genuinely cannot support the method. The
StudentLife archive audited here is a *damaged R-serialised repackaging* of a
dataset that almost certainly can: its conversation sensor is perfect (79 023
episodes, 49 participants, 0% invalid), and sibling EMA tables in the same
archive (PAM, Mood, Sleep) converted correctly — only `Stress.Rds` is mangled.
**The fix is a different download, not a different method.** See [`docs/dataset_audit.md`](docs/dataset_audit.md)
and [`docs/dataset_compatibility.md`](docs/dataset_compatibility.md).

Every other result here is **SYNTHETIC**, generated from the frozen model in
`docs/frozen_scientific_specification.md` §5, and labelled as such on every
figure, table and result object. Nothing here is evidence about humans.

When real files are absent the system reports
`REAL DATA UNAVAILABLE - <DATASET> AUDIT NOT RUN` and **never** substitutes
synthetic data.

| Label | Meaning |
|---|---|
| **REAL** | computed from audited files actually present on disk |
| **SYNTHETIC** | simulation, or a dataset-shaped fixture |
| **PLANNED** | not computed; no data has been analysed |

---

## The problem in one paragraph

Nearly every study in digital mental health predicts a self-reported score and
treats that score as ground truth. But a person's internal scale can drift:
after ten weeks of a stressful term, *"a little stressed"* may not mean what it
meant in week one. If that happens, an apparent improvement is **measurement
drift, not change**, and every effect estimate in that literature is wrong by
an unknown amount in an unknown direction. Passive sensing offers something
psychometrics never had — a continuous reference that doesn't ask the person
anything.

## What we estimate

For person *p* and epoch *e*, standardise the sensor feature **within that
epoch**, fit the ordinal probit

```
P(R ≤ k | x) = Φ(c_k − β_e · x)
```

by maximum likelihood, and take

```
ρ*_p = β_p2 / β_p1
```

pooled as a mean of logs with a bootstrap over **participants**.

**ρ\* is the identified estimand.** ρ itself is **not** point-identified, and
the additive component is **provably not identifiable** — it is absorbed into
the threshold locations. So `1 − ρ*` is reported as a **LOWER BOUND** on the
true multiplicative recalibration, and the additive component is never
estimated. `EstimatorResult` refuses to carry one.

---

## Interactive demo

**<https://claude.ai/code/artifact/2d693337-529c-4470-90b2-674a261c161c>**

A self-contained walkthrough: the problem, the pipeline, the two-curve
epoch-1-vs-epoch-2 plot, the context hypergraph, the audit gate (switchable
between the synthetic cohort, RELAX and PMData), and the results. Every panel
carries a **REAL / SYNTHETIC** stamp, and the real-data panels show the audit
failures rather than hiding them.

Source: [`demo/index.html`](demo/index.html) — single file, no build step, no
network calls at run time beyond the webfont.

## Quick start

```bash
git clone https://github.com/MRC005/Adaptive-Emotional-Digital-Twin-Using-Continual-Knowledge-Hypergraph-Reasoning-.git
cd Adaptive-Emotional-Digital-Twin-Using-Continual-Knowledge-Hypergraph-Reasoning-
pip install -e ".[dev]"          # numpy scipy pandas matplotlib pyyaml pytest
```

### The Review-2 demonstration — one command

```bash
python scripts/run_demo.py --dataset synthetic --participant p07
```

Nine stages, offline, about two minutes:

```
DATA → PREPROCESSING → FEATURES → CONTEXT → DIGITAL TWIN
     → ORDINAL MODEL → SLOPE RATIO → PLACEBO / AUDIT → RESULT
```

### Run the tests

```bash
python -m pytest tests -q
```

### Regenerate every Review-2 figure and table

```bash
python scripts/generate_review2_outputs.py
```

### Acquire and audit the one real dataset that is obtainable

```bash
pip install -e ".[relax]"
python scripts/fetch_relax.py --root data/raw/relax
```

Pulls **~0.5 GB** of a **16.5 GB** archive using HTTP range requests — the
self-reports, the item definitions and the interbeat intervals — and skips
15.9 GB of accelerometer data the analysis does not use. Writes
`data/raw/relax/PROVENANCE.json` (DOI, licence, per-file SHA-256).

```bash
python scripts/audit_dataset.py --dataset relax --root data/raw/relax
python scripts/run_demo.py --dataset relax --root data/raw/relax --sensitivity
```

The demo **exits 3 and produces no ρ\* estimate** — that is the correct result.

### Audit StudentLife (RDS repackaging)

```bash
python scripts/convert_studentlife_rds.py          # needs Rscript
python scripts/audit_dataset.py --dataset studentlife --root data/interim/studentlife
```

Extracts 0.6 MB of a 224 MB archive and reports the conversion defect. Exits
**2** with `DECISION REQUIRED` — the correct result.

### Audit the datasets

```bash
python scripts/audit_dataset.py --all
python scripts/audit_dataset.py --dataset studentlife --root /path/to/StudentLife_Dataset
python scripts/audit_dataset.py --fixture studentlife --out /tmp/sl_fixture
```

### Run the full pipeline (2000-resample bootstrap)

```bash
python scripts/run_pipeline.py --dataset synthetic
python scripts/run_pipeline.py --dataset studentlife --root /path/to/archive --strict-real
```

### Reproduce the validated simulation studies

```bash
python scripts/run_experiments.py --what all
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | completed |
| `2` | **DECISION REQUIRED** — a scientific decision is unspecified; nothing was guessed |
| `3` | no eligible participants |
| `4` | placebo not runnable — the primary is blocked |
| `5` | **placebo FAILED** — the primary is NOT run; that is the finding |
| `6` | **REAL DATA UNAVAILABLE** — no synthetic substitute was used |
| `7` | ordinal model non-convergence where convergence was required |

Distinct codes exist so a wrapper cannot accidentally proceed past a failed gate.

---

## Repository layout

```
├── aedt/
│   ├── constants.py          FROZEN thresholds, seed, label map
│   ├── schemas.py            15 typed, self-validating scientific objects
│   ├── config.py             YAML layering; detects deviations from the frozen spec
│   ├── errors.py             DecisionRequired, RealDataUnavailable, PlaceboFailed...
│   ├── pipeline.py           enforces the frozen experiment execution order
│   ├── demo.py               the nine-stage Review-2 demonstration
│   ├── io/                   dataset adapters + audit + synthetic fixtures
│   ├── preprocess/           cleaning, label remap, epochs
│   ├── alignment/            causal windows; leakage assertion
│   ├── features/             7 extractors behind one interface
│   ├── contexts/             continuous · feature-vector · discretisation
│   ├── hypergraph/           conjunctive context structure + Ablation 1
│   ├── twin/                 TwinState, persistence, longitudinal update
│   ├── knowledge/            append-only continual knowledge store
│   ├── models/               ordinal probit (ML, no statsmodels)
│   ├── estimators/           slope_ratio (ρ*) · spread_ratio · affine_did · linear_anchor
│   ├── inference/            participant-cluster bootstrap · placebo
│   ├── audit/                eligibility · diagnostics ([9b]) · bias envelope
│   ├── baselines/            6 pre-specified baselines
│   ├── simulate/             frozen generator + 13 misspecification scenarios
│   ├── viz/                  two-curve plot, forest, dashboard, architecture...
│   └── reporting/            tables, run metadata, reproducibility
├── configs/                  base + one per dataset
├── docs/                     frozen spec · architecture · audits · demo · status
├── scripts/                  audit_dataset · run_pipeline · run_demo · ...
├── tests/                    unit · integration · synthetic · regression
└── figures/ tables/ results/ notebooks/
```

---

## Review-2 evidence table

What a reviewer can check, and how.

| Claim | Evidence | Verify with |
|---|---|---|
| Estimator is calibrated under the null | ρ\* = 1.00 ± 0.03 across 3 threshold placements and AR(1) noise | `pytest tests/synthetic -q` |
| Estimator is conservative, never inflated | at true ρ = 0.85 it returns ρ\* ∈ (0.85, 1.00) | `tests/synthetic/test_known_answer.py` |
| The obvious affine method fails | reproduces its **−0.107** fabricated null bias on 5- and 7-point scales | `tests/regression/test_known_failures.py` |
| A withdrawn earlier claim stays falsifiable | the −0.19 harness artefact is still reproducible beside the corrected −0.107 | same file |
| Placebo does not fire on real recalibration | does not reject at true ρ = 1.00, 0.85 **and 0.70** | `tests/unit/test_placebo.py` |
| No future leakage | planted future samples are rejected; twin clock only moves forward | `tests/unit/test_leakage.py` |
| Labels are never guessed | scrambled option order still maps correctly; unknown label halts | `tests/unit/test_remap.py` |
| Bootstrap clusters on participants | schema refuses any other resampling unit | `tests/unit/test_bootstrap.py` |
| Hypergraph is **not** load-bearing | deleting every context column leaves ρ\* bit-identical | `tests/unit/test_hypergraph.py` |
| Hypergraph-native estimator **fails** | disqualified: not null-calibrated, wrong direction | `figures/fig07_hypergraph_ablation.png` |
| Real data is not faked | missing archive → `REAL DATA UNAVAILABLE`, exit 6 | `tests/integration/test_missing_real_data.py` |
| **RELAX really was audited, and really failed** | 0/31 eligible; pipeline exits 3 | `tests/regression/test_relax_real_audit.py` |
| **PMData really was audited, and really failed** | 0/14 eligible; A3 violated; direction undocumented | `tests/regression/test_pmdata_real_audit.py` |
| **The StudentLife archive is a broken conversion** | response column named `null`, 88% NA, no temporal overlap | `tests/regression/test_studentlife_rds_audit.py` |

## The four contributions

Everything else in the system is standard technology or engineering
integration, and `tables/t12_standard_vs_contribution.md` says so line by line.

1. **Identifiability analysis** — ρ\* is identified up to an explicit
   conservative scale factor, free of the unknown sensor gain; the additive
   component is **not identifiable under any specification examined**.
2. **The ordinal slope-ratio estimator**, with the algebra showing why
   epoch-invariant attenuation and link misspecification cancel — exactly
   calibrated under the null, conservative otherwise.
3. **A failure analysis of the natural affine approach** — a quantified
   **−0.107** fabricated null bias on 5-point scales, with five failed remedies.
   Asserted by `tests/regression/test_known_failures.py`, so a future "fix"
   that breaks the baseline fails loudly.
4. **A real-world validity protocol** — eligibility screen, contiguous
   split-half placebo, and an empirically calibrated bias envelope.

## Title terms → real modules

| Term | Module | Honest status |
|---|---|---|
| **Adaptive** | `twin/update.py` | Scheduled re-estimation + self-assessed trust. **Not** online learning. |
| **Emotional** | `preprocess/reports.py`, `models/ordinal.py` | We model the **reporting of** stress. **No emotion recognition is claimed.** |
| **Digital Twin** | `twin/state.py` | Persistent and load-bearing — the estimand is a cross-epoch ratio. Term is descriptive. |
| **Continual Knowledge** | `knowledge/store.py` | Append-only, provenance-carrying. **No continual-learning algorithm imported.** |
| **Hypergraph** | `hypergraph/` | Conjunctive context representation + ablation arm. **NOT the identification mechanism** — and Ablation 1 disqualifies the hypergraph-native estimator form. We report that. |
| **Reasoning** | `twin/update.py` trust rule | ⚠️ **Weakest term.** One rule-based trust decision over hyperedge overlap. |

Full table: `tables/t11_title_alignment.md`.

---

## Scientific safety properties, enforced in code

| Property | How it is enforced |
|---|---|
| Never fabricate real data | `RealDataUnavailable`, exit 6 |
| Never substitute synthetic for missing real | adapters raise; fixture directories carry a marker that downgrades results to SYNTHETIC |
| Never guess a stress-label mapping | mapped by **label text**; unknown label → `DecisionRequired`, exit 2 |
| Never reverse stress labels silently | reversal is **detected and reported** |
| Never mix participants across partitions | per-person fits; bootstrap resamples participants |
| Never leak the future | causal windows + `assert_no_leakage`; twin clock only moves forward |
| Never treat repeated observations as participants | `UncertaintyResult` refuses any other resampling unit |
| Never report ρ instead of ρ\* | `EstimatorResult` refuses any other estimand |
| Never ignore a failed placebo | placebo runs first and gates; exit 5 |
| Never hide exclusions | every exclusion logged and tabled with its reason |
| Never change thresholds after seeing results | frozen in `constants.py`; config deviations detected and written to the run folder |
| Never present a benchmark as longitudinal validation | `WesadAdapter.can_support_longitudinal_estimand = False`; not overridable |
| Never present synthetic output unlabelled | `save_figure` refuses unstamped figures; `write_table` refuses tables with no `data_status` |

---

## Reproducibility

Seed **20260828** everywhere. Every run writes a timestamped, status-stamped
folder under `results/` containing `run_metadata.json` (run id, data status,
config digest, seed, package versions, git commit, elapsed time),
`resolved_config.json`, `config_deviations_from_frozen.json`, all figures, all
tables, and the persisted twins.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/frozen_scientific_specification.md`](docs/frozen_scientific_specification.md) | Research question, gap, hypotheses, model, **A1–A5**, ρ\*, identifiability, interpretation, limitations, endpoint, methodology |
| [`docs/architecture.md`](docs/architecture.md) | All 16 modules: purpose, I/O, algorithm, **STANDARD vs CONTRIBUTION** |
| [`docs/dataset_audit.md`](docs/dataset_audit.md) | Dataset roles, audit fields, Day-1 procedures, acquisition instructions |
| [`docs/implementation_status.md`](docs/implementation_status.md) | The strict status board — **including the empty cells** |
| [`docs/review2_demo.md`](docs/review2_demo.md) | The nine-stage demo and what to say at each stage |
| [`docs/panel_explanation.md`](docs/panel_explanation.md) | 20-second / 1-minute / technical explanations and the Q&A |
| [`docs/decision_required.md`](docs/decision_required.md) | Open scientific questions; what was deliberately not implemented, and why |

---

## Requirements

Python 3.11+ with `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`; `pytest`
for the tests. **`statsmodels` is deliberately excluded** — the ordinal probit
is implemented directly. Runs on any laptop, no GPU, no network at run time.

## Status

The **science is frozen**. The **software is built and tested**. **Real-data
validation is still pending — and it is now a harder problem than a download.**

Three real archives were acquired and audited, and **all three are blocked** —
RELAX on self-report density, PMData on density plus assumption A3 plus an
undocumented scale direction, and the StudentLife RDS repackaging on a defective
conversion that destroyed ~99% of its stress EMA. That is a real
finding rather than an excuse: the method needs roughly 120 repeated ordinal
self-reports per person **plus** a documented scale **plus** a sensor whose
variance is stable across epochs, and current open longitudinal datasets rarely
provide all three. StudentLife does have the density (~735/participant) and remains the right
target — the copy obtained so far is simply a broken conversion.

**This project is not publication-ready.** The theory and the simulation
evidence are sound; the empirical section has no evidence at all.
