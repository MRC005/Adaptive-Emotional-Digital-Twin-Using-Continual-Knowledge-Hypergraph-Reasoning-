# Adaptive Emotional Digital Twin Using Continual Knowledge Hypergraph Reasoning

> **Longitudinal self-report is the yardstick of digital mental health. This
> project asks whether the yardstick itself moves — and builds the estimator
> that measures it.**

---

## Scientific status — read this first

| | Status |
|---|---|
| **Implemented & verified** | Full pipeline, 16 modules, 250+ tests passing |
| **Validated on SYNTHETIC data** | Estimator calibration, placebo, bias envelope, ablation |
| **Validated on REAL data** | ⚠️ **Partial.** One archive of four supports the method. Its **pre-specified primary configuration returns insufficient evidence**; two pre-specified secondary configurations produced estimates, both **no detectable drift** |
| **Real data acquired & audited** | ✅ **College Experience Study** (218 pp, 4 years), ✅ **StudentLife** original release (48 pp), ✅ **RELAX** (31 pp), ✅ **PMData** (16 pp) |

**No drift has been demonstrated in real data by this project, and no threshold
was changed to obtain a result.**

### The four archives

| dataset | participants | verdict | why |
|---|---|---|---|
| **College Experience Study** (Nepal et al. 2024, [IMWUT 8(1) art. 38](https://doi.org/10.1145/3643501)) | 218 | ✅ **Ready** | 35 289 aligned observations, median 169 per person, span 4.8 years. **121 participants clear the unchanged 60-per-window screen.** Response direction is stated in the published codebook |
| **StudentLife** — *original Dartmouth release* (Wang et al. 2014) | 48 | ❌ **Incompatible** | the only dense item is not ordered by severity, and every properly ordered item is far too sparse — see below |
| **RELAX** (Halmich et al. 2026, [10.5281/zenodo.20701999](https://doi.org/10.5281/zenodo.20701999), CC-BY-4.0) | 31 | ❌ **Incompatible** | median ~50 aligned reports against the 120 required |
| **PMData** (Thambawita et al. 2020, [Simula](https://datasets.simula.no/pmdata/), CC BY 4.0) | 16 (14 usable) | ❌ **Incompatible** | density, assumption **A3** (variance ratios to 13.6), and an undocumented scale direction |

### A correction to an earlier claim in this file

A previous version of this README audited a **third-party RDS repackaging** of
StudentLife, found its stress column named `null` and 88% empty, and concluded
*"the fix is a different download, not a different method."*

**That prediction was wrong, and the original release is now in hand.** Two
findings replace it:

1. The repackaging was indeed defective, and the cause is now exact: 90% of
   Stress records in the original are keyed `level`, and ~10% carry the answer
   under a literal `null` key. The repackaging kept the 10% and dropped the 90%
   — which is precisely the 122 usable responses the earlier audit counted.
2. **The correct download does not rescue the dataset.** StudentLife's Stress
   item is documented as *"[1] A little stressed, [2] Definitely stressed,
   [3] Stressed out, [4] Feeling good, [5] Feeling great"*. The numbers are not
   ordered by stress, so an ordinal probit fitted to them models a scale that
   does not exist. Remapping by label text fixes the ordering but cannot create
   data: even then **one participant** reaches 60 responses in both windows,
   against the 10 participants required. Every properly ordered alternative
   item (`Behavior/anxious`, `Sleep/rate`, `Social/number`) is sparser still —
   median 4 to 25 responses per person, most 79.

The rejection does not rest on the number 60. Sweeping the per-window minimum
from 20 to 100 across three window definitions, StudentLife reaches 10
qualifying participants only at 20–30 per window — a threefold relaxation that
would fit a five-category probit slope to roughly 20 points per person.

### What the real-data analysis actually returned

Protocol fixed before any result was inspected. Every configuration that was
run is reported, whatever it said.

| configuration | included | result | ρ\* | 95% interval |
|---|---|---|---|---|
| **Primary** — stress vs conversation minutes, own-span halves | 9 of 33 | **Insufficient evidence** | — | — |
| S1 — windows by equal observation count | 8 of 33 | Insufficient evidence | — | — |
| S2 — windows by cohort calendar median | 7 of 33 | Insufficient evidence | — | — |
| S3 — sensor: phone unlock minutes | 61 of 218 | No detectable drift | 0.913 | [0.721, 1.180] |
| S4 — report: time spent with others | 15 of 33 | No detectable drift | 1.199 | [0.891, 1.589] |

The primary misses the participant floor by one. **It is reported as
insufficient evidence and is not replaced by S3, which produced a number.**

Conversation audio is documented *Android only* in the data dictionary and is
87.8% zero on iOS against 13.1% on Android; the adapter therefore restricts
conversation analyses to the Android cohort, because a stored zero on a
platform that never ran the sensor is absent instrumentation, not silence.

Everything else in this repository is **SYNTHETIC**, generated from the frozen
model in `docs/frozen_scientific_specification.md` §5 and labelled as such on
every figure, table and result object.

| Label | Meaning |
|---|---|
| **REAL** | computed from audited files actually present on disk |
| **SYNTHETIC** | simulation, or a dataset-shaped fixture |
| **PLANNED** | not computed; no data has been analysed |

---

## The application

The deliverable is a working analysis application, not a project write-up.

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

It opens on a plain-English page explaining what the tool does, and three ways in:

- **Guided example** — data built so the right answer is known in advance
  (detectable change / stable control / limited evidence). Computed live in
  your browser.
- **Real data** — the audited study results above, plus the audit explaining
  which archives can support the method at all. You can also open your own CSV,
  which is read locally and never transmitted.
- **Sandbox** — apply a deliberate change to a copy of the data and see what
  the method does. Each change states its expected effect *before* the run, and
  the result is checked against it afterwards.

Results lead with a verdict in one line, then what it does and does not mean in
plain words, then the evidence, then the statistics. Light and dark themes are
both supported and the choice is remembered.

**Where computation happens.** Guided examples, the sandbox and uploaded CSVs
run live in the browser on a JavaScript port of the estimator, pinned to the
Python reference by `tests/regression/test_js_python_agreement.py` (fails above
1e-3 divergence; measured agreement 1.8e-5). Bundled study results were computed
**offline** by the Python implementation and are displayed, not recomputed — the
archives are gigabytes and are licensed for research use, not redistribution.
Nothing exported to the browser carries a participant identifier, a timestamp,
or a raw value.

---

## Adding a dataset

1. Write an adapter in `aedt/io/` subclassing `DatasetAdapter`, implementing
   `locate`, `audit` and `load`. `audit` must be safe to call when the files
   are absent and must never guess a field it cannot read.
2. Emit the canonical frame: `pid`, `ts`, `report`, `raw_response`, and one
   sensor column. Take the response direction from the codebook; if the release
   does not document it, say so and leave `severity_direction_confirmed` False.
3. Register it (`@register_adapter`) and export it from `aedt/io/__init__.py`.
4. Add a regression test asserting what the archive does and does not support.
5. Raw files are never modified. Derived artefacts go to `data/interim/` or
   `data/processed/` with a `PROVENANCE.json`.

`aedt/io/college_experience.py` is the worked example.

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

The published build of the application described above. It runs the guided
examples and the sandbox live in the browser, and displays the audited study
results.

For a demonstration with no server and no network:

```bash
npm --prefix frontend run build && python3 scripts/build_single_file.py
```

That writes `frontend/dist/aedt-standalone.html`, a single file that opens from
disk. It is the same bundle, so the analysis is identical; only the packaging
differs.

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

The **science is frozen**. The **software is built, tested and usable**.

**Real-data status, stated precisely:** four archives have been acquired and
audited. One supports the method. Its pre-specified primary configuration
returns *insufficient evidence* by one participant; two pre-specified secondary
configurations return *no detectable drift* with wide intervals. No drift has
been demonstrated in real data, and no threshold was moved to obtain a result.

**What would change this:** more participants meeting the observation floor,
which means either a denser EMA protocol or a cohort larger than 218. The
limitation is study design, not software.

---

## Honest scope

This project investigates whether the *relationship* between a passively
measured signal and a repeated ordinal self-report changes over time. It does
**not** measure anyone's emotional state, and a "drift detected" verdict is not
a clinical finding. ρ itself is not identified; 1 − ρ\* is a lower bound on
multiplicative recalibration. Added measurement noise is indistinguishable from
genuine recalibration, and the sandbox demonstrates this deliberately.

## Provenance

Research design, scientific specification and all decisions are the author's.
Implementation was carried out with AI coding assistance; every threshold,
estimand and audit rule was fixed by the author before implementation, and the
frozen specification in `docs/` governs the code rather than the reverse.
