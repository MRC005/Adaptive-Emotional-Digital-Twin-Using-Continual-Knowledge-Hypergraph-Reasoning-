# Longitudinal Affect Modelling: What Personalisation Does and Does Not Buy

> **A pre-registered test of the personalised digital-twin hypothesis on 218
> participants over four years — and a null result. Carrying the last reported
> value forward beat every personalised model we built.**

---

## The headline result

We pre-registered a protocol ([`docs/preregistration_twin_prediction.md`](docs/preregistration_twin_prediction.md))
and committed it before writing any model, then tested:

> *Does a personalised, continually updated twin predict an individual's next
> reported stress better than strong baselines, on held-out participants and
> strictly future observations?*

**Answer: no.** Two of five pre-registered criteria were met.

### As originally reported

These are the figures this study published. They are preserved unchanged.

| Model | macro-F1 @ K=80 | 95% CI |
|---|---|---|
| Population majority | 0.114 | [0.098, 0.128] |
| Global model, context + behaviour | 0.173 | [0.153, 0.194] |
| Global + static personal prior | 0.189 | [0.168, 0.210] |
| Per-person calibrated global (strong personalised baseline) | 0.198 | [0.177, 0.220] |
| **Proposed twin** | **0.285** | [0.253, 0.315] |
| **Persistence — carry the last value forward** | **0.332** | [0.294, 0.372] |

The twin beat every model we built **except the simplest one**. Against
persistence: −0.047 macro-F1, 95% CI [−0.075, −0.020], and **22 of 31 held-out
participants were harmed**.

### As regenerated — the reproducible record

On 2026-09-05 the archive was restored, verified against its recorded SHA-256
digests, and the frozen experiment rerun **without modification**. The dataset
structure reproduced exactly (25,966 prediction pairs, split 130/43/44, the
same scored cohort at every K) and **all five baselines reproduced to four
decimal places**. One number did not.

| K=80 macro-F1 | Originally reported | Regenerated | |
|---|---|---|---|
| Population majority | 0.1141 | 0.1141 | exact |
| Global model | 0.1731 | 0.1731 | exact |
| Global + static prior | 0.1886 | 0.1886 | exact |
| Per-person calibrated global | 0.1979 | 0.1979 | exact |
| **Persistence** | **0.3320** | **0.3320** | exact |
| **Proposed twin** | **0.2847** | **0.2793** | **−0.0054** |

**Both values stand. Neither replaces the other.** 0.2847 is what this study
reported; 0.2793 is what the committed code reproduces today and is the only
figure new work may be compared against.

Every environmental explanation for the difference has been eliminated —
library versions (all installed before the original run and unchanged), the
interpreter, virtualenvs, the data, the committed code, run-to-run variation,
the OpenMP thread count (1, 4 and 18 all give 0.279288) and the
online-adaptation strength (swept 0→100, unchanged throughout). No search was
made for a code variant that returns 0.2847. The full account, including what
this implies, is in
[`docs/statistic_provenance.md`](docs/statistic_provenance.md).

**The verdict is unchanged either way:** persistence wins by a slightly larger
margin under the regenerated figures (−0.053 rather than −0.047), and 25 of 31
participants were harmed rather than 22.

### Three findings worth more than the headline

*Figures below are as originally reported; the regenerated values are in
brackets where they differ.*

1. **Personal history is the only signal.** Global context+behaviour scores
   0.158; history alone scores 0.285 (regenerated: 0.158 and 0.279).
2. **The behavioural channel contributed nothing.** Removing it *improved* the
   model, 0.2850 vs 0.2757 (regenerated: 0.2820 vs 0.2762). The channel the
   model actually received is **twelve columns** — six daily sensing features
   and their trailing 7-day means. The archive offers 648 sensing columns and
   the strongest of them correlates with next-day stress at r ≈ 0.11; that is a
   separate measurement, and earlier wording conflated the two. See
   [`docs/statistic_provenance.md`](docs/statistic_provenance.md) §4.
3. **Online adaptation contributed nothing** (0.2737 vs 0.2757 static;
   regenerated 0.2767 vs 0.2762 — the difference changes sign but stays far
   inside the confidence intervals, so "nothing" remains the right word).

The twin does win on accuracy (0.504 vs 0.464) and MAE (0.598 vs 0.691) — it
predicts near-misses well but regresses toward the middle of the scale and
misses the rare extremes that persistence reproduces for free. macro-F1 was
declared primary before results were seen and was not changed afterwards.

---

## The two layers

The system receives a person's interactions over time, identifies emotional and
contextual patterns, represents context-rich emotional events as higher-order
relationships, learns personalised temporal patterns, and continually updates an
evolving twin — while explicitly investigating whether the relationship between
behaviour and self-reported emotion stays stable.

| | **Layer 1 — the personal twin** | **Layer 2 — longitudinal reliability** |
|---|---|---|
| **Question** | what has this person's history looked like? | can a long history be read as if the scale meant one fixed thing throughout? |
| **Data** | your own check-ins (or a fictional demo user) | dense cohort archives (218 students, 4 years) |
| **Method** | Transformer emotion detection, rule-based context extraction, event knowledge hypergraph, explainable retrieval | ordinal probit slope ratio ρ\*, eligibility screen, placebo gate, participant-cluster bootstrap |
| **Output** | comparable past episodes and a pattern statement, or "still learning" | drift detected / no detectable drift / insufficient evidence / incompatible |
| **Where it runs** | your device; the Transformer in the Python service, opt-in | offline Python, displayed in the browser |

**How they connect — and how they do not.** Layer 2's finding is a *trust
qualifier*, not a data path. If a cohort's measurement relationship drifts, a
personal history spanning the same period should not be extrapolated as though
the scale were fixed. Layer 2 never sees a personal check-in, and Layer 1 never
contributes to ρ\*. A personal history is also **far too short** to run the
Layer 2 estimator on one individual, and the application says so rather than
implying otherwise.

```
interaction / check-in
        ↓
Transformer emotion detection  +  context extraction        ← Layer 1
        ↓
structured emotional event (per-field provenance)
        ↓
knowledge hypergraph  →  HGNN / higher-order learning (offline)
        ↓
personalised pattern model  →  continual learning, EWC (offline)
        ↓
Adaptive Emotional Digital Twin
        ↓
longitudinal measurement-drift analysis                     ← Layer 2
        ↓
calibrated insight, with its own limits stated
```

---

## Layer 1 — what is implemented, and what it measured

Every component below is real code with a measured result. Where a result is
negative, it is reported as a negative result.

| Component | Status | Where | Measured |
|---|---|---|---|
| **Transformer emotion detection** | IMPLEMENTED | `aedt/emotion/detect.py` | `SamLowe/roberta-base-go_emotions` (RoBERTa-base, 124.7M params). Held-out GoEmotions test split, 5,427 examples: **macro-F1 0.4925**, micro-F1 0.5775. Lexicon baseline 0.0979 |
| **Context extraction** | IMPLEMENTED | `aedt/emotion/context.py` | deterministic rules; every value carries its evidence span; unstated fields stay UNKNOWN |
| **Structured emotional event** | IMPLEMENTED | `aedt/emotion/events.py` | per-field provenance: extracted / user-reported / model / inferred / corrected / unknown |
| **Knowledge hypergraph** | IMPLEMENTED | `aedt/hypergraph/event_graph.py` | one episode = one n-ary relation; typed vertices, incidence matrix, conjunctive queries |
| **HGNN** | IMPLEMENTED, **negative result** | `aedt/models/hgnn.py` | Feng et al. convolution. macro-F1 **0.5116** vs GCN 0.576 vs structure-free MLP **0.8153** (5 seeds). The hypergraph did **not** earn its place |
| **Continual learning (EWC)** | IMPLEMENTED, **positive result** | `aedt/continual/ewc.py` | forgetting **+0.3082 → +0.1333**; average accuracy 0.6892 → 0.742 against a joint upper bound of 0.7798 |
| **Personal twin + retrieval** | IMPLEMENTED | `aedt/twin/personal_twin.py` | weighted field match with explanations; no pattern below 3 comparable episodes |

### The three findings worth reading

**1. GoEmotions has no "stress" label.** The taxonomy NLP emotion research
standardised on does not contain the construct longitudinal wellbeing research
measures. Measured on this checkpoint, *"I am stressed and exhausted"* returns
sadness 0.463 with nervousness only 0.119, while *"I am so anxious about
tomorrow"* correctly returns nervousness 0.504. Handled by precedence: an
explicit first-person statement is a **self-report** and outranks the model,
with the matched phrase shown as evidence.

**2. The HGNN lost to a model with no structure at all.** MLP 0.8153,
GCN 0.576, HGNN 0.5116. With a small categorical
entity set, an episode's own membership already encodes the conjunction, and
propagating over a near-complete graph blurs it. **This is the second
independent hypergraph ablation in this project to reach that conclusion** — the
Layer 2 ablation found the same thing on entirely different data.

**3. EWC works, and the sweep shows why.** Catastrophic forgetting is
demonstrated first (+0.3082 over four sequential tasks with
conflicting rules), then reduced. The penalty-weight sweep is monotone:
λ=100 → +0.31, λ=1000 → +0.31, λ=10000 → +0.2216, λ=100000 → +0.1593, λ=1e+06 → +0.1367.

### What Layer 1 is NOT

- **NOT fine-tuned by this project.** The emotion checkpoint was trained by its
  author on GoEmotions. This project integrates and evaluates it.
- **NOT a neural context extractor.** No labelled check-in data exists for that;
  rules that return UNKNOWN are better than a model that returns confident noise.
- **NOT trained on your check-ins.** Storing an event moves no model parameter.
  EWC updates parameters, and only in the offline research pipeline.
- **NOT running in your browser.** RoBERTa cannot. The browser uses a word-list
  baseline unless you switch the Python service on, and the interface says which
  one produced every label.
- **NOT validated on real human check-in streams.** The HGNN and EWC experiments
  use synthetic data with a known rule. No corpus of longitudinal personal
  check-ins with emotion labels exists for this project.

### Dataset roles, and why the others were not merged

**GoEmotions is the primary**, and the only one used for a reported metric: it
is the label space of the chosen checkpoint, it is large (58k), and its test
split is genuinely held out.

`EmpatheticDialogues`, `MELD` and `DailyDialog` were **considered and not
used** — deliberately, not by oversight. Merging them would require reconciling
four incompatible label sets (32 situation labels, 7 TV-dialogue emotions, and a
neutral-dominated dialogue-act taxonomy), and a union taxonomy invented for
convenience would make every reported number uninterpretable. They remain
appropriate for a future domain-shift study, which is listed as remaining work
rather than claimed as done.

**These are text corpora, not longitudinal sensing.** GoEmotions cannot supply
passive sensing, and the College Experience Study cannot supply labelled chat.
The project never crosses them.

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

The deliverable is a working application, not a project write-up.

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Four modes:

1. **My Digital Twin** — write how you are; the system reads the feeling and the
   situation, shows you what it understood and **where each part came from**,
   lets you correct anything, then stores the episode, updates the hypergraph,
   retrieves comparable past episodes *with the reason they matched*, and either
   states a pattern or says it is still learning. A **synthetic demonstration
   user** can be loaded, labelled as fictional throughout.
2. **Analyse real data** — the audited study results, the audit explaining which
   archives can support the drift analysis, and CSV upload read locally.
3. **Guided demonstration** — controlled data with a known answer.
4. **Interactive sandbox** — deliberate perturbations, each stating its expected
   effect before the run and checked after it.

Results lead with a verdict, then what it does and does not mean in plain words,
then the evidence, then the statistics. Light and dark themes, remembered.

**Where the emotion model runs — three choices, and the interface always says
which one produced a label:**

| Engine | Runs where | Cost | Default |
|---|---|---|---|
| **RoBERTa in your browser** | your device | 125 MB once, then ~250 ms | **yes** |
| RoBERTa via the analysis service | Python service | network round trip; free tier may sleep | no |
| Word-list baseline | your device | instant, but macro-F1 0.098 vs 0.493 | no |

The in-browser engine is the default because it is the real model *and* it
removes every deployment failure mode at once: no stale build, no cold start,
no 512 MB memory ceiling, no CORS. Your check-in text never leaves the device.

**Agreement is measured, not asserted.** The deployed artefact is
`SamLowe/roberta-base-go_emotions-onnx :: model_quantized.onnx`, the same
author's int8 ONNX export of the same fine-tune. Two checks pin it:

```bash
python3 scripts/verify_onnx_agreement.py                   # ONNX vs torch
node frontend/scripts/verify_browser_agreement.mjs         # browser vs ONNX
```

Measured: ONNX vs torch — mean |ΔP| **0.0141**, top-1 label agreement **0.963**,
check-in label agreement **0.970**. Browser vs ONNX — same top label on every
case, max |Δ| **0.045**. Chained, the browser result is the RoBERTa result.

**Everything else:**

| | runs where |
|---|---|
| context extraction, events, hypergraph, retrieval, patterns | your browser |
| guided demo, sandbox, uploaded CSV analysis | your browser |
| bundled study results | offline Python, displayed not recomputed |
| HGNN and EWC experiments | offline Python only |

Personal history stays on your device. The archives are gigabytes and licensed
for research use rather than redistribution, so nothing participant-level is
shipped to the browser or served by the API.

To run the backend that serves the Transformer:

```bash
uvicorn backend.app:app --reload --port 8000
```

Then set `window.AEDT_API_URL` in `frontend/config.js` and choose the
Transformer in the Digital Twin's advanced settings. It is **off by default**,
because a sentence about your health leaving your device should be a decision.

### Reproducing the Layer 1 experiments

```bash
python3 scripts/eval_emotion_model.py      # GoEmotions held-out evaluation
python3 scripts/run_hgnn_experiment.py     # HGNN vs GCN vs MLP vs majority
python3 scripts/run_ewc_experiment.py      # forgetting, with a lambda sweep
```

Each writes a JSON report under `results/` carrying its own protocol block.

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

**Layer 2 (the scientific core)** needs only Python 3.11+ with `numpy`, `scipy`,
`pandas`, `matplotlib`, `pyyaml`, and `pytest`. **`statsmodels` is deliberately
excluded** — the ordinal probit is implemented directly. No GPU, no network at
run time.

**Layer 1** additionally needs `torch`, `transformers`, `scikit-learn` and
`datasets`. Without them the emotion detector reports `backend: "lexicon"` and
says so, and the HGNN and EWC experiments print `NOT RUN` rather than
estimating anything. Their tests skip rather than passing vacuously.

```bash
pip install -r requirements.txt          # Layer 2
pip install torch transformers scikit-learn datasets   # Layer 1
```

The emotion checkpoint (~500 MB) downloads from the Hugging Face Hub on first
use and is cached.

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
