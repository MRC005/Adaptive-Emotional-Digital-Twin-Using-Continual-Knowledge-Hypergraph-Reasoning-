# Dataset Audit

> **NO DATASET FILE HAS BEEN OPENED BY THIS PROJECT.**
>
> Every real dataset below is at status **PLANNED**. The adapters, the audit
> framework and the acquisition instructions are implemented and tested against
> synthetic dataset-shaped fixtures. What is missing is one person-hour of
> downloading — nothing more, and nothing less.
>
> Run `python scripts/audit_dataset.py --all` to reproduce this state.

## Dataset hierarchy and scientific roles

Roles are **not interchangeable**. A benchmark dataset cannot be promoted to
primary longitudinal validation by passing a flag.

| Dataset | Role | Status | Primary sensor | Self-report |
|---|---|---|---|---|
| **synthetic** | SIMULATION | ✅ implemented, runs | conversation minutes (simulated) | simulated 5-point ordinal |
| **StudentLife** | **PRIMARY_LONGITUDINAL** | ⛔ PLANNED — files absent | daily conversation minutes | single-item stress EMA (5 ordered levels) |
| **RELAX** | LONGITUDINAL_ALTERNATIVE | ⛔ PLANNED — files absent | windowed wearable physiology | repeated subjective stress rating |
| **WESAD** | **BENCHMARK_PHYSIOLOGICAL** | ⛔ PLANNED — files absent | chest ECG window statistics | protocol condition label |
| **PMData** | CONDITIONAL_SECONDARY | ⛔ PLANNED — files absent | Fitbit resting heart rate | PMSys daily wellness `stress` |
| SWELL-KW, AffectiveROAD | ROBUSTNESS_OPTIONAL | not implemented | — | — |

## What each role may and may not establish

**LONGITUDINAL ESTIMAND VALIDATION** requires all of: repeated observations per
participant; participant IDs; time ordering; usable repeated self-report;
sensor measurements aligned to reports; enough observations for Epoch 1 vs
Epoch 2; and no future leakage.

**PHYSIOLOGICAL BENCHMARK VALIDATION** may validate feature extraction and
robustness, **but must never be presented as longitudinal validation of ρ\***
if its structure cannot support that.

### ⛔ WESAD cannot support ρ\* — enforced in code

WESAD is a **single laboratory session per subject** with protocol-defined
condition blocks. It has subject IDs, protocol structure, condition labels, rich
sensor modalities and sample timing — but **no repeated self-report across
separated longitudinal epochs**.

ρ\* is a ratio of within-person sensor→report slopes across halves of a
multi-week enrolment span under **fixed thresholds (A2)**. A single session
contains no such span. Splitting one session into "epochs" would produce a
number, and that number would not be an estimate of longitudinal
reporting-scale change.

Therefore `WesadAdapter.can_support_longitudinal_estimand = False`, the audit
reports `eligible_for_primary_analysis = False` with the reason, and
`aedt/io/wesad.py::assert_benchmark_only` raises. **This is a property of the
data and is not overridable by configuration.**

---

## Audit fields emitted for every dataset

`DatasetAudit` carries all of: dataset name · source status · local file
availability · participant count · observation count · sensor modalities ·
self-report variables · self-report scale · stress labels · raw stored codes ·
code-to-label mapping · code-to-severity mapping · timestamps · timezone ·
longitudinal span · observations per participant · missingness ·
participant-level coverage · sensor/report alignment · conversation/context
availability · eligibility for primary analysis · eligibility for benchmark
analysis · reasons for exclusion · acquisition instructions.

**A field that cannot be determined because the files are absent is `None` —
never guessed, never zero.**

---

## StudentLife Day-1 procedure

Implemented step by step in `aedt/io/studentlife.py::audit`:

1. Locate the actual StudentLife files.
2. Locate `EMA_definition.json`.
3. Extract the stress item's response labels **verbatim**.
4. Extract the stored integer codes.
5. Verify the label→severity mapping **from text**, never from position.
6. Inspect PAM storage.
7. Count observations.
8. Check timestamps — unix epoch vs formatted datetime is **detected**, not assumed.
9. Check conversation/context data availability.
10. Produce the first T4 audit table.

**If the files are absent:**

```
REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN
```

and nothing is substituted.

### The label trap — why this matters more than anything else here

The frozen mapping is **by label text**:

| label | severity |
|---|---|
| `feeling great` | 1 |
| `feeling good` | 2 |
| `a little stressed` | 3 |
| `definitely stressed` | 4 |
| `stressed out` | 5 |

`EMA_definition.json` stores the options in **file order**, and that order is
**not guaranteed to be severity order**. Mapping by position would silently
reverse the stress scale and invert every conclusion in the project.

An unrecognised label raises:

```
DECISION REQUIRED: Dataset stress labels differ from expected mapping.
```

The synthetic StudentLife fixture deliberately stores the options **out of
order** so this trap is exercised on every test run
(`tests/unit/test_remap.py`).

### Known finding from the fixture run

Running the pipeline on a StudentLife-shaped fixture reproduces the documented
Round-16 result: a **weak sensor–report association** produces many exclusions
for an indeterminate slope or a sign flip, and a uselessly wide interval
*despite a perfectly calibrated estimator*.

**Calibration and usefulness are different properties.** This is why diagnostic
`[9b]` exists and is the **first** number read on real data. If the median
|β| is below 0.15, switch to the pre-specified PC1 fallback covariate **before
interpreting anything else**.

Pinned by
`tests/integration/test_missing_real_data.py::test_studentlife_fixture_reproduces_the_documented_weak_association`.

---

## PMData audit

Verifies: PMSys stress scale · the exact stress variable · raw codes and labels
· timestamps · participant IDs · resting-HR availability · longitudinal
structure · missingness · participant count · longitudinal span.

**Conditional** because 16 participants × ~5 months × one report per day gives
roughly 75 reports per epoch against a frozen minimum of 60. Whether it clears
the bar is an empirical question the audit answers; it is not assumed either
way.

**PMSys `stress` is stored as a bare integer with no label text**, so the
label-text remap cannot be applied. The audit records the observed code range
verbatim and `configs/pmdata.yaml` carries `direction_confirmed: false`. **The
scale direction must be confirmed against the PMData documentation before any
primary analysis.**

If a required variable is unavailable:

```
DECISION REQUIRED: PMData required variables unavailable.
```

---

## RELAX audit

Verifies: repeated subjective stress observations · participant identifiers ·
timestamps · wearable physiology · longitudinal span · repeated measures
sufficient for epoch analysis · alignment between physiology and reports.

Physiology is aggregated into a **causal window ending at each report**
(2 hours by default, configurable), and the alignment metadata is retained so
the leakage assertion can check it.

⚠️ **The expected file layout is a DECLARED EXPECTATION, not a verified fact.**
No RELAX file has been opened. Column names live in `configs/relax.yaml` so
adapting to the actual release is a **config change, not a code change**. If
the adapter halts with `DECISION REQUIRED`, record the actual column names in
the config — do not edit the adapter to guess.

---

## WESAD audit

Verifies: subject IDs · protocol structure · stress labels · sensor modalities ·
sample timing · and **whether the data can support (A) engineering benchmark
analysis or (B) longitudinal primary ρ\* analysis**.

**The audit does not claim WESAD supports (B), because it does not.** See above.

Condition codes 0, 5, 6, 7 are transient/undefined and are **not analysed**.
Windows whose dominant label covers less than 90% of the window are skipped.

---

## Acquisition instructions

Printed by the audit whenever files are absent, and reproduced here.

### StudentLife (Dartmouth) — Wang et al. 2014
1. Download the archive from the Dartmouth StudentLife study page
   (`https://studentlife.cs.dartmouth.edu/dataset.html`). Public research
   release; no credentials, but a large download.
2. Extract, then pass the directory **containing** `EMA/` and `sensing/` as `--root`.
3. Expected layout: `EMA/EMA_definition.json`, `EMA/response/Stress/Stress_u00.json`,
   `sensing/conversation/conversation_u00.csv`.
4. **Read the `[9b]` association line first.**

### PMData (Simula) — Thambawita et al. 2020, CC BY 4.0
1. Download from `https://datasets.simula.no/pmdata/` (open access).
2. Pass the directory containing `p01 … p16`.
3. Expected: `p01/pmsys/wellness.csv` (with a `stress` column),
   `p01/fitbit/resting_heart_rate.json`.

### WESAD — Schmidt et al. 2018
1. Download `WESAD.zip` from the UniPassau public release page.
2. Pass the directory containing `S2 … S17`. Expected: `S2/S2.pkl`.
3. **BENCHMARK ONLY.**

### RELAX
1. Obtain the release archive from its published repository record; note the
   DOI and licence here.
2. Pass the directory containing the per-participant report and physiology tables.
3. Record the actual column names in `configs/relax.yaml`.

---

## Reproducing this audit

```bash
python scripts/audit_dataset.py --all
```

To exercise an adapter end to end without the real archive:

```bash
python scripts/audit_dataset.py --fixture studentlife --out /tmp/sl_fixture
```

A fixture directory carries a `_SYNTHETIC_FIXTURE` marker file. The pipeline
reads that marker and **downgrades the result status to SYNTHETIC**, so a
fixture can never produce a result stamped REAL.
