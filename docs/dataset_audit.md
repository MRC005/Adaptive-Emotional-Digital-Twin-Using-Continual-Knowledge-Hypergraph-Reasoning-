# Dataset Audit

> **ONE REAL DATASET HAS NOW BEEN ACQUIRED AND AUDITED: RELAX.**
>
> RELAX (Zenodo `10.5281/zenodo.20701999`, CC-BY-4.0) was downloaded, its
> schema inspected, and the strict audit run against the real files. **It fails
> the frozen eligibility screen on self-report density** — see below. That is a
> measured result, not an assumption, and **no primary ρ\* result is reported
> from it.**
>
> StudentLife, PMData and WESAD remain at status **PLANNED** — no file opened.
> StudentLife is currently **unreachable** from this environment.
>
> Run `python scripts/audit_dataset.py --all` to reproduce this state, and
> `python scripts/fetch_relax.py --root data/raw/relax` to re-acquire RELAX.

## Dataset hierarchy and scientific roles

Roles are **not interchangeable**. A benchmark dataset cannot be promoted to
primary longitudinal validation by passing a flag.

| Dataset | Role | Status | Primary sensor | Self-report |
|---|---|---|---|---|
| **synthetic** | SIMULATION | ✅ implemented, runs | conversation minutes (simulated) | simulated 5-point ordinal |
| **StudentLife** | **PRIMARY_LONGITUDINAL** | ⚠️ **RDS repackaging audited — DEFECTIVE conversion** | daily conversation minutes ✅ intact | single-item stress EMA ⛔ ~99% lost |
| **RELAX** | LONGITUDINAL_ALTERNATIVE | ✅ **REAL — acquired & audited; FAILS eligibility** | heart rate from Polar IBI | `ifb-2` 7-point Likert (excited↔calm) |
| **WESAD** | **BENCHMARK_PHYSIOLOGICAL** | ⛔ PLANNED — files absent | chest ECG window statistics | protocol condition label |
| **PMData** | CONDITIONAL_SECONDARY | ✅ **REAL — acquired & audited; FAILS eligibility** | Fitbit resting heart rate | PMSys daily wellness `stress` |
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

## PMData audit — ✅ RUN ON REAL FILES

**Source.** Thambawita et al. (2020), *PMData: A Sports Logging Dataset*, ACM
MMSys. <https://datasets.simula.no/pmdata/>. Licence CC BY 4.0 **per the
publisher's page — the released archive itself contains no licence file.**

**Verified structure** (read from the archive, 1.4 GB zip, 912 members):

```
participant-overview.xlsx          demographics ONLY (age, height, sex, max HR)
p01..p16/pmsys/wellness.csv        effective_time_frame (ISO-8601 Z), fatigue,
                                   mood, readiness, sleep_duration_h,
                                   sleep_quality, soreness, soreness_area, stress
p01..p16/pmsys/srpe.csv            session RPE (not used)
p01..p16/fitbit/resting_heart_rate.json
                                   dateTime (NAIVE), value={date,value,error}
p01..p16/fitbit/heart_rate.json    intraday HR (~1.6 GB, not used)
p01..p16/food-images/              photographs (not used)
```

Only **0.32 MB** of the 1.4 GB archive is scientifically needed.

### ⛔ THREE INDEPENDENT BLOCKERS

**1. Two participants have no primary sensor at all.**
`resting_heart_rate.json` exists for **14 of 16** — p12 and p13 have none.
Reporting "PMData, 16 participants" would misstate the sample.

**2. The scale direction is undocumented.**
PMSys `stress` is a **bare integer**. The archive contains **no README, no
codebook, no questionnaire definition** — `participant-overview.xlsx` is
demographics only. Unlike StudentLife (label text) and RELAX (answer anchors),
**there is no text to key a remap on.** Nothing in the data states whether 5
means most or least stressed.

`configs/pmdata.yaml` therefore carries `direction_confirmed: false`, the
adapter passes the **raw** value through marked as *not a severity scale*, and
both the audit and the pipeline refuse a primary result. Confirming this
requires the PMSys instrument documentation, not inference from the data.

**3. Eligibility fails — 0 of 14.**

| | measured |
|---|---|
| participants with wellness + resting HR | 14 of 16 |
| wellness rows | 1 747 |
| rows joined to a same-day resting HR | 1 348 (22.8% unmatched) |
| median matched reports/participant | 95.5 (max 147) |
| participants ≥120 matched (60/epoch) | **4 of 14** |
| **eligible under the frozen screen** | **0 of 14** |

Exclusion reasons across the 14:

| reason | n |
|---|---|
| too few reports per epoch (<60) | 9 |
| **Var(s) epoch ratio outside [0.25, 4.0] — A3 violated** | 3 |
| sensor–report slope **flips sign** between epochs | 2 |
| \|β\| below the 0.02 floor | 1 |

**The A3 failures matter most.** p06 and p09 show Var(s) ratios of **13.06**
and **13.60** — the resting-HR variance changes by more than an order of
magnitude between epochs. Fitbit resting heart rate is an **algorithmic daily
estimate**, not a raw measurement, so its variance can shift for device reasons
rather than physiological ones. Assumption A3 is exactly what that breaks.

**Even the four densest participants fail** — on A3 or on a sign flip, not on
count. Adding data would not fix this.

### The `stress == 0` values

4 of 1747 rows carry `stress = 0`, outside the documented 1–5 range:

- **2 rows have EVERY wellness item at 0** — blank submissions. Always dropped;
  unambiguous cleaning, not imputation.
- **2 rows have valid answers elsewhere** — genuinely ambiguous. Controlled by
  `pmdata.zero_stress_handling`: `halt` stops with `DECISION REQUIRED`;
  `treat_as_missing` drops them, counted and reported, never imputed.

### Mixed clocks

Wellness timestamps are ISO-8601 with `Z` (tz-aware UTC); Fitbit `dateTime` is
naive. Merging them directly **raises** — a real bug the original adapter had,
exposed only by the real files. Both sides are now pinned to a UTC calendar day
so the join is well defined rather than accidentally working.

---

## RELAX audit — ✅ RUN ON REAL FILES

**Source.** Halmich, Jung, Schmoigl-Tonis, Schranz, Kremser, Kunas & Laireiter
(2026), *Scientific Data*. Zenodo `10.5281/zenodo.20701999`, **CC-BY-4.0**,
open access. Device: Polar Verity Sense.

**Acquisition.** `scripts/fetch_relax.py` reads the remote zip's central
directory over HTTP range requests and pulls **only** the ~0.5 GB that is
scientifically needed, leaving ~15.9 GB of unused accelerometer data on the
server. Provenance (DOI, licence, per-file sizes and SHA-256) is written to
`data/raw/relax/PROVENANCE.json`.

**Verified structure** (read from the archive itself, not from a description):

```
questionnaire_responses.xlsx   sheets: users, interv, mfb, ifb, afb, profile1..7
metadata/questionnaires.xlsx   item + ANSWER-LABEL definitions
metadata/README.md             data dictionary
data/<pid>/ibi_data.parquet    ibi_ppi (ms), ibi_blocker, ibi_errorEstimate,
                               timestamp (tz-aware UTC)
data/<pid>/acc_data.parquet    52 Hz triaxial accelerometer  [NOT USED]
```

31 participants (ids 12–63); four phases, 2024-02-25 → 2024-04-28 UTC.

### The label trap, again — and RELAX is worse than StudentLife

RELAX Likert items are anchored **in both directions**. Some ascend in stress
severity and some descend:

| item | anchors as released | severity |
|---|---|---|
| `ifb-2` "I feel:" | `excited` → `calm` | **REVERSED** (7 = calm = least stressed) |
| `ifb-7` "My mental effort is:" | `low` → `high` | ascending |
| `mfb-3` "I expect for today:" | `no stress at all` → `a lot of stress` | ascending |
| `afb-9` "…I felt overwhelmed today" | `strongly agree` → `strongly disagree` | **REVERSED** |

Mapping by stored value would silently invert the scale for **half** of them.
`aedt.io.relax.ITEM_SPECS` records the exact anchor pair for each supported
item and the adapter **halts with `DECISION REQUIRED`** if the release differs.

### Timestamps

`manual_date` is unix epoch **milliseconds**; `readable_date` is the same
instant expressed tz-naive. The adapter **cross-checks them** and refuses to
proceed if they disagree by more than one second. Measured agreement on the
real files: **0.0 s**.

### Sensor

Heart rate = `60000 / ibi_ppi`. Samples flagged by the device (`ibi_blocker`)
and physiologically implausible intervals (outside 300–2000 ms, i.e. 30–200
bpm) are **DROPPED, never repaired or imputed**. Roughly **36%** of raw IBI
samples are discarded this way on the real data. Heart rate is then averaged
over a strictly causal 2-hour window ending at each report, requiring ≥30 valid
samples or the value is left **missing**.

### ⛔ THE RESULT: RELAX FAILS THE FROZEN ELIGIBILITY SCREEN

Measured on the real files, densest item (`ifb-2`):

| | value |
|---|---|
| participants | 31 |
| aligned reports | ~1.5 k |
| **median reports per participant** | **~71** |
| **maximum reports per participant** | **~106** |
| **frozen requirement** | **≥120 (60 per epoch)** |
| **participants qualifying** | **0 of 31** |
| participants at a relaxed 40/epoch | ~10 of 31 |

`mfb` (max 43) and `afb` (max 41) are far worse.

**No primary ρ\* result is reported from RELAX.** The eligibility screen
refuses, the pipeline exits with code 3, and the screen is **not** relaxed to
make the data fit. A pre-specified sensitivity analysis (S1, 40/epoch) can be
run with `--sensitivity` and is reported alongside — never in place of — the
primary.

This is a genuine, citable finding: **the method needs a self-report density
that most current open longitudinal datasets do not provide.**

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


---

## StudentLife audit — ⚠️ RUN ON AN RDS REPACKAGING, WHICH IS DEFECTIVE

**Archive audited:** `data/raw/studentlife/dataset_rds.zip` (224 MB, 54 members,
zip integrity OK). This is a third-party **R-serialised repackaging** of the
Dartmouth study, not the original release.

**Reproduce:**
```bash
python scripts/convert_studentlife_rds.py          # needs Rscript
python scripts/audit_dataset.py --dataset studentlife --root data/interim/studentlife
```

### The sensor side is perfect

| | measured |
|---|---|
| conversation episodes | **79 023** |
| participants | **49** |
| invalid episodes | **0.0%** |
| span | 2013-03-27 → 2013-06-01 |
| episode duration | median 263 s, no non-positive values |

### The self-report side is destroyed

`EMA/Stress.Rds` has 2017 rows and three columns: `timestamp`, `uid`, and a
response column **literally named `null`**.

| | measured | expected |
|---|---|---|
| response column name | **`null`** | `level` |
| NA in response column | **88.1%** | ~0% |
| parsable 1–5 responses | **122** (6.0%) | ~35 000 |
| **max per participant** | **6** | **~735** |
| stray GPS strings in the response column | 109 | 0 |
| duplicate (uid, timestamp) rows | 13 | — |

**And the two sides do not overlap in time.** All 122 surviving responses fall
on **2013-03-24/25**, before conversation sensing begins on **2013-03-27**.
Participant identifiers match (46 in common), so this is not an ID problem —
the study-period EMA was simply lost in the `null` column.

**Result: not one aligned observation can be formed.** The pipeline exits **2**
with `DECISION REQUIRED`.

### Why this is a conversion defect, not a StudentLife limitation

Other EMA tables in the **same archive** converted correctly, with properly
named columns:

| table | rows | columns | NA |
|---|---|---|---|
| `PAM.Rds` | 9 040 | `picture_idx`, timestamp, uid | **0.0%** |
| `Mood.Rds` | 277 | `happy`, `happyornot`, `sad`, `sadornot`, `location` | 0.4% |
| `Sleep.Rds` | 1 644 | `hour`, `rate`, `social`, `location` | 15.5% |
| **`Stress.Rds`** | 2 017 | **`null`** | **88.1%** |

The converter preserves named fields everywhere else. The failure is specific
to Stress.

### Also missing: the codebook

`EMA_definition.json` is **absent from this repackaging entirely**. That file is
what the frozen specification keys its severity remap on. Without it there is no
way to verify that stored code 1 means *"Feeling great"* rather than *"Stressed
out"* — so the adapter carries the integers through **unmapped** and the audit
sets `eligible_for_primary_analysis = False` on that ground alone.

### What this changes

**StudentLife remains the correct primary target.** Unlike RELAX (intrinsically
too sparse) and PMData (intrinsically A3-unstable and undocumented), this
failure is an artefact of *one particular download*. The published descriptor's
~735 responses per student would comfortably clear the frozen screen.

**The fix is a different archive, not a different method.** Obtain the original
Dartmouth release, which ships `EMA/EMA_definition.json` and
`EMA/response/Stress/Stress_uXX.json`. The adapter already supports that layout
and is fixture-tested against it.
