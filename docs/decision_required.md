# DECISION REQUIRED — open scientific questions

This file records every point where the implementation found a genuinely
unspecified scientific decision, **and did not guess**.

## Status: THREE DATASETS AUDITED, ALL BLOCKED (D2, D7, D9, D10)

Neither obtainable real dataset can support the frozen primary endpoint. Both
failures are measured, not assumed, and neither was worked around.

The frozen specification was complete enough to implement without guessing.
Specifically:

- **A1–A5** are stated identically in `ROUND-17-FINAL-FREEZE-PACKAGE.md` §G and
  `ROUND-14-ordinal-method.md` §8. The "A1–A6" list in `ROUND-13` §155 belongs
  to the **superseded affine-anchor method** that ROUND-14 replaced. That is
  supersession, not conflict. **No conflicting frozen specification was found.**
- **The baseline specification is available** (`ROUND-14` §14, refined in
  `ROUND-16` §13), so no `DECISION REQUIRED: Frozen baseline specification
  unavailable` is raised. Six baselines are implemented; see below for the two
  that were deliberately not.
- **Eligibility thresholds, the epoch rule, the primary sensor, the label
  mapping and the primary endpoint** are all explicitly frozen in `ROUND-17` §W.

---

## Deferred until the real files exist

These are **not** blocking now. Each becomes a live question the moment a real
archive is opened, and the code raises the exact message shown.

### D1 — StudentLife stress labels (will fire if the wording differs)

```
DECISION REQUIRED: Dataset stress labels differ from expected mapping.
```

The frozen map is by **label text**. If `EMA_definition.json` contains any
option not in `aedt.constants.STRESS_LABEL_TO_SEVERITY`, the loader halts.
**Do not guess the severity order.** Confirm the wording against the dataset
documentation, then update the constant with the change recorded here.

*Raised by:* `aedt/preprocess/reports.py::build_code_to_severity`

### D2 — ⚠ BLOCKING: PMData stress scale direction is undocumented

**Files opened; the question is now confirmed as unanswerable from the data.**

The released archive (1.4 GB, 912 members) contains **no README, no codebook
and no questionnaire definition**. `participant-overview.xlsx` holds
demographics only (age, height, sex, max HR, stride). PMSys `stress` is a
**bare integer**, observed range 1–5, mode 3.

Unlike StudentLife (option label text) and RELAX (answer-label anchors),
**there is no text anywhere in the release to key a remap on.**

**Question:** does a higher PMSys `stress` value mean MORE stress or LESS?

**Current handling:** `configs/pmdata.yaml` carries
`direction_confirmed: false`. The adapter carries the **raw** stored value
through, explicitly marked as *not a verified severity scale*; the audit sets
`eligible_for_primary_analysis = False`; and `aedt/pipeline.py` adds a hard
blocking reason. **No primary ρ\* result can be derived from PMData while this
is unresolved** — and if the direction were guessed wrong, the sign of the
result would mean the opposite of what was reported.

**To resolve:** consult the PMSys instrument documentation (not the dataset),
record the source here, then set `pmdata.direction_confirmed` and
`pmdata.severity_ascending`.

### D8 — PMData `stress == 0` (outside the documented 1–5 range)

4 of 1747 wellness rows carry `stress = 0`.

- **2 rows have EVERY wellness item at 0** → blank submissions. Always dropped.
  This is unambiguous cleaning, not a scientific decision and not imputation.
- **2 rows carry valid answers on the other items** → genuinely ambiguous. The
  release does not define whether 0 encodes "not answered" or a real response.

**Handling:** `pmdata.zero_stress_handling` — `halt` raises
`DECISION REQUIRED`; `treat_as_missing` (current default) drops them, counted
and reported in the audit's missingness, never imputed. Affects 0.11% of rows
and does not change any conclusion.

### D3 — RELAX file layout ✅ RESOLVED (files opened and verified)

The RELAX archive has been acquired and inspected
(Zenodo `10.5281/zenodo.20701999`, CC-BY-4.0). The layout in
`configs/relax.yaml` is now a **verified fact**, not an expectation:
`questionnaire_responses.xlsx`, `metadata/questionnaires.xlsx`,
`metadata/README.md`, `data/<pid>/ibi_data.parquet`.

### D4 — RELAX stress-rating direction ✅ RESOLVED (verified from anchor text)

`configs/relax.yaml` now carries `direction_verified: true`.

RELAX Likert items are anchored in **both directions** — some ascend in stress
severity and some descend — so mapping by position would silently invert the
scale for half of them. `aedt.io.relax.ITEM_SPECS` records the exact English
anchor pair for every supported item, and the adapter **halts with
`DecisionRequired`** if the anchors in the release differ.

Verified directions:

| item | anchors (as released) | severity |
|---|---|---|
| `ifb-2` "I feel:" | `excited` → `calm` | **REVERSED** (7 = calm = least stressed) |
| `ifb-7` "My mental effort is:" | `low` → `high` | ascending |
| `mfb-3` "I expect for today:" | `no stress at all` → `a lot of stress` | ascending |
| `afb-9` "…I felt overwhelmed today" | `strongly agree` → `strongly disagree` | **REVERSED** |

### D6 — ⚠ OPEN: which RELAX item is the self-report for ρ\*

**This is a genuine scientific decision the frozen specification does not
cover.** The frozen spec names the primary report variable for StudentLife
(single-item stress EMA) and PMData (PMSys `stress`), but RELAX did not exist
in the specification as an opened dataset.

**RELAX has no single-item "stress" scale of that kind.** The candidates, with
their measured densities:

| item | what it measures | median obs/participant |
|---|---|---|
| `ifb-2` excited↔calm | momentary tension/arousal — a stress **proxy** | ~71 |
| `ifb-7` mental effort | momentary **demand**, not stress response | ~71 |
| `mfb-3` expected stress | explicitly "stress" but **anticipatory** | ~38 |
| `afb-9` felt overwhelmed | PSS-like, but **once per day** | ~34 |

**Current choice: `ifb-2`,** because it is the densest repeated ordinal item
and the closest momentary analogue of the StudentLife stress EMA. It is
configurable (`relax.item`) and the choice is stamped into every audit.

**It is a proxy, not a stress scale, and the reports say so.** Confirm the item
choice before any RELAX result is treated as a stress finding.

### D7 — ⚠ BLOCKING: RELAX cannot meet the frozen eligibility threshold

**Measured, not assumed.** On the densest item (`ifb-2`):

- median **71** self-reports per participant, maximum **106**
- the frozen screen needs **≥120** (60 per epoch)
- **0 of 31 participants qualify**; only 10 of 31 reach even 80 (40 per epoch)

`mfb` (max 43) and `afb` (max 41) are far worse.

**No primary ρ\* result may be reported from RELAX under the frozen
specification.** The alternatives, and why they were rejected:

1. **Lower `MIN_REPORTS_PER_EPOCH` to 40.** Legitimate *only* as the
   pre-specified sensitivity analysis **S1**, never as the primary. It leaves
   ~10 eligible participants — exactly the bootstrap minimum, so any interval
   would be extremely wide.
2. **Pool `ifb` + `mfb` + `afb`.** **Rejected.** They are different questions
   with different wordings and different anchors; pooling them would violate
   A2 (fixed person-specific thresholds) outright.
3. **Lower the threshold to fit the data.** **Rejected** — that is changing the
   frozen method to fit a dataset.

### D5 — Which covariate, if [9b] shows a weak association

Pre-specified: if the median |β| on epoch 1 is below **0.15**, switch to the
PC1 fallback (`aedt/features/common.py::PC1Fallback`, epoch-1-fitted loadings)
**before interpreting anything**. This is already decided; what is not decided
is whether it will be needed, which only real data can answer.

---

## Deliberately not implemented, with reasons

### Ablations 2–7

`ROUND-14` §15 lists seven ablations. Only **Ablation 1 (context
representation)** is implemented and run, because it is the only one that
connects the architecture to the title. Ablations 2–7 (sensor subset, link
function, standardisation on/off, epoch definition, report variable, rolling
update) are **PLANNED** and require real data to be meaningful.

### Two of the eight listed baselines

`ROUND-14` §14 lists eight. Six are implemented
(`aedt/baselines/runners.py`). Not implemented:

- **Ordinal mixed-effects with epoch interaction.** Requires a random-effects
  ordinal fitter; `statsmodels` is deliberately excluded from the frozen
  software list (`ROUND-17` §S.13), and hand-rolling a correct GLMM is a
  research task, not packaging. `ordinal_no_ratio` covers the same scientific
  question — the ordinal model *without* our ratio contribution — using the
  fitter that is already validated.
- **Naive augmented OLS / two-stage-plus-Wilcoxon.** Already run and reported
  in Round 14 (type-I 6.4–16.2% and 24.8% respectively). Re-implementing them
  would add no information; the historical result stands and is cited.

**This is a scope decision, not an oversight.** If a reviewer requires them,
they are a day of work each.

### Patent work

Position is **RED** (`ROUND-17` §AF), confirmed twice. No concrete novel
mechanism was found during implementation, so **no `POTENTIAL IP` trigger is
recorded** and no drafting was undertaken.

---

## How to add an entry

If implementation or analysis meets a genuinely unspecified scientific
decision:

1. **Stop.** Do not guess.
2. Raise `aedt.errors.DecisionRequired` with a message naming the exact
   discrepancy and what would resolve it.
3. Add a section here: the question, what the code does instead, and what
   evidence would settle it.
4. Only then change the specification — and record the change in
   `docs/frozen_scientific_specification.md`.


### D9 — ⚠ BLOCKING: PMData fails the frozen eligibility screen

**Measured on the real files** (14 participants with both wellness and resting
HR; p12 and p13 have no `resting_heart_rate.json` at all):

| | measured |
|---|---|
| matched reports | 1 348 of 1 747 wellness rows (22.8% unmatched) |
| median matched/participant | 95.5 (max 147) |
| participants ≥120 (60/epoch) | 4 of 14 |
| **eligible under the frozen screen** | **0 of 14** |

Exclusions: 9 too few reports · **3 Var(s) epoch ratio outside [0.25, 4.0]
(A3)** · 2 sign flips · 1 \|β\| below floor.

**The A3 failures are the scientifically interesting part.** p06 and p09 show
Var(s) ratios of 13.06 and 13.60 — resting-HR variance changing by more than an
order of magnitude between epochs. Fitbit resting heart rate is an
**algorithmic daily estimate**, not a raw measurement, so its variance can move
for device reasons rather than physiological ones. **Even the four densest
participants fail**, on A3 or a sign flip rather than on count — so more data
would not rescue PMData.

Rejected alternatives: lowering the screen (changes the frozen method);
switching to intraday `heart_rate.json` (a different, undeclared sensor and a
new specification decision, not an implementation choice).


### D10 — ⚠ BLOCKING: the StudentLife RDS repackaging is a defective conversion

**Measured on `data/raw/studentlife/dataset_rds.zip`:**

- `EMA/Stress.Rds` stores its response in a column **literally named `null`**,
  **88.1% NA**, additionally containing 109 GPS coordinate strings
- **122 of 2017 rows** parse as a 1–5 response (6.0%); **max 6 per
  participant** against ~735 in the published descriptor — ~99% loss
- All 122 survivors fall on **2013-03-24/25**, *before* conversation sensing
  starts on 2013-03-27 → **zero temporal overlap, no observation formable**
- **`EMA_definition.json` is absent**, so the mandated label-text remap cannot
  be applied at all

**Not a StudentLife limitation.** `PAM.Rds` (9 040 rows, 0% NA), `Mood.Rds` and
`Sleep.Rds` in the *same archive* converted correctly with named columns. The
conversation sensor is likewise perfect (79 023 episodes, 49 participants, 0%
invalid).

**Resolution: obtain the ORIGINAL Dartmouth release** — not a change to the
method, the threshold, or the report variable.

**Rejected alternative:** substituting PAM for the stress item. PAM is a
different instrument (a 4×4 photographic affect grid whose `picture_idx` is
*not* an ordered severity scale), the frozen specification lists it as
**Ablation 6 — a sensitivity analysis, not the primary**, and mapping its index
to severity would need the PAM codebook. Using it as the primary would be
changing the specification to fit a damaged file.
