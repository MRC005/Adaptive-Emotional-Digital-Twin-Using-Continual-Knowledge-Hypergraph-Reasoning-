# DECISION REQUIRED — open scientific questions

This file records every point where the implementation found a genuinely
unspecified scientific decision, **and did not guess**.

## Status: ONE BLOCKING ITEM (D7) — RELAX cannot meet the frozen eligibility threshold

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

### D2 — PMData stress scale direction

PMSys stores `stress` as a bare integer with **no label text**, so the
label-text remap cannot be applied. `configs/pmdata.yaml` carries
`direction_confirmed: false`.

**Question:** does a higher PMSys `stress` value mean more stress or less?

The audit records the observed code range verbatim and makes no assumption. A
primary PMData analysis must not run until this is confirmed against the PMData
documentation and the answer recorded here.

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
