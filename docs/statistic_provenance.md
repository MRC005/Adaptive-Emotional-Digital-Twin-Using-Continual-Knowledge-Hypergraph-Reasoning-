# Provenance of every reported statistic

Where each number on the website and in the README comes from, what computes
it, and whether it is currently **verified** — meaning a script in this
repository produced it and a test can check the site against that output.

Status vocabulary, used strictly:

| | Meaning |
|---|---|
| **VERIFIED** | A committed script wrote it to a result file, and a test asserts the published copy equals that file. |
| **COMPUTABLE** | The computation exists and is tested, but has not been run, because the archive is absent. The published value is a historical literal. |
| **NOT REPRODUCIBLE** | No script produces it. |

At the time of writing, `results/` is absent and
`data/raw/college-experience/` holds only `PROVENANCE.json`, so **nothing is
VERIFIED today**. That is the honest state, and
`tests/regression/test_findings_provenance.py::test_published_ceiling_matches_the_generated_artefact`
skips with that reason rather than passing quietly. It becomes a real check the
moment the archive returns.

---

## 1. The primary experiment — frozen, not touched

| Statistic | Produced by | Result file | Published in | Status |
|---|---|---|---|---|
| macro-F1 per model, K ∈ {0,5,10,20,40,80} | `scripts/run_twin_experiment.py` | `results/twin/twin_experiment.json` | `findings.json :: headline`, `learning_curve` | COMPUTABLE |
| 95% CIs (participant-clustered bootstrap, 2,000 resamples) | same | same | same | COMPUTABLE |
| Twin vs persistence paired difference, improved/harmed counts | same | same | `findings.json :: headline.twin_vs_persistence` | COMPUTABLE |
| Twin vs B4 paired difference | same | same | `findings.json :: headline.twin_vs_calibrated` | COMPUTABLE |
| Split sizes 130 / 43 / 44 | same | same (`n_train`, `n_val`, `n_test`) | `findings.json :: cohort` | COMPUTABLE |
| Ablation arms A1, A3, A4, A5, A6 | `scripts/run_twin_ablation.py` | `results/twin/twin_ablation.json` | `findings.json :: ablation` | COMPUTABLE |

Neither script has been modified. The pre-registered protocol, the primary
metric, the participant split, the K grid and the recorded null verdict are
unchanged and are to stay unchanged.

## 2. The cohort descriptors

Formerly hand-entered in `scripts/export_findings.py`. Now computed by
`cohort_descriptors()` in `scripts/run_ceiling_analysis.py`.

| Statistic | Published value (unverified) | Definition | Status |
|---|---|---|---|
| participants | 218 | distinct `uid` with at least one non-null `stress` | COMPUTABLE |
| reports | 35,348 | count of non-null `stress` rows | COMPUTABLE |
| years | 4.8 | (last day − first day) / 365.25 | COMPUTABLE |
| prediction_pairs | 25,966 | consecutive report pairs with gap in 1–7 days | COMPUTABLE |

## 3. The ceiling

Formerly twelve literals in `scripts/export_findings.py`, with the emitted
`_source` field claiming they came from the experiment's result files. They did
not. The computation now lives in `aedt/audit/ceiling.py` and is driven by
`scripts/run_ceiling_analysis.py` → `results/twin/ceiling.json`.

| Statistic | Published value (unverified) | Definition now implemented | Status |
|---|---|---|---|
| within_person_autocorrelation | 0.339 | lag-1 Pearson r over consecutive report pairs, each participant's pairs centred on their own mean, pooled | COMPUTABLE |
| variance_explained | 0.115 | r², asserted equal by test | COMPUTABLE |
| icc_between_person | 0.246 | ICC(1), one-way random-effects ANOVA, unequal group sizes | COMPUTABLE |
| strongest_behaviour_r | 0.0907 | largest \|r\| between any sensing column and the next report, sensing taken from the previous day | COMPUTABLE |
| behaviour_variance_explained | 0.008 | that r² | COMPUTABLE |
| per_person_r_median / IQR / range | 0.346 · [0.239, 0.466] · [−0.243, 0.687] | across participants with ≥ 30 reports | COMPUTABLE |
| frac_near_unpredictable | 0.13 | share with r < 0.15 | COMPUTABLE |
| frac_well_predictable | 0.17 | share with r > 0.50 | COMPUTABLE |
| early_late_r | 0.355 | Pearson r across participants between first-half r and second-half r, each half needing ≥ 15 usable pairs | COMPUTABLE |
| n_participants_analysed | 194 | participants clearing the ≥ 30-report screen with a defined r | COMPUTABLE |

### Definitions that had to be chosen, and are therefore declared

The original values were produced by a procedure that is not in the
repository, so some definitions could not be recovered and had to be fixed.
They are declared in `aedt/audit/ceiling.py` before the data returns, not after:

* **Pair rule.** A pair is a report and the participant's *next* report. The
  **primary** definition restricts the gap to 1–7 days, matching the prediction
  task in the pre-registration. `run_ceiling_analysis.py` also computes the
  **all-pairs** variant and writes it as `ceiling_all_pairs`, so a reader can
  see both rather than take one on trust. `tests/unit/test_ceiling.py` includes
  a case where the two definitions genuinely disagree, so the distinction is
  not decorative.
* **Person-centring.** Each participant's pairs are centred on their own mean
  before pooling. Without this the statistic absorbs between-person spread; the
  test that plants ρ = 0.4 across cohorts with means ±20 recovers 0.40 centred
  and 0.9985 uncentred.
* **Screens.** ≥ 30 reports per participant; ≥ 15 usable pairs in each half for
  the early/late split; a participant with no variance is **excluded and
  counted**, never imputed as 0. Exclusions are written to the result file.

**If the regenerated values differ from those above, the difference is
reported, not reconciled.** No definition will be adjusted to reproduce a
number.

## 4. Known discrepancies in the current record

### 4.1 The "648 behavioural features" claim is not what the ablation tested

Stated in `README.md`, `docs/preregistration_twin_prediction.md` §11, the
project handbook and the website: *"648 daily sensing features contributed
nothing — removing the entire behavioural channel improved the model
(0.2850 vs 0.2757)."*

Measured from the code:

| | Count |
|---|---|
| `SENSING_FEATURES` declared in `aedt/twin/prediction_data.py` | **6** |
| Sensing columns in the model's frame (6 raw + 6 trailing 7-day means) | **12** |
| Total feature columns the model receives | **25** |
| Columns removed by the ablation's `A6_no_behaviour` arm | **14** (the 12 sensing, plus `gap_days` and `dow`, which are context rather than behaviour) |

The archive has roughly 648 daily sensing columns, but **the model never
received them.** 648 is the size of the *screening* space that belongs to the
ceiling analysis, not the size of the channel the ablation removed. The
ablation result stands exactly as measured; the sentence describing it
overstates its scope, and the arm's name is also slightly wrong.

This is recorded here rather than silently corrected. The reported numbers are
unchanged, and the wording is the owner's to decide.

### 4.2 The exporter's `_source` field was false

It claimed the ceiling came from `twin_experiment.json + twin_ablation.json`.
Neither file contains a ceiling block. The field has been removed and replaced
with a `_provenance` object that names a real source for each block, and a test
now forbids `_source` from returning.

### 4.3 11% vs 12%

The site computes `(variance_explained * 100).toFixed(0)` from
`findings.json`, giving **12%** from 0.115. The handbook says 11%. The site is
the computed one; the handbook rounds down. No code change needed.

## 5. How to restore full verification

```
# 1. obtain the archive (see data/raw/college-experience/PROVENANCE.json)
#    2.76 GB, 256 files, Kaggle: subigyanepal/college-experience-dataset

# 2. regenerate every artefact
python3 scripts/run_twin_experiment.py      # frozen protocol, unchanged
python3 scripts/run_twin_ablation.py
python3 scripts/run_ceiling_analysis.py     # new; writes results/twin/ceiling.json

# 3. push the numbers into the site, which now refuses if an input is missing
python3 scripts/export_findings.py

# 4. the guard becomes a real check rather than a skip
python3 -m pytest tests/regression/test_findings_provenance.py -rs
```

Step 2 must be compared against the values recorded in
`docs/preregistration_twin_prediction.md` §11. **Any discrepancy is to be
reported, not absorbed** — no methodology may be adjusted to make the
regenerated numbers match the published ones.
