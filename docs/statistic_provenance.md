# Provenance of every reported statistic

Where each number on the website and in the README comes from, what computes
it, and whether it is **verified** — meaning a script in this repository
produced it and a test checks the site against that output.

Status vocabulary, used strictly:

| | Meaning |
|---|---|
| **VERIFIED** | A committed script wrote it to a result file, and a test asserts the published copy equals that file. |
| **COMPUTABLE** | The computation exists and is tested, but has not been run. |
| **NOT REPRODUCIBLE** | No script produces it. |

**Archive restored 2026-09-05** and verified against `PROVENANCE.json`: all four
recorded files match on size and SHA-256 prefix. Every figure below has now been
regenerated. `frontend/src/data/findings.json` is written entirely from result
files; nothing in it is typed by hand.

---

## 1. The primary experiment — rerun, unmodified

`scripts/run_twin_experiment.py`, `scripts/run_twin_ablation.py`,
`aedt/twin/prediction_data.py` and the pre-registration are byte-identical to
the versions that produced the original result. The rerun changed no metric, no
split, no warm-up rule, no horizon and no K value.

**Structure reproduces exactly.** 25,966 prediction pairs; 217 participants with
at least one pair (218 have reports); split 130 / 43 / 44 with empty
intersections; test participants surviving warm-up 44 / 43 / 43 / 42 / 39 / 31
at K = 0 / 5 / 10 / 20 / 40 / 80.

**All five baselines reproduce to four decimal places at every K.**

| Model | Historical macro-F1 @ K=80 | Regenerated | Δ |
|---|---|---|---|
| B0 population majority | 0.1141 | 0.1141 | 0.0000 |
| B1 **persistence** | 0.3320 | 0.3320 | 0.0000 |
| B2 global | 0.1731 | 0.1731 | 0.0000 |
| B3 static personal prior | 0.1886 | 0.1886 | 0.0000 |
| B4 calibrated global | 0.1979 | 0.1979 | 0.0000 |
| **T twin** | **0.2847** | **0.2793** | **−0.0054** |

Only the twin moves, and only slightly: +0.0008 / +0.0001 / +0.0001 / +0.0030 /
+0.0017 / −0.0054 across the K grid.

**The verdict is unchanged: NOT SUPPORTED, two of five criteria met.**

| Criterion | Historical | Regenerated | Met |
|---|---|---|---|
| 1 twin beats persistence | 0.285 vs 0.332 | 0.279 vs 0.332 | ✗ |
| 2 paired CI excludes 0 favourably | −0.047 [−0.075, −0.020] | −0.053 [−0.082, −0.025] | ✗ |
| 3 > 50% of participants improve | 29% (9 of 31) | 16% (5 of 31) | ✗ |
| 4 improves with personal history | 0.262 → 0.285 | 0.263 → 0.279 | ✓ |
| 5 twin beats B4 | +0.087, 84% improved | +0.081, 77% improved | ✓ |

### Criterion 4, with the two readings kept apart

The pre-registration states criterion 4 as *"Performance improves with personal
history across K ∈ {0, 5, 10, 20, 40, 80}"*, and §11 records it as an endpoint
comparison. Those are different tests and are reported separately here. Neither
is redefined to obtain a preferred verdict.

**(a) Endpoint reading, K=0 → K=80 — the one §11 used. PASSES.**

| Metric | K=0 | K=80 | |
|---|---|---|---|
| macro-F1 | 0.2631 | 0.2793 | improves |
| accuracy | 0.4453 | 0.5009 | improves |
| MAE | 0.6941 | 0.5978 | improves (lower is better) |

**(b) Strict monotonicity across the whole grid. FAILS on the primary metric.**

| K | 0 | 5 | 10 | 20 | 40 | 80 | monotone? |
|---|---|---|---|---|---|---|---|
| macro-F1 | 0.2631 | 0.2700 | 0.2705 | 0.2767 | **0.2856** | 0.2793 | **no** — falls at K=80 |
| accuracy | 0.4453 | 0.4661 | 0.4691 | 0.4765 | 0.4822 | 0.5009 | yes |
| MAE | 0.6941 | 0.6651 | 0.6624 | 0.6541 | 0.6398 | 0.5978 | yes |

The historical macro-F1 curve — 0.2623, 0.2699, 0.2703, 0.2737, 0.2839, 0.2847
— **was** monotone. The regenerated one is not, and the single point
responsible is the same K=80 twin value that fails to reproduce.

**So: criterion 4 is met on the reading §11 actually applied, and fails on the
strict reading.** Under the strict reading the verdict becomes one of five
criteria rather than two. The overall verdict is **NOT SUPPORTED either way**,
because criteria 1, 2 and 3 fail under both readings and each independently
defeats the hypothesis.

One structural caveat that applies to both readings and to the historical curve
equally: **the K points are not computed on the same participants.** The
scored cohort shrinks from 44 at K=0 to 31 at K=80 as warm-up consumes each
participant's history. A change between adjacent K values therefore mixes a
learning effect with a cohort change, and the curve is not a within-participant
learning curve. This is a property of the pre-registered design, not of the
rerun, and it is noted rather than corrected.

### Why the twin moved and the baselines did not

An earlier draft of this document attributed the drift to unpinned dependency
versions. **That was wrong and is corrected here.** The versions did not change.

**Environment, measured 2026-09-05.** One interpreter, Python 3.12.10, no
virtualenvs anywhere on the machine, `aedt` imported from the working tree.
Install timestamps of the packages the experiment uses, against the historical
run on **2026-09-02**:

| Package | Version | Installed | Relative to the historical run |
|---|---|---|---|
| numpy | 1.26.4 | 2026-07-25 | before |
| pandas | 2.2.2 | 2026-07-25 | before |
| scikit-learn | 1.5.1 | 2026-07-25 | before |
| scipy | 1.17.1 | 2026-03-03 | before |
| joblib | 1.5.3 | 2026-02-05 | before |
| threadpoolctl | 3.6.0 | 2026-02-05 | before |
| torch | 2.11.0+cpu | 2026-04-04 | before (and not importable — see below) |
| onnxruntime | 1.20.1 | 2026-09-03 | **after**, and never imported by this experiment |

Every package the experiment touches was installed before the historical run
and has not changed since. **The historical environment does not need
reconstructing: this is it.**

**Hypotheses tested and eliminated.**

| Hypothesis | Test | Result |
|---|---|---|
| Different library versions | install timestamps above | ruled out |
| Different interpreter or venv | filesystem search | ruled out — one Python, no venvs |
| Different data | `restore_dataset.py` digests, and 25,966 pairs / identical n per K | ruled out |
| Different code | `git log`: one commit, never modified | ruled out |
| Run-to-run stochasticity | two consecutive full runs | bit-identical |
| OpenMP thread count | K=80 rerun at `OMP_NUM_THREADS` = 1, 4 and 18 | **0.279288 in all three** |
| A different online-adaptation strength | K=80 recomputed with the shrinkage constant swept 0 → 100 | **0.279288 for every value** |

The last row is worth its own sentence. The committed shrinkage is
`w = seen / (seen + 10)`; sweeping that constant from 0 to 100 moves `w` between
1.00 and 0.44 at K=80 and **does not change a single prediction**. At K=80 the
online residual never pushes a prediction across a rounding boundary, so the
twin there is exactly `round(model_full.predict(...))`. That is a stronger
statement than the ablation's "online adaptation adds nothing": at this warm-up
length it does not act at all.

It also removes the residual path as an explanation. The historical figure
would have to come from a different *fit* of `model_full` — but that fit is
deterministic here, on identical data, with the pre-registered seed, under an
unchanged library stack, and the five baselines that share `fit_global` and
`model_ctx` reproduce to four decimals.

**Conclusion.** The committed pipeline is fully deterministic and yields
**0.2793**. No recorded or reconstructable environment difference produces
0.2847. The remaining explanation consistent with the evidence is that the
historical twin figure was produced by a state of the twin's own prediction
path that was never committed — a draft edited before `6768087` was made. The
baselines are insensitive to that path, which is exactly why they reproduce and
it does not. **No attempt was made to find a code variant that yields 0.2847;
doing so would be fitting the method to a remembered number.**

**Why nothing recorded the environment.** `aedt/reporting/metadata.py` already
contains `make_run_metadata()`, which captures the Python version, package
versions and git commit, and `aedt/demo.py` uses it. `run_twin_experiment.py`
does not call it. The machinery existed and was not wired into the primary
experiment; had it been, this section would have been a one-line lookup.

### Ablation

| Arm | Features | Historical | Regenerated |
|---|---|---|---|
| A1 global only | 14 | 0.1582 | 0.1582 |
| A5 no trajectory | 19 | 0.2676 | 0.2676 |
| A3 global + history | 25 | 0.2757 | 0.2762 |
| A4 full twin, online | 25 | 0.2737 | 0.2767 |
| A6 no behaviour | 11 | 0.2850 | 0.2820 |

Both conclusions survive: removing the behavioural channel still **improves**
the model (0.2820 vs 0.2762), and online adaptation still contributes nothing.
One detail changed sign: historically A4 sat 0.0020 *below* static A3; it now
sits 0.0005 *above*. Both differences are far inside the confidence intervals,
so "adds nothing" remains the correct description, but the direction quoted in
§11 no longer holds.

## 2. The cohort descriptors — VERIFIED

Formerly hand-entered in the exporter; now computed by `cohort_descriptors()`.

| Statistic | Published | Regenerated | |
|---|---|---|---|
| participants | 218 | 218 | ✓ |
| reports | 35,348 | 35,348 | ✓ |
| prediction_pairs | 25,966 | 25,966 | ✓ |
| years | 4.8 | 4.82 | measured span, 2017-09-07 → 2022-07-02 |

## 3. The ceiling — the original procedure, recovered

Formerly twelve literals with no script behind them. The computation is now
`aedt/audit/ceiling.py`, driven by `scripts/run_ceiling_analysis.py` →
`results/twin/ceiling.json`.

The original procedure was not in the repository, so the first implementation
declared its own definitions. Running it on the restored archive made the
original **recoverable**: with participants included on **usable pairs**
(≥ 30) rather than raw observations, and the headline taken as the **mean** of
the per-person correlations rather than a pooled centred correlation, eight
published figures reproduce together.

| Statistic | Published | Recovered | |
|---|---|---|---|
| n_participants_analysed | 194 | 194 | exact |
| per_person_r_iqr | [0.239, 0.466] | [0.2389, 0.4657] | exact to 3 dp |
| per_person_r_range | [−0.243, 0.687] | [−0.2433, 0.6871] | exact to 3 dp |
| per_person_r_median | 0.346 | 0.3456 | ✓ |
| frac_near_unpredictable | 0.13 | 0.1289 | ✓ |
| frac_well_predictable | 0.17 | 0.1701 | ✓ |
| variance_explained | 0.115 | 0.1153 | ✓ |
| within_person_autocorrelation | 0.339 | 0.3395 | truncated, not rounded |

A participant count and six order statistics do not coincide by accident. The
alternative screen (≥ 30 *observations*) gives n = 205 and misses every one of
them. `tests/regression/test_findings_provenance.py` now pins the recovered
definition against the published block, so the recovery cannot be lost again.

### Three statistics still differ, and were not chased

| Statistic | Published | Regenerated | Note |
|---|---|---|---|
| icc_between_person | 0.246 | 0.2364 | one-way random-effects ANOVA on the same 194 participants |
| early_late_r | 0.355 | 0.3818 | halves split on observations, ≥ 15 pairs each; floors of 5/10/20 pairs give 0.374 / 0.373 / 0.349 — none lands on 0.355 |
| strongest_behaviour_r | 0.0907 | 0.1095 | all 648 sensing columns screened; the strongest is `call_in_num_ep_0` |

The search stopped here deliberately. Continuing to vary definitions until
these three matched would be fitting the method to a remembered number, which
is the practice this document exists to prevent. The published values are
superseded by the computed ones; the differences are small and none changes a
stated conclusion — behaviour still explains ~1% of variance rather than ~0.8%,
and predictability is still substantially stable across halves.

## 4. The "648 behavioural features" claim — origin established

**The model never received 648 columns, at any point in the project's history.**

`git log` shows `aedt/twin/prediction_data.py` was created once, in
`6768087`, with `SENSING_FEATURES` holding exactly six column names, and never
modified. Measured on the restored archive and from the ablation's own output:

| | Count |
|---|---|
| Sensing columns in `Sensing/sensing.csv` | **648** (651 columns − `uid`, `is_ios`, `day`) |
| `SENSING_FEATURES` declared in the frame builder | **6** |
| Sensing columns in the model's frame (6 raw + 6 trailing 7-day means) | **12** |
| Total feature columns the model receives | **25** |
| Columns removed by the ablation's `A6_no_behaviour` arm | **14** (the 12 sensing, plus `gap_days` and `dow`, which are context) |

The figure originates in `docs/scientific_upgrade_blueprint.md`, a feasibility
survey committed alongside the pre-registration in `45f3503`, which records
*"Daily sensing features, all strictly prior: 648"*. That statement is **true**
and has now been verified exactly against the archive — it describes what the
dataset offers.

The pre-registration §5, frozen in the same commit, correctly lists the six
sensing features actually used and does not mention 648.

The conflation entered in `6768087`, when §11 was appended after the results
and described the ablation's channel as *"the 648-column behavioural channel"*.
It propagated from there to `README.md` and to
`frontend/src/ui/discovered.js:188,199`.

So this is **provenance drift between two documents written in the same
commit** — an availability figure reused as a usage figure. Nothing was
fabricated, no experiment changed, and the ablation result stands exactly as
measured. Only the sentence describing its scope is wrong.

Corrected wording is **not applied here**; it touches the frozen
pre-registration and the README, and that is the owner's call. The accurate
statement is: *removing the twelve behavioural columns the model actually used
improved it*, alongside the separate and also-true finding that *the strongest
of all 648 sensing columns in the archive correlates with next-day stress at
r = 0.11, explaining about 1% of its variance.*

## 5. Remaining reproducibility gaps

1. **The historical K=80 twin figure (0.2847) is not reproducible.** The
   committed code deterministically gives 0.2793 and every environmental
   explanation has been eliminated. The five baselines, the splits, the pair
   count and the verdict all reproduce; this single number does not.
2. **No run records its own environment**, although
   `make_run_metadata()` exists to do it. Wiring it into
   `run_twin_experiment.py`, and pinning exact versions, would prevent a
   repeat. Neither has been done, because both would touch the frozen
   experiment.
3. **Three ceiling statistics** (§3) are computed differently from the original
   and the original definition is unrecovered.
4. `docs/preregistration_twin_prediction.md` §11 describes criterion 4 in a way
   that admits two readings, and the regenerated curve separates them.

## 6. How to regenerate everything

```
python3 scripts/restore_dataset.py          # verify the archive first
python3 scripts/run_twin_experiment.py      # frozen protocol, unchanged
python3 scripts/run_twin_ablation.py
python3 scripts/run_ceiling_analysis.py
python3 scripts/export_findings.py          # refuses if any input is missing
python3 -m pytest tests/regression/test_findings_provenance.py -rs
```
