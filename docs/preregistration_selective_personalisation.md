# Pre-registration — evidence-aware selective personalisation

**Frozen before any gate code is written.** Committed ahead of the
implementation so the protocol cannot be adjusted after seeing results. Any
later change must be recorded in §13 with its date and reason.

This is a **new** study. It does not re-open, re-score or reinterpret the
pre-registered twin-vs-persistence experiment recorded in
[`preregistration_twin_prediction.md`](preregistration_twin_prediction.md).
That experiment, its protocol, its metric, its splits and its **NOT SUPPORTED**
verdict stand as frozen historical evidence. New code lives in new files and
imports from the frozen scripts; it never edits them.

---

## 1. Motivation, in one paragraph

The first study asked whether a personalised twin predicts next-day stress
better than carrying the last value forward. It does not. It also showed that
the failure is not uniform: at K=20, **12 of 42** held-out participants were
better served by the twin than by persistence. Separately, the ceiling analysis
found that per-person predictability varies from r = −0.24 to 0.69 and that a
participant's early-history predictability correlates with their later
predictability. If that heterogeneity is visible *before* a prediction is made,
a system could stop personalising everyone and personalise only where the
evidence supports it.

## 2. Primary research question

> **Can longitudinal evidence available before prediction identify when
> personalisation is likely to outperform persistence?**

**H1 (primary).** A gate fitted on evidence available strictly before
prediction time routes participants between the twin and persistence better
than a gate that uses observation count alone.

**Null.** The evidence-aware gate does not beat the count-only gate by a
practically meaningful margin on held-out participants.

**H2 (secondary, mechanistic).** The gate recovers a measurable share of the
oracle routing headroom — the gain available to a router that knows, per
participant, which strategy actually scored higher.

## 3. The baseline record this study builds on

All new work is measured against the **regenerated** committed pipeline, not
against the historically published figures.

### 3.1 The baseline numbers (regenerated 2026-09-05, seed 20260828)

| | K=20 (primary) | K=80 (secondary) |
|---|---|---|
| test participants scored | 42 | 31 |
| B1 persistence macro-F1 | **0.3108** [0.2748, 0.3565] | **0.3320** [0.2940, 0.3722] |
| T twin macro-F1 | **0.2767** [0.2417, 0.3268] | **0.2793** [0.2458, 0.3162] |
| twin − persistence (paired) | −0.0341 [−0.0508, −0.0188] | −0.0527 [−0.0818, −0.0253] |
| participants improved by the twin | **12 of 42 (28.6%)** | **5 of 31 (16.1%)** |

### 3.2 The historical/regenerated discrepancy, preserved

**Both values are kept. Neither replaces the other.**

| | K=80 twin macro-F1 |
|---|---|
| **Historically reported** (README, pre-registration §11, the site until 2026-09-05) | **0.2847** |
| **Regenerated** from the committed pipeline on the verified archive | **0.2793** |

The historical value **is** what the original study reported and must continue
to be cited as such. It is not to be retroactively rewritten as 0.2793, and
nothing in this study may present 0.2793 as the originally reported figure.

The regenerated value is what the committed code produces today and is the only
figure new experiments may be compared against, because it is the only one that
can be reproduced.

The investigation is recorded in full in
[`statistic_provenance.md`](statistic_provenance.md). In summary: the archive is
digest-verified identical; the code is one commit, never modified; the five
baselines B0–B4, the splits and the pair count all reproduce to four decimals;
two consecutive runs are bit-identical; library versions all predate the
historical run and are unchanged; the result is invariant to OpenMP thread count
(1, 4, 18) and to the online-adaptation shrinkage constant swept 0→100. No
environmental cause exists. The remaining explanation consistent with the
evidence is an uncommitted state of the twin's own prediction path. **No search
was made, and none will be made, for a code variant that yields 0.2847.**

Consequence for this study: the twin's absolute level carries a ±0.005
reproducibility uncertainty that the baselines do not. Any conclusion here that
would flip under a 0.005 shift in the twin's score must be reported as
fragile.

## 4. Data, splits and the reuse limitation

**Archive.** College Experience Study, `data/raw/college-experience/`, verified
against `PROVENANCE.json` (four recorded files, SHA-256 prefixes match). 218
participants, 35,348 stress reports, 25,966 prediction pairs.

**Splits.** Reused **unchanged** from the frozen experiment —
`make_splits(data, seed=20260828)`, participant-disjoint 60/20/20:

| Split | n | Role in this study |
|---|---|---|
| train | 130 | fits the global models only, exactly as in the frozen experiment |
| **validation** | **43** | fits **everything** in this study: evidence standardisation, gate coefficients, regularisation strength, all thresholds, the abstention rate |
| test | 44 (42 scored at K=20, 31 at K=80) | opened **once**, at the end |

**Declared limitation.** The test participants are **not naive**: their
outcomes have been seen by the team during the first study. Re-splitting was
considered and rejected, because it would destroy comparability with the
pre-registered result that motivates this study. This limitation is declared
here, before the run, and must appear in any write-up. It is the single
largest threat to the inferential strength of this study.

**The validation split has never been used for anything.** The frozen
experiment computed `val_pids`, asserted disjointness and never fitted or
selected on it. That is what makes it a clean fitting surface today.

## 5. Task, warm-up and decision point

The prediction task is **unchanged** from the frozen protocol: for a report at
day *d*, predict the stress value at the next report *d′* with
`0 < (d′ − d) ≤ 7 days`. Features, the 7-day rule, the temporal invariant and
`assert_no_leakage` are inherited from `aedt/twin/prediction_data.py` without
modification.

**Primary warm-up: K = 20.** Chosen for power — 42 test participants survive
the "at least five scored observations" rule at K=20 versus 31 at K=80 — and
because K=20 is the ablation's K, fixed in the earlier protocol before any
result was seen. **K = 80 is a secondary sensitivity analysis** and is reported
whether or not it agrees.

**Decision point (primary).** One routing decision per participant, made once
at the end of warm-up, using only observations at or before the warm-up cutoff.
This matches the participant-clustered resampling unit and is the
better-powered design.

**Decision point (secondary).** Per-observation re-evaluation, where the
evidence vector is recomputed at every prediction opportunity. Reported as a
secondary analysis. In this variant the gate at time *t* may use the outcomes
of observations at times **strictly before** *t*, including previously scored
ones — those are the past by then, and the twin's own online residual already
uses them. It may never use the outcome at *t* or later.

## 6. The evidence vector

Computed for participant *p* at decision time *t*, from that participant's
observations with `prediction_time < t` only. Fixed here; **no variable may be
added, removed or transformed after results are seen.**

| # | Name | Definition |
|---|---|---|
| E1 | `n_prior` | count of prior observations, entered as `log1p(n)` |
| E2 | `density` | `n_prior / (day span of the prior observations)`, reports per day |
| E3 | `r_hat` | expanding-window lag-1 Pearson autocorrelation over prior consecutive pairs restricted to a 1–7 day gap, per `aedt.audit.ceiling.person_autocorrelation` |
| E4 | `r_stab` | `abs(r_hat(first half of prior) − r_hat(second half of prior))`; requires ≥ 8 usable pairs in each half, otherwise missing |
| E5 | `vol_recent` | standard deviation of the last 10 prior stress values |
| E6 | `regime` | maximum absolute two-sided CUSUM statistic of the standardised prior series, divided by `n_prior` |
| E7 | `adv_hat` | mean over prior observations of `abs(persistence error) − abs(twin error)`, i.e. the twin's realised advantage in absolute error, computed **only on warm-up observations that are never scored** |

**Missing values.** E3–E7 can be undefined for a short or flat history. Each is
median-imputed using **validation-split medians only**, and each carries a
binary missing-indicator column. Imputation statistics are fitted once on
validation and frozen; they are never recomputed on test.

**E7 is the feature closest to the target and is therefore the one to watch.**
It is a mean absolute-error advantage, not a macro-F1 advantage, because
macro-F1 is undefined on the handful of observations available per participant
at the decision point. It is computed only from warm-up rows, which the frozen
protocol already designates as personalisation-only and never scores.

## 7. The gate

**Model family, fixed in advance:** L2-regularised logistic regression on the
standardised evidence vector, predicting `P(twin beats persistence for this
participant)`. Deliberately small — a large model would make any positive
result impossible to attribute to the evidence rather than to capacity.

**Fitting target on validation participants:** `y_p = 1` if
`macro_f1(twin, p) > macro_f1(persistence, p)` over participant *p*'s scored
observations, else 0.

**Fitting procedure, entirely within the validation split:**

1. Standardise E1–E7 using validation means and standard deviations.
2. Select the regularisation strength `C` from `{0.01, 0.03, 0.1, 0.3, 1, 3, 10}`
   by leave-one-participant-out cross-validation within validation, maximising
   cross-validated AUC. Ties broken toward the strongest regularisation.
3. Select the routing threshold τ by the same leave-one-participant-out scheme,
   maximising cross-validated routed macro-F1. Ties broken toward persistence,
   which is the conservative choice.

**Routing rule (primary):** route to the twin when `P̂ ≥ τ`, otherwise to
persistence.

**Readiness states (secondary, reported not primary):** two further thresholds
on the same score, also fitted on validation, produce
`SUPPORTED / UNCERTAIN / NOT SUPPORTED`. These exist so the decision can be
displayed with its reasons; they are not the primary estimand and no product
work is in scope here.

## 8. Routers, fixed before implementation

| ID | Router | Definition |
|---|---|---|
| **R0** | always persistence | the incumbent winner and the bar that matters |
| **R1** | always twin | the frozen study's proposed model, reproduced under this harness |
| **R2** | always global | `model_ctx` predictions for every participant |
| **R3** | random routing | Bernoulli personalisation at **R6's realised test personalisation rate**, 1,000 independent draws, mean with a participant-clustered CI |
| **R4** | **count-only gate** | identical model family, fitting procedure and threshold selection as R6, restricted to the single feature E1. A nested special case of R6, which is what makes the comparison clean. |
| **R5** | oracle routing | per participant, whichever of twin or persistence actually scored higher **on test outcomes**. A ceiling, never a result. It may not influence any feature, coefficient, threshold or reported conclusion, and is computed in a separate pass after everything else is frozen. |
| **R6** | **evidence-aware gate** | the proposed method, §7 |
| **R7** | R6 with abstention | issue no prediction when the readiness score is in the lowest band. **Run only if the pre-condition in §10.3 is met on validation**; otherwise not run, and that is reported. |

## 9. Metrics

**Primary metric: macro-F1 of the routed system**, computed per participant and
averaged over test participants, with a **participant-clustered bootstrap**,
2,000 resamples, seed 20260828. Unchanged from the frozen study, deliberately,
so the two are comparable.

**Primary comparison: R6 − R4**, paired per participant.

Secondary, all reported whether favourable or not:

| Metric | Definition |
|---|---|
| oracle routing headroom | `macro_f1(R5) − macro_f1(R0)`. Reported first; if it is near zero, routing has no room and that is the study's finding. |
| fraction of headroom recovered | `(R6 − R0) / (R5 − R0)`, with a bootstrap CI. Reported only when `R5 − R0 > 0`. |
| per-participant improvement | count and fraction of test participants improved, harmed and tied by R6 against R0 and against R4 |
| routing accuracy | fraction of test participants for which the gate selected the strategy that actually scored higher |
| readiness calibration | Spearman ρ between the gate score and the realised per-participant `macro_f1(twin) − macro_f1(persistence)`. Tests the mechanism independently of any threshold. |
| risk–coverage | for R7 only: error as a function of the fraction of predictions issued, against R3 abstaining at a matched rate |
| accuracy, MAE, QWK | reported for every router, as in the frozen study |

## 10. Success and failure criteria, declared before implementation

### 10.1 Primary failure criterion

> **If the evidence-aware gate (R6) does not outperform the count-only gate
> (R4) on macro-F1 with a paired participant-clustered 95% confidence interval
> excluding zero in R6's favour, the mechanism is reported as NOT SUPPORTED.**

No feature will be added, no threshold retuned, no metric changed and no
participant excluded in response to that outcome.

### 10.2 A second declaration, so a partial result cannot be oversold

Beating R4 establishes that the *evidence* carries information beyond
observation count. It does **not** establish a deployable system. Therefore:

- If R6 beats R4 but does **not** beat R0 (always persistence), the study must
  report that selective personalisation does not yield a deployable improvement
  on this cohort, in the same prominence as the R4 comparison.
- If R6 beats R0, the improvement must additionally exceed R3 (random routing
  at the same rate), or the gain is attributable to mixing rather than to
  selection.

### 10.3 Pre-condition for running R7

Abstention is run **only if** readiness calibration on the **validation** split
reaches Spearman ρ > 0.2 with p < 0.05. If it does not, R7 is not implemented,
and that decision and its reason are reported. This is declared now so that
abstention cannot be added later as a rescue.

### 10.4 Power, declared in advance

The minimum detectable paired difference at n = 42 test participants, α = 0.05,
power 0.8, is to be computed **from the validation split before the test set is
opened**, and reported alongside the result whether or not the result is
favourable. A null accompanied by a large minimum detectable effect is a weaker
claim than a null with a small one, and the write-up must say which it has.

## 11. Leakage controls

| # | Route | Control |
|---|---|---|
| 1 | Autocorrelation computed over a whole series | E3 is an **expanding-window** statistic on `shift(1)` data. A full-series r summarises the future. Tested against a fixture where the two differ. |
| 2 | Thresholds or coefficients fitted on test | All fitting is inside the validation split. The experiment asserts in code that no test participant id is reachable during fitting, and that assertion is itself tested against a deliberately contaminated call. |
| 3 | Oracle contamination | R5 is computed in a separate pass, after coefficients and thresholds are frozen and written to disk. |
| 4 | Participant selection on the outcome | Inclusion is by the frozen protocol's rules only (warm-up K, ≥ 5 scored observations). No rule may reference realised twin-vs-persistence performance. |
| 5 | Warm-up misuse | E7 uses warm-up rows only, which the frozen protocol designates personalisation-only and never scores. No scored observation's own outcome may enter the evidence vector that predicts it. |
| 6 | Multiple K values as multiple looks | K=20 is primary. K=80 is secondary and labelled. No other K is analysed. |
| 7 | Evidence-variable shopping | E1–E7 are closed by §6. Adding a variable after seeing test results is metric-changing by another name. |
| 8 | The test set is not naive | Not fixable. Declared in §4 and required in every write-up. |
| 9 | Standardisation and imputation statistics | Fitted on validation once and frozen; never recomputed on test. |
| 10 | Per-observation variant | At time *t* the gate may use outcomes strictly before *t* and nothing at or after *t*. Enforced by extending `assert_no_leakage` to the evidence columns. |

**Test the tests.** Every control above gets a fixture that must make it fire.
A check that has never failed proves nothing — that is already this project's
rule and it applies hardest to the newest code.

## 12. Reproducibility requirements

The frozen experiment recorded nothing about its environment, and that is why
its K=80 twin figure cannot be explained today. Every script written for this
study **must** call `aedt.reporting.metadata.make_run_metadata()` and write its
output beside the result, capturing:

- git commit
- Python version
- package versions — `package_versions()` will be extended to include
  **scikit-learn, joblib and threadpoolctl**, which it currently omits, plus the
  resolved OpenMP thread count
- dataset identifier and the SHA-256 digests from
  `data/raw/college-experience/PROVENANCE.json`
- the full experiment configuration, including K, seed, feature list, model
  family, selected `C` and every threshold
- the random seed (20260828 throughout)

Exact dependency versions are to be pinned in a new
`requirements-experiment.txt`. The existing `requirements.txt` is not changed,
because it is part of the frozen record.

## 13. What this study will not do

- It will not modify `scripts/run_twin_experiment.py`,
  `scripts/run_twin_ablation.py`, `aedt/twin/prediction_data.py`, the frozen
  pre-registration, the README or the blueprint.
- It will not re-run or re-score the frozen experiment as part of its own
  analysis; it consumes the regenerated result file as a fixed input.
- It will not add HGNN or EWC to the central model. Both remain documented
  negative results, and neither returns unless a real-data experiment
  demonstrates a benefit under its own pre-registration.
- It will not touch UI or product surfaces.
- It will not push, merge, or modify `main`.

## 14. Deviations

**2026-09-05 — documentation cleanup before step A1, at the owner's direction.**
`README.md` and the Research page were edited to distinguish the *originally
reported* results from the *regenerated* ones, and `package_versions()` was
extended. Appendix B previously listed `README.md` as untouched; that line is
corrected below. **No protocol element changed** — not the question, the
evidence vector, the routers, the metric, the thresholds, the splits or the
failure criterion. No number was altered: 0.2847 and 0.2793 are both preserved
with provenance.

## 15. Result

*(to be appended after the test set is opened, once, whatever it says)*

---

## Appendix A — implementation plan

Ordered so that the most informative and most falsifying steps come first.

| Step | What | Gate on proceeding |
|---|---|---|
| **A0** | Extend `package_versions()`; add `requirements-experiment.txt`; add the metadata wiring helper used by every new script. | — |
| **A1** | `aedt/twin/evidence.py` — E1–E7, expanding-window only, with the extended leakage assertion and its adversarial fixtures. | leakage tests fire on deliberately broken input |
| **A2** | **Validation-only feasibility read.** Oracle headroom, the distribution of per-participant twin-minus-persistence advantage, readiness calibration, and the §10.4 power calculation — all on the 43 validation participants. Test set untouched. | If oracle headroom on validation is ≈ 0, report that and stop: routing has no room, which is itself the answer to the research question. |
| **A3** | `aedt/twin/gate.py` — fit, select `C` and τ by leave-one-participant-out within validation, freeze coefficients and thresholds to disk with run metadata. | frozen artefact written before any test participant is read |
| **A4** | `scripts/run_gate_experiment.py` — R0–R6 at K=20, test opened **once**. | — |
| **A5** | R5 oracle pass, separately, after A4's artefacts are written. | — |
| **A6** | K=80 sensitivity; per-observation secondary analysis; ablation of the gate with each evidence term removed. | — |
| **A7** | R7 abstention — **only if** §10.3's pre-condition was met at A2. | — |
| **A8** | Append §15 with the result, favourable or not. | — |

## Appendix B — files

**Created**

| Path | Purpose |
|---|---|
| `docs/preregistration_selective_personalisation.md` | this document |
| `aedt/twin/evidence.py` | the evidence vector |
| `aedt/twin/gate.py` | fit, threshold, route, and emit reasons from the same vector the decision consumed |
| `scripts/run_gate_validation_readout.py` | step A2, validation only |
| `scripts/run_gate_experiment.py` | steps A4–A6 |
| `requirements-experiment.txt` | pinned exact versions |
| `tests/unit/test_evidence.py` | evidence definitions, planted answers, refusal cases |
| `tests/unit/test_gate.py` | fitting, thresholds, reason generation |
| `tests/regression/test_gate_leakage.py` | every control in §11, each with a fixture that must make it fire |
| `results/gate/*` | run artefacts plus `run_metadata.json` (gitignored, regenerable) |

**Modified**

| Path | Change |
|---|---|
| `aedt/reporting/metadata.py` | `package_versions()` extended with scikit-learn, joblib and threadpoolctl; `thread_environment()` added; optional `seed` override on `make_run_metadata()` — **done, step A0** |
| `aedt/reporting/experiment_record.py` | new: one `run_metadata.json` per experiment run — **done, step A0** |
| `requirements-experiment.txt` | new: exact pins — **done, step A0** |
| `README.md` | **labelling only** — reported vs regenerated results separated; no figure changed |
| `frontend/src/ui/discovered.js` | a Reproducibility panel showing both records |
| `frontend/src/data/historical_findings.json` | new: the originally reported figures, transcribed from git `38f8785` |
| `scripts/export_historical_baseline.py` | new: generates that file from git, so it is not hand-typed |
| `docs/statistic_provenance.md` | a section for the new study's artefacts |

**Frozen — not touched**

`scripts/run_twin_experiment.py` · `scripts/run_twin_ablation.py` ·
`aedt/twin/prediction_data.py` · `docs/preregistration_twin_prediction.md` ·
`docs/scientific_upgrade_blueprint.md` · `requirements.txt`

(`README.md` received a labelling-only edit; see §14. Its reported figures are
unchanged.)
