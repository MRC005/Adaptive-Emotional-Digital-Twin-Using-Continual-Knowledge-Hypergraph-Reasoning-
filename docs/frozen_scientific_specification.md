# Frozen Scientific Specification

**Adaptive Emotional Digital Twin Using Continual Knowledge Hypergraph Reasoning**

> **STATUS: FROZEN.** This document reproduces the scientific specification
> settled in the historical project record (`ROUND-17-FINAL-FREEZE-PACKAGE.md`
> §B–§H, with the mathematics also stated in `ROUND-14-ordinal-method.md` §8–§9
> and the Round-16 correction in `ROUND-16-real-data-audit-NOT-RUN.md` §3).
> Nothing here may be changed by implementation convenience. If real-data
> validation reveals a fatal scientific problem, that is a new research round,
> not an edit to this file.
>
> **Authority check.** The A1–A5 wording and the estimating equation are
> identical in ROUND-14 §8 and ROUND-17 §G. The "A1–A6" list in
> ROUND-13 §155 belongs to the **superseded affine-anchor method**, which
> ROUND-14 replaced; that is supersession, not conflict. **No conflicting
> frozen specification was found.**

---

## 1. Research question

Longitudinal self-report is the endpoint of nearly all digital mental-health
research, and every predictive pipeline in that literature treats it as a
stable measuring instrument. **It may not be.** If a person's five-point stress
scale means something different in week ten than in week one, an apparent
improvement may be **recalibration** rather than change — and nobody in the
passive-sensing literature has the machinery to tell the difference.

**The question:** *which components of longitudinal self-report scale change are
identifiable from passive sensing, and can they be estimated without bias from
the ordinal instruments mobile-health studies actually use?*

## 2. Research gap

> Response-shift methodology presumes a psychometric design — multiple
> indicators per wave, a retrospective then-test, or a pooled respondent panel.
> Passive-sensing research presumes the opposite, treating self-report as
> ground truth to be predicted. Neither has characterised the identifiability
> of reporting-function parameters when the reference is a passively sensed
> proxy of unknown fidelity, nor produced an estimator that survives the
> bounded ordinal response formats and serially dependent sensor streams that
> mobile-health data actually exhibit.

Claimed as a gap in **characterisation** (verifiable), never in **existence**
(not verifiable). Four search formulations returned nothing at the
intersection. **"Not found" is not "novel", and this document says so.**

## 3. Objectives

| # | Objective |
|---|---|
| **O1** | Formalise longitudinal reporting-scale change as a parameter-identification problem against a passively sensed proxy of unknown gain and offset. |
| **O2** | Determine which parameters are point-identified and which are provably not. |
| **O3** | Develop and validate an estimator that survives ordinal responses, floor effects and serial dependence. |
| **O4** | Build a persistent per-person representation that maintains the estimate, its uncertainty and its audit state over time. |
| **O5** | Establish the conditions under which real mobile-sensing data can support the inference — and honestly report when they cannot. |

## 4. Hypotheses

| # | Hypothesis | Status |
|---|---|---|
| **H1** | The multiplicative recalibration component is identifiable from a passively sensed proxy up to an explicit, sign-known scale factor. | Supported (analysis + simulation), SYNTHETIC |
| **H2** | The additive component is **not** identifiable under any specification examined. | Supported (Theorem T1(b)), SYNTHETIC |
| **H3** | The ordinal slope-ratio estimator is exactly calibrated under the null and conservative otherwise, including under floor effects and serial dependence. | Supported across 13 misspecification scenarios, SYNTHETIC |
| **H4** | Participant-cluster inference gives approximately nominal type-I error at realistic study density. | Supported (8.3% against a nominal 5%), SYNTHETIC |
| **H5** | The natural affine approach fabricates apparent scale compression on ordinal responses. | Supported (−0.107 null bias), SYNTHETIC |
| **H6** | Residual latent curvature beyond the threshold structure biases the null by no more than ≈0.07. | Supported, SYNTHETIC |
| **H7** | Real mobile-sensing data can support the inference at usable precision. | **PARTIALLY SUPPORTED.** Four archives audited. The College Experience Study (218 participants, 4.8 years) clears the unchanged screen and the pipeline ran on it. The pre-specified primary returns insufficient evidence (9 eligible against 10 required); two pre-specified secondaries return no detectable drift with wide intervals (rho* 0.913 [0.721, 1.180] and 1.199 [0.891, 1.589]). So real data can support the inference, but not yet at usable precision. RELAX (0/31) and PMData (0/14) are too sparse; StudentLife's original release is incompatible because its densest item is not ordered by severity. |

## 5. Mathematical model — FROZEN

**Latent construct** $\theta_{pt}$ for person $p$ at occasion $t$.

**Reporting (ordinal):**

$$r^*_{pt} = a_{e(t)}\theta_{pt} + b_{e(t)} + \varepsilon_{pt}, \qquad
R_{pt} = k \iff \tau^{(p)}_{k-1} < r^*_{pt} \le \tau^{(p)}_k$$

Thresholds $\tau^{(p)}$ are **person-specific and fixed across epochs**;
response shift is a change in $(a_e, b_e)$. Normalisation $a_1 = 1$, $b_1 = 0$.

**Sensor:**

$$s_{pt} = \lambda_p\theta_{pt} + \kappa_p + \eta_{pt},
\qquad \lambda_p, \kappa_p \text{ unknown}$$

with $\eta$ permitted to be serially dependent.

**Estimating equation.** For person $p$, epoch $e$, standardise the sensor
feature **within that epoch**,

$$x = \frac{s - \bar{s}_e}{\mathrm{SD}(s_e)}$$

and fit the ordinal probit by maximum likelihood:

$$P(R \le k \mid x) = \Phi\left(c_k^{(pe)} - \beta_{pe}\,x\right)$$

**Why the ratio is calibrated.** Algebra gives

$$\beta_e = \frac{a_e\lambda\sigma_\theta^2}
{\mathrm{SD}(s_e)\sqrt{a_e^2 v + \sigma_r^2}}, \qquad v = \operatorname{Var}(\theta\mid s)$$

so under A3,

$$\frac{\beta_2}{\beta_1} = \rho\sqrt{\frac{v + \sigma_r^2}{\rho^2 v + \sigma_r^2}}$$

*Implemented in* `aedt/models/ordinal.py` *and* `aedt/estimators/slope_ratio.py`.

## 6. Assumptions A1–A5 — FROZEN, quoted verbatim

> **A1** latent reporting affine in θ (threshold nonlinearity modelled; residual curvature not);
> **A2** thresholds fixed across epochs;
> **A3** λ, σ_θ, σ_p stable across epochs (**observable proxy: Var(s) epoch ratio**);
> **A4** sensor error stationary within epoch (dependence permitted);
> **A5** sufficient within-epoch spread in x and ≥ 2 categories used in both epochs.
>
> — ROUND-17 §G, identical to ROUND-14 §8

| Assumption | How it is checked on real data | If violated |
|---|---|---|
| **A1** affine latent reporting | linear vs quadratic sensed-level term in the epoch-1 fit (`audit/diagnostics.py::curvature_check`) | report the ≤ −0.07 bias envelope |
| **A2** fixed thresholds | **untestable directly; it IS the null** | stated limitation |
| **A3** stable λ, σ_θ, σ_p | Var(s) epoch ratio per person, required in [0.25, 4.0] | exclude the participant, by name |
| **A4** stationary sensor error | lag-1/2/7 autocorrelation within epoch | the participant bootstrap absorbs it |
| **A5** spread and category use | category-usage table; SD(x) per epoch | exclusion criterion |

## 7. Estimand — ρ\*

$$\boxed{\rho^\ast_p = \frac{\beta_{p2}}{\beta_{p1}}, \qquad
\widehat{\log\rho^\ast} = \frac{1}{P}\sum_p \log\rho^\ast_p}$$

with a nonparametric bootstrap over **participants** (2000 resamples).

**The primary estimand is ρ\*, not ρ.**

## 8. Identifiability — FROZEN

- **T1(a).** ρ is identified only up to the scale factor
  $\sqrt{(v+\sigma_r^2)/(\rho^2 v+\sigma_r^2)} \ge 1$ for ρ < 1. The estimate is
  therefore a **conservative bound**, equal to ρ exactly when $\sigma_r \to 0$
  or under the standard equal-residual-variance invariance assumption.
- **T1(b).** The additive component $b_2 - b_1$ is **not identified**: in the
  ordinal model it is absorbed into the threshold locations, and any location
  shift is indistinguishable from genuine change in θ.
- **T2.** **No specification examined identifies both.** The augmented-linear
  intercept is confounded with κ; the single-regressor intercept with
  $\mathbb{E}[\mu]$; the single-regressor slope with λ.
- **T3 (rank condition).** Within-epoch variation in x, and ≥ 2 categories used
  in both epochs. Strictly weaker than the anchor rank condition.
- **T4.** In the linear formulation the two regressors share an error term,
  making the measurement error non-classical and manufacturing false
  detections. **The ratio formulation avoids this entirely by never
  differencing.**

*Enforced in code:* `EstimatorResult` refuses `estimand != "rho_star"` and
refuses any non-`None` `additive_component`.

## 9. Interpretation

| Quantity | Status | What may be said |
|---|---|---|
| ρ\* | **IDENTIFIED** | "The sensor predicts the report ρ\* times as strongly in the second half of the study as in the first." |
| 1 − ρ\* | **LOWER BOUND** | "At least this much multiplicative recalibration occurred." |
| ρ | **NOT point-identified** | Never state "we estimated ρ = 0.94". |
| b₂ − b₁ (additive) | **NOT IDENTIFIED** | Never estimated, never reported. |

## 10. Primary endpoint

**95% participant-cluster bootstrap CI on $\log\rho^\ast$ excluding 0**
(equivalently, the CI on ρ\* excluding 1), with 2000 resamples.

The placebo runs **before** the primary and **halts it on failure** (exit code 5).

## 11. Statistical methodology

| Step | Method | Module |
|---|---|---|
| Epochs | halves of each participant's **own** enrolment span | `preprocess/epochs.py` |
| Standardisation | **within each epoch separately** — required, not optional | `estimators/slope_ratio.py` |
| Model | ordinal probit, ML, per person per epoch, fits **independent** | `models/ordinal.py` |
| Estimator | β₂/β₁, geometric mean over participants (mean of logs) | `estimators/slope_ratio.py` |
| Inference | nonparametric bootstrap over **participants**, 2000 resamples | `inference/bootstrap.py` |
| Negative control | contiguous epoch-1 split-half, **gates the primary** | `inference/placebo.py` |
| Sensitivity | bias envelope over 9 pre-enumerated assumption violations | `audit/envelope.py` |

## 12. Eligibility criteria — FIXED BEFORE ANY REAL DATA

| Criterion | Threshold |
|---|---|
| reports per epoch | ≥ 60 |
| categories used per epoch | ≥ 2 (A5) |
| standardised sensor SD | ≥ 0.10 |
| Var(s) epoch ratio | ∈ [0.25, 4.0] (A3 proxy) |
| slope determination | \|β\| ≥ 0.02, **either sign** |
| slope stability | sign(β₁) == sign(β₂) |

**Self-correction 26 (Round 16).** The earlier criterion required β > 0 in both
epochs. On a synthetic fixture this reduced the usable sample from 48
participants to 2, because conversation minutes *fall* as stress rises, so the
true slope is negative. That is not a crash but a **silent, severe selection on
the outcome**. The corrected criterion is \|β\| ≥ 0.02 either sign, plus a
matching-sign requirement.

This was a **specification error, not a threshold tuned to a result** — the
only data seen was a synthetic fixture generated by the project itself. The
numeric thresholds that *are* tuned to results (reports, categories, SD,
variance ratio) are untouched.

## 13. Limitations — stated, not buried

1. **No real-data validation.** One real dataset (RELAX) has now been opened
   and audited, and it **failed** the eligibility screen — so there is still no
   real-data estimate, and everything downstream remains contingent. The
   binding constraint is now known and measured: the method needs ≈120
   repeated ordinal self-reports per person, and RELAX provides a median of
   ~50 after causal alignment.
2. **ρ is not directly identified.** A CI for ρ\* is not a CI for ρ.
3. **Scale-change vs relation-change confound.** The method cannot distinguish
   "the person's reporting scale changed" from "the behaviour–construct
   relation changed". If conversation stops tracking stress because
   circumstances changed, β moves with no recalibration. **This is a genuine
   identification limit and belongs in the abstract, not a footnote.**
4. **Underpowered at 48 participants** for ρ = 0.85 (68.3% against a
   pre-committed 70%). Powered for ρ ≤ 0.80.
5. **A2 is untestable.** It *is* the null. If thresholds shift too, that shift
   is absorbed and not separately identified.
6. **Residual latent curvature** beyond the threshold structure biases the null
   by up to −0.068.
7. **Association strength is unknown on real data.** A perfectly calibrated
   estimator can still yield a useless interval when the per-person
   sensor–report slope is weak. **Calibration and usefulness are different
   properties.**
8. **Sign-flip participants are excluded.** If many flip, that itself indicates
   the sensor is not tracking the construct stably.

## 14. Pre-specified analysis order

Before any primary real-data result:

1. dataset audit → 2. stress-label verification → 3. timestamp verification →
4. **[9b] association strength** → 5. eligibility → 6. **placebo** →
7. primary ρ\* → 8. baselines → 9. hypergraph ablation → 10. bias envelope.

**A final primary real-data result may NOT be labelled validated if any
preceding check fails.** Enforced by `PipelineResult.validated`.

## 15. Pre-specified sensitivity analyses

These, and only these, may vary a frozen threshold. Each must be reported
alongside the primary, never in place of it.

| # | What varies | Range | Rationale |
|---|---|---|---|
| S1 | `MIN_REPORTS_PER_EPOCH` | 40, 60, 100 | precision vs sample size |
| S2 | `MIN_ABS_BETA` | 0.01, 0.02, 0.05 | slope-determination strictness |
| S3 | Var(s) ratio window | [0.5, 2.0] and [0.25, 4.0] | A3 strictness |
| S4 | epoch rule | own-span halves (frozen) vs observation halves | A2/A3 robustness |
| S5 | sensor feature | primary vs pre-specified PC1 fallback | triggered by [9b] |
| S6 | link | probit (frozen) vs logit | link misspecification |

Running one requires a config override, which
`Config.deviations_from_frozen()` detects and writes to
`config_deviations_from_frozen.json` in the run folder. **A deviation cannot be
made silently.**

## 16. What is frozen, and what a change would require

| Frozen | Change requires |
|---|---|
| Official title | not changeable |
| Architecture (§I) | a new research round |
| Ordinal probit slope-ratio method | a new research round |
| ρ\* as the primary estimand | a new research round |
| Hypergraph as representation + ablation, **not identification** | a new research round |
| Eligibility thresholds | a pre-specified sensitivity analysis (§15) |
| Primary endpoint | not changeable |
| Real-data validation as **pending** | actual audited files |
