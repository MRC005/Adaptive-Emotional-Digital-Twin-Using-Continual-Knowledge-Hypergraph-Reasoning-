# Scientific Upgrade Blueprint

Written before any code change, from repository inspection and measurement. Every
number here was produced by running against the files on disk; nothing is taken
from documentation or dataset marketing.

**Target claim.** *A personalised, continually updated model predicts an
individual's future emotional state better than strong non-personalised and
simpler personalised baselines, on genuinely unseen future observations, with
leakage-free evaluation and quantified uncertainty.*

**Headline conclusion.** That claim is testable **with data already in this
repository**. The previous audit concluded a new data-collection study was
required; that conclusion was wrong in one specific way, and the correction is
the most important content in this document. See §3.

---

## 1. Verified current state

Each finding below was checked against code, not documentation.

| Prior audit finding | Verdict | Evidence |
|---|---|---|
| No dataset supports person + time + context + **text** + repeated affect | **CONFIRMED** | Schema inspection: College Experience / StudentLife / RELAX / PMData have no free text; GoEmotions has no person ID and no timestamps |
| No evaluation of the core twin claim | **CONFIRMED** | 16 scripts in `scripts/`; none evaluates prediction, retrieval or personalisation |
| Participant leakage in Layer 1 experiments | **CONFIRMED** | `split_edges()` stratifies by class, not person. Measured: 60/60 people in both train and test; 0 unseen. `Person=` vertex is a model input |
| HGNN underperforms a structure-free baseline | **CONFIRMED** | `results/hgnn/hgnn_experiment.json`: HGNN 0.512, GCN 0.576, MLP 0.815 macro-F1 |
| EWC shown only on synthetic data | **CONFIRMED** | `results/ewc/ewc_experiment.json` protocol block states synthetic |
| Layer 2 is more rigorous than Layer 1 | **CONFIRMED** | Layer 2 has pre-specification, placebo gate, participant-clustered bootstrap; Layer 1 has none of these |
| The GoEmotions→project label mapping may be flawed | **CONFIRMED, and worse than stated** | See §2 |

**Correction to the prior audit.** The prior audit's conclusion — "additional
real-world data is essential" — conflated two different requirements. Free text
is required for the *emotion-recognition module*. It is **not** required for the
*future-affect prediction* task that defines the Digital Twin. Decoupling those
two is what makes the central claim testable now.

### What is real vs synthetic

| | Real | Synthetic |
|---|---|---|
| GoEmotions evaluation (macro-F1 0.4925, n=5,427) | ✓ | |
| ONNX↔torch↔browser agreement | ✓ | |
| Layer 2 drift analysis on College Experience | ✓ | |
| Four dataset compatibility audits | ✓ | |
| HGNN comparison | | ✓ |
| EWC comparison | | ✓ |
| Twin retrieval / personalisation | | neither — **no experiment exists** |

---

## 2. Root causes of the four reported failures

All four were reproduced against the live pipeline. Three distinct defects, plus
one that is worse than reported.

### D1 — A keyword regex silently overrides the model

`aedt/emotion/pipeline.py` gives `self_reported_emotion()` precedence over the
Transformer. It has no negation handling, no scope handling and no temporal
handling. Measured:

| Input | Regex output | Correct reading |
|---|---|---|
| "I am not sure I am **good** enough" | **joy** | self-doubt |
| "...which I am grateful for, but I am anxious" | **gratitude** | anxiety |
| "Yesterday I was anxious, but now I feel relieved" | *(none)* | relief |
| "I am not happy about this" | *(none)* | negative |

The second row is the reported Case 1 failure. The first is a **worse failure
found during this audit**: a sentence expressing self-doubt is labelled *joy*.

The rule was introduced for a defensible reason — GoEmotions has no `stress`
class — but it was given the wrong precedence and the wrong scope. It is applied
to a whole multi-episode narrative and fires on any matching substring anywhere
in it.

### D2 — The event lexicon collapses distinct events

`aedt/emotion/context.py:50` lists `"interview"` and `"demo"` as trigger phrases
for the category `presentation`. So *interview* is displayed as *presentation*,
with the evidence span still reading "interview" — reproduced exactly. This is
aggressive ontology normalisation with no mapping shown and no confidence.

### D3 — An indefensible label mapping

`aedt/emotion/detect.py:87` maps `realization → confusion`. **Realisation is not
confusion; it is arguably its opposite.** It was chosen to give all 28 GoEmotions
labels a target, which is a coverage motive, not a semantic one. `curiosity →
confusion` and `surprise → confusion` are questionable for the same reason.
`nervousness → anxiety` (Case 3) is defensible and should be kept.

### D4 — Single-label reduction of multi-episode narratives, out of distribution

The pipeline reduces an entire narrative spanning several episodes and several
time frames to one emotion. Two measurements make this untenable:

- **Length.** GoEmotions training text is median **18** tokens, p95 **33**, with
  **0%** over 64 tokens. The reported narratives are **57–60** tokens — roughly
  3× the median and beyond the p95 of anything the model was trained on. Note
  this is *not* truncation (the 128-token limit is not reached); it is a
  distribution shift in input length and structure.
- **Task mismatch.** A narrative containing anxiety, relief, gratitude and
  frustration has no single correct label. Forcing one is a specification error,
  not a model error.

---

## 3. Task separation

The application currently conflates three tasks. Every module is assigned below.

| Task | Question | Currently performed by | Status |
|---|---|---|---|
| **T1 Expressed emotion** | which emotions appear in this text? | `onnx_detect.predict()` | multi-label, works, evaluated on GoEmotions |
| **T2 Current state** | how does the person feel *now*? | conflated into T1, plus the regex override | **not properly implemented** |
| **T3 Future state** | what will they feel at t+h? | *nothing* | **not implemented, and this is the Digital Twin claim** |

T1 is not a twin. T2 is not a twin. **Only T3 is.** The project currently ships
T1 and calls the result a Digital Twin.

---

## 4. Twin state definition

State is carried forward chronologically and may only contain information
available strictly before the prediction time.

```
S_t = (
  H_t     recent affect trajectory: last k reports, their times, EWMA level and slope
  C_t     recurring context: per-person rates over a trailing window
  B_t     behavioural summary: prior-day and trailing-7-day sensing aggregates
  P_t     person-level priors: mean/SD of affect estimated on history so far only
  U_t     uncertainty: n observations so far, time since last report, coverage
)
S_{t+1} = update(S_t, o_t)      # o_t observed at time t
ŷ_{t+h} = predict(S_t, C_{t+h}^known)
```

Justification, component by component — nothing is included because it sounds
impressive:

- **H_t** — persistence is the strongest simple baseline (44.7% exact-match,
  measured). A twin that cannot use trajectory cannot beat it.
- **C_t / B_t** — the only inputs that distinguish a twin from a persistence
  model; without them "personalisation" is just an intercept.
- **P_t** — separates *this person runs high on stress* from *this person is
  currently stressed*. Must be estimated on history only.
- **U_t** — required for the honest-abstention behaviour the product already
  implements, and for the calibration analysis in §8.

**Deliberately excluded:** the hypergraph and EWC. Neither enters the state
unless §11 and §12 show a measured benefit.

---

## 5. Dataset eligibility

Checklist applied: stable IDs · repeated observations · timestamps · repeated
affect · pre-prediction context · sufficient density · codebook · license ·
feasible horizon · sample size for uncertainty.

| Dataset | Access | N | Duration | ID | Time | Affect target | Context | Text | Sensing | License | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **College Experience** | Kaggle, verified on disk | **218** | **4.8 yr** | ✓ | ✓ | stress 1–5, social 1–5, PHQ-4, PAM | 648 daily features | ✗ | ✓ | CC-BY, research | **A** |
| StudentLife | on disk | 48 | 9.5 wk | ✓ | ✓ | stress (non-monotone coding) | conversation, activity | ✗ | ✓ | research | **D** |
| DAPPER | Synapse `10.7303/syn22418021`, CC-BY-4.0 | 142 | **5 days** | ✓ | ✓ | PANAS-10, valence, arousal 1–5 | — | **✗ removed for re-identification** | HR, GSR, accel | CC-BY-4.0 | **B** |
| RELAX | on disk | 31 | 6 wk | ✓ | ✓ | 7-pt Likert | limited | ✗ | ✓ | CC-BY-4.0 | **D** |
| PMData | on disk | 16 | 5 mo | ✓ | ✓ | ordinal 1–5 | Fitbit | ✗ | ✓ | CC-BY-4.0 | **D** |
| GoEmotions | HF, verified | — | — | ✗ | ✗ | 28 multi-label | — | ✓ | — | open | **C** |

Classes: **A** end-to-end twin evaluation ready · **B** supports
behavioural/contextual prediction · **C** emotion-model evaluation only ·
**D** Layer 2 drift analysis only · **E** unsuitable.

**Verified DAPPER specification** (from the Scientific Data paper, not the
description): 142 participants, 5 consecutive days, 6 prompts/day, 3,789 ESM
events, 26.7 ± 3.5 per participant. Free-text DRM descriptions were **excluded
from the public release** to prevent re-identification. Five days is too short
for within-person adaptation, and no text means it cannot serve Layer 1 either.
It is a good **replication** set for the behaviour→affect link, nothing more.

---

## 6. Recommended data strategy — **Strategy A**

**One dataset already on disk supports the complete central study: College
Experience.** No new collection is required for the target claim.

Measured feasibility:

| Property | Measured |
|---|---|
| Participants with stress reports | **218** |
| Stress observations | **35,348** |
| Prediction pairs (report → next report) | **35,130** |
| Pairs with prior-day sensing available | **35,054 (99%)** |
| Daily sensing features, all strictly prior | **648** |
| Participants with ≥50 pairs within 7 days | **180** |
| Participant-disjoint split feasible | train 130 / val 44 / test 44, intersection **0** |
| Test-participant observations | median **178** → warm-up K=20 leaves ~158 eval points |
| Target distribution (1–5) | 6671 / 11967 / 10119 / 4829 / 1762 — not degenerate |
| Median within-person SD | **0.95** — there is real within-person variation |
| **Persistence baseline (next == current)** | **44.7%** — the number to beat |

A prospective study is **not** recommended for the central claim. It becomes
worth doing only if the project wants the *text* channel validated (§13).

---

## 7. Unified schema

```
participant_id        stable, opaque
t_predict             time at which the prediction is made
t_target              time of the outcome;  horizon h = t_target - t_predict
features_prior        sensing + context aggregated over (-inf, t_predict]
history_prior         affect trajectory over (-inf, t_predict]
target                affect at t_target
source                dataset identifier
provenance            extracted | user_reported | model | inferred | corrected
model_prediction      retained even when the user corrects it
confirmed_label       user-confirmed value, stored separately
```

Two invariants, enforced in code, not by convention:

1. every column in `features_prior` / `history_prior` has timestamp ≤ `t_predict`;
2. `intersection(train_persons, test_persons) == ∅` for unseen-person evaluation.

---

## 8. Formal prediction task

> **Given** everything observable about participant *p* strictly before time *t*,
> **predict** their self-reported stress at the next reporting occasion within a
> 7-day horizon.

- **Primary target:** next stress report, 5-point ordinal.
- **Primary metric:** macro-F1 (the classes are imbalanced 4:1).
- **Secondary:** MAE and QWK (the target is ordinal, so distance matters),
  balanced accuracy, per-person macro-F1.
- **Horizon:** next occasion, gap ≤ 7 days (74% of pairs; median gap 5 days).
- **Excluded:** pairs with gap > 7 days, declared before running.

---

## 9. Leakage-free protocol

**Split 1 — unseen-person generalisation (primary).**
Participants partitioned 60/20/20. For each test participant the first
**K = 20** observations are warm-up (adaptation only, never scored); the
remainder are evaluated. Person identity is never a feature.

**Split 2 — within-person forward prediction (secondary).**
Per participant, chronological: first 60% adapt, last 40% evaluate. No shuffling.

**Executable invariants** — a test must fail if any is violated:

```
assert set(train_persons) & set(test_persons) == set()
assert all(f.timestamp <= row.t_predict for f in row.features)
assert row.t_target > row.t_predict
assert "participant_id" not in feature_columns
assert warmup_indices ∩ eval_indices == ∅
```

---

## 10. Baselines — defined **before** the proposed model is run

| # | Baseline | Population | Personal history | Updated |
|---|---|---|---|---|
| 0 | Global majority class | ✓ | ✗ | ✗ |
| 1 | Population model on context only | ✓ | ✗ | ✗ |
| 2 | **Persistence** — last observed value | ✗ | ✓ | trivially |
| 3 | Personal running mean / EWMA | ✗ | ✓ | ✓ |
| 4 | Gradient-boosted trees on prior features + history | ✓ | ✓ | ✗ |
| 5 | **Proposed twin** — state + context + updating + uncertainty | ✓ | ✓ | ✓ |

Baseline 2 at **44.7%** is the honest bar. A twin that does not beat persistence
is not a twin, and this must be stated before results are seen.

---

## 11. Metrics and uncertainty

- Participant-clustered bootstrap, 2,000 resamples, **95% CI on every number**.
- **Per-participant** results reported as a distribution, not only a pooled mean:
  median, IQR, and the count of participants who improve vs worsen.
- Paired per-participant differences with a CI and an effect size.
- Calibration: reliability curve and Brier score. If probabilities are not
  calibrated, say so and report ranks instead of implying probabilities.

**Rule.** A model that wins on the pooled mean while harming most individuals is
reported as a failure of personalisation, not a success.

---

## 12. Ablations

| Question | Comparison |
|---|---|
| Does personal history help? | 5 vs 1 |
| Does it beat *simple* personalisation? | 5 vs 2 and 3 |
| Does continual updating help? | 5 vs 5-frozen |
| Does the hypergraph help? | 5 vs 5-without-hypergraph |
| Does EWC help? | continual updating with vs without EWC |
| How much history is needed? | performance vs K (learning curve) |

The learning curve is the most informative single result: a genuine twin should
**improve with more personal history**. If it does not, the word is unearned.

---

## 13. Emotion pipeline repairs

Ordered by severity. None of these require new data.

1. **Remove the regex override from the decision path.** Keep it as a *detected
   self-statement* signal shown beside the model output; never let it silently
   replace the model. Add negation, scope and clause detection before it is
   allowed to influence anything.
2. **Split `interview` out of `presentation`.** Show *extracted phrase →
   normalised category → mapping confidence* in the interface. Never display a
   category whose evidence span contradicts it.
3. **Delete `realization → confusion`.** Re-derive the whole mapping table with a
   written justification per row; leave labels unmapped rather than forcing
   coverage. Publish the table.
4. **Segment before classifying.** Sentence- or episode-level classification with
   recency weighting for the current-state estimate, chosen on a held-out
   development set — not on four hand-picked narratives.
5. **Separate T1 from T2.** Report expressed emotions as multi-label; report
   current state separately with its own temporal logic.
6. **Never overwrite a model prediction with a user correction.** Store both;
   the disagreement is the error signal that makes evaluation and continual
   learning possible.
7. **Stop calling sigmoid scores percentages** anywhere they remain.

---

## 14. Continual learning and HGNN decisions

Both are demoted to open questions, to be answered on real data:

- **EWC** — run the three-period protocol on College Experience with the split of
  §9. If forgetting is not observed, or EWC does not reduce it, remove EWC from
  the contribution and report the negative result.
- **HGNN** — rerun with participant-disjoint splits and an identical
  hyperparameter budget. It has already lost twice. If it loses a third time on
  real data, remove it from the title and keep the hypergraph only if §12 shows
  it earns its place for retrieval or interpretability.

Neither may appear in the central claim without a real-data result.

---

## 15. Publication roadmap

| Work | Impact | Difficulty | Time | Publication value |
|---|---|---|---|---|
| Prediction task + baselines 0–4 | **critical** | low | 1 wk | unlocks everything |
| Leakage-free splits + invariants | **critical** | low | 2 d | removes a fatal criticism |
| Twin model + ablations | **critical** | medium | 2 wk | the contribution |
| Uncertainty + per-person analysis | high | low | 3 d | credibility |
| Emotion pipeline repairs | high | medium | 1 wk | removes embarrassing failures |
| Retrieval evaluation | medium | medium | 1 wk | supports a secondary claim |
| EWC on real data | medium | medium | 1 wk | keep or kill |
| HGNN rerun | low | low | 3 d | almost certainly kill |

### Minimum viable publishable study

College Experience, next-stress prediction, participant-disjoint splits,
baselines 0–4 vs the twin, participant-clustered CIs, per-person distribution,
learning curve over K. **Roughly 4–5 weeks.** This is publishable whichever way
the result falls — a rigorous null against persistence is a useful finding in a
field that rarely reports one.

### Stronger follow-up

Add DAPPER as independent replication of the behaviour→affect link; add the
Layer 2 drift result as a trust qualifier; add the emotion channel only if a
text corpus is obtained. Roughly 6–9 months.

---

## 16. What must be removed or downgraded

- "Adaptive Emotional Digital Twin" as a **validated** description — until §8
  produces a result.
- HGNN and "hypergraph reasoning" from the title and central claim.
- EWC from the contribution list until §14.
- Any implication that the deployed twin adapts model parameters. It does not.
- The claim that retrieval constitutes reasoning.

**Retained and strengthened:** the Layer 2 measurement-comparability work, the
four dataset audits, the provenance model, and the honest-abstention behaviour.
These are real and currently undersold.
