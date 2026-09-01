# Pre-registration — personalised future-affect prediction

**Frozen before any model was fitted or any result inspected.** Committed ahead
of the experiment code so that the protocol cannot be adjusted after seeing
results. Any later change must be recorded in §10 with its date and reason.

## 1. Hypothesis

> A personalised, continually updated twin predicts an individual's next
> reported stress state better than strong non-personalised **and** simpler
> personalised baselines, on held-out participants and strictly future
> observations.

**Null:** the twin does not beat persistence by a practically meaningful margin
on held-out participants.

## 2. Data

College Experience Study, `data/raw/college-experience/`.
Verified from the files: 218 participants, 35,348 non-null `stress` reports,
daily granularity, **max one report per (uid, day) — no ties**, span
2017-09-07 to 2022-07-02. Sensing: 651 columns keyed `(uid, day, is_ios)`.

## 3. Target

`stress`, ordinal 1–5, from `EMA/general_ema.csv`.
Codebook (published data dictionary): *1 Not at All · 2 A Little Bit ·
3 Somewhat · 4 Very Much · 5 Extremely* — ascending, direction confirmed.

## 4. Prediction task

For participant *p* with a report at day *d*, predict the stress value at their
**next report** *d'*, subject to `0 < (d' − d) ≤ 7 days`.
Pairs with a gap > 7 days are **excluded**, declared here in advance.

Timestamps are calendar days at midnight UTC. Since no participant has two
reports on the same day, the ordering is total and tie-breaking is not required.

**Temporal invariant**, enforced in code and tested:
`feature_time ≤ prediction_time < target_time ≤ prediction_time + 7 days`

## 5. Features (all strictly prior to `prediction_time`)

- **History:** last 1/2/3 stress values, EWMA (α=0.5), running mean and SD,
  count of prior observations, days since last report.
- **Behaviour:** previous-day sensing — `unlock_duration_ep_0`,
  `loc_dist_ep_0`, `act_still_ep_0`, `act_walking_ep_0`,
  `audio_convo_duration_ep_0`, `sleep_duration` where present — plus trailing
  7-day means of the same.
- **Context:** day of week, gap to target.
- **Never used:** participant identity, any value at or after `target_time`,
  any aggregate computed over the evaluation period.

## 6. Splits

Participants partitioned **60/20/20** by a seeded shuffle (seed 20260828) into
population-train / validation / test. Intersections must be empty.

For each **test** participant the first *K* observations are warm-up
(personalisation only, never scored); all later observations are evaluated.
Validation participants are used only for model selection.

## 7. Baselines, fixed before the twin is implemented

| | Baseline |
|---|---|
| B0 | Training-population majority class |
| B1 | **Persistence** — the participant's most recent observed stress |
| B2 | Global gradient-boosted model on context + behaviour, no personal history |
| B3 | B2 + static personal prior (warm-up mean/SD only, not updated after) |
| B4 | Per-person calibrated global model — global prediction shifted by the participant's running residual (a mixed-effects-style random intercept, updated online) |
| **T** | **Proposed twin** — evolving state, dynamic history, behaviour, uncertainty |

## 8. Metrics

**Primary: macro-F1** (classes are imbalanced ≈ 4:1), declared before results.
Secondary, all reported whether favourable or not: exact accuracy, MAE,
quadratic weighted kappa, per-person macro-F1 distribution.

Uncertainty: **participant-clustered bootstrap, 2,000 resamples, 95% CI**.
Participants are the resampling unit; observations are never resampled as if
independent.

## 9. Success criteria — all must hold

1. Twin beats **B1 persistence** on the primary metric.
2. The 95% CI for the paired difference **excludes 0**.
3. **> 50%** of test participants improve.
4. Performance improves with personal history across K ∈ {0, 5, 10, 20, 40, 80}.
5. Twin beats **B4**, the strong personalised baseline.

If any fails, the hypothesis is reported **NOT SUPPORTED** or **PARTIALLY
SUPPORTED**. No dataset, horizon, threshold or split will be changed in
response to a null result.

## 10. Deviations

*(none — to be appended with date and reason if any occur)*
