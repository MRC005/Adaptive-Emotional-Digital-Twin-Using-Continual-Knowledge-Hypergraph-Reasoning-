# Dataset Compatibility Assessment

**Question:** which public dataset can support the FROZEN ordinal probit
slope-ratio method and its estimand ρ\*?

Every row below was checked against **the actual released files or the official
record**, not against a description. Where a dataset was not opened, the row
says so.

---

## What the frozen method actually requires

Not negotiable, because these are what identify ρ\*:

| # | Requirement | Why it is load-bearing |
|---|---|---|
| R1 | **Ordinal** self-report with a small number of ordered categories | The estimator models thresholds. A 0–100 slider is not this. |
| R2 | **Repeated** measures: ≥60 per participant **per epoch** (≥120 total) | Frozen eligibility screen; a per-person ordinal fit needs K−1 cutpoints + slope |
| R3 | A **multi-week** span that can be halved into two epochs | ρ\* is a cross-epoch ratio |
| R4 | Participant IDs and reliable **time ordering** | Epochs, causal alignment, placebo |
| R5 | A **passively sensed** covariate aligned causally to each report | The sensor is the external reference |
| R6 | ≥10 eligible participants | Participant-cluster bootstrap |
| R7 | Public legal access + documentation sufficient for the strict audit | Reproducibility |

---

## Compatibility matrix

| | **RELAX** | **StudentLife** | **IFH Affect** | **PMData** | **WESAD** | **ADARP** |
|---|---|---|---|---|---|---|
| **Source** | Zenodo `10.5281/zenodo.20701999` (Sci Data 2026) | Dartmouth study page | Dryad `10.7280/D1WH6T` | Simula | UniPassau | Zenodo `6640290` |
| **Licence** | CC-BY-4.0 | Research release | Dryad | CC BY 4.0 | Research release | CC-BY-4.0 |
| **Participants** | **31** | 48 | 21 | 16 | 15 | ~11 |
| **Duration** | **~9 weeks** (4 phases, 2024-02-25→04-28) | 10 weeks | **~7.8 months** avg | ~5 months | **single lab session** | ≤14 days |
| **Repeated longitudinal obs** | ✅ several/day | ✅ ~5–6/day | ✅ daily + weekly | ✅ 1/day | ❌ none | ✅ 4/day |
| **Self-report format** | ✅ **7-point Likert** (ordinal) | ✅ 5-point ordinal EMA | ❌ **0–100 sliders** (continuous) | ⚠ bare integer, direction undocumented | ❌ protocol condition label | ⚠ EMA, format unverified |
| **Stress/emotion labels** | tension, mental effort, expected stress, overwhelmed | single-item stress | PANAS-20, COVID/health worry | PMSys `stress` | baseline/stress/amusement/meditation | stress, craving, emotion |
| **Physiological modality** | ✅ IBI + 52 Hz ACC (Polar) | phone sensing (conversation, GPS, activity) | ✅ PPG, HR/HRV, sleep, IMU (Galaxy Watch + Oura) | ✅ Fitbit resting HR | ✅ ECG/EDA/EMG/RESP/TEMP | ✅ Empatica E4 |
| **Timestamps** | ✅ **tz-aware UTC**, cross-checked | ✅ unix epoch | ✅ submission timestamps | ✅ date | ✅ sample index | ✅ |
| **Participant IDs** | ✅ 12–63 | ✅ u00… | ✅ par_1–21 | ✅ p01… | ✅ S2–S17 | ✅ |
| **R1 ordinal** | ✅ | ✅ | ❌ | ⚠ | ❌ | ⚠ |
| **R2 ≥120 reports/pp** | ❌ **max 106, median 71** | ✅ ~735 | ⚠ plausible, not verified | ❌ **4/14 only; median 95.5** | ❌ | ❌ ~56 max |
| **R3 multi-week** | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠ |
| **R5 causal sensor** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Acquisition** | ✅ **open, obtained** (partial, 0.5 of 16.5 GB) | ❌ **host unreachable** (timeout on all URLs) | ⚠ 8.59 GB, not obtained | ✅ **obtained & audited** (0.3 MB of 1.4 GB) | ⚠ not obtained | ⚠ 1.5 GB, not obtained |
| **Frozen-method risk** | **LOW** — correct format, fails only on density | **LOW** — the specified target | **HIGH** — would require inventing a discretisation | **MEDIUM** — scale direction undocumented | **FATAL** — cannot identify ρ\* | **HIGH** — too short |
| **Verdict** | ⚠ **audited; fails R2** | ⛔ **inaccessible** | ❌ wrong response format | ⚠ **audited; fails R2 + A3 + direction** | ✅ benchmark only | ❌ too short |

---

## Ranking

**1. RELAX — SELECTED, and audited.** The only candidate that is (a) publicly
downloadable, (b) genuinely longitudinal, (c) ordinal in the right way, and
(d) paired with continuous physiology. Its files were obtained and its schema
verified. **It fails only on density (R2)** — and that failure is a measured
result, not a guess.

**2. StudentLife** — remains the best *specified* target and the frozen primary.
It is simply **not reachable from this environment**: `https://studentlife.cs.dartmouth.edu`
times out (`http=000`) on every URL tried, consistent with the project's
historical record across four rounds. Nothing about the method changes.

**3. IFH Affect** — the strongest longitudinal density found (21 participants,
~7.8 months, daily EMA). **Rejected on response format:** PANAS items are
0–100 sliders. Applying an ordinal threshold model to a continuous slider would
mean inventing a discretisation the participant never used — which is precisely
"modifying the frozen method to make a dataset fit". Worth revisiting only if
the method were extended to continuous responses, which is a new research
round, not an implementation choice.

**4. PMData — ACQUIRED AND AUDITED; fails on three independent grounds.**
(a) only 4 of 14 participants reach 120 matched reports and **0 pass the
screen**; (b) three fail assumption **A3** with Var(s) epoch ratios up to
**13.6**, because Fitbit resting HR is an algorithmic daily estimate whose
variance moves for device reasons; (c) `stress` ships as a bare integer with
**no label text and no codebook in the archive**, so the severity direction
cannot be verified at all. Even the densest participants fail on A3 rather than
on count, so more data would not help.

**5. WESAD** — **benchmark only.** Enforced in code
(`can_support_longitudinal_estimand = False`). A single lab session has no
epochs, so it cannot identify ρ\* at all.

**6. ADARP** — EMA 4×/day for ≤14 days gives ~56 observations maximum, less
than half of R2, in a clinical AUD population.

---

## The finding, stated plainly

**No public dataset examined satisfies every requirement of the frozen
specification. Two were obtained and audited; both failed.**

RELAX comes closest and fails on one axis: **self-report density**. It has the
right instrument type, the right modality, the right span and the right
documentation — but the densest repeated item yields a **median of 71** and a
**maximum of 106** observations per participant, against the 120 the frozen
screen requires.

This is not a defect of RELAX. It reflects a real tension in the field: studies
dense enough for a per-person ordinal fit (StudentLife's ~735/participant) are
rare, and the recent generation of well-documented open datasets samples less
aggressively to reduce participant burden.

**It is also a concrete, citable result for the paper:** the method needs a
report density that most current open longitudinal datasets do not provide, and
the eligibility screen now has a real number attached to that claim.
