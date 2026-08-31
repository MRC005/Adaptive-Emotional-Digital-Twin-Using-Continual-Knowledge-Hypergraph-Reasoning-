# Panel-facing explanation

Three depths of the same honest answer. Rehearse the 20-second version until it
is comfortable; it is the one that gets used.

---

## The 20-second explanation

> "Self-reports are the yardstick in digital mental health — but the yardstick
> can change without anyone noticing. If your five-point stress scale means
> something different in week ten than in week one, an apparent improvement
> might just be recalibration. We ask whether phone and wearable data can
> detect that, and we build the estimator that measures it."

## The 1-minute explanation

> "Almost every study in digital mental health predicts a self-reported score
> and treats that score as ground truth. But a person's internal scale can
> drift. After ten weeks of a stressful term, 'a little stressed' may mean
> something different than it did in week one. If that happens, an apparent
> improvement is measurement drift, not change — and every effect estimate in
> that literature is wrong by an unknown amount in an unknown direction.
>
> Passive sensing gives us something psychometrics never had: a continuous
> reference that doesn't ask the person anything. We fit each person's
> sensor-to-report relationship separately in the first and second halves of
> their study, and take the ratio. If that ratio is 1, the scale hasn't moved.
>
> We proved which part of the drift is recoverable and which provably isn't. We
> report only the part we can defend. And before we report anything, a built-in
> placebo test can veto our own headline result."

## The technical explanation

We formalise recalibration as a change in the affine parameters of a
thresholded reporting function. For person *p* in epoch *e*:

- latent report `r* = a_e·θ + b_e + ε`, observed as `R = k` iff
  `τ_{k−1} < r* ≤ τ_k`, with thresholds **fixed across epochs**;
- sensor `s = λ_p·θ + κ_p + η`, with λ and κ **unknown**.

We standardise the sensor **within each epoch**, fit an ordinal probit
`P(R ≤ k | x) = Φ(c_k − β_e·x)` by maximum likelihood per person per epoch, and
take `ρ* = β₂/β₁`, pooling as a mean of logs with a bootstrap over
**participants**.

The algebra gives `β₂/β₁ = ρ·√((v+σ_r²)/(ρ²v+σ_r²))`. Three consequences:
under the null it is **exactly 1**; under the alternative it is **attenuated
toward 1**, never inflated; and errors-in-variables attenuation together with
any epoch-invariant link misspecification appear in **both** slopes and cancel.
The unknown sensor gain λ cancels in the ratio — that is the whole trick.

---

## The questions, and the answers

### What problem are we solving?
Response shift — specifically **recalibration** — in longitudinal self-report,
detected against a passively sensed reference. Every passive-sensing pipeline
assumes the self-report scale is stable. Nobody has checked.

### What is the Digital Twin?
A **persistent, per-person, individually parameterised model of that person's
measuring instrument** — their thresholds, their sensor→report slope, their
category-usage history, their audit flags, and their recalibration history with
uncertainty. **Not** a model of their mood.

Persistence is **load-bearing, not decorative**: the estimand is a ratio across
epochs, so without a persisted epoch-1 state there is nothing to compare
epoch 2 against.

**Say this explicitly: the term is descriptive. We claim no novelty for it.**

### What does continual knowledge mean?
Exactly four things update, and nothing else: (1) personalised parameters —
thresholds and slope; (2) longitudinal state history; (3) context
relationships — hyperedge occupancy and effects; (4) uncertainty and audit
state. The store is **append-only** with provenance, so you can ask what the
twin knew at any past time.

**No continual-learning algorithm runs in the knowledge store.** Appending an
observation moves no model parameter. EWC continual learning IS implemented,
in `aedt/continual/ewc.py`, and is measured: forgetting falls from +0.308 to
+0.133 across four sequential tasks. What the store does NOT use is replay, no
rehearsal. That is a deliberate scope decision, and Ablation 7 (planned) tests
whether the rolling update buys anything over a single two-epoch fit. If it
doesn't, we will say so.

### What does hypergraph mean?
A context is **several conditions holding at once** — *sleep poor ∧ activity low
∧ evening ∧ at home*. That is one hyperedge over feature-value vertices:
conjunctive and exact, where a feature-vector distance is compensatory (a low
value on one feature can be offset by a high value on another; a hyperedge
cannot be offset).

**Then say the hard part first, before anyone asks:**

> "The hypergraph is **not** part of the identification mathematics. Our
> estimator regresses the ordinal response on a continuous sensor covariate and
> uses every observation. The hypergraph is the twin's contextual knowledge
> representation, and we tested whether it helps — **and it does not.**
>
> We ran the ablation and the hypergraph-native form is disqualified: it
> falsely rejects when the truth is no change, and at a real 15% recalibration
> it reports the effect in the *wrong direction*. We report that, because a
> negative result presented honestly is worth more than no result."

**If pressed on why:** the hyperedge form measures the *spread of context
effects* rather than a slope, and that quantity confounds a genuine change in
the construct with a change in the reporting scale. The slope-ratio does not,
because the unknown sensor gain cancels in the ratio.

**And the detail that shows we looked properly:** our own placebo does *not*
catch this failure, because both of its pseudo-epochs come from inside epoch 1
and therefore contain no genuine construct change. We found it with a separate
null-cohort calibration check, and we added that check to the ablation.

It *is* doing one real job: the twin reads epoch-to-epoch hyperedge occupancy
overlap and flags an update as untrustworthy when the two epochs describe
different situations. That is a trust decision over higher-order context.

### What is the research contribution?
**Four things.** Point at the standard-vs-contribution table.

1. **Identifiability analysis** — ρ\* is identified up to an explicit
   conservative factor; the additive component is **provably not identifiable**.
2. **The ordinal slope-ratio estimator** and the algebra showing why it is
   exactly calibrated under the null.
3. **Failure analysis of the obvious approach** — the natural affine method
   fabricates a −0.107 apparent scale compression when nothing has changed.
4. **A real-world validity protocol** — eligibility screen, contiguous
   split-half placebo, and empirically calibrated bias envelope.

**Everything else on the architecture diagram is standard technology, and we
have labelled it.**

### What is ρ\*?
The ratio of how strongly the sensor predicts the report in the second half of
the study versus the first. **If it's 1, the scale hasn't changed.**

### Why is ρ not directly estimated? *(the most important answer to get right)*
> "Because ρ is only identified up to a factor we cannot estimate. So we report
> ρ\*, which we can estimate, and treat 1 − ρ\* as a **lower bound** on the real
> recalibration. We'd rather report a bound honestly than a point estimate we
> can't defend."

The code enforces this: `EstimatorResult` refuses any estimand other than
`rho_star` and refuses to carry a value for the additive component.

### Why not simply predict stress?
Prediction assumes the target is a stable measurement. **Our whole question is
whether it is.** A model that predicts a drifting scale very accurately has
learned the drift, not the construct — and will report improvement where there
is none.

### Why is the ordinal model necessary?
A Likert item is a hidden continuous feeling cut into five bands by four
thresholds. That is literally what an ordinal probit models. **We tried the
affine alternative** — it fabricates roughly 10% apparent scale compression on
5-point scales when nothing has changed, because its level coefficient absorbs
threshold saturation. The boundary is not noise to be corrected; **it is the
measurement model.**

And it matters practically: under floor-heavy usage the ordinal estimator
retains about **three times** the effect at equal calibration.

### What happens when the method cannot trust the data?
It says so, loudly, and stops.

- **Weak association** — diagnostic `[9b]` is read *first*; below a median
  |β| of 0.15 it demands the pre-specified fallback covariate before anything
  is interpreted.
- **Eligibility** — every excluded participant is printed by name with a reason.
- **Placebo** — runs *before* the primary and **vetoes it** (exit code 5). Almost
  no undergraduate project has a built-in control that can kill its own headline.
- **Unrecognised labels** — `DECISION REQUIRED`, exit 2, no guessing.
- **Missing real data** — `REAL DATA UNAVAILABLE`, exit 6, **no synthetic
  substitute**.
- **Untrustworthy epoch** — the twin appends `FLAGGED-UNTRUSTWORTHY` and leaves
  its calibration state unchanged.

### What results are real?
**None.** Every result is simulation, labelled as such on every figure and
table.

**But we did open real data, and this is the interesting part.** We acquired
RELAX — 31 working adults, six weeks, wearable heart rate plus repeated
7-point Likert self-reports, published in *Scientific Data*, CC-BY-4.0. We
verified its schema, verified the answer-label anchors, cross-checked its
timestamps, and ran our strict audit.

**It failed our own eligibility screen.** The method needs 60 repeated ordinal
self-reports per person per epoch; RELAX's densest item gives a median of about
50 after causal alignment and a maximum of 93. **Zero of 31 participants
qualified.** Our pipeline exited with code 3 and produced no estimate.

We did not lower the threshold to get a number.

### Why is that a result rather than a failure?
Because it quantifies a real constraint on the method: **it needs a report
density that most current open longitudinal datasets do not provide.**
StudentLife has it (~735 per participant) and is currently unreachable; the
newer, better-documented open datasets deliberately sample less to reduce
participant burden. That tension is worth stating in the paper, and now we have
a measured number behind it instead of a hunch.

### Why is this research and not just software?
Because the central output is a **proof about what is knowable**, and a
demonstration that the obvious method gives a confidently wrong answer. The
software is how we test that.

### Is it patentable?
**No, and we checked properly.** The mathematics is excluded subject matter, and
the one mechanism we thought might qualify turned out to be occupied by
federated-learning prior art. We'd rather tell you that than pretend.

---

## Title term → module → status → role

Full table at `tables/t11_title_alignment.md`. The honest summary:

| Term | Module | Honest status |
|---|---|---|
| **Adaptive** | `twin/update.py` | Scheduled re-estimation and self-assessed trust — **not** an online learning algorithm. |
| **Emotional** | `preprocess/reports.py`, `models/ordinal.py` | We model the **reporting of** self-reported stress. **No emotion recognition is claimed.** |
| **Digital Twin** | `twin/state.py` | Real, persistent, load-bearing. Term is descriptive; no novelty claimed. |
| **Continual Knowledge** | `knowledge/store.py` | Real append-only store. **No continual-learning algorithm imported** — deliberate. |
| **Hypergraph** | `hypergraph/` | Real conjunctive representation + ablation arm. **NOT the identification mechanism.** |
| **Reasoning** | `twin/update.py` trust rule | ⚠️ **Weakest term.** One rule-based trust decision over hyperedge overlap, not a general reasoning engine. **We say this before being asked.** |

**Raise the title-to-contribution gap yourself, on your own slide, before the
panel does.** A panel respects a student who reports that their titular
component did not help far more than one who fudges it.
