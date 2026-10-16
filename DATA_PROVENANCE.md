# Data provenance

This note says what data the current files of this branch do and do not contain, where the
research findings it reports come from, and which questions about reuse are
still open. It says nothing about a software licence; none has been chosen
for this repository.

## 1. No participant-level research record is distributed

The current files of this branch, and the web application built from them,
contain **no** record of any study participant: no observation series, no
per-person statistics, no per-person model output, and no identifier. Earlier
commits in this repository's history still contain such records (section 5).

Every person shown in the application is **simulated**:

| File | Used by | What it is |
|---|---|---|
| `frontend/src/data/trajectories.json` | Twin Evolution, readiness check | three simulated people |

(That script also writes a `participants.json` of simulated people for a
participant explorer. This branch has no such page, and the file is not part
of it.)

They are written by [`scripts/make_synthetic_demo_data.py`](scripts/make_synthetic_demo_data.py)
from fixed constants and a random seed. That script reads no dataset and no
result file; the series are not copied from, fitted to or perturbed from any
person's data. The forecasts drawn for these simulated people come from a few
lines of arithmetic in the same script, not from the research model. Each
file declares itself synthetic, and each page that draws it shows a
"Synthetic demonstration data" label.

The fictional check-in history in the "My Digital Twin" demo
(`frontend/src/lib/demo_history.js`) was always invented and is labelled so.

## 2. Source of the research findings

The research findings reported here are **aggregate** results computed from:

> Nepal, S., Liu, W., Pillai, A., Wang, W., Vojdanovski, V., Huckins, J. F.,
> Rogers, C., Meyer, M. L., & Campbell, A. T. (2024). Capturing the College
> Experience: A Four-Year Mobile Sensing Study of Mental Health, Resilience,
> and Behavior of College Students during the Pandemic. *Proc. ACM Interact.
> Mob. Wearable Ubiquitous Technol.* 8(1), Article 38.
> <https://doi.org/10.1145/3643501>

obtained from its authors' public release
(<https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset>),
which that page lists under CC BY-NC-SA 4.0. **The archive itself is not
included in this repository**; `data/raw/college-experience/PROVENANCE.json`
records its citation and file digests only.

Smaller aggregate results come from three other archives, none of which is
included: RELAX, PMData and StudentLife. Their citations and licences are in
`data/raw/*/PROVENANCE.json` and `docs/dataset_audit.md`. The text-emotion
figures come from the public GoEmotions test split.

## 3. Aggregate result files that are retained

None holds a per-participant list, a series or an identifier.

| File | Contents | Source |
|---|---|---|
| `frontend/src/data/findings.json` | cohort totals; model scores; learning curve; ablation; ceiling statistics for the first study | `docs/preregistration_twin_prediction.md`, `docs/statistic_provenance.md` |
| `frontend/src/data/historical_findings.json` | the same figures as first reported | `docs/statistic_provenance.md` |
| `frontend/src/data/real_datasets.json` | per-dataset audit and estimator summaries | `docs/dataset_audit.md` |
| `frontend/src/data/layer1_results.json` | text-emotion evaluation, and results of two synthetic experiments | `scripts/eval_emotion_model.py` and the scripts named in the README |
| `tables/`, `figures/` | every table row carries a status label: SYNTHETIC, REAL (aggregate dataset-audit counts) or PLANNED; figures are stamped SYNTHETIC | simulation; `docs/dataset_audit.md` |

The result files these were generated from (`results/`) are not in the
repository, so the retained figures cannot be regenerated without obtaining
the source archive.

## 4. Scripts

No script on this branch writes participant-level records into the web
application.

## 5. What is not established

- **Licence of data-derived outputs.** The source release is listed under
  CC BY-NC-SA 4.0. Whether, and how, that licence's attribution,
  non-commercial and share-alike terms apply to the aggregate results and
  reports in this repository has not been determined. Until it is, treat them
  as non-commercial, attribute the source above, and do not assume the
  software licence covers them.
- **Ethics.** The source study reports informed consent and approval by its
  own institution's review board. This project performed secondary analysis
  of the public release. No statement is made here that any further approval
  was obtained or was not required.
- **Participant counts.** The source paper reports 215 participants; the
  analyses here use 218 participant identifiers with a stress report. The
  release does not explain the difference.
- **Earlier versions of this project** distributed derived participant-level
  records in the application: per-person summary statistics, and report series
  with model predictions for three people. They were removed from the current
  files of this branch. They remain in the repository's earlier commits and in
  any copy, fork, cache or deployment made before their removal. Removing them
  from the current files does not remove them from the history.

## 6. Safeguard against reintroduction

`frontend/scripts/check-no-participant-data.mjs` runs before every build
(`npm run build` calls it as `prebuild`) and fails the build if a file in
`frontend/src/data/` is one of the removed participant-derived files
(recognised by git blob hash) or if a people-level demonstration file lacks
`"_synthetic": true`. `tests/frontend/test_no_participant_data.py` applies the
same rule in the Python test suite.
