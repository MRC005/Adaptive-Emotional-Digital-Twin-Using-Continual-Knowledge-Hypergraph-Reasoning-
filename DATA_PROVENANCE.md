# Data provenance

This note says what data this branch does and does not contain, where the
research findings it reports come from, and which questions about reuse are
still open. It says nothing about a software licence; none has been chosen
for this repository.

## 1. No participant-level research record is distributed

The repository and the web application contain **no** record of any study
participant: no observation series, no per-person statistics, no per-person
model output, and no identifier.

Every person shown in the application is **simulated**:

| File | Used by | What it is |
|---|---|---|
| `frontend/src/data/trajectories.json` | Twin Evolution, readiness check | three simulated people |
| `frontend/src/data/participants.json` | Explorer, two-person comparison | 24 simulated people |

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
| `frontend/src/data/cohort_summary.json` | four counts: test participants (44), scored after 20 warm-up reports (42), of whom the twin was better for 12 | `docs/statistic_provenance.md` §1; `docs/preregistration_selective_personalisation.md` §§1, 4 |
| `frontend/src/data/real_datasets.json` | per-dataset audit and estimator summaries | `docs/dataset_audit.md` |
| `frontend/src/data/layer1_results.json` | text-emotion evaluation, and results of two synthetic experiments | `scripts/eval_emotion_model.py` and the scripts named in the README |
| `tables/`, `figures/` | every table row carries a status label: SYNTHETIC, REAL (aggregate dataset-audit counts) or PLANNED; figures are stamped SYNTHETIC | simulation; `docs/dataset_audit.md` |

The result files these were generated from (`results/`) are not in the
repository, so the retained figures cannot be regenerated without obtaining
the source archive.

## 4. Script that is disabled here

`scripts/export_participant_explorer.py` writes participant-level records into
the web application's data directory. It is disabled on this branch: it exits
without reading or writing anything.

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
- **Earlier versions of this project** distributed a small number of derived
  participant-level records in the application. They are not part of this
  repository.
- **Repository history.** The history of this repository was revised to remove
  the participant-derived records that earlier versions contained. As a
  result, some commit identifiers cited in older documents here may no longer
  resolve. Those documents are left exactly as they were written.
  `scripts/export_historical_baseline.py` names the current identifier of the
  commit it reads.
