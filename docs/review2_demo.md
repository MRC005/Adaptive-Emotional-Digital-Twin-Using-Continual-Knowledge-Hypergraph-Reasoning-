# Review-2 Demonstration

**One command. Offline. About two minutes.**

```bash
python scripts/run_demo.py --dataset synthetic --participant p07
```

The same interface accepts every dataset:

```bash
python scripts/run_demo.py --dataset studentlife --root /path/to/StudentLife_Dataset
python scripts/run_demo.py --dataset relax       --root /path/to/relax
python scripts/run_demo.py --dataset wesad       --root /path/to/WESAD
python scripts/run_demo.py --dataset pmdata      --root /path/to/pmdata
```

If the real files are absent it prints, for example:

```
REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN: no directory at '...'
No synthetic substitute was used.
```

followed by the acquisition instructions, and exits **6**.

---

## The nine stages, and the one line to say for each

| Stage | What is shown | Say this |
|---|---|---|
| **1 Ingest** | dataset, role, DATA STATUS, participants, observations, span, code→severity map, preprocessing retention | *"One person, one term."* |
| **2 Reports** | category-usage counts per epoch, floor and ceiling rates | *"Most days sit at one end of the scale — that matters enormously later."* |
| **3 Sensor** | per-epoch mean/SD/range, the epoch rule and midpoint | *"The passive signal, and where we cut the term in two."* |
| **4 Context** | all three representations; the participant's hypergraph — vertices, hyperedges, arity, epoch overlap | *"A hyperedge is several conditions holding at once. We test whether it helps."* |
| **5 Twin** | `TwinState`: time, observations seen, category usage, eligibility, audit flags, knowledge-node counts by kind; persisted to JSON | *"This models their measuring instrument, not their mood."* |
| **6 MODEL** ⭐ | **both fitted ordinal curves on one axis**, β and cutpoints per epoch, ρ\* for this person | *"If the curve got flatter, the same behaviour now earns a different number."* |
| **7 Audit** | `[9b]` association strength, eligibility with every exclusion by name, the placebo verdict | *"Before believing it, we check whether we're allowed to."* |
| **8 Result** | ρ\* with CI, median, participants used, exclusions, and the identification caveats | *"If the estimate doesn't clear the band, we say so."* |
| **9 Twin update** | the history row gaining `ACCEPTED` or `FLAGGED-UNTRUSTWORTHY`, with flags | *"The twin now remembers this epoch and how much to trust it."* |

**Stage 6 is the money shot.** Two curves on one axis is the entire project in
one picture.

---

## What the demo output contains

1. Dataset audit summary
2. **Data status: REAL or SYNTHETIC** (stamped on every figure and table)
3. Pipeline execution summary
4. Eligibility result, with every exclusion named
5. Placebo result — **shown before the primary**
6. ρ\* result
7. Uncertainty (participant-cluster bootstrap CI)
8. **Epoch 1 vs Epoch 2 figure**
9. Context / hypergraph figure
10. Exclusions
11. Reproducibility metadata — run id, seed, config digest, versions, git commit, elapsed

## Artefacts written

Into `results/<timestamp>_<dataset>_<STATUS>/`:

**Figures** — architecture · pipeline · **two-curve (fig01)** · category usage ·
placebo · forest · bias envelope · hypergraph context · hypergraph ablation ·
audit dashboard

**Tables** — dataset audit · participants · eligibility · primary estimator ·
exclusions · uncertainty · placebo · sensitivity/bias envelope · baselines ·
hypergraph ablation · status board · title alignment · standard-vs-contribution

Plus `run_metadata.json`, `resolved_config.json`,
`config_deviations_from_frozen.json` and `twins/<pid>.json`.

To regenerate the presentation set into top-level `figures/` and `tables/`:

```bash
python scripts/generate_review2_outputs.py
```

---

## Live demonstrations worth doing

### 1. The placebo vetoing the headline

Show stage 7 running **before** stage 8, and say: *"almost no undergraduate
project has a built-in control that can kill its own result."*

### 2. `DECISION REQUIRED` halting on a mislabelled file

Ten seconds, and it demonstrates engineering maturity better than any
architecture diagram:

```bash
python scripts/audit_dataset.py --fixture studentlife --out /tmp/sl
python - <<'EOF'
import json, pathlib
p = pathlib.Path("/tmp/sl/EMA/EMA_definition.json")
d = json.loads(p.read_text())
d["Stress"]["responses"][0] = "Mildly perturbed"   # not in the specification
p.write_text(json.dumps(d))
EOF
python scripts/audit_dataset.py --dataset studentlife --root /tmp/sl   # exits 2
```

### 3. `REAL DATA UNAVAILABLE` refusing to substitute

```bash
python scripts/run_demo.py --dataset studentlife --root /nonexistent   # exits 6
```

### 4. The tests, live

```bash
python -m pytest tests -q
```

---

## Failure paths

If anything breaks, fall through to a recorded run — **and say plainly that it
is a recording.** Record it in advance with:

```bash
python scripts/run_demo.py --dataset synthetic --participant p07 | tee demo_recording.txt
```

## The one rule for this review

**Every slide and every plot title carries `SYNTHETIC`. No exceptions.**

Do not let a single number, plot or sentence imply that any result came from
real data. One blurred label would cost more credibility than every missing
module combined.
