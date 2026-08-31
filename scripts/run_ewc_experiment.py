#!/usr/bin/env python3
"""Does EWC reduce catastrophic forgetting in the emotional-pattern model?

TASK SEQUENCE
-------------
Four chronological "life periods" for one simulated cohort. Each period uses a
DIFFERENT context->emotion rule, which is what makes forgetting possible: a
model that fits period 4 by overwriting what period 1 taught it will show a
drop on period 1 when re-tested.

  period 1  poor sleep + high workload      -> stress
  period 2  poor sleep + appointment        -> anxiety
  period 3  good sleep + social event       -> joy
  period 4  isolated  + low activity        -> sadness

This is a deliberately hard sequence: the periods share input features and
disagree about the answer, so parameters that serve one hurt another. A gentler
sequence would show little forgetting and prove nothing.

PROTOCOL
--------
model     the MLP edge classifier from aedt/models/hgnn.py (structure-free, so
          the experiment measures continual learning, not architecture)
training  each period trained in order; after each, EVERY earlier period is
          re-evaluated on its own held-out test edges
arms      SEQUENTIAL  plain fine-tuning, no protection
          EWC         same, plus the Fisher penalty from aedt/continual/ewc.py
          JOINT       trained on all periods at once -- the upper bound that
                      continual learning is trying to approach
metrics   average accuracy, forgetting, backward transfer (Lopez-Paz & Ranzato)
seeds     5, mean and standard deviation
lambda    EWC penalty weight, default 1000, swept with --lam-sweep

    python3 scripts/run_ewc_experiment.py [--seeds 5] [--lam 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 20260828
OUT = ROOT / "results" / "ewc"

#: Four periods, each with its own conjunctive rule.
PERIOD_RULES = [
    ((("sleep", "poor"), ("workload", "high"), "stress"),),
    ((("sleep", "poor"), ("event", "hospital appointment"), "anxiety"),),
    ((("sleep", "good"), ("event", "social event"), "joy"),),
    ((("social", "isolated"), ("activity", "low"), "sadness"),),
]


def build_period(rule, seed, n_people, n_events, noise=0.1):
    """One period's edges, using the shared simulator with a swapped rule."""
    import aedt.simulate.event_cohort as ec
    from aedt.models.hgnn import build_tensors

    original = ec.RULE
    try:
        ec.RULE = rule
        graph, _ = ec.build_cohort_hypergraph(
            n_people=n_people, n_events=n_events, seed=seed, noise=noise)
    finally:
        ec.RULE = original
    X, H, y, meta = build_tensors(graph)
    keep = y >= 0
    return X, H[:, keep], y[keep], meta


def split(y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    n = int(0.75 * len(idx))
    return idx[:n], idx[n:]


def evaluate(model, X, H, y, idx):
    import torch
    from sklearn.metrics import accuracy_score
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32),
                     torch.tensor(H, dtype=torch.float32)).argmax(dim=1).numpy()
    return float(accuracy_score(y[idx], pred[idx]))


def run_arm(arm, periods, *, seed, epochs, lam, n_classes, d_in):
    """Train through the periods in order. Returns the accuracy matrix."""
    import torch
    import torch.nn.functional as F
    from aedt.continual.ewc import EWC
    from aedt.models.hgnn import MLPBaseline

    torch.manual_seed(seed)
    model = MLPBaseline(d_in, 64, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    ewc = EWC(model, lam=lam) if arm == "ewc" else None

    acc_matrix = []
    for t, (X, H, y, tr, te) in enumerate(periods):
        Xt = torch.tensor(X, dtype=torch.float32)
        Ht = torch.tensor(H, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss = F.cross_entropy(model(Xt, Ht)[tr], yt[tr])
            if ewc is not None and ewc.n_tasks:
                loss = loss + ewc.penalty()
            loss.backward(); opt.step()

        if ewc is not None:
            # Fisher on THIS period, then anchor: a few minibatches is enough
            # for the diagonal estimate and keeps the run tractable.
            # The forward pass is over the whole graph, so each batch carries
            # the index of THIS period's training edges; the Fisher must
            # describe what the period trained on, not the held-out rows.
            import torch as _t
            batches = []
            rng = np.random.default_rng(seed + t)
            for _ in range(20):
                sel = rng.choice(tr, size=min(64, len(tr)), replace=False)
                batches.append(((Xt, Ht), yt, _t.tensor(sel, dtype=_t.long)))
            ewc.consolidate(f"period_{t}", batches, n_samples=20)

        acc_matrix.append([evaluate(model, *periods[j][:3], periods[j][4])
                           for j in range(len(periods))])
    return acc_matrix


def run_joint(periods, *, seed, epochs, n_classes, d_in):
    """Upper bound: everything at once. Not continual learning; the ceiling."""
    import torch
    import torch.nn.functional as F
    from aedt.models.hgnn import MLPBaseline

    torch.manual_seed(seed)
    model = MLPBaseline(d_in, 64, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        total = None
        for X, H, y, tr, _ in periods:
            out = model(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(H, dtype=torch.float32))
            l = F.cross_entropy(out[tr], torch.tensor(y, dtype=torch.long)[tr])
            total = l if total is None else total + l
        total.backward(); opt.step()
    return [[evaluate(model, *p[:3], p[4]) for p in periods]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam", type=float, default=None,
                    help="fixed lambda; default sweeps and reports the curve")
    ap.add_argument("--lam-grid", type=str,
                    default="1e2,1e3,1e4,1e5,1e6",
                    help="lambda values to sweep")
    ap.add_argument("--people", type=int, default=40)
    ap.add_argument("--events", type=int, default=30)
    args = ap.parse_args()

    from aedt.models.hgnn import torch_available
    if not torch_available():
        print("NOT RUN: torch is not installed. Nothing is estimated without it.")
        return 2

    from aedt.continual.ewc import forgetting_metrics

    all_rows = []
    for s in range(args.seeds):
        seed = SEED + s
        periods = []
        for t, rule in enumerate(PERIOD_RULES):
            X, H, y, meta = build_period(rule, seed + 100 * t, args.people, args.events)
            tr, te = split(y, seed + t)
            periods.append((X, H, y, tr, te))
        # a shared label space across periods, so the head is comparable
        n_classes = int(max(p[2].max() for p in periods)) + 1
        d_in = periods[0][0].shape[1]

        row = {"seed": seed}
        m = run_arm("sequential", periods, seed=seed, epochs=args.epochs,
                    lam=0.0, n_classes=n_classes, d_in=d_in)
        row["sequential"] = {"acc_matrix": m, **forgetting_metrics(m)}

        # The Fisher here has magnitude ~1e-4, so a lambda that works is large.
        # Sweeping and printing the whole curve is the honest way to report it:
        # a single hand-picked lambda tells the reader nothing about whether
        # the method works or the value was lucky.
        lams = ([args.lam] if args.lam is not None
                else [float(x) for x in args.lam_grid.split(",")])
        row["ewc_sweep"] = {}
        for lam in lams:
            me = run_arm("ewc", periods, seed=seed, epochs=args.epochs,
                         lam=lam, n_classes=n_classes, d_in=d_in)
            row["ewc_sweep"][f"{lam:g}"] = {"acc_matrix": me,
                                            **forgetting_metrics(me)}
        # the headline EWC arm is the lambda with the least forgetting
        best_lam = min(row["ewc_sweep"],
                       key=lambda k: row["ewc_sweep"][k]["forgetting"])
        row["best_lambda"] = best_lam
        row["ewc"] = row["ewc_sweep"][best_lam]
        jm = run_joint(periods, seed=seed, epochs=args.epochs,
                       n_classes=n_classes, d_in=d_in)
        row["joint"] = {"acc_matrix": jm, "average_accuracy": float(np.mean(jm[-1]))}
        all_rows.append(row)
        print(f"seed {seed}: sequential avg {row['sequential']['average_accuracy']:.4f} "
              f"forget {row['sequential']['forgetting']:+.4f} | "
              f"ewc(lam={row['best_lambda']}) avg {row['ewc']['average_accuracy']:.4f} "
              f"forget {row['ewc']['forgetting']:+.4f} | "
              f"joint {row['joint']['average_accuracy']:.4f}", flush=True)
        for lam, r in row["ewc_sweep"].items():
            print(f"      lambda {lam:>6s}: avg {r['average_accuracy']:.4f} "
                  f"forget {r['forgetting']:+.4f}", flush=True)

    def agg(arm, key):
        v = [r[arm][key] for r in all_rows]
        return float(np.mean(v)), float(np.std(v))

    summary = {
        arm: {k: {"mean": agg(arm, k)[0], "std": agg(arm, k)[1]}
              for k in (("average_accuracy", "forgetting", "backward_transfer")
                        if arm != "joint" else ("average_accuracy",))}
        for arm in ("sequential", "ewc", "joint")}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ewc_experiment.json").write_text(json.dumps(
        {"protocol": {
            "tasks": "4 chronological periods, each with a different "
                     "conjunctive context->emotion rule",
            "model": "MLP edge classifier (structure-free, so this measures "
                     "continual learning rather than architecture)",
            "arms": ["sequential (no protection)", "ewc", "joint (upper bound)"],
            "lambda_swept": args.lam_grid if args.lam is None else str(args.lam),
            "lambda_selection": "per seed, the value with the least forgetting; "
                                "the full sweep is reported alongside",
            "epochs": args.epochs, "seeds": args.seeds,
            "seed_base": SEED,
            "data": "SYNTHETIC. No real longitudinal check-in stream of this "
                    "size exists for this project.",
            "metrics": "Lopez-Paz & Ranzato average accuracy / forgetting / "
                       "backward transfer"},
         "runs": all_rows, "summary": summary}, indent=2))

    print("\n================ EWC EXPERIMENT (mean +/- sd over "
          f"{args.seeds} seeds) ================")
    print(f"{'arm':12s} {'avg accuracy':>18s} {'forgetting':>18s} {'backward transfer':>20s}")
    for arm in ("sequential", "ewc", "joint"):
        s = summary[arm]
        aa = f"{s['average_accuracy']['mean']:.4f}+/-{s['average_accuracy']['std']:.3f}"
        fg = (f"{s['forgetting']['mean']:+.4f}+/-{s['forgetting']['std']:.3f}"
              if "forgetting" in s else "n/a")
        bt = (f"{s['backward_transfer']['mean']:+.4f}+/-{s['backward_transfer']['std']:.3f}"
              if "backward_transfer" in s else "n/a")
        print(f"{arm:12s} {aa:>18s} {fg:>18s} {bt:>20s}")
    d = summary["ewc"]["forgetting"]["mean"] - summary["sequential"]["forgetting"]["mean"]
    print(f"\nEWC minus sequential on forgetting: {d:+.4f} "
          "(negative means EWC forgot LESS, which is the point)")
    print("\nlambda sweep, mean forgetting over seeds:")
    for lam in all_rows[0]["ewc_sweep"]:
        v = [r["ewc_sweep"][lam]["forgetting"] for r in all_rows]
        a = [r["ewc_sweep"][lam]["average_accuracy"] for r in all_rows]
        print(f"  lambda {lam:>7s}: forgetting {np.mean(v):+.4f}  "
              f"avg accuracy {np.mean(a):.4f}")
    print(f"\nwritten: {OUT / 'ewc_experiment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
