#!/usr/bin/env python3
"""Write the SYNTHETIC demonstration data the web application ships with.

    python scripts/make_synthetic_demo_data.py

Everything this script writes is SIMULATED. It reads no dataset, no result
file and no participant record: the only inputs are the constants below and a
fixed random seed. Nothing is copied from, fitted to, or perturbed from any
real person's data.

It writes, under frontend/src/data/:

  trajectories.json      three simulated people, for the Twin Evolution page
  participants.json      simulated people, for the Explorer and the two-person
                         comparison
  research_results.json  ONLY its ``participants`` section is replaced, by three
                         simulated twins; every aggregate result in that file is
                         left exactly as it is

WHAT THE "TWIN" COLUMN MEANS HERE. The demonstration files need a forecast to
draw. It comes from a few lines of arithmetic in this file (a rounded
exponentially weighted mean, or a smoothed personal transition table), NOT
from the research model. The pages say so.

Standard library only, so it can be run and reviewed without the research
environment.
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "frontend" / "src" / "data"
SEED = 20261013

PROVENANCE = ("SYNTHETIC DEMONSTRATION DATA — simulated by "
              "scripts/make_synthetic_demo_data.py. Not derived from, fitted to or "
              "copied from any participant. It describes no real person.")
NOT_APPLICABLE = "Not applicable: these records are simulated and describe no one."
STAND_IN = ("The forecast shown is an illustrative stand-in computed by a few lines of "
            "arithmetic, not the research model.")

LABELS = (1, 2, 3, 4, 5)
WARMUP_K = 20
K_GRID = (20, 80)
MIN_EVAL = 10

# The node vocabulary of the research system (field -> values), restated here so
# this script has no import from the research code. Index 0 means "not observed".
FIELD_VALUES = (
    ("stress", ("1", "2", "3", "4", "5")),
    ("pam", ("1-4", "5-8", "9-12", "13-16")),
    ("phq4", ("0", "1-2", "3-5", "6-12")),
    ("social", ("1", "2", "3", "4", "5")),
    ("daytype", ("weekday", "weekend")),
    ("sleep", ("low", "typical", "high")),
    ("phone", ("low", "typical", "high")),
    ("still", ("low", "typical", "high")),
    ("mobility", ("low", "typical", "high")),
    ("convo", ("low", "typical", "high")),
)


def vocabulary():
    offsets, labels, i = {}, {}, 1
    for field, values in FIELD_VALUES:
        offsets[field] = i
        for v in values:
            labels[i] = f"{field}={v}"
            i += 1
    return offsets, labels


# ------------------------------------------------------------------ simulation
def simulate(rng, n, phi, level, noise):
    """An ordinal series: a latent first-order process, rounded onto 1..5,
    reported at irregular intervals of one to seven days."""
    z, day, out = level, 0, []
    for i in range(n):
        if i:
            z = level + phi * (z - level) + rng.gauss(0.0, noise)
            day += min(7, 1 + int(rng.expovariate(0.55)))
        out.append((day, max(1, min(5, int(round(z))))))
    return out


def lag1(values):
    if len(values) < 3:
        return None
    a, b = values[:-1], values[1:]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def ewma_forecasts(values, alpha=0.5):
    """For each report i, a point forecast of report i+1 from reports <= i."""
    out, level = [], None
    for v in values:
        level = v if level is None else alpha * v + (1 - alpha) * level
        out.append(max(1, min(5, int(round(level)))))
    return out


def scores(pred, truth):
    """macro-F1 over the five fixed labels, accuracy and mean absolute error."""
    n = len(truth)
    f1 = []
    for lab in LABELS:
        tp = sum(1 for p, t in zip(pred, truth) if p == lab and t == lab)
        fp = sum(1 for p, t in zip(pred, truth) if p == lab and t != lab)
        fn = sum(1 for p, t in zip(pred, truth) if p != lab and t == lab)
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"macro_f1": round(sum(f1) / len(LABELS), 4),
            "accuracy": round(sum(1 for p, t in zip(pred, truth) if p == t) / n, 4),
            "mae": round(sum(abs(p - t) for p, t in zip(pred, truth)) / n, 4),
            "qwk": None}


def band(r):
    if r is None:
        return "typical"
    return "well-predictable" if r >= 0.5 else "near-unpredictable" if r < 0.2 else "typical"


# ------------------------------------------------------------------ files
def make_trajectories(rng):
    people = []
    for pid, phi, level, noise, n in (("SYN-A", 0.88, 2.6, 0.45, 96),
                                      ("SYN-B", 0.45, 3.0, 0.80, 80),
                                      ("SYN-C", 0.03, 2.8, 1.00, 64)):
        s = simulate(rng, n, phi, level, noise)
        values = [v for _, v in s]
        pred = ewma_forecasts(values)
        r = lag1(values)
        people.append({
            "id": pid, "n_observations": n,
            "own_autocorrelation": None if r is None else round(r, 3),
            "predictability": {"well-predictable": "well predicted by their own history",
                               "typical": "typical",
                               "near-unpredictable": "close to unpredictable"}[band(r)],
            "warmup_k": WARMUP_K,
            # "baseline" is the same report carried forward (the persistence
            # forecast) and "actual_next" the report that followed: the two
            # fields the walkthrough page prints beside the stand-in forecast.
            "series": [{"day": d, "observed": v, "warmup": i < WARMUP_K, "predicted": pred[i],
                        "baseline": v,
                        "actual_next": values[i + 1] if i + 1 < len(values) else None}
                       for i, (d, v) in enumerate(s)],
        })
    return {"_synthetic": True, "_provenance": PROVENANCE, "_anonymisation": NOT_APPLICABLE,
            "_note": ("These are SIMULATED people, generated to demonstrate the page. "
                      + STAND_IN),
            "participants": people}


def make_participants(rng, n_people=24):
    people = []
    for j in range(n_people):
        phi = rng.uniform(0.0, 0.9)
        n = rng.randint(34, 150)
        s = simulate(rng, n, phi, rng.uniform(2.0, 3.6), rng.uniform(0.4, 1.0))
        values = [v for _, v in s]
        pred = ewma_forecasts(values)
        r = lag1(values)
        mean = sum(values) / n
        by_k = {}
        for k in K_GRID:
            n_eval = n - k - 1                       # report i forecasts report i+1
            if n_eval < MIN_EVAL:
                by_k[str(k)] = {"scored": False,
                                "reason": f"fewer than {k + MIN_EVAL + 1} simulated reports"}
                continue
            idx = range(k, n - 1)
            truth = [values[i + 1] for i in idx]
            twin = scores([pred[i] for i in idx], truth)
            pers = scores([values[i] for i in idx], truth)
            d = round(twin["macro_f1"] - pers["macro_f1"], 4)
            by_k[str(k)] = {"scored": True, "n_warmup": k, "n_eval": n_eval, "twin": twin,
                            "persistence": pers, "twin_minus_persistence_macro_f1": d,
                            "twin_better": d > 0}
        people.append({
            "n_observations": n, "span_days": s[-1][0],
            "autocorrelation": None if r is None else round(r, 3),
            "predictability_band": band(r),
            "variability_sd": round(math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)), 3),
            "mean_stress": round(mean, 3), "by_K": by_k,
            # a scored row carries the stand-in forecast of the NEXT report, the
            # persistence forecast (this report) and the value that followed
            "series": [dict({"day": d, "observed": v, "warmup": i < WARMUP_K},
                            **({"predicted": pred[i], "baseline": v,
                                "actual_next": values[i + 1]}
                               if WARMUP_K <= i < n - 1 else {"predicted": None}))
                       for i, (d, v) in enumerate(s)],
            "warmup_k": WARMUP_K, "id": f"S{j + 1:02d}", "series_published": True,
        })
    kp = str(K_GRID[0])
    scored = [p for p in people if p["by_K"][kp]["scored"]]
    ranked = sorted(scored, key=lambda p: p["autocorrelation"] or 0.0)
    illustrative = [ranked[0]["id"], ranked[len(ranked) // 2]["id"], ranked[-1]["id"]]
    return {
        "_synthetic": True,
        "_what_this_is": ("One record per SIMULATED person. Demonstration data for the "
                          "Explorer; not a research result."),
        "_generated_by": "scripts/make_synthetic_demo_data.py",
        "_provenance": PROVENANCE,
        "_anonymisation": NOT_APPLICABLE,
        "_definitions": {
            "autocorrelation": "Lag-1 correlation of the person's own simulated reports.",
            "predictability_band": "well-predictable: r >= 0.5; near-unpredictable: r < 0.2; "
                                   "otherwise typical",
            "band_source": "Fixed thresholds chosen for this demonstration.",
            "variability_sd": "Standard deviation of the simulated reports.",
            "macro_f1": "Unweighted mean F1 over the five stress levels, on this person's "
                        "scored simulated reports. 'Twin' here is an illustrative stand-in "
                        "(a rounded exponentially weighted mean), not the research model.",
            "warm_up": "The first K reports, used as history and not scored.",
        },
        "_uncertainty": ("No interval is shown: these are simulated records, and no "
                         "inference about any population can be drawn from them."),
        "_redistribution": ("No participant-level research record is distributed with this "
                            "application. These records are simulated."),
        "series_mode": "all",
        "illustrative_rule": ("the scored simulated people at the minimum, median and "
                              "maximum autocorrelation"),
        "illustrative_ids": illustrative,
        "n_series_published": len(people),
        "split": "synthetic",
        "K_primary": K_GRID[0], "K_grid": list(K_GRID), "seed": SEED,
        "n_test_participants": len(people),
        "n_scored_primary": len(scored),
        "n_twin_better_primary": sum(1 for p in scored if p["by_K"][kp]["twin_better"]),
        "participants": people,
    }


def make_twins(rng):
    """Three simulated twins for the Twin System page."""
    offsets, labels = vocabulary()
    tau, window, out = 60.0, 64, []
    for pid, phi, level, noise, n in (("SYN-1", 0.85, 2.4, 0.5, 72),
                                      ("SYN-2", 0.50, 3.1, 0.8, 60),
                                      ("SYN-3", 0.10, 2.9, 1.0, 48)):
        s = simulate(rng, n, phi, level, noise)
        trans = [[1.0] * 5 for _ in range(5)]            # Laplace-smoothed, personal
        episodes, series, seen = [], [], set()
        ewma = mean = None
        for i, (day, v) in enumerate(s):
            nodes = [offsets["stress"] + v - 1]
            for field, values in FIELD_VALUES[1:]:
                if field == "daytype":
                    nodes.append(offsets[field] + (1 if day % 7 >= 5 else 0))
                elif rng.random() < 0.8:                  # some fields go unobserved
                    nodes.append(offsets[field] + rng.randrange(len(values)))
            if i:
                trans[s[i - 1][1] - 1][v - 1] += 1.0
            row = trans[v - 1]
            total = sum(row)
            forecast = [round(x / total, 4) for x in row]
            cdf, median = 0.0, 5
            for lab, q in zip(LABELS, forecast):
                cdf += q
                if cdf >= 0.5 - 1e-12:
                    median = lab
                    break
            evidence = sum(math.exp(-(day - e["day"]) / tau) * len(set(nodes) & set(e["nodes"]))
                           for e in episodes[-window:])
            ewma = v if ewma is None else 0.5 * v + 0.5 * ewma
            mean = v if mean is None else mean + (v - mean) / (i + 1)
            nxt = s[i + 1] if i + 1 < n else None
            scored = nxt is not None and 1 <= nxt[0] - day <= 7
            seen |= set(nodes)
            series.append({
                "day": day, "stress": v, "scored": scored,
                "target": nxt[1] if nxt else None,
                "forecast": forecast, "median": median,
                "mode": forecast.index(max(forecast)) + 1,
                "uncertainty": round(-sum(q * math.log(q) for q in forecast if q > 0)
                                     / math.log(5), 4),
                "context_evidence": round(evidence, 3), "edges": i,
                "nodes_used": len(seen),
                "change": None if i == 0 else round(ewma - mean, 3),
                "calibrator_shift": 0.0,
            })
            episodes.append({"day": day, "nodes": sorted(nodes),
                             "outcome": nxt[1] if nxt else -1})
        used = sorted(seen)
        out.append({
            "id": pid, "n_reports": n, "series": series,
            "hypergraph": {
                "window": window, "tau_days": tau,
                "summary": {"n_episodes": n, "n_resolved": n - 1, "n_nodes_used": len(used),
                            "mean_arity": round(sum(len(e["nodes"]) for e in episodes) / n, 2),
                            "in_reasoning_window": min(n, window)},
                "nodes": [{"index": i, "label": labels[i]} for i in used],
                "episodes": episodes[-40:],
            },
        })
    return out


def dump(path, doc):
    # ASCII-escaped and without a trailing newline: the form the existing data
    # files use, so the aggregate sections of research_results.json are written
    # back byte-for-byte.
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=True), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def main() -> int:
    rng = random.Random(SEED)
    dump(DATA / "trajectories.json", make_trajectories(rng))
    dump(DATA / "participants.json", make_participants(rng))
    results = DATA / "research_results.json"
    if results.exists():
        doc = json.loads(results.read_text(encoding="utf-8"))
        doc["participants"] = make_twins(rng)
        doc["_redistribution"] = ("No participant-level research record is distributed with "
                                  "this application. The three twins shown are SIMULATED. "
                                  + STAND_IN)
        doc["_participants_are_synthetic"] = True
        dump(results, doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
