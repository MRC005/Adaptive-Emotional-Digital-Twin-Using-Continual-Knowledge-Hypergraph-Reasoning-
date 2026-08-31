/**
 * Data sources for the instrument.
 *
 * Two kinds, kept strictly apart:
 *   1. GUIDED CONTROLS — simulated cohorts with a KNOWN answer, used to
 *      validate that the pipeline behaves correctly. These are controls, not
 *      findings, and the UI labels them as such everywhere.
 *   2. USER DATA — a CSV the reviewer supplies, validated before analysis.
 *
 * The generator is a port of aedt/simulate/generator.py and implements the
 * same model: a latent construct theta, a thresholded ordinal report whose
 * person-specific cutpoints are FIXED across epochs, and a sensor that is a
 * noisy affine function of theta with an unknown per-person gain.
 */
import { quantile, randn, rng } from "./stats.js";

export const PLACEMENTS = {
  balanced: [0.2, 0.4, 0.6, 0.8],
  skewed: [0.45, 0.7, 0.86, 0.95],
  extreme_floor: [0.65, 0.84, 0.93, 0.975],
};

function ar1(rand, n, sdv, phi) {
  const out = new Array(n);
  let prev = randn(rand) * sdv;
  out[0] = prev;
  const k = Math.sqrt(1 - phi * phi);
  for (let t = 1; t < n; t++) { prev = phi * prev + k * randn(rand) * sdv; out[t] = prev; }
  return out;
}

/**
 * Simulate one cohort.
 * @param trueRho  the recalibration actually applied (1.0 = none)
 * @returns {byPid, K, meta}
 */
export function simulateCohort({
  trueRho = 0.85, nParticipants = 30, nPerEpoch = 150, seed = 20260828,
  placement = "skewed", phi = 0.4, sigmaReport = 0.5, sigmaSensor = 0.8,
  missingness = 0,
} = {}) {
  const rand = rng(seed);
  const cutq = PLACEMENTS[placement] || PLACEMENTS.skewed;
  const K = cutq.length + 1;
  const byPid = {};
  for (let p = 0; p < nParticipants; p++) {
    const pid = `P${String(p + 1).padStart(2, "0")}`;
    const lam = 0.35 + rand() * 0.55;                 // unknown per-person gain
    const kap = 0.4 + randn(rand) * 0.3;
    const a2 = trueRho + randn(rand) * 0.08;          // the recalibration
    const delta = -0.4 + randn(rand) * 0.3;           // GENUINE change in theta
    const th1 = [], th2 = [];
    for (let i = 0; i < nPerEpoch; i++) { th1.push(randn(rand)); th2.push(delta + randn(rand)); }
    const r1 = th1.map((t) => t + randn(rand) * sigmaReport);
    const r2 = th2.map((t) => a2 * t + 0.3 + randn(rand) * sigmaReport);
    // thresholds are FIXED for the person, placed on the epoch-1 latent scale
    const sorted = [...r1].sort((a, b) => a - b);
    const cuts = cutq.map((q) => quantile(sorted, q));
    const toCat = (v) => { let k = 0; while (k < cuts.length && v > cuts[k]) k++; return k + 1; };
    const n1 = ar1(rand, nPerEpoch, sigmaSensor, phi);
    const n2 = ar1(rand, nPerEpoch, sigmaSensor, phi);
    // A SECOND observable feature with a different loading on theta, so the
    // variable selector reflects a real choice rather than a decorative one.
    const lamB = 0.15 + rand() * 0.35;
    const m1 = ar1(rand, nPerEpoch, 0.9, phi / 2);
    const m2 = ar1(rand, nPerEpoch, 0.9, phi / 2);
    const rows = [];
    for (let e = 0; e < 2; e++) {
      const th = e === 0 ? th1 : th2, rr = e === 0 ? r1 : r2, nz = e === 0 ? n1 : n2;
      for (let i = 0; i < nPerEpoch; i++) {
        if (missingness > 0 && rand() < (e === 1 ? missingness : missingness * 0.3)) continue;
        const s = lam * th[i] + kap + nz[i];
        const b = lamB * th[i] + (e === 0 ? m1[i] : m2[i]);
        rows.push({ pid, t: e * nPerEpoch + i, epoch: e,
                    report: toCat(rr[i]),
                    // conversation minutes: falls as the construct rises
                    sensor: Math.max(0, 60 - 25 * s),
                    // a weaker second feature, on a 0-1 scale
                    features: { conversation_minutes: Math.max(0, 60 - 25 * s),
                                activity_regularity: Math.min(1, Math.max(0, 0.45 + 0.12 * b)) } });
      }
    }
    byPid[pid] = rows;
  }
  return { byPid, K,
           variables: [
             { id: "conversation_minutes", label: "Conversation minutes/day",
               note: "primary feature; falls as the construct rises" },
             { id: "activity_regularity", label: "Activity regularity",
               note: "weaker secondary feature" }],
           meta: { trueRho, nParticipants, nPerEpoch, placement, phi, seed, missingness } };
}

/** The three guided controls. Each states its expected verdict IN ADVANCE. */
export const CONTROLS = {
  shift: {
    id: "shift",
    name: "Known distribution shift",
    kind: "Positive control",
    expect: "Drift detected",
    blurb:
      "A cohort in which a 25% multiplicative recalibration is applied on purpose, " +
      "on top of a genuine change in the underlying construct. The pipeline should " +
      "detect it and should report an ATTENUATED estimate — nearer 1 than the true " +
      "0.75 — because the estimator is conservative by construction.",
    honesty:
      "Positive control: the change is known before analysis and is used to validate " +
      "the pipeline. Sized to be reliably detectable (8 of 8 seeds), which is what a " +
      "positive control is for. This is NOT a real-world research finding.",
    params: { trueRho: 0.75, nParticipants: 45, nPerEpoch: 300, placement: "skewed", phi: 0.4 },
  },
  stable: {
    id: "stable",
    name: "Stable control",
    kind: "Negative control",
    expect: "No meaningful drift",
    blurb:
      "The same generator with NO recalibration applied, but with a genuine change " +
      "in the construct still present. A method that confuses real change with " +
      "scale change fails here.",
    honesty:
      "Negative control: no recalibration exists in this data, though the underlying " +
      "construct genuinely does change. A detection here would be a false positive. " +
      "The test is calibrated to roughly a 5-10% false-positive rate, so changing the " +
      "seed in the sandbox will occasionally produce one — that is the nominal error " +
      "rate doing exactly what it should, not a defect.",
    params: { trueRho: 1.0, nParticipants: 40, nPerEpoch: 220, placement: "skewed", phi: 0.4 },
  },
  ambiguous: {
    id: "ambiguous",
    name: "Ambiguous / limited evidence",
    kind: "Boundary case",
    expect: "Insufficient evidence",
    blurb:
      "A small cohort with short windows, a floor-heavy scale and elevated " +
      "missingness — the conditions real studies often have. The honest answer " +
      "is that the data cannot settle the question.",
    honesty:
      "Boundary case: this exists to show that the system declines to answer when " +
      "the evidence is too thin, rather than producing a confident number.",
    params: { trueRho: 0.85, nParticipants: 14, nPerEpoch: 70,
              placement: "extreme_floor", phi: 0.6, missingness: 0.35 },
  },
};

export function buildControl(id, overrides = {}) {
  const c = CONTROLS[id];
  if (!c) throw new Error(`unknown control ${id}`);
  return { control: c, ...simulateCohort({ ...c.params, ...overrides }) };
}

/**
 * Perturbation catalogue. `expect` states, in advance, what the estimator SHOULD do.
 * Two of these are deliberate no-ops: because the feature is standardised inside each
 * window, any affine transform of it cancels before the slope is fitted. That is the
 * same invariance that lets the unknown per-participant gain drop out of the ratio,
 * so demonstrating it is the point, not a defect.
 */
export const PERTURBATIONS = [
  { id: "", label: "None", expect: "" },
  { id: "scale_shift", label: "Scale shift (feature x factor)", effect: "none",
    expect: "No change is expected. Multiplying the feature within a window is removed by " +
            "within-window standardisation, so the fitted slope, the ratio and the interval " +
            "should all be identical. An unchanged result here is the correct result." },
  { id: "location_shift", label: "Location shift (feature + offset)", effect: "none",
    expect: "No change is expected, for the same reason as the scale shift: adding a constant " +
            "inside a window is removed by standardisation. An unchanged result is correct." },
  { id: "gradual_drift", label: "Gradual drift", effect: "small",
    expect: "A small change is expected. The ramp is largely, but not entirely, absorbed by " +
            "standardisation, so the ratio should move slightly." },
  { id: "noise", label: "Added measurement noise", effect: "attenuates",
    expect: "The ratio should fall. Extra measurement error in the comparison window attenuates " +
            "the fitted slope, which is regression dilution. Note the consequence: added noise " +
            "is indistinguishable from genuine recalibration, and at a large enough magnitude " +
            "it will produce a drift verdict on data with no drift in it." },
  { id: "missingness", label: "Increased missingness", effect: "drops_obs",
    expect: "Observations are removed from the comparison window, so the interval widens on " +
            "average and participants fall below the observation floor once enough is lost. " +
            "The point estimate should not move systematically. At small magnitudes a single " +
            "random draw can narrow the interval by chance; the widening is a tendency, not a " +
            "guarantee for any one seed." },
];

export const PERTURBATION_BY_ID = Object.fromEntries(PERTURBATIONS.map((p) => [p.id, p]));

/**
 * Controlled perturbation applied to a COPY of a dataset (sandbox only).
 * Never mutates the source, and the caller must label the output.
 */
export function perturb(byPid, kind, magnitude, seed = 7) {
  const rand = rng(seed);
  const out = {};
  for (const [pid, rows] of Object.entries(byPid)) {
    out[pid] = rows.map((r) => {
      const row = { ...r };
      if (r.epoch !== 1) return row;                 // perturb the comparison window
      if (kind === "scale_shift") row.sensor = r.sensor * (1 + magnitude);
      else if (kind === "location_shift") row.sensor = r.sensor + magnitude * 20;
      else if (kind === "gradual_drift") {
        const frac = (r.t % 1e9) / Math.max(1, rows.length);
        row.sensor = r.sensor * (1 + magnitude * frac);
      } else if (kind === "noise") row.sensor = r.sensor + randn(rand) * magnitude * 20;
      return row;
    });
    if (kind === "missingness") {
      out[pid] = out[pid].filter((r) => r.epoch === 0 || rand() > magnitude);
    }
  }
  return out;
}
