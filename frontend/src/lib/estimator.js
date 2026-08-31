/**
 * Ordinal probit slope-ratio estimator — JavaScript port.
 *
 * Faithful to aedt/models/ordinal.py and aedt/estimators/slope_ratio.py:
 *
 *   P(R <= k | x) = Phi(c_k - beta_e * x)      fitted per person PER EPOCH
 *   rho*_p        = beta_p2 / beta_p1
 *   log rho*      = mean_p log rho*_p          bootstrap over PARTICIPANTS
 *
 * Design rules carried over verbatim, because they are load-bearing:
 *   - the sensor covariate is standardised WITHIN EACH EPOCH separately;
 *   - the two epochs are fitted INDEPENDENTLY (no constant crosses the split);
 *   - beta is NOT constrained to be positive, only well determined, and the
 *     two epochs must share a sign or the ratio is uninterpretable;
 *   - the bootstrap resamples PARTICIPANTS, never observations.
 *
 * rho itself is not point-identified. 1 - rho* is a LOWER BOUND on the true
 * multiplicative recalibration, and the additive component is not identified
 * at all — neither is ever reported here.
 */
import { mean, median, nelderMead, normCdf, normPpf, quantile, rng, sd } from "./stats.js";

export const THRESHOLDS = {
  MIN_REPORTS_PER_EPOCH: 60,
  MIN_CATEGORIES_USED: 2,
  MIN_SENSOR_SD: 0.1,
  VAR_RATIO_LO: 0.25,
  VAR_RATIO_HI: 4.0,
  MIN_ABS_BETA: 0.02,
  MIN_PARTICIPANTS_FOR_CI: 10,
  BOOTSTRAP: 2000,
};

/** Unpack (c0, log-increments) so cutpoints are ordered by construction. */
function cutsFrom(par, K) {
  const c = [par[0]];
  for (let i = 1; i <= K - 2; i++) c.push(c[c.length - 1] + Math.exp(par[i]));
  return c;
}

function negLogLik(par, x, y, K) {
  const cuts = cutsFrom(par, K), beta = par[par.length - 1];
  let nll = 0;
  for (let i = 0; i < x.length; i++) {
    const k = y[i], bx = beta * x[i];
    const lo = k > 1 ? normCdf(cuts[k - 2] - bx) : 0;
    const hi = k < K ? normCdf(cuts[k - 1] - bx) : 1;
    nll -= Math.log(Math.max(hi - lo, 1e-12));
  }
  return nll;
}

/** Fit one person-epoch. Returns {converged:false, reason} rather than a guess. */
export function ordinalProbitFit(x, y, K, minAbsBeta = THRESHOLDS.MIN_ABS_BETA) {
  const n = y.length;
  if (n === 0) return { converged: false, reason: "no observations", beta: NaN };
  if (new Set(y).size < 2)
    return { converged: false, reason: "fewer than 2 response categories used", beta: NaN };
  if (sd(x) < 1e-9)
    return { converged: false, reason: "no variation in the sensor covariate", beta: NaN };

  const counts = new Array(K + 1).fill(0);
  for (const v of y) counts[v]++;
  const start = [];
  let cum = 0;
  const q = [];
  for (let k = 1; k <= K - 1; k++) { cum += counts[k]; q.push(Math.min(Math.max(cum / n, 0.01), 0.99)); }
  const c = q.map(normPpf);
  start.push(c[0]);
  for (let i = 1; i < c.length; i++) start.push(Math.log(Math.max(c[i] - c[i - 1], 1e-3)));
  start.push(0.5);

  const res = nelderMead((p) => negLogLik(p, x, y, K), start);
  const beta = res.x[res.x.length - 1];
  if (!isFinite(beta)) return { converged: false, reason: "optimiser diverged", beta: NaN };
  if (Math.abs(beta) < minAbsBeta)
    return { converged: false, beta: NaN,
             reason: `|beta| = ${Math.abs(beta).toFixed(4)} below the floor ${minAbsBeta}` };
  return { converged: true, beta, cutpoints: cutsFrom(res.x, K), n, K, logLik: -res.fx, reason: "" };
}

/** x = (s - mean(s_e)) / SD(s_e), from THIS epoch only. Required, not optional. */
export function standardiseWithinEpoch(s) {
  const m = mean(s), d = sd(s);
  if (d < 1e-12) return { x: s.map(() => 0), mean: m, sd: d };
  return { x: s.map((v) => (v - m) / d), mean: m, sd: d };
}

export function expectedCategory(fit, grid) {
  return grid.map((xv) => {
    const cum = fit.cutpoints.map((ck) => normCdf(ck - fit.beta * xv));
    let e = 0, prev = 0;
    for (let k = 0; k < cum.length; k++) { e += (k + 1) * (cum[k] - prev); prev = cum[k]; }
    e += fit.K * (1 - prev);
    return e;
  });
}

/** Pre-specified eligibility screen. Every exclusion carries its reason. */
export function screenParticipant(rows, K, th = THRESHOLDS) {
  const reasons = [];
  const ep = [rows.filter((r) => r.epoch === 0), rows.filter((r) => r.epoch === 1)];
  const info = { n: [ep[0].length, ep[1].length], categories: [0, 0], varRatio: NaN,
                 floor: [NaN, NaN] };
  for (let e = 0; e < 2; e++) {
    const R = ep[e].map((r) => r.report), S = ep[e].map((r) => r.sensor);
    info.categories[e] = new Set(R).size;
    info.floor[e] = R.length ? R.filter((v) => v === 1).length / R.length : NaN;
    if (R.length < th.MIN_REPORTS_PER_EPOCH)
      reasons.push(`epoch ${e + 1}: ${R.length} reports < ${th.MIN_REPORTS_PER_EPOCH}`);
    if (info.categories[e] < th.MIN_CATEGORIES_USED)
      reasons.push(`epoch ${e + 1}: only ${info.categories[e]} response categories used (A5)`);
    const dS = sd(S);
    if (!(dS > 0)) reasons.push(`epoch ${e + 1}: no sensor variation (A5)`);
  }
  const v0 = Math.pow(sd(ep[0].map((r) => r.sensor)), 2);
  const v1 = Math.pow(sd(ep[1].map((r) => r.sensor)), 2);
  if (v0 > 0) {
    info.varRatio = v1 / v0;
    if (info.varRatio < th.VAR_RATIO_LO || info.varRatio > th.VAR_RATIO_HI)
      reasons.push(`Var(s) epoch ratio ${info.varRatio.toFixed(2)} outside ` +
                   `[${th.VAR_RATIO_LO}, ${th.VAR_RATIO_HI}] — assumption A3 not supported`);
  }
  return { eligible: reasons.length === 0, reasons, ...info };
}

/** log(beta2/beta1) for one participant, or a stated reason it is unusable. */
export function personLogRatio(rows, K, th = THRESHOLDS) {
  const fits = [];
  for (let e = 0; e < 2; e++) {
    const ep = rows.filter((r) => r.epoch === e);
    if (ep.length < 5) return { value: NaN, reason: `epoch ${e + 1}: only ${ep.length} observations`, fits };
    const { x } = standardiseWithinEpoch(ep.map((r) => r.sensor));
    fits.push(ordinalProbitFit(x, ep.map((r) => r.report), K, th.MIN_ABS_BETA));
  }
  if (!fits[0].converged) return { value: NaN, reason: `epoch 1: ${fits[0].reason}`, fits };
  if (!fits[1].converged) return { value: NaN, reason: `epoch 2: ${fits[1].reason}`, fits };
  if (Math.sign(fits[0].beta) !== Math.sign(fits[1].beta))
    return { value: NaN, fits,
             reason: `sensor-report slope flips sign between epochs ` +
                     `(b1=${fits[0].beta.toFixed(3)}, b2=${fits[1].beta.toFixed(3)}); ` +
                     `the ratio is not interpretable` };
  return { value: Math.log(fits[1].beta / fits[0].beta), reason: "", fits };
}

/** Percentile CI on exp(mean(log rho*)), resampling PARTICIPANTS. */
export function bootstrapParticipants(logs, B = 999, seed = 20260828) {
  const v = logs.filter(Number.isFinite);
  const P = v.length;
  if (P < THRESHOLDS.MIN_PARTICIPANTS_FOR_CI)
    return { nParticipants: P, nResamples: 0, point: P ? Math.exp(mean(v)) : NaN,
             ciLow: NaN, ciHigh: NaN, resamplingUnit: "participant" };
  const rand = rng(seed), means = [];
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < P; i++) s += v[(rand() * P) | 0];   // whole participants
    means.push(s / P);
  }
  means.sort((a, b) => a - b);
  return { nParticipants: P, nResamples: B, point: Math.exp(mean(v)),
           ciLow: Math.exp(quantile(means, 0.025)),
           ciHigh: Math.exp(quantile(means, 0.975)),
           resamplingUnit: "participant" };
}

/** Contiguous epoch-1 split-half negative control. Runs BEFORE the primary. */
export function placeboSplitHalf(byPid, K, B = 999, seed = 20260828, th = THRESHOLDS) {
  const logs = [];
  for (const rows of Object.values(byPid)) {
    const e0 = rows.filter((r) => r.epoch === 0).sort((a, b) => a.t - b.t);
    if (e0.length < 2 * th.MIN_REPORTS_PER_EPOCH) continue;
    const h = Math.floor(e0.length / 2);
    const pseudo = e0.map((r, i) => ({ ...r, epoch: i < h ? 0 : 1 }));
    logs.push(personLogRatio(pseudo, K, th).value);
  }
  const usable = logs.filter(Number.isFinite);
  if (usable.length < th.MIN_PARTICIPANTS_FOR_CI)
    return { runnable: false, rejected: false, nParticipants: usable.length,
             rhoStar: NaN, ciLow: NaN, ciHigh: NaN,
             verdict: `NOT RUNNABLE: only ${usable.length} participants have the ` +
                      `${2 * th.MIN_REPORTS_PER_EPOCH} epoch-1 observations needed to ` +
                      `split epoch 1 in half; at least ${th.MIN_PARTICIPANTS_FOR_CI} are required.` };
  const u = bootstrapParticipants(usable, B, seed);
  const rejected = !(u.ciLow <= 1 && 1 <= u.ciHigh);
  return { runnable: true, rejected, nParticipants: u.nParticipants,
           rhoStar: u.point, ciLow: u.ciLow, ciHigh: u.ciHigh,
           verdict: rejected
             ? "REJECTS — the estimator fires where no change can exist. The primary analysis is NOT run."
             : "does not reject — the primary analysis may proceed" };
}

/**
 * The full pipeline: screen -> placebo -> primary. Returns a typed verdict.
 *
 * Statuses are deliberately four-valued. "Insufficient evidence" is a first-
 * class outcome, not a failure to produce one.
 */
export const STATUS = {
  DRIFT: "DRIFT_DETECTED",
  STABLE: "NO_MEANINGFUL_DRIFT",
  INSUFFICIENT: "INSUFFICIENT_EVIDENCE",
  QUALITY: "DATA_QUALITY_ISSUE",
};

export function runPipeline(byPid, K, opts = {}) {
  const th = { ...THRESHOLDS, ...(opts.thresholds || {}) };
  const B = opts.bootstrap ?? 999;
  const seed = opts.seed ?? 20260828;
  const t0 = performance.now();

  const screened = Object.entries(byPid).map(([pid, rows]) => ({
    pid, ...screenParticipant(rows, K, th),
  }));
  const eligiblePids = screened.filter((s) => s.eligible).map((s) => s.pid);
  const kept = Object.fromEntries(eligiblePids.map((p) => [p, byPid[p]]));

  const base = {
    nScreened: screened.length, nEligible: eligiblePids.length,
    screened, thresholds: th, bootstrap: B, seed, K,
    elapsedMs: () => Math.round(performance.now() - t0),
  };

  if (eligiblePids.length < th.MIN_PARTICIPANTS_FOR_CI) {
    return { ...base, status: STATUS.INSUFFICIENT, placebo: null, primary: null,
             elapsedMs: base.elapsedMs(),
             headline: "Insufficient evidence",
             why: `Only ${eligiblePids.length} of ${screened.length} participants passed the ` +
                  `eligibility screen; at least ${th.MIN_PARTICIPANTS_FOR_CI} are required to form ` +
                  `a participant-clustered interval. No estimate was produced.` };
  }

  const placebo = placeboSplitHalf(kept, K, B, seed, th);
  if (placebo.rejected || !placebo.runnable) {
    return { ...base, status: placebo.rejected ? STATUS.QUALITY : STATUS.INSUFFICIENT,
             placebo, primary: null, elapsedMs: base.elapsedMs(),
             headline: placebo.rejected ? "Data quality issue" : "Insufficient evidence",
             why: placebo.rejected
               ? "The negative control fired: the estimator reports a change on a split of the " +
                 "baseline window, where no change can exist. The primary analysis was not run."
               : placebo.verdict };
  }

  const logs = [], pids = [], exclusions = [];
  for (const pid of eligiblePids) {
    const r = personLogRatio(byPid[pid], K, th);
    if (Number.isFinite(r.value)) { logs.push(r.value); pids.push(pid); }
    else exclusions.push({ pid, reason: r.reason });
  }
  if (logs.length < th.MIN_PARTICIPANTS_FOR_CI) {
    return { ...base, status: STATUS.INSUFFICIENT, placebo, primary: null,
             exclusions, elapsedMs: base.elapsedMs(),
             headline: "Insufficient evidence",
             why: `${logs.length} participants produced an interpretable ratio; ` +
                  `at least ${th.MIN_PARTICIPANTS_FOR_CI} are required.` };
  }

  const u = bootstrapParticipants(logs, B, seed);
  const excludesNull = !(u.ciLow <= 1 && 1 <= u.ciHigh);
  const perParticipant = pids.map((pid, i) => ({ pid, rhoStar: Math.exp(logs[i]) }));
  return {
    ...base, status: excludesNull ? STATUS.DRIFT : STATUS.STABLE,
    placebo, exclusions, elapsedMs: base.elapsedMs(),
    primary: { rhoStar: u.point, ciLow: u.ciLow, ciHigh: u.ciHigh,
               medianRhoStar: Math.exp(median(logs)), excludesNull,
               lowerBound: 1 - u.point, nUsed: u.nParticipants, perParticipant },
    headline: excludesNull ? "Drift detected" : "No detectable drift",
    why: excludesNull
      ? `The sensor-to-report relationship is ${u.point.toFixed(3)} times as strong in the ` +
        `comparison window as in the baseline, and the 95% interval excludes 1.`
      : `The 95% interval [${u.ciLow.toFixed(3)}, ${u.ciHigh.toFixed(3)}] includes 1, so the ` +
        `data do not support a claim that the relationship changed.`,
  };
}
