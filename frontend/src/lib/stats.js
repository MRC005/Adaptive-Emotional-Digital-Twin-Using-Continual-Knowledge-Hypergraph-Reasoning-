/**
 * Numerical primitives for the ordinal probit slope-ratio estimator.
 *
 * This is a PORT of the validated Python implementation in
 * aedt/models/ordinal.py and aedt/estimators/slope_ratio.py. It exists so the
 * deployed application genuinely computes rather than displaying canned
 * answers. It is cross-checked against the Python original by
 * tests/regression/test_js_python_agreement.py, which runs fixed cases through
 * both and requires the slopes to agree to 1e-3.
 */

/** Normal CDF via Abramowitz & Stegun 7.1.26 on erf. Max abs error ~1.5e-7. */
export function normCdf(z) {
  const s = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + s * y);
}

/** Inverse normal CDF (Acklam's rational approximation, |err| < 1.15e-9). */
export function normPpf(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
             3.754408661907416];
  const pl = 0.02425;
  let q, r;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
export function sd(a, ddof = 1) {
  if (a.length <= ddof) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - ddof));
}
export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (i - lo) * (sorted[hi] - sorted[lo]);
}
export const median = (a) => quantile([...a].sort((x, y) => x - y), 0.5);

/** Deterministic PRNG (mulberry32) so every run in the UI is reproducible. */
export function rng(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
/** Box-Muller standard normal driven by `rand`. */
export function randn(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Nelder-Mead simplex minimiser.
 *
 * The Python original uses BFGS. A derivative-free method is used here because
 * the objective is cheap, low-dimensional (K parameters) and this avoids
 * hand-coding a gradient that could silently disagree with Python. Agreement
 * with the Python fit is asserted by the cross-validation test.
 */
export function nelderMead(fn, x0, { maxIter = 4000, tol = 1e-10 } = {}) {
  const n = x0.length;
  const simplex = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    p[i] = p[i] !== 0 ? p[i] * 1.05 : 0.00025;
    simplex.push(p);
  }
  let f = simplex.map(fn);
  const order = () => {
    const idx = f.map((v, i) => i).sort((a, b) => f[a] - f[b]);
    return [idx.map((i) => simplex[i]), idx.map((i) => f[i])];
  };
  for (let it = 0; it < maxIter; it++) {
    let s, fv;
    [s, fv] = order();
    for (let i = 0; i <= n; i++) { simplex[i] = s[i]; f[i] = fv[i]; }
    if (Math.abs(f[n] - f[0]) <= tol * (Math.abs(f[0]) + tol)) break;
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    const refl = centroid.map((c, j) => c + (c - simplex[n][j]));
    const fr = fn(refl);
    if (fr < f[0]) {
      const exp = centroid.map((c, j) => c + 2 * (c - simplex[n][j]));
      const fe = fn(exp);
      if (fe < fr) { simplex[n] = exp; f[n] = fe; } else { simplex[n] = refl; f[n] = fr; }
    } else if (fr < f[n - 1]) {
      simplex[n] = refl; f[n] = fr;
    } else {
      const con = centroid.map((c, j) => c + 0.5 * (simplex[n][j] - c));
      const fc = fn(con);
      if (fc < f[n]) { simplex[n] = con; f[n] = fc; }
      else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
          f[i] = fn(simplex[i]);
        }
      }
    }
  }
  const [s2, f2] = order();
  return { x: s2[0], fx: f2[0] };
}
