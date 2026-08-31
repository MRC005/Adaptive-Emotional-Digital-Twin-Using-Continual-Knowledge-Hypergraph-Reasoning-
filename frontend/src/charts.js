/**
 * Statistical plots. Conventional scientific styling: axis lines with ticks,
 * no fills for decoration, no rounded bars, no alpha washes. Every value comes
 * from the computation.
 */
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const P = () => ({ ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"),
  rule: css("--rule"), rule2: css("--rule-2"), accent: css("--accent"),
  warn: css("--warn"), stop: css("--stop"), s1: css("--s1"), s2: css("--s2") });

function setup(canvas, aspect) {
  const w = canvas.parentElement.clientWidth || 460;
  const h = Math.max(170, Math.round(w * aspect));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = "100%"; canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const x = canvas.getContext("2d"); x.scale(dpr, dpr); x.clearRect(0, 0, w, h);
  x.lineJoin = "round";
  return { x, w, h, p: P() };
}
const MONO = (s, wt) => `${wt || 400} ${s}px "IBM Plex Mono", monospace`;
const SANS = (s, wt) => `${wt || 400} ${s}px "IBM Plex Sans", sans-serif`;

/** Draw L-shaped axes with outward ticks — the convention in statistical graphics. */
function axes(x, p, m, pw, ph, xt, yt, xfmt, yfmt) {
  x.strokeStyle = p.ink3; x.lineWidth = 1; x.beginPath();
  x.moveTo(m.l, m.t); x.lineTo(m.l, m.t + ph); x.lineTo(m.l + pw, m.t + ph); x.stroke();
  x.font = MONO(10); x.fillStyle = p.ink3;
  x.textAlign = "center"; x.textBaseline = "top";
  for (const [v, px] of xt) {
    x.beginPath(); x.moveTo(px, m.t + ph); x.lineTo(px, m.t + ph + 4); x.stroke();
    x.fillText(xfmt(v), px, m.t + ph + 6);
  }
  x.textAlign = "right"; x.textBaseline = "middle";
  for (const [v, py] of yt) {
    x.beginPath(); x.moveTo(m.l - 4, py); x.lineTo(m.l, py); x.stroke();
    x.fillText(yfmt(v), m.l - 6, py);
  }
}
function axisLabels(x, p, m, pw, ph, h, xl, yl) {
  x.fillStyle = p.ink2; x.font = SANS(11);
  x.textAlign = "center"; x.textBaseline = "bottom";
  x.fillText(xl, m.l + pw / 2, h - 3);
  x.save(); x.translate(11, m.t + ph / 2); x.rotate(-Math.PI / 2);
  x.textAlign = "center"; x.textBaseline = "top"; x.fillText(yl, 0, 0); x.restore();
}

/** Fitted response functions, baseline vs comparison window. */
export function drawCurves(canvas, fits, expectedCategory, K) {
  const { x, w, h, p } = setup(canvas, 0.62);
  const m = { l: 42, r: 12, t: 10, b: 40 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const X = (v) => m.l + ((v + 2.4) / 4.8) * pw, Y = (v) => m.t + ph - ((v - 1) / (K - 1)) * ph;
  const xt = [-2, -1, 0, 1, 2].map((v) => [v, X(v)]);
  const yt = Array.from({ length: K }, (_, i) => [i + 1, Y(i + 1)]);
  axes(x, p, m, pw, ph, xt, yt, String, String);
  const grid = Array.from({ length: 61 }, (_, i) => -2.4 + (i * 4.8) / 60);
  [[fits[0], p.s1, []], [fits[1], p.s2, [6, 3]]].forEach(([f, col, dash]) => {
    if (!f?.converged) return;
    const ys = expectedCategory(f, grid);
    x.strokeStyle = col; x.lineWidth = 1.8; x.setLineDash(dash); x.beginPath();
    grid.forEach((xv, i) => { const px = X(xv), py = Y(ys[i]); i ? x.lineTo(px, py) : x.moveTo(px, py); });
    x.stroke(); x.setLineDash([]);
  });
  // in-plot key, plain text, no badges
  x.font = MONO(10); x.textAlign = "left"; x.textBaseline = "top";
  x.strokeStyle = p.s1; x.lineWidth = 1.8;
  x.beginPath(); x.moveTo(m.l + 8, m.t + 8); x.lineTo(m.l + 26, m.t + 8); x.stroke();
  x.fillStyle = p.ink2; x.fillText("baseline", m.l + 30, m.t + 3);
  x.strokeStyle = p.s2; x.setLineDash([6, 3]);
  x.beginPath(); x.moveTo(m.l + 8, m.t + 22); x.lineTo(m.l + 26, m.t + 22); x.stroke(); x.setLineDash([]);
  x.fillText("comparison", m.l + 30, m.t + 17);
  axisLabels(x, p, m, pw, ph, h, "feature (standardised within window)", "expected report");
}

/** Per-participant estimates against the cohort interval. */
export function drawForest(canvas, per, primary) {
  const { x, w, h, p } = setup(canvas, 0.6);
  const m = { l: 42, r: 14, t: 10, b: 40 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const vals = per.map((d) => d.rhoStar);
  const lo = Math.min(0.7, ...vals) - 0.04, hi = Math.max(1.3, ...vals) + 0.04;
  const X = (v) => m.l + ((v - lo) / (hi - lo)) * pw;
  const n = per.length, step = ph / Math.max(n, 1);
  const ticks = [];
  for (let v = Math.ceil(lo * 10) / 10; v <= hi; v += 0.2) ticks.push([v, X(v)]);
  const yt = [[1, m.t + step * 0.5], [n, m.t + step * (n - 0.5)]];
  axes(x, p, m, pw, ph, ticks, yt, (v) => v.toFixed(1), (v) => String(v));
  if (primary && isFinite(primary.ciLow)) {
    x.strokeStyle = p.accent; x.lineWidth = 1; x.setLineDash([3, 3]);
    [primary.ciLow, primary.ciHigh].forEach((v) => {
      x.beginPath(); x.moveTo(X(v), m.t); x.lineTo(X(v), m.t + ph); x.stroke();
    });
    x.setLineDash([]);
  }
  x.strokeStyle = p.ink3; x.lineWidth = 1;
  x.beginPath(); x.moveTo(X(1), m.t); x.lineTo(X(1), m.t + ph); x.stroke();
  const sorted = [...per].sort((a, b) => a.rhoStar - b.rhoStar);
  x.strokeStyle = p.ink2; x.lineWidth = 1;
  sorted.forEach((d, i) => {
    const cy = m.t + step * (i + 0.5), cx = X(d.rhoStar);
    x.beginPath(); x.moveTo(cx - 2.5, cy); x.lineTo(cx + 2.5, cy);
    x.moveTo(cx, cy - 2.5); x.lineTo(cx, cy + 2.5); x.stroke();
  });
  if (primary) {
    x.strokeStyle = p.accent; x.lineWidth = 2;
    x.beginPath(); x.moveTo(X(primary.rhoStar), m.t); x.lineTo(X(primary.rhoStar), m.t + ph); x.stroke();
  }
  axisLabels(x, p, m, pw, ph, h, "ρ* per participant", "participant (sorted)");
}

/** Category frequencies in each window. */
export function drawUsage(canvas, byPid, K) {
  const { x, w, h, p } = setup(canvas, 0.62);
  const m = { l: 42, r: 12, t: 10, b: 40 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const c = [new Array(K).fill(0), new Array(K).fill(0)];
  for (const rows of Object.values(byPid))
    for (const r of rows) if (r.report >= 1 && r.report <= K) c[r.epoch][r.report - 1]++;
  const tot = c.map((a) => a.reduce((s, v) => s + v, 0) || 1);
  const mx = Math.max(...c.map((a, i) => Math.max(...a.map((v) => v / tot[i])))) || 1;
  const Y = (f) => m.t + ph - (f / mx) * ph;
  const bw = pw / K;
  const yt = [0, mx / 2, mx].map((v) => [v, Y(v)]);
  axes(x, p, m, pw, ph, Array.from({ length: K }, (_, k) => [k + 1, m.l + bw * (k + 0.5)]),
       yt, String, (v) => (100 * v).toFixed(0) + "%");
  for (let k = 0; k < K; k++) {
    [0, 1].forEach((e) => {
      const f = c[e][k] / tot[e];
      x.fillStyle = e === 0 ? p.s1 : p.s2;
      x.fillRect(m.l + k * bw + bw * (0.16 + e * 0.34), Y(f), bw * 0.3, m.t + ph - Y(f));
    });
  }
  x.font = MONO(10); x.textAlign = "left"; x.textBaseline = "top";
  x.fillStyle = p.s1; x.fillRect(m.l + 8, m.t + 7, 9, 9);
  x.fillStyle = p.ink2; x.fillText("baseline", m.l + 21, m.t + 6);
  x.fillStyle = p.s2; x.fillRect(m.l + 78, m.t + 7, 9, 9);
  x.fillStyle = p.ink2; x.fillText("comparison", m.l + 91, m.t + 6);
  axisLabels(x, p, m, pw, ph, h, "response category", "share of responses");
}

/** One participant's series with the window boundary marked. */
export function drawTimeline(canvas, rows, featureLabel) {
  const { x, w, h, p } = setup(canvas, 0.44);
  const m = { l: 46, r: 12, t: 10, b: 40 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const ts = rows.map((r) => r.t), ss = rows.map((r) => r.sensor);
  const t0 = Math.min(...ts), t1 = Math.max(...ts);
  const s0 = Math.min(...ss), s1 = Math.max(...ss);
  const X = (v) => m.l + ((v - t0) / Math.max(t1 - t0, 1)) * pw;
  const Y = (v) => m.t + ph - ((v - s0) / Math.max(s1 - s0, 1e-9)) * ph;
  const xt = [t0, t0 + (t1 - t0) / 2, t1].map((v) => [v, X(v)]);
  const yt = [s0, (s0 + s1) / 2, s1].map((v) => [v, Y(v)]);
  axes(x, p, m, pw, ph, xt, yt, (v) => String(Math.round(v)), (v) => v.toFixed(1));
  const split = rows.find((r) => r.epoch === 1);
  if (split) {
    x.strokeStyle = p.ink3; x.setLineDash([4, 3]); x.lineWidth = 1;
    x.beginPath(); x.moveTo(X(split.t), m.t); x.lineTo(X(split.t), m.t + ph); x.stroke();
    x.setLineDash([]);
    x.fillStyle = p.ink3; x.font = MONO(9.5); x.textAlign = "left"; x.textBaseline = "top";
    x.fillText("window split", X(split.t) + 4, m.t + 2);
  }
  rows.forEach((r) => {
    x.fillStyle = r.epoch === 0 ? p.s1 : p.s2;
    x.fillRect(X(r.t) - 1, Y(r.sensor) - 1, 2, 2);
  });
  axisLabels(x, p, m, pw, ph, h, "observation index", featureLabel || "feature");
}
