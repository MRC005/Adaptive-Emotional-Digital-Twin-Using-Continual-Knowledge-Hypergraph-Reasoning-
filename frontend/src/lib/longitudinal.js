/**
 * The longitudinal chart. Real observations, drawn honestly.
 *
 * THREE RULES THIS FILE EXISTS TO ENFORCE
 *
 * 1. NO INTERPOLATION ACROSS GAPS. Reports are sparse and irregular. A line
 *    joining two points 40 days apart claims a measurement that was never
 *    taken. The observed path therefore BREAKS wherever consecutive reports
 *    are more than `MAX_JOIN_DAYS` apart, and the break is visible.
 * 2. NO CONFIDENCE BANDS. The underlying output carries no per-observation
 *    uncertainty, so none is drawn. A shaded band here would be decoration
 *    presented as statistics.
 * 3. NO PREDICTION LINE WITHOUT PREDICTIONS. Warm-up observations are
 *    personalisation-only and were never scored; nothing is drawn for them,
 *    and a participant the protocol did not score gets a stated absence
 *    rather than an empty chart that looks like a failure.
 *
 * Rendered as inline SVG so it inherits the page's theme tokens and needs no
 * charting dependency.
 */

/** Consecutive reports further apart than this are NOT joined by a line. */
export const MAX_JOIN_DAYS = 7;          // the prediction task's own horizon

/**
 * Above this many visible scored points, per-observation prediction marks are
 * not drawn. At roughly 4px per point, three extra marks per observation
 * collapse into vertical smears that read as noise rather than as data. This
 * is a legibility rule applied to every participant identically -- it never
 * depends on whether the twin did well, the observed series is always drawn in
 * full, and the narrower views bring the marks back.
 */
export const MARK_LIMIT = 70;

export const VIEWS = {
  full: { label: "Full history", hint: "every report, no marks when too dense" },
  evaluation: { label: "Evaluation window",
                hint: "the most recent scored observations, with predictions" },
  recent: { label: "Recent", hint: "last 60 days of reports" },
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Rows for a view. Never resamples; only selects. */
export function rowsForView(series, view) {
  if (view === "evaluation") {
    // The most RECENT scored observations, capped so the per-point marks can
    // actually be resolved. It is a stated window, not a hidden subset: the
    // count is printed beside the chart, and Full history still shows every
    // report. Taking the tail is fixed for every participant -- it is never
    // chosen by how the twin performed.
    const scored = series.filter((s) => !s.warmup);
    return scored.slice(-MARK_LIMIT);
  }
  if (view === "recent") {
    if (!series.length) return [];
    const last = series[series.length - 1].day;
    return series.filter((s) => s.day >= last - 60);
  }
  return series;
}

/** Break a run of points wherever the gap exceeds the join rule. */
function segments(points) {
  const out = [];
  let run = [];
  for (const p of points) {
    if (run.length && p.day - run[run.length - 1].day > MAX_JOIN_DAYS) {
      out.push(run); run = [];
    }
    run.push(p);
  }
  if (run.length) out.push(run);
  return out;
}

/**
 * Draw. Returns SVG markup; the caller attaches hover behaviour to `.lg-hit`
 * rectangles, each carrying the data it describes in data- attributes.
 */
export function longitudinalSvg(series, opts = {}) {
  const { view = "full", showTwin = true, showPersistence = true,
          height = 260, title = "" } = opts;
  const rows = rowsForView(series, view);
  const evalPoints = rows.filter((p) => p.predicted != null).length;
  const marksFit = evalPoints > 0 && evalPoints <= MARK_LIMIT;

  if (!rows.length) {
    return `<p class="lg-empty">No reports fall inside this view.</p>`;
  }

  const W = 760, H = height, L = 42, R = 16, T = 18, B = 34;
  const d0 = rows[0].day, d1 = rows[rows.length - 1].day;
  const span = Math.max(1, d1 - d0);
  const x = (d) => L + ((d - d0) / span) * (W - L - R);
  // stress is an ordinal 1-5 scale; the axis is fixed so two participants are
  // always drawn on the same scale and can be compared by eye
  const y = (v) => T + (1 - (v - 1) / 4) * (H - T - B);

  const grid = [1, 2, 3, 4, 5].map((v) =>
    `<line class="lg-grid" x1="${L}" x2="${W - R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>
     <text class="lg-ax" x="${L - 8}" y="${(y(v) + 3.5).toFixed(1)}" text-anchor="end">${v}</text>`
  ).join("");

  const ticks = 5;
  const xlab = Array.from({ length: ticks }, (_, i) => {
    const d = Math.round(d0 + (i / (ticks - 1)) * span);
    return `<text class="lg-ax" x="${x(d).toFixed(1)}" y="${H - 12}" text-anchor="middle">day ${d}</text>`;
  }).join("");

  // observed path, broken across gaps
  const obs = segments(rows).map((seg) =>
    `<path class="lg-obs" d="${seg.map((p, i) =>
      `${i ? "L" : "M"}${x(p.day).toFixed(1)},${y(p.observed).toFixed(1)}`).join(" ")}"/>`
  ).join("");

  const warmEnd = rows.find((s) => !s.warmup);
  const warmBand = warmEnd && view === "full"
    ? `<rect class="lg-warm" x="${L}" y="${T}" width="${Math.max(0, x(warmEnd.day) - L).toFixed(1)}" height="${H - T - B}"/>
       <text class="lg-ax lg-warmlab" x="${L + 5}" y="${T + 12}">warm-up · never scored</text>`
    : "";

  const dots = rows.map((p) => {
    const cls = p.warmup ? "lg-dot lg-dot-warm" : "lg-dot";
    return `<circle class="${cls}" cx="${x(p.day).toFixed(1)}" cy="${y(p.observed).toFixed(1)}" r="2.6"/>`;
  }).join("");

  const marks = !marksFit ? "" : rows.filter((p) => p.predicted != null).map((p) => {
    const px = x(p.day).toFixed(1);
    const twin = showTwin
      ? `<path class="lg-twin" d="M${px - 4},${y(p.predicted).toFixed(1)} l4,-4 l4,4 l-4,4 Z"/>` : "";
    const pers = showPersistence
      ? `<rect class="lg-pers" x="${px - 3}" y="${(y(p.baseline) - 3).toFixed(1)}" width="6" height="6"/>` : "";
    const actual = `<circle class="lg-actual" cx="${px}" cy="${y(p.actual_next).toFixed(1)}" r="3.4"/>`;
    return twin + pers + actual;
  }).join("");

  const hits = rows.map((p) => {
    const w = Math.max(6, (W - L - R) / rows.length);
    return `<rect class="lg-hit" x="${(x(p.day) - w / 2).toFixed(1)}" y="${T}"
      width="${w.toFixed(1)}" height="${H - T - B}"
      data-day="${p.day}" data-observed="${p.observed}"
      data-warmup="${p.warmup ? 1 : 0}"
      ${p.predicted != null ? `data-predicted="${p.predicted}" data-actual="${p.actual_next}" data-baseline="${p.baseline}"` : ""}
      tabindex="0" role="img"
      aria-label="Day ${p.day}, reported stress ${p.observed}${p.predicted != null
        ? `, stand-in forecast ${p.predicted}, persistence predicted ${p.baseline}, actual next ${p.actual_next}` : ", warm-up"}"/>`;
  }).join("");

  return `
  <figure class="lg-fig">
    ${title ? `<figcaption class="lg-title">${esc(title)}</figcaption>` : ""}
    <svg class="lg-svg" viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Reported stress over time, ${rows.length} reports">
      ${warmBand}${grid}${xlab}${obs}${dots}${marks}${hits}
      ${evalPoints > MARK_LIMIT ? `<text class="lg-ax lg-dense" x="${W - R}"
        y="${T - 5}" text-anchor="end">${evalPoints} scored points — too dense
        for per-point marks; see the Evaluation window</text>` : ""}
      <text class="lg-ax lg-ylab" transform="rotate(-90)" x="${-(H / 2)}" y="12"
            text-anchor="middle">reported stress</text>
    </svg>
    <div class="lg-tip" hidden></div>
  </figure>`;
}

/** Legend, listing only the marks actually drawn. */
export function legend({ hasPredictions }, marksDrawn = true) {
  const items = [
    `<span class="lg-key"><i class="k-obs"></i>reported stress</span>`,
    `<span class="lg-key"><i class="k-warm"></i>warm-up (never scored)</span>`,
  ];
  if (hasPredictions && marksDrawn) {
    items.push(
      `<span class="lg-key"><i class="k-actual"></i>actual next value</span>`,
      `<span class="lg-key"><i class="k-twin"></i>stand-in forecast</span>`,
      `<span class="lg-key"><i class="k-pers"></i>persistence prediction</span>`);
  }
  return `<p class="lg-legend">${items.join("")}</p>`;
}

/** Do the per-point marks fit in this view? Mirrors the drawing rule. */
export function marksVisible(series, view) {
  const n = rowsForView(series, view).filter((p) => p.predicted != null).length;
  return n > 0 && n <= MARK_LIMIT;
}

/** Wire hover/focus tooltips onto a rendered figure. */
export function attachTooltips(root) {
  for (const fig of root.querySelectorAll(".lg-fig")) {
    const tip = fig.querySelector(".lg-tip");
    if (!tip) continue;
    const show = (el) => {
      const d = el.dataset;
      const lines = [`<b>Day ${d.day}</b>`,
                     `reported <b>${d.observed}</b>`];
      if (d.predicted != null) {
        lines.push(`actual next <b>${d.actual}</b>`,
                   `stand-in forecast <b>${d.predicted}</b>`,
                   `persistence <b>${d.baseline}</b>`);
      } else if (d.warmup === "1") {
        lines.push(`<span class="dim">warm-up — not scored</span>`);
      }
      tip.innerHTML = lines.join("<br>");
      tip.hidden = false;
      const r = el.getBoundingClientRect(), f = fig.getBoundingClientRect();
      tip.style.left = `${Math.min(f.width - 130, Math.max(0, r.left - f.left))}px`;
    };
    const hide = () => { tip.hidden = true; };
    fig.addEventListener("mouseover", (e) => {
      if (e.target.classList.contains("lg-hit")) show(e.target);
    });
    fig.addEventListener("focusin", (e) => {
      if (e.target.classList.contains("lg-hit")) show(e.target);
    });
    fig.addEventListener("mouseleave", hide);
    fig.addEventListener("focusout", hide);
  }
}
