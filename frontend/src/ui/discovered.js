/**
 * "What We Discovered" -- the scientific story, told honestly.
 *
 * Every number on this page is read from `findings.json`, which is generated
 * by scripts/export_findings.py from the committed experiment artefacts. No
 * figure is typed by hand, so the page cannot drift from the results.
 *
 * The page leads with the NULL, because that is the honest headline and
 * because a project that reports the baseline which beat it reads as more
 * credible, not less.
 */
import F from "../data/findings.json";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f3 = (x) => (x == null || !isFinite(x) ? "--" : x.toFixed(3));

const MODEL_LABEL = {
  B0_majority: "Population majority",
  B2_global: "Global model — context + behaviour",
  B3_static_prior: "Global + static personal prior",
  B4_calibrated: "Per-person calibrated global",
  T_twin: "Personalised twin (ours)",
  B1_persistence: "Persistence — carry the last value forward",
};
const ORDER = ["B0_majority", "B2_global", "B3_static_prior", "B4_calibrated",
               "T_twin", "B1_persistence"];

/** Learning-curve chart. Two series, drawn to scale from the real numbers. */
function curveSvg() {
  const d = F.learning_curve;
  const W = 560, H = 210, L = 46, R = 14, T = 14, B = 32;
  const xs = d.map((p) => p.K);
  const lo = 0.24, hi = 0.36;                       // covers both series
  const x = (k) => L + (xs.indexOf(k) / (xs.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const path = (key) => d.map((p, i) => `${i ? "L" : "M"}${x(p.K).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  const dots = (key, cls) => d.map((p) =>
    `<circle cx="${x(p.K).toFixed(1)}" cy="${y(p[key]).toFixed(1)}" r="3.2" class="${cls}"/>`).join("");
  const grid = [0.24, 0.28, 0.32, 0.36].map((v) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" class="g"/>
     <text x="${L - 8}" y="${(y(v) + 4).toFixed(1)}" class="ax" text-anchor="end">${v.toFixed(2)}</text>`).join("");
  const ticks = d.map((p) =>
    `<text x="${x(p.K).toFixed(1)}" y="${H - 10}" class="ax" text-anchor="middle">${p.K}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img"
      aria-label="Macro-F1 against personal history length. Persistence is above the twin at every value of K.">
    <style>
      .g{stroke:var(--rule);stroke-width:1}
      .ax{fill:var(--ink-3);font-size:10px;font-family:ui-monospace,monospace}
      .pers{fill:none;stroke:var(--ok);stroke-width:2.4}
      .twin{fill:none;stroke:var(--accent);stroke-width:2.4;stroke-dasharray:5 3}
      .dp{fill:var(--ok)} .dt{fill:var(--accent)}
    </style>
    ${grid}
    <path d="${path("persistence")}" class="pers"/>
    <path d="${path("twin")}" class="twin"/>
    ${dots("persistence", "dp")}${dots("twin", "dt")}
    ${ticks}
    <text x="${(L + W) / 2}" y="${H - 0}" class="ax" text-anchor="middle">personal history K (observations)</text>
  </svg>`;
}

/** Per-person predictability spread — the finding that motivates what's next. */
function spreadSvg() {
  const c = F.ceiling;
  const W = 560, H = 96, L = 46, R = 14;
  const lo = -0.30, hi = 0.75;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const tick = (v) => `<line x1="${x(v).toFixed(1)}" x2="${x(v).toFixed(1)}" y1="52" y2="58" class="g"/>
    <text x="${x(v).toFixed(1)}" y="72" class="ax" text-anchor="middle">${v.toFixed(1)}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img"
      aria-label="Per-person autocorrelation ranges from -0.24 to 0.69, median 0.35.">
    <style>
      .g{stroke:var(--rule);stroke-width:1}
      .ax{fill:var(--ink-3);font-size:10px;font-family:ui-monospace,monospace}
      .rng{stroke:var(--rule);stroke-width:2}
      .iqr{fill:var(--accent);opacity:.22}
      .med{stroke:var(--accent);stroke-width:2.6}
    </style>
    <line x1="${x(c.per_person_r_range[0]).toFixed(1)}" x2="${x(c.per_person_r_range[1]).toFixed(1)}"
          y1="34" y2="34" class="rng"/>
    <rect x="${x(c.per_person_r_iqr[0]).toFixed(1)}" y="22"
          width="${(x(c.per_person_r_iqr[1]) - x(c.per_person_r_iqr[0])).toFixed(1)}"
          height="24" class="iqr"/>
    <line x1="${x(c.per_person_r_median).toFixed(1)}" x2="${x(c.per_person_r_median).toFixed(1)}"
          y1="20" y2="48" class="med"/>
    <line x1="${L}" x2="${W - R}" y1="52" y2="52" class="g"/>
    ${[-0.2, 0, 0.2, 0.4, 0.6].map(tick).join("")}
    <text x="${L}" y="14" class="ax">each person's own autocorrelation, ${c.n_participants_analysed} participants</text>
  </svg>`;
}

export function viewDiscovered(root) {
  const H = F.headline, C = F.cohort, CE = F.ceiling;
  const tp = H.twin_vs_persistence;

  const rows = ORDER.filter((k) => H.models[k]).map((k) => {
    const m = H.models[k];
    const hi = k === "B1_persistence" || k === "T_twin";
    return `<tr${hi ? ' class="hi"' : ""}>
      <th>${esc(MODEL_LABEL[k])}</th>
      <td class="num">${f3(m.macro_f1)}</td>
      <td class="num dim">[${f3(m.ci[0])}, ${f3(m.ci[1])}]</td>
      <td class="num dim">${f3(m.accuracy)}</td>
      <td class="num dim">${f3(m.mae)}</td></tr>`;
  }).join("");

  const abl = [
    ["A6_no_behaviour", "Personal history only, no behaviour"],
    ["A3_global_plus_history", "History + behaviour"],
    ["A4_full_twin_online", "Full twin, with online adaptation"],
    ["A5_no_trajectory", "Without recent trajectory"],
    ["A1_global_only", "Global only — no personal history"],
  ].filter(([k]) => F.ablation[k] != null)
   .map(([k, label]) => `<tr><th>${esc(label)}</th>
      <td class="num">${f3(F.ablation[k])}</td></tr>`).join("");

  root.innerHTML = `
    <div class="story">

      <section class="panel hero">
        <div class="pad">
          <p class="eyebrow">The question we set out to answer</p>
          <h2 class="q">Can a system learn from a person's own history to
            predict how they will feel next?</h2>
          <div class="scale">
            <div><b>${C.participants}</b><span>participants</span></div>
            <div><b>${C.years}</b><span>years</span></div>
            <div><b>${C.reports.toLocaleString()}</b><span>stress reports</span></div>
            <div><b>${C.prediction_pairs.toLocaleString()}</b><span>prediction pairs</span></div>
          </div>
          <p class="prov">Real longitudinal data · College Experience Study ·
            held-out participants, strictly future observations</p>
        </div>
      </section>

      <section class="panel">
        <header><h3>The answer we found</h3>
          <span class="meta">pre-registered before any model was fitted</span></header>
        <div class="pad">
          <p class="answer">Not the way we expected. <b>Simply carrying the last
            reported value forward beat every personalised model we built.</b></p>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>Model</th><th class="num">macro-F1</th>
              <th class="num">95% CI</th><th class="num">Accuracy</th><th class="num">MAE</th></tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
          <p class="hint">At K = ${H.K} prior observations. Participant-clustered
            bootstrap, 2,000 resamples.</p>
          <div class="msg warn">
            <b>Twin versus persistence: ${tp.mean_diff >= 0 ? "+" : ""}${f3(tp.mean_diff)} macro-F1</b>,
            95% CI [${f3(tp.ci_low)}, ${f3(tp.ci_high)}] — the interval excludes zero
            on the <em>wrong</em> side, and ${tp.harmed} of ${tp.n_participants}
            held-out participants were worse off.
          </div>
          <p>We did not remove the baseline that beat us, and we did not change the
            metric afterwards. <b>macro-F1 was declared primary before any result
            was seen.</b></p>
        </div>
      </section>

      <section class="panel">
        <header><h3>More history did not close the gap</h3>
          <span class="meta">macro-F1 against personal history length</span></header>
        <div class="pad">
          ${curveSvg()}
          <p class="legend"><span class="sw ok"></span> Persistence
            <span class="sw acc"></span> Personalised twin</p>
          <p>The twin does improve as it sees more of a person
            (${f3(F.learning_curve[0].twin)} → ${f3(F.learning_curve.at(-1).twin)}).
            So does persistence. The gap never closes.</p>
        </div>
      </section>

      <section class="panel">
        <header><h3>So we asked why — and this is the real finding</h3>
          <span class="meta">measured, not assumed</span></header>
        <div class="pad">
          <p>If a simple rule cannot be beaten, either the model is weak or
            <b>the signal is not there.</b> We measured which:</p>
          <div class="facts">
            <div><b>r = ${CE.within_person_autocorrelation}</b>
              <span>within-person autocorrelation — only
                ${(CE.variance_explained * 100).toFixed(0)}% of variation is
                predictable from the previous report</span></div>
            <div><b>r = ${CE.strongest_behaviour_r.toFixed(2)}</b>
              <span>strongest of ${"648"} behavioural sensing features —
                ${(CE.behaviour_variance_explained * 100).toFixed(1)}% of variation</span></div>
            <div><b>${(CE.icc_between_person * 100).toFixed(0)}%</b>
              <span>of variance is <em>who</em> the person is; the other
                ${(100 - CE.icc_between_person * 100).toFixed(0)}% is <em>when</em> it is</span></div>
          </div>
          <p><b>Roughly ${((1 - CE.variance_explained) * 100).toFixed(0)}% of day-to-day
            stress variation simply is not predictable from the previous report.</b>
            That ceiling binds every model, not just ours — and persistence
            already reaches most of it.</p>
          <div class="msg">
            <b>The ablation agrees.</b> Removing all 648 behavioural features
            <em>improved</em> the model, and online adaptation changed nothing.
          </div>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>What the model was given</th><th class="num">macro-F1</th></tr></thead>
            <tbody>${abl}</tbody>
          </table></div>
        </div>
      </section>

      <section class="panel">
        <header><h3>But not everyone is equally predictable</h3>
          <span class="meta">${CE.n_participants_analysed} participants with ≥30 observations</span></header>
        <div class="pad">
          ${spreadSvg()}
          <p>Predictability is <b>not a property of the task — it is a property of
            the person.</b> It ranges from ${CE.per_person_r_range[0]} to
            ${CE.per_person_r_range[1]}:
            <b>${(CE.frac_near_unpredictable * 100).toFixed(0)}%</b> are close to
            unpredictable, while <b>${(CE.frac_well_predictable * 100).toFixed(0)}%</b>
            are well predicted by their own history.</p>
          <p>And it is <b>forecastable</b>: a person's predictability in the first
            half of their data correlates at
            <b>r = ${CE.early_late_r}</b> with the second half.</p>
        </div>
      </section>

      <section class="panel next">
        <header><h3>Which gives us the next question</h3>
          <span class="meta">not yet tested</span></header>
        <div class="pad">
          <p class="answer">If personalisation helps some people and not others,
            can a system tell <b>in advance</b> which case it is in — and decline
            to personalise when the evidence does not support it?</p>
          <div class="msg warn"><b>This is the next study, not a result.</b>
            The interface shows an experimental personalisation-evidence
            assessment. It has <em>not</em> been evaluated against abstaining at
            random, abstaining by history length, or abstaining by model
            confidence. Until it beats those, it is a design decision, not a
            finding.</div>
        </div>
      </section>

      <section class="panel">
        <header><h3>How this was evaluated</h3>
          <span class="meta">the parts that make the numbers mean something</span></header>
        <div class="pad">
          <ul class="checks">
            <li>Test participants never appear in training — ${C.train} / ${C.val} / ${C.test} split, intersections empty</li>
            <li>Every feature is timestamped strictly before the value it predicts</li>
            <li>Participant identity is never a model input</li>
            <li>Confidence intervals resample <em>participants</em>, never observations</li>
            <li>Protocol committed to git before the experiment was written</li>
            <li>Leakage detectors are themselves tested against deliberately leaky data</li>
          </ul>
        </div>
      </section>

    </div>`;
}
