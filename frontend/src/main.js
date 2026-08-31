/**
 * AEDT — analysis application.
 *
 * Layered by design. The first screen is one paragraph of plain English and
 * three ways in; statistics, thresholds and window rules stay behind
 * disclosures until someone asks for them. Every result leads with a plain
 * verdict and only then shows the evidence that produced it.
 *
 * WHAT IS COMPUTED WHERE, stated once and never blurred:
 *   guided controls, the sandbox, and any CSV opened here run LIVE in this
 *   browser on the JavaScript estimator;
 *   bundled real-dataset results were computed OFFLINE by
 *   scripts/export_real_results.py against archives that cannot be
 *   redistributed, and are displayed here rather than recomputed.
 */
import { runPipeline, personLogRatio, THRESHOLDS } from "./lib/estimator.js";
import { CONTROLS, buildControl, perturb, PERTURBATIONS, PERTURBATION_BY_ID }
  from "./lib/scenarios.js";
import { parseCsv, suggestColumns, toRecords, validate } from "./lib/dataset.js";
import { analyze, activeEngine } from "./lib/engine.js";
import { renderReport, renderPrecomputed } from "./ui/results.js";
import { initTheme, getTheme, setTheme, onThemeChange } from "./lib/theme.js";
import REAL from "./data/real_datasets.json";

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const SENS = {
  standard:   { minRep: THRESHOLDS.MIN_REPORTS_PER_EPOCH, boot: THRESHOLDS.BOOTSTRAP },
  permissive: { minRep: 30, boot: 1000 },
  strict:     { minRep: 100, boot: THRESHOLDS.BOOTSTRAP },
};

const state = {
  tab: "home",
  mode: "guided",
  scenario: "shift",
  variable: "conversation_minutes",
  focusPid: "",
  sensitivity: "standard",
  windowSplit: "own-span",
  perturbKind: "", perturbMag: 0.3, seed: 20260828,
  perturbation: null,
  nParticipants: 45, nPerEpoch: 300,
  loaded: null,
  csv: null,
  realDataset: "college_experience",
  runs: [],
  sandboxBaseline: null,
  busy: false,
  showAdvanced: false,
};

/* ------------------------------------------------------------------ chrome */
function setBusy(on, label) {
  state.busy = on;
  const e = $("#engine");
  e.classList.toggle("busy", on);
  e.lastElementChild.textContent = on ? (label || "Working") : "Ready";
}

function fmtTime(d) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/* ---------------------------------------------------------------- data prep */
function applyVariable() {
  const ds = state.loaded;
  if (!ds || !ds.raw) return;
  const v = state.variable;
  ds.byPid = {};
  for (const [pid, rows] of Object.entries(ds.raw))
    ds.byPid[pid] = rows.map((r) => ({ ...r, sensor: r.features ? r.features[v] ?? r.sensor : r.sensor }));
  applyWindows(ds.byPid);
  const pt = state.perturbation;
  if (pt && pt.kind) ds.byPid = perturb(ds.byPid, pt.kind, pt.magnitude, pt.seed);
}

function applyWindows(byPid) {
  if (state.windowSplit === "own-span") {
    for (const rows of Object.values(byPid)) {
      rows.sort((a, b) => a.t - b.t);
      const lo = rows[0].t, hi = rows[rows.length - 1].t, mid = lo + (hi - lo) / 2;
      rows.forEach((r) => (r.epoch = r.t > mid ? 1 : 0));
    }
  } else {
    for (const pid of Object.keys(byPid)) {
      const rows = byPid[pid].sort((a, b) => a.t - b.t);
      const n = rows.length, a = Math.floor(n * 0.4), b = Math.ceil(n * 0.6);
      byPid[pid] = rows.map((r, i) => ({ ...r, epoch: i < a ? 0 : i >= b ? 1 : -1 }))
                       .filter((r) => r.epoch >= 0);
    }
  }
}

function loadCohort(label, built, source) {
  state.perturbation = null;
  const raw = built.byPid;
  state.loaded = { label, raw, byPid: {}, K: built.K,
                   variables: built.variables || [{ id: "sensor", label: "Sensor" }], source };
  state.variable = state.loaded.variables[0].id;
  applyVariable();
  state.focusPid = Object.keys(state.loaded.byPid)[0];
}

/* ------------------------------------------------------------------ execute */
async function execute(extraMessages = "", ctxExtra = {}) {
  if (state.busy || !state.loaded) return null;
  const out = $("#out");
  setBusy(true, "Computing");
  out.innerHTML = `<div class="panel"><div class="pad running">
    <span class="spin" aria-hidden="true"></span>
    <span>Fitting each participant, then resampling participants for the interval…</span>
    </div></div>`;
  try {
    const sens = SENS[state.sensitivity];
    const byPid = state.loaded.byPid;
    const r = await analyze({ byPid, K: state.loaded.K,
      options: { bootstrap: sens.boot, seed: state.seed,
                 thresholds: { MIN_REPORTS_PER_EPOCH: sens.minRep } } });
    const varLabel = (state.loaded.variables.find((v) => v.id === state.variable) || {}).label
                     || state.variable;
    let messages = extraMessages;
    if (state.sensitivity !== "standard")
      messages += `<div class="msg warn"><b>Sensitivity setting, not the primary rule.</b>
        The minimum observations per window was changed from the pre-specified
        ${SENS.standard.minRep} to ${sens.minRep}.</div>`;
    if (state.windowSplit !== "own-span")
      messages += `<div class="msg warn"><b>Non-default periods.</b> Comparing the first 40%
        with the last 40% of each participant's observations; the middle 20% is excluded.</div>`;

    renderReport(out, r, {
      byPid, personLogRatio, focusPid: state.focusPid, featureLabel: varLabel, messages,
      header: { dataset: state.loaded.label, feature: varLabel,
                windows: state.windowSplit === "own-span"
                  ? "Halves of each participant's own observation span"
                  : "First 40% vs last 40% of each participant's observations",
                time: new Date().toLocaleString() },
      onFocusChange: (pid) => { state.focusPid = pid; },
      ...ctxExtra });

    state.runs.unshift({
      at: new Date(), dataset: state.loaded.label, feature: varLabel,
      participants: r.nScreened, eligible: r.nEligible, status: r.status,
      headline: r.headline, rho: r.primary ? r.primary.rhoStar : null,
      ci: r.primary ? [r.primary.ciLow, r.primary.ciHigh] : null,
      sensitivity: state.sensitivity,
      windows: state.windowSplit, mode: state.mode,
      ms: r.wallMs ?? r.elapsedMs, result: r, ctx: { ...ctxExtra, messages },
    });
    out.scrollIntoView({ block: "start", behavior: "smooth" });
    return r;
  } catch (e) {
    out.innerHTML = `<div class="msg stop"><b>The analysis could not finish.</b>
      ${esc(e.message)}<br><span class="hint">Nothing was saved. Adjust the inputs and run again.</span></div>`;
    return null;
  } finally { setBusy(false); }
}

/* --------------------------------------------------------------------- home */
function viewHome() {
  const ce = REAL.datasets.find((d) => d.id === "college_experience");
  const primary = ce && ce.runs.find((r) => r.primary);
  const withEstimate = ce ? ce.runs.filter((r) => r.rho_star != null) : [];

  $("#work").innerHTML = `
    <section class="hero">
      <h1>What this tool does</h1>
      <p class="lede">People answer questions like &ldquo;how stressed are you right now?&rdquo;
        over months. The same answer can come to mean something different over time. This tool
        checks whether the relationship between someone's passive phone data and their repeated
        self&#8209;report <b>changes</b> between an earlier and a later period.</p>
      <p class="lede2">It reports one of four things, and refusing to answer is a real answer:
        <b>drift detected</b>, <b>no detectable drift</b>, <b>insufficient evidence</b>, or
        <b>data incompatible with this analysis</b>.</p>
    </section>

    <h2 class="sech">Three ways to start</h2>
    <div class="starts">
      <div class="start">
        <h3>1. Try a controlled example</h3>
        <p>Data built so the right answer is known in advance. The quickest way to see whether
          the method behaves. <b>Start here.</b></p>
        <button class="b run" data-go="guided">Run a guided example</button>
      </div>
      <div class="start">
        <h3>2. Look at real data</h3>
        <p>Results from real longitudinal studies, and the audit explaining which of them can
          support this analysis at all. You can also open your own CSV.</p>
        <button class="b" data-go="real">Open real data</button>
      </div>
      <div class="start">
        <h3>3. Experiment</h3>
        <p>Change the data on purpose and watch what the method does. Useful for seeing where
          it breaks.</p>
        <button class="b" data-go="sandbox">Open the sandbox</button>
      </div>
    </div>

    <h2 class="sech">Where the real data stands today</h2>
    <div class="panel"><div class="pad">
      ${primary ? `
        <p style="margin-top:0"><b>College Experience Study</b> (218 students, four years) is the
          first archive of the four audited that carries enough repeated measurement to attempt
          this analysis. Under the pre-specified primary configuration it still reports
          <b>${esc(primary.headline.toLowerCase())}</b>: ${primary.eligible} participants passed
          the screen and at least 10 are needed.</p>
        <p>${withEstimate.length} pre-specified secondary configurations did produce an estimate,
          and both report <b>no detectable drift</b> with wide intervals
          ${withEstimate.map((r) => `<span class="mono">${r.rho_star.toFixed(3)}
            [${r.ci_low.toFixed(3)}, ${r.ci_high.toFixed(3)}]</span>`).join(" and ")}.</p>
        <p class="note">No drift has been demonstrated in real data by this project, and no
          threshold was changed to obtain a result. The full audit, including the two datasets
          that cannot support the method and why, is on the
          <a href="#" data-tab-link="datasets">Datasets</a> page.</p>
      ` : `<p class="note">Real-dataset results have not been generated in this build.</p>`}
    </div></div>`;

  for (const b of document.querySelectorAll("[data-go]"))
    b.addEventListener("click", () => {
      state.mode = b.dataset.go;
      if (b.dataset.go === "real") state.realSub = "bundled";
      go("analyze");
    });
  for (const a of document.querySelectorAll("[data-tab-link]"))
    a.addEventListener("click", (e) => { e.preventDefault(); go(a.dataset.tabLink); });
}

/* ------------------------------------------------------------------ analyze */
function modeBar() {
  const modes = [["guided", "Guided example"], ["real", "Real data"], ["sandbox", "Sandbox"]];
  return `<div class="segmented" role="tablist" aria-label="Analysis mode">
    ${modes.map(([id, label]) => `<button role="tab" aria-selected="${state.mode === id}"
      class="${state.mode === id ? "on" : ""}" data-mode="${id}">${label}</button>`).join("")}
  </div>`;
}

function advancedPanel() {
  return `
    <details class="adv"${state.showAdvanced ? " open" : ""}>
      <summary>Advanced settings</summary>
      <div class="form">
        <div class="lbl"><label for="a-win">Periods compared</label></div>
        <div class="ctl"><select id="a-win">
          <option value="own-span"${state.windowSplit === "own-span" ? " selected" : ""}>
            Earlier half vs later half of each person's own timeline</option>
          <option value="first-last"${state.windowSplit === "first-last" ? " selected" : ""}>
            First 40% vs last 40%, skipping the middle</option>
        </select></div>
        <div class="lbl"><label for="a-sens">Evidence threshold</label></div>
        <div class="ctl"><select id="a-sens">
          <option value="standard"${state.sensitivity === "standard" ? " selected" : ""}>
            Standard — 60 observations per period (pre-specified)</option>
          <option value="permissive"${state.sensitivity === "permissive" ? " selected" : ""}>
            Permissive — 30 (sensitivity analysis only)</option>
          <option value="strict"${state.sensitivity === "strict" ? " selected" : ""}>
            Strict — 100 (sensitivity analysis only)</option>
        </select></div>
      </div>
      <p class="hint">The standard setting is the one fixed before any data was examined.
        The others are labelled as sensitivity analyses wherever they appear in a result.</p>
    </details>`;
}

function wireAdvanced() {
  const d = document.querySelector("details.adv");
  if (d) d.addEventListener("toggle", () => { state.showAdvanced = d.open; });
  const w = $("#a-win"), s = $("#a-sens");
  if (w) w.addEventListener("change", (e) => {
    state.windowSplit = e.target.value; applyVariable();
  });
  if (s) s.addEventListener("change", (e) => { state.sensitivity = e.target.value; });
}

function viewAnalyze() {
  $("#work").innerHTML = `${modeBar()}<div id="modebody"></div>`;
  for (const b of document.querySelectorAll("[data-mode]"))
    b.addEventListener("click", () => { state.mode = b.dataset.mode; viewAnalyze(); });
  ({ guided: viewGuided, real: viewReal, sandbox: viewSandbox }[state.mode])();
}

/* ------------------------------------------------------- mode: guided */
function viewGuided() {
  const s = CONTROLS[state.scenario];
  $("#modebody").innerHTML = `
    <div class="panel">
      <header><h3>Guided example</h3>
        <span class="meta">the answer is known before the analysis runs</span></header>
      <div class="pad">
        <p class="intro">Each example is data generated with a property deliberately built in.
          Because the right answer is known, these check that the method behaves — they are
          <b>not</b> findings about real people.</p>
        <div class="form">
          <div class="lbl"><label for="g-sc">Example</label></div>
          <div class="ctl"><select id="g-sc">
            ${Object.values(CONTROLS).map((c) => `<option value="${c.id}"${c.id === state.scenario ? " selected" : ""}>${esc(c.name)}</option>`).join("")}
          </select></div>
          <div class="lbl">In plain English</div>
          <div class="ctl"><p class="plain">${esc(s.plain || s.description)}</p></div>
          <div class="lbl">What should happen</div>
          <div class="ctl"><p class="expect"><b>${esc(s.expected)}</b></p></div>
        </div>
        ${advancedPanel()}
      </div>
      <div class="actions">
        <button class="b run" id="g-run">Run analysis</button>
        <span class="hint">Runs in this browser. Takes a few seconds.</span>
      </div>
    </div>
    <div id="out"></div>`;
  wireAdvanced();
  $("#g-sc").addEventListener("change", (e) => { state.scenario = e.target.value; viewGuided(); });
  $("#g-run").addEventListener("click", () => {
    loadCohort(`${s.name} (controlled example)`,
      buildControl(state.scenario, { nParticipants: state.nParticipants,
                                     nPerEpoch: state.nPerEpoch, seed: state.seed }),
      "guided");
    execute(`<div class="msg"><b>This is a control, not a finding.</b> ${esc(s.honesty)}</div>`);
  });
}

/* --------------------------------------------------------- mode: real data */
function viewReal() {
  const sub = state.realSub || "bundled";
  $("#modebody").innerHTML = `
    <div class="panel">
      <header><h3>Real data</h3><span class="meta">audited archives, or your own file</span></header>
      <div class="pad">
        <div class="segmented sub" role="tablist">
          <button role="tab" class="${sub === "bundled" ? "on" : ""}" data-sub="bundled">Study datasets</button>
          <button role="tab" class="${sub === "upload" ? "on" : ""}" data-sub="upload">Open my CSV</button>
        </div>
        <div id="realbody"></div>
      </div>
    </div>`;
  for (const b of document.querySelectorAll("[data-sub]"))
    b.addEventListener("click", () => { state.realSub = b.dataset.sub; viewReal(); });
  (sub === "bundled" ? viewBundled : viewUpload)();
}

function viewBundled() {
  const ds = REAL.datasets.find((d) => d.id === state.realDataset) || REAL.datasets[0];
  $("#realbody").innerHTML = `
    <div class="form">
      <div class="lbl"><label for="rb-ds">Dataset</label></div>
      <div class="ctl"><select id="rb-ds">
        ${REAL.datasets.map((d) => `<option value="${d.id}"${d.id === ds.id ? " selected" : ""}>${esc(d.name)} — ${esc(d.status)}</option>`).join("")}
      </select></div>
    </div>
    <div class="msg"><b>Displayed, not recomputed here.</b> These archives are gigabytes and are
      licensed for research use, not redistribution, so they are not shipped to your browser.
      The numbers below were computed offline by the Python implementation against the local
      archive, with the seed and settings shown. Guided examples, the sandbox and your own CSV
      are computed live in this browser.</div>
    <div id="rbout"></div>`;
  $("#rb-ds").addEventListener("change", (e) => { state.realDataset = e.target.value; viewBundled(); });
  renderPrecomputed($("#rbout"), ds, REAL);
}

function viewUpload() {
  $("#realbody").innerHTML = `
    <ol class="steps">
      <li class="step"><span class="n">1</span>
        <div><b>Choose a file</b>
          <p class="hint">A CSV with one row per observation. It needs a person identifier, a
            time, a whole-number self-report (for example 1–5), and one numeric sensing
            measure.</p>
          <input type="file" id="u-file" accept=".csv,text/csv">
          <p class="hint">Read in this browser. Nothing is uploaded anywhere.</p>
        </div></li>
    </ol>
    <div id="u-rest"></div>`;
  $("#u-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const text = await f.text();
      ingest(text, f.name);
    } catch (err) {
      $("#u-rest").innerHTML = `<div class="msg stop"><b>Could not read that file.</b> ${esc(err.message)}</div>`;
    }
  });
  if (state.csv) renderUploadRest();
}

function ingest(text, name) {
  const { header, rows } = parseCsv(text);
  if (!header.length || !rows.length) {
    $("#u-rest").innerHTML = `<div class="msg stop"><b>That file has no readable rows.</b>
      A header line and at least one data row are required.</div>`;
    return;
  }
  const records = toRecords(header, rows);
  state.csv = { name, header, records, cols: suggestColumns(header, records) };
  renderUploadRest();
}

function renderUploadRest() {
  const c = state.csv;
  const sample = c.records.slice(0, 6);
  const pick = (id, sel) => `<select id="${id}">
      <option value="">— none —</option>
      ${c.header.map((h) => `<option value="${h}"${h === sel ? " selected" : ""}>${esc(h)}</option>`).join("")}
    </select>`;

  $("#u-rest").innerHTML = `
    <ol class="steps" start="2">
      <li class="step"><span class="n">2</span>
        <div><b>Check it looks right</b>
          <p class="hint">${esc(c.name)} — ${c.records.length.toLocaleString()} rows,
            ${c.header.length} columns. First few rows:</p>
          <div class="gridwrap"><table class="grid"><thead><tr>
            ${c.header.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>
            ${sample.map((r) => `<tr>${c.header.map((h) => `<td class="mono">${esc(r[h] ?? "")}</td>`).join("")}</tr>`).join("")}
          </tbody></table></div>
        </div></li>
      <li class="step"><span class="n">3</span>
        <div><b>Say which column is which</b>
          <p class="hint">Guessed from the column names. Correct anything that is wrong.</p>
          <div class="form">
            <div class="lbl"><label for="m-pid">Person</label></div><div class="ctl">${pick("m-pid", c.cols.participant)}</div>
            <div class="lbl"><label for="m-time">Time</label></div><div class="ctl">${pick("m-time", c.cols.time)}</div>
            <div class="lbl"><label for="m-report">Self-report answer</label></div><div class="ctl">${pick("m-report", c.cols.report)}</div>
            <div class="lbl"><label for="m-sensor">Passive sensing measure</label></div><div class="ctl">${pick("m-sensor", c.cols.sensor)}</div>
          </div>
          ${advancedPanel()}
          <button class="b" id="m-check">Check this file</button>
        </div></li>
    </ol>
    <div id="r-val"></div><div id="out"></div>`;
  wireAdvanced();
  $("#m-check").addEventListener("click", runValidation);
}

function runValidation() {
  const cols = { participant: $("#m-pid").value, time: $("#m-time").value,
                 report: $("#m-report").value, sensor: $("#m-sensor").value };
  state.csv.cols = cols;
  const v = validate(state.csv.records, cols,
                     { thresholds: { MIN_REPORTS_PER_EPOCH: SENS[state.sensitivity].minRep } });
  state.csv.validation = v;
  const icon = { pass: "✓", warn: "!", fail: "✗" };
  const word = { pass: "passed", warn: "warning", fail: "failed" };

  $("#r-val").innerHTML = `
    <div class="panel">
      <header><h3>Step 4 — File check</h3>
        <span class="meta">${v.ready ? "this file can be analysed" : "this file cannot be analysed"}</span></header>
      <div class="pad">
        ${v.checks.map((c) => `<div class="chk">
          <span class="m ${c.status}" aria-hidden="true">${icon[c.status]}</span>
          <span class="vh">${word[c.status]}:</span>
          <span><b>${esc(c.name)}</b> — ${esc(c.detail)}</span></div>`).join("")}
        ${v.ready ? "" : `<p class="note">These are structural problems with the file, not
          settings to adjust. No estimate will be produced from it, because producing one would
          be misleading.</p>`}
      </div>
      <div class="actions">
        <button class="b run" id="r-run"${v.ready ? "" : " disabled"}>Run analysis</button>
        <span class="hint">${v.ready
          ? `${v.summary.nParticipants} people, ${v.summary.nRows.toLocaleString()} observations,
             ${v.K} answer categories.`
          : "Fix the failures above, then check again."}</span>
      </div>
    </div>`;
  if (v.ready) $("#r-run").addEventListener("click", () => {
    state.loaded = { label: state.csv.name, raw: v.byPid, byPid: v.byPid, K: v.K,
                     variables: [{ id: "sensor", label: state.csv.cols.sensor }],
                     source: "upload" };
    state.variable = "sensor";
    applyWindows(state.loaded.byPid);
    state.focusPid = Object.keys(state.loaded.byPid)[0];
    execute("", { scaleUnverified: true });
  });
}

/* ----------------------------------------------------------- mode: sandbox */
function sandboxKey() {
  return [state.scenario, state.nParticipants, state.nPerEpoch, state.seed,
          state.variable, state.windowSplit, state.sensitivity].join("|");
}

function countObs(byPid) {
  return Object.values(byPid).reduce((n, rows) => n + rows.length, 0);
}

function renderPerturbationCheck(base, now) {
  const spec = PERTURBATION_BY_ID[state.perturbKind];
  if (!spec) return;
  const bp = base.result.primary, np = now.primary;
  const nowObs = countObs(state.loaded.byPid);
  const num = (v) => (v == null || !isFinite(v) ? "--" : v.toFixed(4));
  const ciOf = (p) => (p ? `[${p.ciLow.toFixed(3)}, ${p.ciHigh.toFixed(3)}]` : "--");
  const widthOf = (p) => (p ? p.ciHigh - p.ciLow : NaN);
  const d = bp && np ? np.rhoStar - bp.rhoStar : NaN;
  const dw = widthOf(np) - widthOf(bp);

  let verdict, ok;
  if (!bp || !np) {
    ok = null;
    verdict = "One of the two runs did not produce an estimate, so there is nothing to compare.";
  } else if (spec.effect === "none") {
    ok = Math.abs(d) < 1e-6;
    verdict = ok
      ? "The estimate is unchanged to within 1e-6, which is the expected result. The feature is "
        + "rescaled within each period before fitting, so this change cancels out entirely."
      : "The estimate moved, which was not expected for a change of this kind.";
  } else if (spec.effect === "small") {
    ok = Math.abs(d) < 0.05;
    verdict = ok ? "The estimate moved slightly, as expected."
                 : "The estimate moved more than expected for this change.";
  } else if (spec.effect === "attenuates") {
    ok = d < 0;
    verdict = ok
      ? "The ratio fell, as expected: extra measurement noise flattens the fitted relationship. "
        + "This is why the method cannot tell added noise apart from genuine recalibration."
      : "The ratio did not fall, which was not expected for added noise.";
  } else {
    ok = nowObs < base.nObs;
    const pct = base.nObs ? (100 * (base.nObs - nowObs) / base.nObs).toFixed(1) : "--";
    verdict = ok
      ? `${(base.nObs - nowObs).toLocaleString()} observations (${pct}%) were removed. `
        + `The interval ${dw > 0 ? "widened" : "did not widen on this seed, which a single draw can do"}`
        + ` (width ${widthOf(bp).toFixed(4)} to ${widthOf(np).toFixed(4)}).`
      : "No observations were removed, which was not expected.";
  }

  const mark = ok === null ? "" : ok ? "matches the stated expectation"
                                     : "DOES NOT match the stated expectation";
  const row = (a, b2, c) => `<tr><th>${a}</th><td class="mono num">${b2}</td><td class="mono num">${c}</td></tr>`;
  const el = (r) => `${r.nEligible} of ${r.nScreened}`;

  $("#out").insertAdjacentHTML("beforeend", `
    <div class="panel">
      <header><h3>Did it do what it should?</h3>
        <span class="meta">expectation stated before the run, checked after it</span></header>
      <div class="pad">
        <p class="hint" style="margin-top:0">Expected: ${esc(spec.expect)}</p>
        <div class="gridwrap"><table class="grid">
          <caption class="vh">Unperturbed versus perturbed result</caption>
          <thead><tr><th></th><th class="num">Original</th><th class="num">Changed</th></tr></thead>
          <tbody>
            ${row("Ratio &rho;*", num(bp && bp.rhoStar), num(np && np.rhoStar))}
            ${row("95% interval", ciOf(bp), ciOf(np))}
            ${row("Interval width", num(widthOf(bp)), num(widthOf(np)))}
            ${row("Observations used", base.nObs.toLocaleString(), nowObs.toLocaleString())}
            ${row("People included", el(base.result), el(now))}
            ${row("Verdict", esc(base.result.headline), esc(now.headline))}
          </tbody>
        </table></div>
        <p style="margin-bottom:0">Change in &rho;*:
          <span class="mono">${isFinite(d) ? (d >= 0 ? "+" : "") + d.toFixed(6) : "--"}</span>.
          ${mark ? `<b>Result ${esc(mark)}.</b>` : ""} ${esc(verdict)}</p>
      </div>
    </div>`);
}

function viewSandbox() {
  $("#modebody").innerHTML = `
    <div class="panel">
      <header><h3>Sandbox</h3><span class="meta">change the data on purpose, then re-run</span></header>
      <div class="pad">
        <p class="intro">Start from a controlled example, apply a deliberate change to a copy of
          it, and compare. Each change comes with a statement of what the method
          <i>should</i> do, written before you run it.</p>
        <div class="form">
          <div class="lbl"><label for="x-base">Starting data</label></div>
          <div class="ctl"><select id="x-base">
            ${Object.values(CONTROLS).map((c) => `<option value="${c.id}"${c.id === state.scenario ? " selected" : ""}>${esc(c.name)}</option>`).join("")}
          </select></div>
          <div class="lbl"><label for="x-pert">Change to apply</label></div>
          <div class="ctl"><select id="x-pert">
            ${PERTURBATIONS.map((p) => `<option value="${p.id}"${p.id === state.perturbKind ? " selected" : ""}>${esc(p.label)}</option>`).join("")}
          </select></div>
          <div class="lbl"></div>
          <div class="ctl"><p class="hint" id="x-expect">${esc(PERTURBATION_BY_ID[state.perturbKind]?.expect
            || "Pick a change to see what the method is expected to do.")}</p></div>
          <div class="lbl"><label for="x-mag">How strong</label></div>
          <div class="ctl mag"><input type="range" id="x-mag" min="0.05" max="0.9" step="0.05"
              value="${state.perturbMag}"><span class="mono" id="x-magv">${state.perturbMag.toFixed(2)}</span></div>
        </div>
        ${advancedPanel()}
      </div>
      <div class="actions">
        <button class="b" id="x-load">Load starting data</button>
        <button class="b" id="x-apply" disabled>Apply the change</button>
        <button class="b run" id="x-run"${state.loaded && String(state.loaded.source).startsWith("sandbox") ? "" : " disabled"}>Run analysis</button>
        <span class="hint" id="x-hint">Changes are applied to a copy. The source is never modified.</span>
      </div>
    </div>
    <div id="out"></div>`;
  wireAdvanced();
  $("#x-mag").addEventListener("input", (e) => {
    state.perturbMag = +e.target.value; $("#x-magv").textContent = state.perturbMag.toFixed(2);
  });
  $("#x-pert").addEventListener("change", (e) => {
    state.perturbKind = e.target.value;
    $("#x-apply").disabled = !e.target.value || !state.loaded;
    $("#x-expect").textContent = PERTURBATION_BY_ID[state.perturbKind]?.expect
      || "Pick a change to see what the method is expected to do.";
  });
  $("#x-load").addEventListener("click", () => {
    state.scenario = $("#x-base").value;
    loadCohort(`${CONTROLS[state.scenario].name} (unchanged copy)`,
      buildControl(state.scenario, { nParticipants: state.nParticipants,
                                     nPerEpoch: state.nPerEpoch, seed: state.seed }),
      "sandbox:clean");
    state.sandboxBaseline = null;
    viewSandbox();
    $("#x-hint").textContent = "Loaded. Run it once to record a baseline, then apply a change.";
  });
  $("#x-apply").addEventListener("click", () => {
    if (!state.loaded || !state.perturbKind) return;
    state.perturbation = { kind: state.perturbKind, magnitude: state.perturbMag, seed: state.seed };
    state.loaded.label = `${CONTROLS[state.scenario].name} — changed copy (${state.perturbKind})`;
    state.loaded.source = "sandbox:perturbed";
    applyVariable();
    viewSandbox();
    $("#x-hint").textContent = `Change "${state.perturbKind}" applied to a copy at strength ${state.perturbMag.toFixed(2)}.`;
  });
  $("#x-run").addEventListener("click", async () => {
    const perturbed = state.loaded.source === "sandbox:perturbed";
    const msg = perturbed
      ? `<div class="msg warn"><b>Experimental modification.</b> A deliberate synthetic change
         (<span class="mono">${esc(state.perturbKind)}</span>, strength ${state.perturbMag.toFixed(2)})
         was applied to a copy of the later period. This is not a property of any real data.</div>`
      : "";
    const r = await execute(msg);
    if (!r) return;
    if (!perturbed) {
      state.sandboxBaseline = { key: sandboxKey(), result: r, nObs: countObs(state.loaded.byPid) };
      return;
    }
    const b = state.sandboxBaseline;
    if (b && b.key === sandboxKey()) renderPerturbationCheck(b, r);
  });
}

/* ------------------------------------------------------------------- tabs */
function viewRuns() {
  const rs = state.runs;
  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Previous runs</h3>
        <span class="meta">${rs.length} run${rs.length === 1 ? "" : "s"} this session</span></header>
      ${rs.length ? `<div class="gridwrap"><table class="grid">
        <thead><tr><th>Time</th><th>Data</th><th>Measure</th><th class="num">People</th>
          <th class="num">Included</th><th>Result</th><th class="num">&rho;*</th>
          <th class="num">95% interval</th><th>Settings</th><th></th></tr></thead>
        <tbody>${rs.map((r, i) => `<tr>
          <td class="mono">${fmtTime(r.at)}</td><td>${esc(r.dataset)}</td>
          <td>${esc(r.feature)}</td><td class="num">${r.participants}</td>
          <td class="num">${r.eligible}</td><td>${esc(r.headline)}</td>
          <td class="mono num">${r.rho ? r.rho.toFixed(3) : "—"}</td>
          <td class="mono num">${r.ci ? `[${r.ci[0].toFixed(3)}, ${r.ci[1].toFixed(3)}]` : "—"}</td>
          <td class="mono">${esc(r.sensitivity)}</td>
          <td><button class="b tiny" data-open="${i}">Open</button></td>
        </tr>`).join("")}</tbody></table></div>
      <div class="pad"><p class="note" style="margin:0">History lives in this page only. It is
        not saved to disk or sent anywhere, and reloading clears it.</p></div>`
      : `<div class="pad"><p class="empty">No runs yet. Start with a
          <a href="#" data-tab-link="analyze">guided example</a>.</p></div>`}
    </div>
    <div id="out"></div>`;
  for (const b of document.querySelectorAll("[data-open]"))
    b.addEventListener("click", () => {
      const r = state.runs[+b.dataset.open];
      renderReport($("#out"), r.result, {
        byPid: null, personLogRatio, focusPid: r.focusPid, featureLabel: r.feature,
        messages: (r.ctx && r.ctx.messages) || "",
        header: { dataset: r.dataset, feature: r.feature,
                  windows: r.windows === "own-span"
                    ? "Halves of each participant's own observation span"
                    : "First 40% vs last 40% of each participant's observations",
                  time: r.at.toLocaleString() },
        restored: true,
      });
      $("#out").scrollIntoView({ block: "start", behavior: "smooth" });
    });
  for (const a of document.querySelectorAll("[data-tab-link]"))
    a.addEventListener("click", (e) => { e.preventDefault(); go(a.dataset.tabLink); });
}

const STATUS_WORD = {
  ready: ["Ready", "ok"], incompatible: ["Incompatible", "stop"],
  insufficient: ["Insufficient evidence", "warn"], configure: ["Needs configuration", "warn"],
};

function viewDatasets() {
  const rows = [
    ...REAL.datasets.map((d) => ({
      name: d.name, status: d.status,
      participants: d.participants,
      span: d.span_days
        ? (d.span_days >= 120 ? `${Math.round(d.span_days / 30)} months`
                              : `${Math.round(d.span_days / 7)} weeks`)
        : "—",
      obs: d.observations, median: d.median_obs_per_participant,
      targets: d.id === "college_experience"
        ? "stress and social level, against conversation, unlock and location features"
        : "stress (non-monotone coding), against conversation",
      reason: d.id === "college_experience"
        ? "Enough repeated measurement per person, and the response scale direction is stated in the published codebook."
        : "The dense item's answer options are not ordered by severity, and every properly ordered item is far too sparse.",
      notes: d.notes || [],
    })),
    { name: "RELAX", status: "incompatible", participants: 31, span: "6 weeks", obs: null,
      median: 50, targets: "7-point Likert, 3–4/day",
      reason: "Median 50 aligned reports per person against the 120 the screen requires.", notes: [] },
    { name: "PMData", status: "incompatible", participants: 16, span: "5 months", obs: null,
      median: null, targets: "PMSys stress 1–5, daily",
      reason: "Too sparse, the window variance ratio reaches 13.6, and the scale direction is not documented anywhere in the release.", notes: [] },
  ];

  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Datasets</h3><span class="meta">what each archive can and cannot support</span></header>
      <div class="pad">
        <p class="intro">A dataset can be valuable research data and still be unable to answer
          this particular question. &ldquo;Incompatible&rdquo; below is a statement about the fit
          between an archive and this method, not a judgement of the study.</p>
      </div>
      <div class="gridwrap"><table class="grid">
        <thead><tr><th>Dataset</th><th>Status</th><th class="num">People</th><th>Span</th>
          <th class="num">Median reports each</th><th>Analysis targets</th><th>Why</th></tr></thead>
        <tbody>${rows.map((r) => {
          const [word, cls] = STATUS_WORD[r.status] || [r.status, "warn"];
          return `<tr>
            <td><b>${esc(r.name)}</b></td>
            <td><span class="tag ${cls}">${esc(word)}</span></td>
            <td class="num">${r.participants ?? "—"}</td>
            <td>${esc(r.span)}</td>
            <td class="num">${r.median ?? "—"}</td>
            <td>${esc(r.targets)}</td>
            <td>${esc(r.reason)}</td></tr>`;
        }).join("")}</tbody>
      </table></div>
      <div class="pad">
        <details><summary>The screen every dataset is measured against</summary>
          <div class="gridwrap"><table class="grid">
            <tbody>
              <tr><th>Self-report</th><td>whole numbers, 2–11 ordered categories</td></tr>
              <tr><th>Observations per person per period</th><td class="mono">&ge; 60</td></tr>
              <tr><th>People passing the screen</th><td class="mono">&ge; 10</td></tr>
              <tr><th>Categories actually used per period</th><td class="mono">&ge; 2</td></tr>
              <tr><th>Feature variance ratio between periods</th><td class="mono">[0.25, 4]</td></tr>
              <tr><th>Slope magnitude</th><td class="mono">|&beta;| &ge; 0.02, same sign in both periods</td></tr>
              <tr><th>Scale direction</th><td>must come from the codebook; it cannot be recovered from data</td></tr>
            </tbody></table></div>
          <p class="hint">These were fixed before any dataset was examined and have not been
            changed. The College Experience Study passes them as they stand.</p>
        </details>
        ${REAL.datasets.map((d) => `<details><summary>${esc(d.name)} — audit detail</summary>
          <ul class="bul">${(d.notes || []).map((n) => `<li>${esc(n)}</li>`).join("")}
          ${(d.exclusion_reasons || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
          <p class="hint">${esc(d.citation || "")}</p></details>`).join("")}
      </div>
    </div>`;
}

function viewMethod() {
  $("#work").innerHTML = `
    <div class="panel"><header><h3>A. In plain English</h3></header><div class="pad">
      <p style="margin-top:0">Someone wears a phone that records how much they talk, how much they
        move, how often they unlock. They also answer a short question repeatedly over months.
        We fit how strongly the sensing measure predicts the answer in an <b>earlier</b> period,
        then again in a <b>later</b> period, for each person separately. If that strength has
        changed, the same answer may no longer mean the same thing.</p>
      <p>We report the ratio of the two strengths, called &rho;*. A ratio near 1 means nothing
        detectable changed. Further from 1 means the relationship changed.</p>
    </div></div>

    <div class="panel"><header><h3>B. The method</h3></header><div class="pad">
      <p style="margin-top:0">Within each period the feature is standardised using only that
        period's own mean and spread, and an ordinal probit is fitted:</p>
      <pre class="eq">P(R &le; k | x) = &Phi;(c<sub>k</sub> &minus; &beta;<sub>e</sub>&middot;x)
&rho;* = &beta;<sub>2</sub> / &beta;<sub>1</sub></pre>
      <p>The two periods are fitted independently. Person-level ratios are pooled as the mean of
        logs, and the interval comes from resampling <b>whole participants</b>, never individual
        observations. Standardising inside each period is what makes the unknown per-person gain
        cancel in the ratio.</p>
    </div></div>

    <div class="panel"><header><h3>C. What must be true of the data</h3></header><div class="pad">
      <ul class="bul">
        <li>At least 60 observations per person in each period, and at least 10 such people.</li>
        <li>At least two answer categories actually used in each period.</li>
        <li>Feature variance comparable between periods (ratio within [0.25, 4]).</li>
        <li>A determinable slope in both periods, with the same sign.</li>
        <li>A response scale whose direction is documented, and whose numbering is ordered by
          the thing being measured.</li>
      </ul>
      <p class="note">The last point rejected StudentLife: its densest item runs
        &ldquo;1 a little stressed, 2 definitely stressed, 3 stressed out, 4 feeling good,
        5 feeling great&rdquo;, so the numbers are not ordered by stress.</p>
    </div></div>

    <div class="panel"><header><h3>D. Assumptions</h3></header><div class="pad">
      <ul class="bul">
        <li>The latent construct relates to the feature in the same functional form in both periods.</li>
        <li>Feature reliability is stable between periods; the observable proxy for this is the
          variance ratio above.</li>
        <li>Thresholds may move freely; that movement is absorbed and never estimated.</li>
        <li>Each person is compared only against their own earlier self.</li>
      </ul>
    </div></div>

    <div class="panel"><header><h3>E. What it can establish</h3></header><div class="pad">
      <ul class="bul">
        <li>Whether the measured association between one sensing feature and one repeated ordinal
          report differs between two periods, within person, with an interval.</li>
        <li>A lower bound on the size of multiplicative recalibration: 1 &minus; &rho;*.</li>
      </ul>
    </div></div>

    <div class="panel"><header><h3>F. What it cannot establish</h3></header><div class="pad">
      <ul class="bul">
        <li>It cannot show that a person's emotional state changed. It is about the
          <i>relationship</i> between a sensor and a report, not about wellbeing.</li>
        <li>It cannot separate a change in reporting from a change in how the feature relates to
          the underlying construct.</li>
        <li>&rho; itself is not identified. Only &rho;* is, and 1 &minus; &rho;* is a lower bound.</li>
        <li>The additive part of a shift is not identifiable and is never estimated.</li>
        <li>It cannot verify a scale's direction from data.</li>
        <li>A wide interval means evidence is absent, not that the measure is stable.</li>
      </ul>
    </div></div>

    <div class="panel"><header><h3>G. Known limitations</h3></header><div class="pad">
      <ul class="bul">
        <li>No drift has been demonstrated in real data by this project. One archive of four can
          support the analysis at all, and its pre-specified primary configuration returns
          insufficient evidence.</li>
        <li>Added measurement noise attenuates &rho;* in exactly the way genuine recalibration
          does. The sandbox demonstrates this deliberately.</li>
        <li>Daily aggregates hide within-day timing; a report at 9am is matched to that whole day.</li>
        <li>Results are single-cohort and have not been replicated on an independent sample.</li>
      </ul>
    </div></div>

    <div class="panel"><header><h3>H. How this is implemented</h3></header><div class="pad">
      <p style="margin-top:0">The reference implementation is Python. The estimator you run here
        is a JavaScript port of it, and a regression test drives fixed cases through both and
        fails if the fitted slopes disagree by more than 1e&minus;3; measured agreement is
        1.8e&minus;5. That is why an in-browser number can be trusted.</p>
      <p>Guided examples, the sandbox and any CSV you open are computed live in this browser, and
        files you open are never transmitted. Bundled study results were computed offline by the
        Python implementation, because those archives are gigabytes and are licensed for research
        use rather than redistribution.</p>
      <p class="note">A small Python service is deployed separately for fixed demonstration
        scenarios. It is deliberately not what runs the analysis here.</p>
    </div></div>`;
}

/* ------------------------------------------------------------------ router */
function go(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll("#tabs button"))
    b.classList.toggle("on", b.dataset.tab === tab);
  window.scrollTo({ top: 0 });
  ({ home: viewHome, analyze: viewAnalyze, datasets: viewDatasets,
     runs: viewRuns, method: viewMethod }[tab])();
}

function boot() {
  initTheme();
  const sel = $("#theme");
  sel.value = getTheme();
  sel.addEventListener("change", (e) => setTheme(e.target.value));
  onThemeChange(() => { if (state.tab === "analyze" || state.tab === "runs") go(state.tab); });

  for (const b of document.querySelectorAll("#tabs button"))
    b.addEventListener("click", () => go(b.dataset.tab));

  const eng = activeEngine();
  $("#engine").title = `${eng.name}: ${eng.detail}`;
  go("home");
}

boot();
