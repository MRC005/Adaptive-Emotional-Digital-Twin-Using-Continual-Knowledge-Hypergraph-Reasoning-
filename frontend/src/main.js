/**
 * Analysis workstation.
 *
 * Layout follows conventional research-software patterns: a mode/parameter
 * sidebar, a central parameter form ending in one Execute action, and a
 * persistent run history. Nothing on screen is a stored answer — every result
 * is produced by `analyze()` when the user runs it.
 */
import { analyze, activeEngine } from "./lib/engine.js";
import { CONTROLS, buildControl, perturb, simulateCohort, PERTURBATIONS, PERTURBATION_BY_ID } from "./lib/scenarios.js";
import { personLogRatio, THRESHOLDS, STATUS } from "./lib/estimator.js";
import { parseCsv, suggestColumns, toRecords, validate } from "./lib/dataset.js";
import { renderReport } from "./ui/results.js";

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const SENS = {
  standard: { label: "Standard (pre-specified)", minRep: 60, boot: 999 },
  permissive: { label: "Permissive — sensitivity analysis", minRep: 40, boot: 999 },
  strict: { label: "Strict — sensitivity analysis", minRep: 90, boot: 999 },
};

const state = {
  tab: "analyze",
  mode: "guided",
  scenario: "shift",
  variable: "conversation_minutes",
  focusPid: "",
  sensitivity: "standard",
  windowSplit: "own-span",
  perturbKind: "", perturbMag: 0.3, seed: 20260828,
  perturbation: null,   // { kind, magnitude, seed } - re-derived, never written into `raw`
  nParticipants: 45, nPerEpoch: 300,
  loaded: null,          // { label, byPid, K, variables, source }
  csv: null,
  runs: [],
  sandboxBaseline: null, // { key, result } - the unperturbed run a perturbed run is compared to
  busy: false,
};

/* ------------------------------------------------------------------ chrome */
function setBusy(on, label) {
  state.busy = on;
  const e = $("#engine");
  e.classList.toggle("busy", on);
  e.querySelector("span").textContent = on ? (label || "Computing") : "Ready";
  document.querySelectorAll("button.run").forEach((b) => (b.disabled = on));
}
function setLoaded() {
  $("#loaded").textContent = state.loaded
    ? `${state.loaded.label} · ${Object.keys(state.loaded.byPid).length} participants`
    : "No dataset loaded";
}
function fmtTime(d) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/* ----------------------------------------------------------------- sidebar */
function renderSidebar() {
  if (state.tab !== "analyze") {
    $("#side").innerHTML = `<div class="sidenote">
      This section is reference material. Return to <b>Analyze</b> to configure and run an analysis.
    </div>`;
    return;
  }
  const ds = state.loaded;
  const vars = ds?.variables || [];
  const pids = ds ? Object.keys(ds.byPid) : [];
  $("#side").innerHTML = `
    <div class="sgroup">
      <h4>Analysis mode</h4>
      ${[["guided", "Guided demonstration", "controls with a known answer"],
         ["real", "Real dataset", "upload and validate a CSV"],
         ["sandbox", "Interactive sandbox", "perturb data and re-run"]].map(([v, t, s]) => `
        <label class="radio"><input type="radio" name="mode" value="${v}"${state.mode === v ? " checked" : ""}>
          <span>${t}<small>${s}</small></span></label>`).join("")}
    </div>
    <div class="sgroup">
      <h4>Data</h4>
      <div class="sfield"><label>Dataset</label>
        <input type="text" value="${esc(ds ? ds.label : "none loaded")}" disabled></div>
      <div class="sfield"><label>Feature (variable)</label>
        <select id="sd-var"${vars.length ? "" : " disabled"}>
          ${vars.length ? vars.map((v) => `<option value="${esc(v.id)}"${v.id === state.variable ? " selected" : ""}>${esc(v.label)}</option>`).join("")
                        : `<option>—</option>`}
        </select></div>
      <div class="sfield"><label>Participant in focus (plots)</label>
        <select id="sd-pid"${pids.length ? "" : " disabled"}>
          ${pids.length ? pids.map((p) => `<option value="${esc(p)}"${p === state.focusPid ? " selected" : ""}>${esc(p)}</option>`).join("")
                        : `<option>—</option>`}
        </select></div>
      <p class="hint" style="margin-top:6px">The estimate is computed across the whole cohort.
        This selection changes which participant the figures show.</p>
    </div>
    <div class="sgroup">
      <h4>Configuration</h4>
      <div class="sfield"><label>Baseline / comparison windows</label>
        <select id="sd-win">
          <option value="own-span"${state.windowSplit === "own-span" ? " selected" : ""}>Halves of each participant's span</option>
          <option value="first-last"${state.windowSplit === "first-last" ? " selected" : ""}>First 40% vs last 40%</option>
        </select></div>
      <div class="sfield"><label>Sensitivity</label>
        <select id="sd-sens">
          ${Object.entries(SENS).map(([k, v]) => `<option value="${k}"${state.sensitivity === k ? " selected" : ""}>${esc(v.label)}</option>`).join("")}
        </select></div>
      <p class="hint" id="sd-senshint"></p>
    </div>
    <div class="sidenote">
      Analysis runs in this browser. Uploaded files are read locally and are not transmitted.
    </div>`;

  $("#side").querySelectorAll("[name=mode]").forEach((el) =>
    el.addEventListener("change", () => { state.mode = el.value; render(); }));
  const bind = (id, key, cast = (v) => v) => {
    const el = $("#" + id);
    if (el) el.addEventListener("change", () => { state[key] = cast(el.value); afterConfigChange(key); });
  };
  bind("sd-var", "variable"); bind("sd-pid", "focusPid");
  bind("sd-win", "windowSplit"); bind("sd-sens", "sensitivity");
  updateSensHint();
}
function updateSensHint() {
  const el = $("#sd-senshint"); if (!el) return;
  el.textContent = state.sensitivity === "standard"
    ? `Minimum ${SENS.standard.minRep} observations per window, as pre-specified.`
    : `Minimum ${SENS[state.sensitivity].minRep} per window. Any result is a sensitivity analysis, not a primary one.`;
}
function afterConfigChange(key) {
  updateSensHint();
  if (key === "variable" && state.loaded) applyVariable();
}

/* ------------------------------------------------- dataset / variable wiring */
function applyVariable() {
  const ds = state.loaded;
  if (!ds || !ds.raw) return;
  const v = state.variable;
  ds.byPid = {};
  for (const [pid, rows] of Object.entries(ds.raw))
    ds.byPid[pid] = rows.map((r) => ({ ...r, sensor: r.features ? r.features[v] ?? r.sensor : r.sensor }));
  applyWindows(ds.byPid);
  // Applied last, and only to the derived copy: a perturbation targets the selected feature
  // in the comparison window, so it must come after the feature and the windows are fixed.
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
  state.perturbation = null;          // a freshly loaded cohort is always unperturbed
  const raw = built.byPid;
  state.loaded = { label, raw, byPid: {}, K: built.K,
                   variables: built.variables || [{ id: "sensor", label: "Sensor" }], source };
  state.variable = state.loaded.variables[0].id;
  applyVariable();
  state.focusPid = Object.keys(state.loaded.byPid)[0];
  setLoaded();
}

/* ------------------------------------------------------------------ run */
async function execute(extraMessages = "", ctxExtra = {}) {
  if (state.busy || !state.loaded) return;
  const out = $("#out");
  setBusy(true, "Computing");
  out.innerHTML = `<div class="panel"><div class="pad"><span class="spin"></span>
    &nbsp;Fitting each participant and bootstrapping over participants…</div></div>`;
  try {
    const sens = SENS[state.sensitivity];
    const byPid = state.loaded.byPid;
    const r = await analyze({ byPid, K: state.loaded.K,
      options: { bootstrap: sens.boot, seed: state.seed,
                 thresholds: { MIN_REPORTS_PER_EPOCH: sens.minRep } } });
    const varLabel = (state.loaded.variables.find((v) => v.id === state.variable) || {}).label || state.variable;
    let messages = extraMessages;
    if (state.sensitivity !== "standard")
      messages += `<div class="msg warn"><b>Sensitivity analysis.</b> Minimum observations per
        window was changed from the pre-specified ${SENS.standard.minRep} to ${sens.minRep}.
        This is not a primary result.</div>`;
    if (state.windowSplit !== "own-span")
      messages += `<div class="msg warn"><b>Non-default windows.</b> Baseline is the first 40% and
        comparison the last 40% of each participant's observations; the middle 20% is excluded.</div>`;
    renderReport(out, r, {
      byPid, personLogRatio, focusPid: state.focusPid, featureLabel: varLabel,
      messages,
      header: { dataset: state.loaded.label, feature: varLabel,
                windows: state.windowSplit === "own-span"
                  ? "Halves of each participant's own observation span"
                  : "First 40% vs last 40% of each participant's observations",
                time: new Date().toLocaleString() },
      ...ctxExtra });
    state.runs.unshift({
      at: new Date(), dataset: state.loaded.label, feature: varLabel,
      participants: r.nScreened, eligible: r.nEligible, status: r.status,
      headline: r.headline, rho: r.primary ? r.primary.rhoStar : null,
      ci: r.primary ? [r.primary.ciLow, r.primary.ciHigh] : null,
      sensitivity: state.sensitivity, ms: r.wallMs ?? r.elapsedMs });
    out.scrollIntoView({ block: "start", behavior: "smooth" });
    return r;
  } catch (e) {
    out.innerHTML = `<div class="msg stop"><b>Analysis failed.</b> ${esc(e.message)}</div>`;
    return null;
  } finally { setBusy(false); }
}

/* -------------------------------------------------------------- mode: guided */
function viewGuided() {
  const c = CONTROLS[state.scenario];
  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Guided demonstration</h3>
        <span class="meta">simulated data with a known answer</span></header>
      <div class="pad">
        <div class="form">
          <div class="lbl">Scenario</div>
          <div class="ctl"><select id="g-scn">
            ${Object.values(CONTROLS).map((s) => `<option value="${s.id}"${s.id === state.scenario ? " selected" : ""}>${esc(s.name)} — ${esc(s.kind)}</option>`).join("")}
          </select></div>
          <div class="lbl">Description</div>
          <div class="ctl"><span style="font-size:12.5px">${esc(c.blurb)}</span></div>
          <div class="lbl">Expected verdict</div>
          <div class="ctl"><span class="mono">${esc(c.expect)}</span></div>
          <div class="lbl">Cohort size</div>
          <div class="ctl"><span class="mono">${c.params.nParticipants} participants ×
            ${c.params.nPerEpoch} observations per window</span></div>
        </div>
      </div>
      <div class="actions">
        <button class="b" id="g-load">Load scenario</button>
        <button class="b run" id="g-run"${state.loaded && state.loaded.source === "control:" + state.scenario ? "" : " disabled"}>Run analysis</button>
        <span class="hint" id="g-hint">${state.loaded && state.loaded.source === "control:" + state.scenario
          ? "Scenario loaded. Adjust the feature or windows in the sidebar, then run."
          : "Load the scenario to populate the analysis configuration."}</span>
      </div>
    </div>
    <div class="msg"><b>These are controls, not findings.</b> ${esc(c.honesty)}</div>
    <div id="out"></div>`;
  $("#g-scn").addEventListener("change", (e) => { state.scenario = e.target.value; render(); });
  $("#g-load").addEventListener("click", () => {
    loadCohort(`${CONTROLS[state.scenario].name} (simulated control)`,
               buildControl(state.scenario), "control:" + state.scenario);
    render();
  });
  $("#g-run").addEventListener("click", () => {
    const s = CONTROLS[state.scenario];
    execute(`<div class="msg"><b>Positive/negative control.</b> ${esc(s.honesty)}</div>`,
            { expected: s.expect });
  });
}

/* ---------------------------------------------------------------- mode: real */
function viewReal() {
  const csv = state.csv;
  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Real dataset</h3><span class="meta">read locally; never transmitted</span></header>
      <div class="pad">
        <div class="form">
          <div class="lbl">Source</div>
          <div class="ctl">
            <input type="file" id="r-file" accept=".csv,text/csv" style="max-width:290px">
            <button class="b" id="r-example">Use example file</button>
          </div>
          <div class="lbl">Required structure</div>
          <div class="ctl"><span class="hint">One row per observation, with columns for participant
            identifier, time, an ordinal report (2–11 integer categories) and a numeric feature.</span></div>
        </div>
      </div>
    </div>
    <div id="r-preview"></div>
    <div id="r-map"></div>
    <div id="r-val"></div>
    <div id="out"></div>`;
  $("#r-file").addEventListener("change", (e) => e.target.files[0] && readFile(e.target.files[0]));
  $("#r-example").addEventListener("click", () => {
    const built = simulateCohort({ trueRho: 0.78, nParticipants: 22, nPerEpoch: 160, seed: 41 });
    let out = "participant_id,timestamp,stress_report,conversation_minutes,activity_regularity\n";
    for (const [pid, rows] of Object.entries(built.byPid))
      for (const r of rows) out += `${pid},${r.t},${r.report},${r.features.conversation_minutes.toFixed(3)},${r.features.activity_regularity.toFixed(4)}\n`;
    ingest(out, "example_longitudinal.csv");
  });
  if (csv) { renderPreview(); renderMapping(); runValidation(); }
}

function readFile(f) {
  if (f.size > 40e6) {
    $("#r-preview").innerHTML = `<div class="msg stop"><b>File too large.</b>
      ${(f.size / 1e6).toFixed(1)} MB exceeds the 40 MB in-browser limit.</div>`;
    return;
  }
  const fr = new FileReader();
  fr.onload = () => ingest(String(fr.result), f.name);
  fr.onerror = () => { $("#r-preview").innerHTML = `<div class="msg stop"><b>Could not read the file.</b></div>`; };
  fr.readAsText(f);
}
function ingest(text, name) {
  const { header, rows } = parseCsv(text);
  if (!header.length || !rows.length) {
    $("#r-preview").innerHTML = `<div class="msg stop"><b>${esc(name)} contains no usable rows.</b></div>`;
    return;
  }
  const records = toRecords(header, rows);
  state.csv = { name, header, records, cols: suggestColumns(header, records) };
  renderPreview(); renderMapping(); runValidation();
}
function renderPreview() {
  const { name, header, records } = state.csv;
  const head = records.slice(0, 12);
  $("#r-preview").innerHTML = `
    <div class="panel">
      <header><h3>Data preview</h3>
        <span class="meta">${esc(name)} · ${records.length.toLocaleString()} rows · ${header.length} columns</span></header>
      <div class="gridwrap"><table>
        <thead><tr>${header.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
        <tbody>${head.map((r) => `<tr>${header.map((h) => `<td class="num">${esc(r[h])}</td>`).join("")}</tr>`).join("")}</tbody>
      </table></div>
      <div class="pad" style="padding-top:8px"><span class="hint">First ${head.length} of
        ${records.length.toLocaleString()} rows.</span></div>
    </div>`;
}
function renderMapping() {
  const { header, cols } = state.csv;
  const opt = (sel) => header.map((h) => `<option value="${esc(h)}"${h === sel ? " selected" : ""}>${esc(h)}</option>`).join("");
  $("#r-map").innerHTML = `
    <div class="panel">
      <header><h3>Column mapping</h3><span class="meta">confirm — nothing is assumed</span></header>
      <div class="pad"><div class="form">
        <div class="lbl">Participant identifier</div><div class="ctl"><select id="m-participant">${opt(cols.participant)}</select></div>
        <div class="lbl">Time</div><div class="ctl"><select id="m-time">${opt(cols.time)}</select></div>
        <div class="lbl">Ordinal report</div><div class="ctl"><select id="m-report">${opt(cols.report)}</select></div>
        <div class="lbl">Feature (sensor)</div><div class="ctl"><select id="m-sensor">${opt(cols.sensor)}</select></div>
      </div></div>
      <div class="actions"><button class="b" id="m-go">Validate dataset</button>
        <span class="hint">Suggestions come from column names and value ranges.</span></div>
    </div>`;
  $("#m-go").addEventListener("click", runValidation);
}
function runValidation() {
  const cols = { participant: $("#m-participant").value, time: $("#m-time").value,
                 report: $("#m-report").value, sensor: $("#m-sensor").value };
  state.csv.cols = cols;
  const v = validate(state.csv.records, cols,
                     { thresholds: { MIN_REPORTS_PER_EPOCH: SENS[state.sensitivity].minRep } });
  state.csv.validation = v;
  const icon = { pass: "✓", warn: "!", fail: "✗" };
  $("#r-val").innerHTML = `
    <div class="panel">
      <header><h3>Validation report</h3>
        <span class="meta">${v.ready ? "dataset accepted" : "dataset rejected"}</span></header>
      <div class="pad">
        ${v.checks.map((c) => `<div class="chk"><span class="m ${c.status}">${icon[c.status]}</span>
          <span><b>${esc(c.name)}</b> — ${esc(c.detail)}</span></div>`).join("")}
      </div>
      <div class="actions">
        <button class="b run" id="r-run"${v.ready ? "" : " disabled"}>Run analysis</button>
        <span class="hint">${v.ready
          ? `${v.summary.nParticipants} participants, ${v.summary.nRows.toLocaleString()} observations, K = ${v.K}.`
          : "The failures above are structural. No result will be produced from this file."}</span>
      </div>
    </div>`;
  $("#out").innerHTML = "";
  if (v.ready) {
    state.loaded = { label: state.csv.name, raw: v.byPid, byPid: v.byPid, K: v.K,
                     variables: [{ id: cols.sensor, label: cols.sensor }], source: "csv" };
    state.variable = cols.sensor;
    state.focusPid = Object.keys(v.byPid)[0];
    setLoaded(); renderSidebar();
    $("#r-run").addEventListener("click", () => execute("", { scaleUnverified: true }));
  }
}

/* The comparison is only valid if everything except the perturbation is held fixed. */
function sandboxKey() {
  return [state.scenario, state.nParticipants, state.nPerEpoch, state.seed,
          state.variable, state.windowSplit, state.sensitivity].join("|");
}

/**
 * Compare a perturbed run against the unperturbed run of the same configuration and
 * report it against the effect stated in advance. This is what makes an unchanged
 * result readable as a demonstrated invariance rather than a broken control.
 */
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
    verdict = "One of the two runs did not produce an estimate, so the comparison is not available.";
  } else if (spec.effect === "none") {
    ok = Math.abs(d) < 1e-6;
    verdict = ok
      ? "The estimate is unchanged to within 1e-6, which is the expected result. The feature is "
        + "standardised inside each window, so this perturbation cancels before the slope is fitted."
      : "The estimate moved, which was not expected for an affine change to the feature.";
  } else if (spec.effect === "small") {
    ok = Math.abs(d) < 0.05;
    verdict = ok ? "The estimate moved slightly, as expected."
                 : "The estimate moved more than expected for this perturbation.";
  } else if (spec.effect === "attenuates") {
    ok = d < 0;
    verdict = ok
      ? "The ratio fell, as expected: added measurement error attenuates the fitted slope. "
        + "This is why the method cannot separate added noise from genuine recalibration."
      : "The ratio did not fall, which was not expected for added measurement error.";
  } else {
    // Judged on the deterministic effect. Whether the interval widens on any single seed is
    // a tendency, not a guarantee, so it is reported rather than used as a pass condition.
    ok = nowObs < base.nObs;
    const pct = base.nObs ? (100 * (base.nObs - nowObs) / base.nObs).toFixed(1) : "--";
    verdict = ok
      ? `${(base.nObs - nowObs).toLocaleString()} observations (${pct}%) were removed. `
        + `The interval ${dw > 0 ? "widened" : "did not widen on this seed, which a single draw can do"}`
        + ` (width ${widthOf(bp).toFixed(4)} to ${widthOf(np).toFixed(4)}).`
      : "No observations were removed, which was not expected from added missingness.";
  }

  const mark = ok === null ? "" : ok ? "matches the stated expectation"
                                     : "DOES NOT match the stated expectation";
  const cls = ok === null ? "msg" : ok ? "msg" : "msg warn";
  const row = (a, b2, c) => `<tr><th>${a}</th><td class="mono num">${b2}</td><td class="mono num">${c}</td></tr>`;
  const el = (r) => `${r.nEligible} of ${r.nScreened}`;

  const html = `
    <div class="panel">
      <header><h3>Perturbation check</h3>
        <span class="meta">stated before the run, evaluated after it</span></header>
      <div class="pad">
        <p class="hint" style="margin-top:0">Expected: ${esc(spec.expect)}</p>
        <div class="gridwrap"><table class="grid">
          <thead><tr><th></th><th class="num">Unperturbed</th><th class="num">Perturbed</th></tr></thead>
          <tbody>
            ${row("&rho;*", num(bp && bp.rhoStar), num(np && np.rhoStar))}
            ${row("95% interval", ciOf(bp), ciOf(np))}
            ${row("Interval width", num(widthOf(bp)), num(widthOf(np)))}
            ${row("Observations used", base.nObs.toLocaleString(), nowObs.toLocaleString())}
            ${row("Eligible participants", el(base.result), el(now))}
            ${row("Verdict", esc(base.result.headline), esc(now.headline))}
          </tbody>
        </table></div>
        <p style="margin-bottom:0">Change in &rho;*:
          <span class="mono">${isFinite(d) ? (d >= 0 ? "+" : "") + d.toFixed(6) : "--"}</span>.
          ${mark ? `<b>Result ${esc(mark)}.</b>` : ""} ${esc(verdict)}</p>
      </div>
    </div>`;
  $("#out").insertAdjacentHTML("beforeend", html);
}

/* ------------------------------------------------------------- mode: sandbox */
function viewSandbox() {
  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Sandbox configuration</h3>
        <span class="meta">experiment with the pipeline's behaviour</span></header>
      <div class="pad"><div class="form">
        <div class="lbl">Base dataset</div>
        <div class="ctl"><select id="x-base">
          ${Object.values(CONTROLS).map((c) => `<option value="${c.id}"${c.id === state.scenario ? " selected" : ""}>${esc(c.name)}</option>`).join("")}
        </select></div>
        <div class="lbl">Participants</div>
        <div class="ctl"><select id="x-n">${[15, 30, 45, 60].map((n) => `<option${n === state.nParticipants ? " selected" : ""}>${n}</option>`).join("")}</select></div>
        <div class="lbl">Observations per window</div>
        <div class="ctl"><select id="x-obs">${[80, 150, 300, 400].map((n) => `<option${n === state.nPerEpoch ? " selected" : ""}>${n}</option>`).join("")}</select></div>
        <div class="lbl">Random seed</div>
        <div class="ctl"><input type="number" id="x-seed" value="${state.seed}" style="max-width:150px"></div>
        <div class="lbl">Perturbation</div>
        <div class="ctl"><select id="x-pert">
          ${PERTURBATIONS.map((p) => `<option value="${p.id}"${p.id === state.perturbKind ? " selected" : ""}>${esc(p.label)}</option>`).join("")}
        </select></div>
        <div class="lbl"></div>
        <div class="ctl"><p class="hint" id="x-expect">${esc(PERTURBATION_BY_ID[state.perturbKind]?.expect || "Select a perturbation to see what the estimator is expected to do.")}</p></div>
        <div class="lbl">Magnitude</div>
        <div class="ctl" style="max-width:340px">
          <input type="range" id="x-mag" min="0.05" max="0.9" step="0.05" value="${state.perturbMag}">
          <span class="mono" id="x-magv">${state.perturbMag.toFixed(2)}</span></div>
      </div></div>
      <div class="actions">
        <button class="b" id="x-load">Load / reset data</button>
        <button class="b" id="x-apply" disabled>Apply perturbation</button>
        <button class="b run" id="x-run"${state.loaded && state.loaded.source?.startsWith("sandbox") ? "" : " disabled"}>Run analysis</button>
        <span class="hint" id="x-hint">Perturbations are applied to a copy of the data. The source is never modified.</span>
      </div>
    </div>
    <div id="out"></div>`;
  $("#x-mag").addEventListener("input", (e) => {
    state.perturbMag = +e.target.value; $("#x-magv").textContent = state.perturbMag.toFixed(2);
  });
  $("#x-pert").addEventListener("change", (e) => {
    state.perturbKind = e.target.value;
    $("#x-apply").disabled = !e.target.value || !state.loaded;
    $("#x-expect").textContent = PERTURBATION_BY_ID[state.perturbKind]?.expect
      || "Select a perturbation to see what the estimator is expected to do.";
  });
  $("#x-load").addEventListener("click", () => {
    state.scenario = $("#x-base").value;
    state.nParticipants = +$("#x-n").value; state.nPerEpoch = +$("#x-obs").value;
    state.seed = +$("#x-seed").value || 1;
    loadCohort(`${CONTROLS[state.scenario].name} (unperturbed copy)`,
      buildControl(state.scenario, { nParticipants: state.nParticipants,
        nPerEpoch: state.nPerEpoch, seed: state.seed }), "sandbox:clean");
    state.sandboxBaseline = null;
    render();
    $("#x-hint").textContent =
      "Original data loaded. Run it first to record a baseline, then apply a perturbation.";
  });
  $("#x-apply").addEventListener("click", () => {
    if (!state.loaded || !state.perturbKind) return;
    state.perturbation = { kind: state.perturbKind, magnitude: state.perturbMag, seed: state.seed };
    state.loaded.label = `${CONTROLS[state.scenario].name} — perturbed copy (${state.perturbKind})`;
    state.loaded.source = "sandbox:perturbed";
    applyVariable(); setLoaded(); render();
    $("#x-hint").textContent = `Perturbation "${state.perturbKind}" applied to a copy at magnitude ${state.perturbMag.toFixed(2)}.`;
  });
  $("#x-run").addEventListener("click", async () => {
    const perturbed = state.loaded.source === "sandbox:perturbed";
    const msg = perturbed
      ? `<div class="msg warn"><b>Controlled synthetic perturbation applied to a copy of the data.</b>
         Type <span class="mono">${esc(state.perturbKind)}</span>, magnitude
         ${state.perturbMag.toFixed(2)}, applied to the comparison window. This is a deliberate
         manipulation for examining system behaviour, not a property of the source data.</div>`
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
  const rows = state.runs;
  $("#work").innerHTML = `
    <div class="panel">
      <header><h3>Previous runs</h3><span class="meta">this session · ${rows.length} run(s)</span></header>
      ${rows.length ? `<div class="scroll"><table>
        <thead><tr><th>Time</th><th>Dataset</th><th>Feature</th><th class="num">N</th>
          <th class="num">Eligible</th><th>Result</th><th class="num">ρ*</th>
          <th class="num">95% CI</th><th>Settings</th><th class="num">ms</th></tr></thead>
        <tbody>${rows.map((r) => `<tr>
          <td class="num">${fmtTime(r.at)}</td><td>${esc(r.dataset)}</td><td>${esc(r.feature)}</td>
          <td class="num">${r.participants}</td><td class="num">${r.eligible}</td>
          <td>${esc(r.headline)}</td>
          <td class="num">${r.rho ? r.rho.toFixed(3) : "—"}</td>
          <td class="num">${r.ci ? `[${r.ci[0].toFixed(3)}, ${r.ci[1].toFixed(3)}]` : "—"}</td>
          <td>${r.sensitivity === "standard" ? "standard" : esc(r.sensitivity) + " (sensitivity)"}</td>
          <td class="num">${r.ms}</td></tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">No analyses have been run in this session yet.</div>`}
    </div>`;
}

function viewDatasets() {
  const rows = [
    ["RELAX", "31", "6 weeks", "7-point Likert, 3–4/day", "Zenodo 10.5281/zenodo.20701999, CC-BY-4.0",
     "Rejected", "0 of 31 eligible — median 50 aligned reports against the 120 required"],
    ["PMData", "16 (14 usable)", "~5 months", "PMSys stress, 1–5 daily", "datasets.simula.no/pmdata, CC BY 4.0",
     "Rejected", "0 of 14 eligible — density, assumption A3 (variance ratios to 13.6), and an undocumented scale direction"],
    ["StudentLife (RDS repackaging)", "46", "10 weeks", "single-item stress EMA", "third-party R conversion",
     "Rejected", "Defective conversion: response column named 'null', 88% missing; 122 of ~35,000 responses survive, none overlapping the sensing period"],
    ["StudentLife (original release)", "48", "10 weeks", "~735 reports per participant", "studentlife.cs.dartmouth.edu",
     "Not obtained", "The only identified source with sufficient density; remains the correct target"],
  ];
  $("#work").innerHTML = `
    <div class="msg stop"><b>No real dataset currently supports a primary result.</b>
      Three real archives were audited against the same screen this application runs; all three were
      rejected. The guided scenarios are simulations with known answers — they validate the
      pipeline and are not empirical findings.</div>
    <div class="panel">
      <header><h3>Audited datasets</h3></header>
      <div class="scroll"><table>
        <thead><tr><th>Dataset</th><th class="num">Participants</th><th>Span</th><th>Report</th>
          <th>Source</th><th>Status</th><th>Finding</th></tr></thead>
        <tbody>${rows.map(([n, p, s, rep, src, st, f]) => `<tr>
          <td><b>${esc(n)}</b></td><td class="num">${esc(p)}</td><td>${esc(s)}</td>
          <td>${esc(rep)}</td><td>${esc(src)}</td><td>${esc(st)}</td><td>${esc(f)}</td></tr>`).join("")}
        </tbody></table></div>
    </div>
    <div class="panel">
      <header><h3>Requirements enforced before any analysis</h3></header>
      <div class="scroll"><table>
        <thead><tr><th>Requirement</th><th>Threshold</th></tr></thead>
        <tbody>
          <tr><td>Ordinal report</td><td>2–11 integer categories, 1-based</td></tr>
          <tr><td>Observations per participant per window</td><td class="num">≥ ${THRESHOLDS.MIN_REPORTS_PER_EPOCH}</td></tr>
          <tr><td>Eligible participants</td><td class="num">≥ ${THRESHOLDS.MIN_PARTICIPANTS_FOR_CI}</td></tr>
          <tr><td>Categories used per window</td><td class="num">≥ ${THRESHOLDS.MIN_CATEGORIES_USED}</td></tr>
          <tr><td>Feature variance ratio between windows</td><td class="num">[${THRESHOLDS.VAR_RATIO_LO}, ${THRESHOLDS.VAR_RATIO_HI}]</td></tr>
          <tr><td>Slope determination |β|</td><td class="num">≥ ${THRESHOLDS.MIN_ABS_BETA}</td></tr>
          <tr><td>Scale direction</td><td>Cannot be verified from data; must be confirmed from the codebook</td></tr>
        </tbody></table></div>
    </div>`;
}

function viewMethod() {
  $("#work").innerHTML = `
    <div class="panel"><header><h3>What is estimated</h3></header><div class="pad">
      <p style="font-size:13px;margin-bottom:10px">For each participant and each window, the
        feature is standardised within that window and an ordinal probit is fitted:</p>
      <p class="mono" style="padding:8px 10px;background:var(--head);border:1px solid var(--rule);border-radius:2px">
        P(R ≤ k | x) = Φ(c<sub>k</sub> − β<sub>e</sub>·x)&nbsp;&nbsp;&nbsp;&nbsp;ρ* = β₂ / β₁</p>
      <p class="note" style="margin-top:10px">Windows are fitted independently. Estimates are pooled
        as a mean of logs with a nonparametric bootstrap over participants. The unknown
        per-participant gain of the feature cancels in the ratio.</p>
    </div></div>
    <div class="panel"><header><h3>What this deployment computes</h3></header><div class="pad">
      <p style="font-size:13px">Every number on this page is computed in your browser when you
        press Run. Nothing is precomputed and no result is stored in the page. The estimator is a
        port of the reference Python implementation, and a regression test in the repository runs
        fixed cases through both and fails if the fitted slopes diverge by more than 1e-3;
        measured agreement is 1.8e-5.</p>
      <p style="font-size:13px;margin-top:9px">A Python service is deployed separately and serves a
        bounded set of fixed demonstration scenarios. It is deliberately not what runs the analysis
        here, for two reasons: it does not implement the general analysis endpoint, and a dataset you
        upload is participant data, which stays on your machine. Uploaded files are read locally and
        are never transmitted.</p>
      <p class="note" style="margin-top:9px">The analysis layer sits behind an engine interface, so a
        Python service that implements the same request can be switched on by configuring one
        endpoint without changing the product. Full-length inference and archives that cannot be
        redistributed belong there.</p>
    </div></div>
    <div class="panel"><header><h3>What the method cannot establish</h3></header><div class="pad">
      <ul style="margin:0;padding-left:18px;font-size:12.5px;line-height:1.65">
        <li>It cannot separate a change in reporting from a change in how the feature relates to the underlying construct.</li>
        <li>ρ itself is not point-identified. 1 − ρ* is a lower bound on the change, never a point estimate.</li>
        <li>The additive component of a shift is not identifiable and is never estimated.</li>
        <li>It cannot verify the direction of a response scale.</li>
        <li>A wide interval means evidence is absent, not that the measure is stable.</li>
      </ul>
    </div></div>`;
}

/* ------------------------------------------------------------------- boot */
function render() {
  renderSidebar();
  if (state.tab === "analyze") {
    ({ guided: viewGuided, real: viewReal, sandbox: viewSandbox }[state.mode])();
  } else {
    ({ datasets: viewDatasets, runs: viewRuns, method: viewMethod }[state.tab])();
  }
}
document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    state.tab = b.dataset.tab;
    document.querySelectorAll("#tabs button").forEach((o) => o.classList.toggle("on", o === b));
    render(); window.scrollTo(0, 0);
  }));
setLoaded();
render();
