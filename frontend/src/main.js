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
import { PersonalTwin, buildEvent, correctField } from "./lib/twin.js";
import { classify, ENGINES, ENGINE_INFO, STATE, browserModelReady, getEngine,
         probeHealth, serviceConfigured, setEngine, warmBrowserModel }
  from "./lib/emotion_engine.js";
import { buildDemoHistory, DEMO_PERSON_ID } from "./lib/demo_history.js";
import { renderHistory, renderSimilar, renderUnderstanding, syntheticBanner }
  from "./ui/twin_view.js";
import REAL from "./data/real_datasets.json";
import L1 from "./data/layer1_results.json";

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
  showEngine: false,
  twin: null,          // PersonalTwin, Layer 1
  draft: null,         // the event awaiting confirmation
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
      <h1>What AEDT does</h1>
      <p class="lede">AEDT explores how emotions and personal context change over time.
        Your <b>Digital Twin</b> builds a history of emotional experiences from what you write,
        finds similar past situations, and offers evidence-based pattern insights.
        Separate research tools investigate whether emotional measurements stay
        <b>comparable</b> over months and years.</p>
      <p class="lede2">Those are two different questions, and the tool keeps them apart.
        One asks <i>what has this person's history looked like</i>. The other asks
        <i>can a long history be read as if the scale meant one fixed thing throughout</i>.</p>
    </section>

    <h2 class="sech">Four ways to start</h2>
    <div class="starts">
      <div class="start">
        <h3>1. My Digital Twin</h3>
        <p>Write how you are. The system reads the feeling and the situation, shows you what it
          understood, and builds a personal history you can inspect. <b>Start here.</b></p>
        <button class="b run" data-go="twin">Open my Digital Twin</button>
      </div>
      <div class="start">
        <h3>2. Analyse real data</h3>
        <p>Results from audited longitudinal studies, and the audit explaining which archives
          can support this analysis at all. You can also open your own CSV.</p>
        <button class="b" data-go="real">Open real data</button>
      </div>
      <div class="start">
        <h3>3. Guided demonstration</h3>
        <p>Controlled data where the right answer is known in advance, to check the scientific
          estimator behaves.</p>
        <button class="b" data-go="guided">Run a guided example</button>
      </div>
      <div class="start">
        <h3>4. Interactive sandbox</h3>
        <p>Change the data on purpose and watch what the analysis does. Useful for seeing where
          the method breaks.</p>
        <button class="b" data-go="sandbox">Open the sandbox</button>
      </div>
    </div>

    <h2 class="sech">The two layers, and how they connect</h2>
    <div class="starts">
      <div class="start">
        <h3>Layer 1 — what the twin learns about a person</h3>
        <p>A Transformer classifies the feeling; rules recover the situation; each episode
          becomes one higher-order relation in a knowledge hypergraph; retrieval finds
          comparable past episodes and explains why they matched.</p>
        <p class="hint">Runs on your device, including the Transformer: the model is
          downloaded once (about 125 MB) and then classifies in your browser, so your
          check-ins never leave it.</p>
      </div>
      <div class="start">
        <h3>Layer 2 — when a long history can be trusted</h3>
        <p>If the relationship between behaviour and self-report drifts over a study, a history
          spanning that period cannot be read as if the scale meant one fixed thing throughout.
          Layer 2 tests that on dense cohort data.</p>
        <p class="hint">Layer 2 never sees your check-ins, and a personal history is far too
          short to run it on one individual. It qualifies how far a history may be
          extrapolated; it is not a step in the twin's pipeline.</p>
      </div>
    </div>

    <h2 class="sech">Where the real data stands today</h2>
    <div class="panel"><div class="pad">
      ${(() => {
        const ce = REAL.datasets.find((d) => d.id === "college_experience");
        const primary = ce && ce.runs.find((r) => r.primary);
        const withEst = ce ? ce.runs.filter((r) => r.rho_star != null) : [];
        if (!primary) return `<p class="note">Real-dataset results were not generated in this build.</p>`;
        return `<p style="margin-top:0"><b>College Experience Study</b> (218 students, four years)
          is the first archive of the four audited with enough repeated measurement to attempt
          the drift analysis. Under the pre-specified primary configuration it reports
          <b>${esc(primary.headline.toLowerCase())}</b>: ${primary.eligible} participants passed
          the screen and at least 10 are needed.</p>
        <p>${withEst.length} pre-specified secondary configurations produced an estimate, and both
          report <b>no detectable drift</b> with wide intervals
          ${withEst.map((r) => `<span class="mono">${r.rho_star.toFixed(3)}
            [${r.ci_low.toFixed(3)}, ${r.ci_high.toFixed(3)}]</span>`).join(" and ")}.</p>
        <p class="note">No drift has been demonstrated in real data, and no threshold was changed
          to obtain a result. The full audit is on the
          <a href="#" data-tab-link="datasets">Datasets</a> page; the models behind Layer 1 and
          their measured performance are on <a href="#" data-tab-link="method">Method</a>.</p>`;
      })()}
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


/* ---------------------------------------------------------- Layer 1: twin */
function ensureTwin() {
  if (!state.twin) state.twin = new PersonalTwin("You");
  return state.twin;
}

async function makeEvent(text, timestamp, fields, dataStatus = "USER",
                        onState = () => {}) {
  const pred = await classify(text, { onState });
  const ev = buildEvent(text, { personId: ensureTwin().personId, timestamp,
                                userFields: fields, prediction: pred, dataStatus });
  if (pred.note) ev.note = pred.note;
  ev.engineState = pred.state;
  ev.distribution = pred.distribution || null;
  ev.rawTop = pred.rawTop || null;
  ev.inferenceMs = pred.inferenceMs ?? null;
  ev.roundTripMs = pred.roundTripMs ?? null;
  return ev;
}

/**
 * Say up front whether the research model is reachable, so a cold start is
 * visible before the user has typed anything and a genuine outage is not
 * discovered only after they press the button.
 */
async function reportEngineHealth() {
  const slot = $("#t-enginestate");
  if (!slot) return;
  const engine = getEngine();

  if (engine === ENGINES.LEXICON) {
    slot.innerHTML = `<span class="hint">Using the word-list baseline. Results are
      labelled as a baseline, not as the model.</span>`;
    return;
  }

  if (engine === ENGINES.BROWSER) {
    if (browserModelReady()) {
      slot.innerHTML = `<span class="hint">Model loaded in this tab. Nothing leaves
        your device.</span>`;
      return;
    }
    slot.innerHTML = `<span class="hint">The model will download once
      (about ${ENGINE_INFO.browser.name ? "125" : "125"} MB) the first time you
      analyse a check-in, then stay cached.</span>
      <button class="b tiny" id="t-warm" style="margin-left:6px">Download now</button>`;
    const warm = $("#t-warm");
    if (warm) warm.addEventListener("click", async () => {
      warm.disabled = true;
      try {
        await warmBrowserModel((p) => {
          slot.innerHTML = `<span class="hint">Downloading the model… ${p.pct || 0}%
            <span class="mono">(${((p.loaded || 0) / 1e6).toFixed(0)} of
            ${((p.total || 0) / 1e6).toFixed(0)} MB)</span></span>`;
        });
        slot.innerHTML = `<span class="hint">Model loaded in this tab. Nothing
          leaves your device.</span>`;
      } catch (e) {
        slot.innerHTML = `<span class="hint">The model could not be downloaded
          (${esc(e.message)}). The word-list baseline will be used and labelled
          as such.</span>`;
      }
    });
    return;
  }

  // the Python service
  slot.innerHTML = `<span class="hint">Checking the analysis service…</span>`;
  const h = await probeHealth();
  if (h.modelReady) {
    slot.innerHTML = `<span class="hint">Analysis service ready — ${esc(h.model || "RoBERTa")}.</span>`;
  } else if (h.loading) {
    slot.innerHTML = `<span class="hint">The analysis service is waking up. Your first
      check-in may take a few seconds longer.</span>`;
  } else if (h.reachable) {
    slot.innerHTML = `<span class="hint">The service is reachable but its model is not
      loaded${h.reason ? ` (${esc(h.reason)})` : ""}. Consider the in-browser model,
      which needs no server.</span>`;
  } else {
    slot.innerHTML = `<span class="hint">The analysis service could not be reached.
      Consider the in-browser model, which needs no server.</span>`;
  }
}

function viewTwinMode() {
  const twin = ensureTwin();
  const engine = getEngine();

  $("#work").innerHTML = `
    ${syntheticBanner(twin)}
    <div class="panel">
      <header><h3>My Digital Twin</h3>
        <span class="meta">${twin.events.length} episodes recorded</span></header>
      <div class="pad">
        <p class="intro">Write how you are and what is going on, in your own words. The system
          works out the feeling and the situation, shows you exactly what it understood and
          where each part came from, and lets you correct it before anything is stored.</p>
        <textarea id="t-text" rows="3" placeholder="e.g. I am stressed and exhausted. I slept only four hours because I have an important exam tomorrow."></textarea>
        <div class="actions" style="border:0;background:none;padding:10px 0 0">
          <button class="b run" id="t-go">Analyse this check-in</button>
          <button class="b" id="t-demo">Load demonstration user</button>
          ${twin.events.length ? `<button class="b" id="t-clear">Clear history</button>` : ""}
        </div>
        <details class="adv"${state.showEngine ? " open" : ""}>
          <summary>Where the feeling is classified</summary>
          <div class="engines">
            ${[ENGINES.LEXICON, ENGINES.BROWSER, ENGINES.SERVICE].map((id) => {
              const info = ENGINE_INFO[id];
              const disabled = id === ENGINES.SERVICE && !serviceConfigured();
              return `<label class="radio">
                <input type="radio" name="eng" value="${id}"
                  ${engine === id ? "checked" : ""}${disabled ? " disabled" : ""}>
                <span><b>${esc(info.name)}</b>
                  <small>${esc(info.detail)}${
                    disabled ? " No analysis service is configured, so this is unavailable." : ""}</small>
                </span></label>`;
            }).join("")}
          </div>
          <p id="t-enginestate" style="margin-top:8px"></p>
          <p class="hint">A 124.7M-parameter Transformer cannot be bundled into a
            page, so it is either downloaded once into your browser or run in the
            Python service. The word list is neither, and is labelled a baseline
            wherever it appears.</p>
        </details>
      </div>
    </div>
    <div id="t-understand"></div>
    <div id="t-result"></div>
    <div id="t-history">${renderHistory(twin)}</div>`;

  const det = document.querySelector("details.adv");
  if (det) det.addEventListener("toggle", () => { state.showEngine = det.open; });
  for (const r of document.querySelectorAll('input[name="eng"]'))
    r.addEventListener("change", (e) => {
      setEngine(e.target.value);
      reportEngineHealth();
    });
  reportEngineHealth();

  $("#t-go").addEventListener("click", analyseCheckIn);
  $("#t-demo").addEventListener("click", loadDemoUser);
  const clr = $("#t-clear");
  if (clr) clr.addEventListener("click", () => {
    state.twin = new PersonalTwin("You");
    state.draft = null;
    viewTwinMode();
  });
}

function progressPanel(message, sub = "") {
  return `<div class="panel"><div class="pad running">
    <span class="spin" aria-hidden="true"></span>
    <div><div>${esc(message)}</div>
    ${sub ? `<div class="hint" style="margin-top:3px">${esc(sub)}</div>` : ""}</div>
    </div></div>`;
}

async function analyseCheckIn() {
  const text = ($("#t-text").value || "").trim();
  if (!text) return;
  if (state.busy) return;
  setBusy(true, "Reading");
  const show = (m, sub) => { $("#t-understand").innerHTML = progressPanel(m, sub); };
  show("Analysing your check-in…");
  try {
    // State C is reported as it happens, so a cold start reads as "waking up"
    // rather than as a failure. This is the difference the old code missed.
    const ev = await makeEvent(text, undefined, undefined, "USER", (st) => {
      if (st.state !== STATE.WAKING) return;
      if (st.phase === "download")
        show(`Downloading the emotion model… ${st.pct || 0}%`,
             "This happens once; afterwards it is cached in your browser.");
      else
        show("The analysis service is waking up. This may take a moment.",
             `Attempt ${st.attempt || 1} — free hosting suspends the service when idle.`);
    });
    state.draft = ev;
    renderDraft();
  } catch (e) {
    $("#t-understand").innerHTML = `<div class="msg stop"><b>Could not read that check-in.</b>
      ${esc(e.message)}<br><span class="hint">Nothing was saved.</span></div>`;
  } finally { setBusy(false); }
}

function renderDraft() {
  const ev = state.draft;
  if (!ev) return;
  $("#t-understand").innerHTML = renderUnderstanding(ev, {});
  for (const sel of document.querySelectorAll("select.fix"))
    sel.addEventListener("change", (e) => {
      const v = e.target.value;
      if (!v) return;
      state.draft = correctField(state.draft, e.target.dataset.field, v);
      renderDraft();
    });
  $("#t-discard").addEventListener("click", () => {
    state.draft = null;
    $("#t-understand").innerHTML = "";
    $("#t-result").innerHTML = "";
  });
  $("#t-save").addEventListener("click", () => {
    const twin = ensureTwin();
    const similar = twin.similarEpisodes(state.draft);
    const insight = twin.patternInsight(state.draft);
    twin.addEvent(state.draft);
    state.draft = null;
    $("#t-understand").innerHTML = "";
    $("#t-result").innerHTML = renderSimilar(similar, insight);
    $("#t-history").innerHTML = renderHistory(twin);
    $("#t-result").scrollIntoView({ block: "start", behavior: "smooth" });
  });
}

async function loadDemoUser() {
  if (state.busy) return;
  setBusy(true, "Building history");
  $("#t-understand").innerHTML = `<div class="panel"><div class="pad running">
    <span class="spin" aria-hidden="true"></span>
    <span>Building the fictional history through the real pipeline…</span></div></div>`;
  try {
    const twin = new PersonalTwin(DEMO_PERSON_ID, "SYNTHETIC_DEMO");
    const events = await buildDemoHistory(
      (text, ts, fields) => makeEvent(text, ts, fields, "SYNTHETIC_DEMO"));
    for (const ev of events) twin.addEvent(ev);
    state.twin = twin;
    state.draft = null;
    viewTwinMode();
    $("#t-text").value = "I am stressed and exhausted. I slept only four hours "
      + "because I have an important exam tomorrow.";
  } finally { setBusy(false); }
}

/* ------------------------------------------------------------------ analyze */
function modeBar() {
  const modes = [["twin", "My Digital Twin"], ["real", "Real data"],
                 ["guided", "Guided example"], ["sandbox", "Sandbox"]];
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
  if (state.mode === "twin") {
    // the twin view owns the workspace, so re-render the bar above it
    const bar = modeBar();
    viewTwinMode();
    $("#work").insertAdjacentHTML("afterbegin", bar);
    for (const b of document.querySelectorAll("[data-mode]"))
      b.addEventListener("click", () => { state.mode = b.dataset.mode; viewAnalyze(); });
    return;
  }
  ({ twin: viewTwinMode, guided: viewGuided, real: viewReal,
     sandbox: viewSandbox }[state.mode] || viewTwinMode)();
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

    <div class="panel"><header><h3>A2. Layer 1 — what the twin learns about a person</h3>
      <span class="meta">measured, with baselines</span></header><div class="pad">
      <p style="margin-top:0">Four steps turn a sentence into something the twin can reason over.
        Each is a real component, and each has a measured limitation.</p>
      <ol class="bul" style="padding-left:19px">
        <li><b>Emotion detection.</b> By default this runs <b>in your browser</b>:
          the model is downloaded once (about 125 MB) and classifies on your device,
          so check-ins never leave it. The same model is also available through the
          Python service. Agreement was measured both ways — ONNX vs torch mean
          |&Delta;P| 0.0141 with 96.3% top-1 label agreement, and browser vs ONNX the
          same top label on every case (max |&Delta;| 0.045) — so a browser result is
          a RoBERTa result. The model is
          <span class="mono">${esc(L1.emotion.model)}</span>, a
          124.7M-parameter RoBERTa fine-tuned on GoEmotions by its author — <b>not by this
          project</b>. Evaluated here on the held-out GoEmotions test split
          (${L1.emotion.n_test.toLocaleString()} examples, threshold ${L1.emotion.threshold}
          chosen on validation): <b>macro-F1 ${L1.emotion.macro_f1}</b>,
          micro-F1 ${L1.emotion.micro_f1}. The word-list baseline scores
          ${L1.emotion.baseline_macro_f1} on the same split.</li>
        <li><b>Context extraction.</b> Deterministic rules over an explicit lexicon. Every value
          carries the exact phrase that produced it, and anything unstated is left blank rather
          than guessed.</li>
        <li><b>Knowledge hypergraph.</b> Each episode becomes ONE relation joining everything
          that co-occurred in it, so "poor sleep AND an exam tomorrow" is a single object rather
          than two independent facts.</li>
        <li><b>Retrieval and pattern statement.</b> Weighted field matching, with the matched
          fields shown. Nothing is said about a tendency below three comparable episodes.</li>
      </ol>
      <p class="note" style="margin-top:10px"><b>A measured gap worth knowing.</b> GoEmotions has
        28 labels and none of them is "stress" — the construct this project's longitudinal layer
        is built around. Measured on this checkpoint, "I am stressed and exhausted" returns
        sadness 0.463 with nervousness only 0.119. So an explicit first-person statement
        ("I am stressed") is treated as a <b>self-report</b> and outranks the model, with the
        matched phrase shown as evidence.</p>
    </div></div>

    <div class="panel"><header><h3>A3. Layer 1 research components</h3>
      <span class="meta">trained offline; results reported whatever they say</span></header>
      <div class="pad">
      <p style="margin-top:0"><b>Hypergraph neural network.</b> A Feng et al. hypergraph
        convolution, compared against a clique-expansion GCN and a structure-free MLP on
        synthetic events with a known conjunctive rule (5 seeds, emotion vertices removed from
        the input so the label cannot leak).</p>
      <div class="gridwrap"><table class="grid">
        <thead><tr><th>Model</th><th class="num">Macro-F1</th><th class="num">Accuracy</th></tr></thead>
        <tbody>
          <tr><th>Majority class</th><td class="num mono">${L1.hgnn.majority.macro_f1}</td>
            <td class="num mono">${L1.hgnn.majority.accuracy}</td></tr>
          <tr><th>MLP (no structure)</th><td class="num mono">${L1.hgnn.mlp.macro_f1}</td>
            <td class="num mono">${L1.hgnn.mlp.accuracy}</td></tr>
          <tr><th>GCN (pairwise)</th><td class="num mono">${L1.hgnn.gcn.macro_f1}</td>
            <td class="num mono">${L1.hgnn.gcn.accuracy}</td></tr>
          <tr><th>HGNN (higher-order)</th><td class="num mono">${L1.hgnn.hgnn.macro_f1}</td>
            <td class="num mono">${L1.hgnn.hgnn.accuracy}</td></tr>
        </tbody></table></div>
      <p class="note" style="margin-top:9px"><b>The HGNN lost.</b> The structure-free MLP scores
        ${L1.hgnn.mlp.macro_f1} against the HGNN's ${L1.hgnn.hgnn.macro_f1}. With a small
        categorical entity set, an episode's own membership already encodes the conjunction, and
        propagating over a near-complete graph only blurs it. This is reported because it is what
        the experiment found; the same conclusion was reached independently by the Layer 2
        hypergraph ablation.</p>

      <p style="margin-top:14px"><b>Continual learning with EWC.</b> Four chronological periods,
        each with a different context-to-emotion rule, trained in order. Forgetting and backward
        transfer as defined by Lopez-Paz &amp; Ranzato, 5 seeds.</p>
      <div class="gridwrap"><table class="grid">
        <thead><tr><th>Arm</th><th class="num">Average accuracy</th><th class="num">Forgetting</th></tr></thead>
        <tbody>
          <tr><th>Sequential (no protection)</th>
            <td class="num mono">${L1.ewc.sequential.avg_accuracy}</td>
            <td class="num mono">+${L1.ewc.sequential.forgetting}</td></tr>
          <tr><th>EWC</th><td class="num mono">${L1.ewc.ewc.avg_accuracy}</td>
            <td class="num mono">+${L1.ewc.ewc.forgetting}</td></tr>
          <tr><th>Joint training (upper bound)</th>
            <td class="num mono">${L1.ewc.joint.avg_accuracy}</td><td class="num mono">—</td></tr>
        </tbody></table></div>
      <p class="note" style="margin-top:9px"><b>EWC worked.</b> Forgetting falls from
        +${L1.ewc.sequential.forgetting} to +${L1.ewc.ewc.forgetting}, and the penalty weight
        sweep is monotone: ${Object.entries(L1.ewc_sweep).map(([k, v]) =>
          `&lambda;=${k} &rarr; +${v}`).join(", ")}. Catastrophic forgetting is demonstrated
        first and then reduced, rather than asserted.</p>
      <p class="note"><b>These are synthetic experiments.</b> No real check-in stream of this size
        exists for this project. They compare models on identical data; they are not findings
        about people.</p>
      <p class="note"><b>Storing an event is not continual learning.</b> Adding an episode to your
        history moves no model parameters. EWC updates parameters under a penalty, and only in
        the offline research pipeline.</p>
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
