/**
 * Review demo — one rehearsed surface for the project review.
 *
 * Nothing here is recomputed. Every figure is read from the frozen artefacts:
 * per-point predictions come from src/data/participants.json, which
 * scripts/export_participant_explorer.py disaggregated from the frozen
 * experiment, and the cohort and aggregate figures are the published values
 * from results/final/apples_to_apples_42.json.
 */
import PARTS from "../data/participants.json";

const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = (s, r = document) => r.querySelector(s);

/* ---- published aggregates (results/final/apples_to_apples_42.json) ------- */
const ARMS = [
  { k: "persistence", name: "Persistence", f1: 0.3108, lo: 0.2748, hi: 0.3565, tone: "ok" },
  { k: "best",        name: "Learned model", f1: 0.2917, lo: 0.2598, hi: 0.3387, tone: "accent" },
  { k: "historical",  name: "Historical baseline", f1: 0.2850, lo: 0.2521, hi: 0.3315, tone: "mut" },
  { k: "ordinal",     name: "Ordinal", f1: 0.2605, lo: 0.2363, hi: 0.2861, tone: "mut" },
];
const PAIRED = { diff: -0.0191, lo: -0.0361, hi: -0.0021, better: 13, n: 42 };
const TRANS = [
  { name: "Learned model", det: 38.0, dir: 79.1, fa: 23.7 },
  { name: "Ordinal",       det: 48.4, dir: 86.4, fa: 35.2 },
  { name: "Persistence",   det: 0.0,  dir: null, fa: 0.0  },
];
const COHORT = {
  42: { hist: 0.285039, best: 0.291697, label: "at least 5 scored observations" },
  43: { hist: 0.278410, best: 0.308169, label: "at least 1 scored observation" },
};

const state = { pid: null, k: 0, guess: null, revealed: false, cohort: 42 };

/* ---- helpers ------------------------------------------------------------ */
const withSeries = () => PARTS.participants.filter((p) => (p.series || []).length);
const scored = (p) => (p.series || []).filter((s) => !s.warmup && s.actual_next != null);
function classify(s) {
  const changed = s.actual_next !== s.observed;
  const moved = s.predicted !== s.observed;
  if (changed && moved) {
    const a = Math.sign(s.predicted - s.observed), b = Math.sign(s.actual_next - s.observed);
    return a === b ? "hit" : "wrong-way";
  }
  if (changed && !moved) return "missed";
  if (!changed && moved) return "false-alarm";
  return "quiet";
}
const TONE = { hit: "ok", "wrong-way": "warn", missed: "mut", "false-alarm": "stop", quiet: "mut" };
const LABEL = { hit: "caught the change", "wrong-way": "moved, wrong direction",
  missed: "change missed", "false-alarm": "false alarm", quiet: "correctly stayed put" };

/* ---- 1. result card ----------------------------------------------------- */
function resultCard() {
  const lo = 0.24, hi = 0.37, X = (v) => ((v - lo) / (hi - lo)) * 100;
  const rows = ARMS.map((a) => `
    <div class="dm-row">
      <div class="dm-name">${esc(a.name)}</div>
      <div class="dm-track">
        <div class="dm-ci t-${a.tone}" style="left:${X(a.lo)}%;width:${X(a.hi) - X(a.lo)}%"></div>
        <div class="dm-dot t-${a.tone}" style="left:${X(a.f1)}%"></div>
      </div>
      <div class="dm-val">${a.f1.toFixed(4)}</div>
    </div>`).join("");
  const bars = TRANS.map((t) => `
    <div class="dm-tbar">
      <div class="dm-tname">${esc(t.name)}</div>
      <div class="dm-tcol"><span class="dm-fill t-accent" style="width:${t.det}%"></span><b>${t.det.toFixed(1)}%</b></div>
      <div class="dm-tcol"><span class="dm-fill t-ok" style="width:${t.dir ?? 0}%"></span><b>${t.dir == null ? "n/a" : t.dir.toFixed(1) + "%"}</b></div>
      <div class="dm-tcol"><span class="dm-fill t-stop" style="width:${t.fa}%"></span><b>${t.fa.toFixed(1)}%</b></div>
    </div>`).join("");
  return `
  <section class="dm-card">
    <h3>The result, in one screen</h3>
    <p class="dm-sub">Matched 42-participant test cohort · 5,285 scored observations · 95% participant-clustered bootstrap</p>
    <div class="dm-split">
      <div>
        <h4>Primary metric (macro-F1)</h4>
        ${rows}
        <p class="dm-note">Persistence — carry the last value forward — scores highest of any arm.
          Learned model minus persistence: <b>${PAIRED.diff.toFixed(4)}</b>,
          95% CI [${PAIRED.lo.toFixed(4)}, ${PAIRED.hi.toFixed(4)}], better on ${PAIRED.better} of ${PAIRED.n}.
          The interval excludes zero, so the learned model is <b>credibly behind</b>.
          Pre-registered verdict: <b class="t-stop">NOT SUPPORTED</b>.</p>
      </div>
      <div>
        <h4>What the aggregate conceals</h4>
        <div class="dm-tlegend"><span>detects change</span><span>direction right</span><span>false alarms</span></div>
        ${bars}
        <p class="dm-note">Persistence scores 0% on change <b>by construction</b>, not by measurement.
          Because stable and changing pairs occur at similar rates (change base rate 53.6%),
          the gain and the loss cancel in any aggregate score. That is the mechanism behind the result on the left.</p>
      </div>
    </div>
  </section>`;
}

/* ---- 2. transition timeline --------------------------------------------- */
function timeline() {
  const ps = withSeries();
  const p = ps.find((x) => x.id === state.pid) || ps[0];
  state.pid = p.id;
  const pts = scored(p);
  const W = 1000, H = 190, L = 34, R = 12, T = 12, B = 26;
  const days = pts.map((s) => s.day);
  const d0 = Math.min(...days), d1 = Math.max(...days);
  const X = (d) => L + ((d - d0) / Math.max(1, d1 - d0)) * (W - L - R);
  const Y = (v) => T + ((5 - v) / 4) * (H - T - B);
  let path = "", dots = "", rail = "", prail = "";
  pts.forEach((s, i) => {
    path += `${i ? "L" : "M"}${X(s.day).toFixed(1)} ${Y(s.observed).toFixed(1)}`;
    const c = classify(s);
    dots += `<circle cx="${X(s.day).toFixed(1)}" cy="${Y(s.observed).toFixed(1)}" r="3.6" class="t-${TONE[c]}-f"/>`;
    if (s.actual_next !== s.observed)
      rail += `<rect x="${(X(s.day) - 2.5).toFixed(1)}" y="0" width="5" height="14" class="t-accent-f"/>`;
    if (s.predicted !== s.observed)
      prail += `<rect x="${(X(s.day) - 2.5).toFixed(1)}" y="0" width="5" height="14" class="t-${TONE[c]}-f"/>`;
  });
  const n = pts.length;
  const ch = pts.filter((s) => s.actual_next !== s.observed).length;
  const hit = pts.filter((s) => classify(s) === "hit").length;
  const fa = pts.filter((s) => classify(s) === "false-alarm").length;
  const opts = ps.map((x) => `<option value="${x.id}"${x.id === p.id ? " selected" : ""}>${x.id} — ${x.n_observations} observations, r = ${x.autocorrelation}</option>`).join("");
  return `
  <section class="dm-card">
    <h3>One participant, every decision</h3>
    <p class="dm-sub">Each scored instant: what they reported, what the model predicted next, and what actually happened.</p>
    <label class="dm-pick">Participant <select id="dm-pid">${opts}</select></label>
    <svg class="dm-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Reported stress over time">
      ${[1, 2, 3, 4, 5].map((v) => `<line x1="${L}" y1="${Y(v)}" x2="${W - R}" y2="${Y(v)}" class="dm-grid"/><text x="${L - 8}" y="${Y(v) + 4}" class="dm-ax" text-anchor="end">${v}</text>`).join("")}
      <path d="${path}" class="dm-line"/>${dots}
    </svg>
    <div class="dm-rails">
      <div><span class="dm-rlab">the label actually changed</span>
        <svg viewBox="0 0 ${W} 14" preserveAspectRatio="none">${rail}</svg></div>
      <div><span class="dm-rlab">the model moved away from the last value</span>
        <svg viewBox="0 0 ${W} 14" preserveAspectRatio="none">${prail}</svg></div>
      <div><span class="dm-rlab">persistence moved</span>
        <svg viewBox="0 0 ${W} 14" preserveAspectRatio="none"></svg>
        <span class="dm-never">never — by construction</span></div>
    </div>
    <div class="dm-key">
      <span class="t-ok">● caught the change</span><span class="t-warn">● moved, wrong direction</span>
      <span class="t-stop">● false alarm</span><span class="t-mut">● stayed put</span>
    </div>
    <p class="dm-note">${p.id}: ${n} scored instants, the label changed on <b>${ch}</b> of them.
      The model caught <b>${hit}</b> and raised <b>${fa}</b> false alarms on stable instants.
      Persistence caught none and raised none — its rail is empty because it predicts no change by definition.
      That is the trade the aggregate metric averages away.</p>
  </section>`;
}

/* ---- 3. beat the model -------------------------------------------------- */
function guessGame() {
  const ps = withSeries();
  const p = ps.find((x) => x.id === state.pid) || ps[0];
  const pts = scored(p);
  if (!pts.length) return "";
  const s = pts[Math.min(state.k, pts.length - 1)];
  const hist = (p.series || []).filter((x) => x.day <= s.day).slice(-8).map((x) => x.observed);
  const spark = hist.map((v, i) =>
    `<span class="dm-sp" style="height:${8 + v * 9}px" title="${v}">${i === hist.length - 1 ? "<i></i>" : ""}</span>`).join("");
  const c = classify(s);
  const btns = [1, 2, 3, 4, 5].map((v) =>
    `<button class="dm-g${state.guess === v ? " on" : ""}" data-g="${v}" ${state.revealed ? "disabled" : ""}>${v}</button>`).join("");
  const reveal = !state.revealed ? "" : `
    <div class="dm-reveal">
      <div class="dm-rcell"><span>your guess</span><b>${state.guess ?? "—"}</b></div>
      <div class="dm-rcell"><span>persistence</span><b>${s.baseline}</b></div>
      <div class="dm-rcell"><span>the model</span><b>${s.predicted}</b></div>
      <div class="dm-rcell t-ok"><span>what happened</span><b>${s.actual_next}</b></div>
    </div>
    <p class="dm-note">This instant: <b class="t-${TONE[c]}">${LABEL[c]}</b>.
      ${s.actual_next === s.observed
        ? "The label did not change, so persistence was right for free — and any movement by the model is a false alarm."
        : "The label changed, so persistence was wrong by construction; only a model that moves can be right here."}</p>`;
  return `
  <section class="dm-card">
    <h3>Can you beat the model?</h3>
    <p class="dm-sub">${p.id}'s last reports are below. Day ${s.day}, they reported <b>${s.observed}</b>. What do they report next?</p>
    <div class="dm-spark">${spark}</div>
    <div class="dm-guess">${btns}
      <button class="dm-act" id="dm-reveal" ${state.guess == null || state.revealed ? "disabled" : ""}>Reveal</button>
      <button class="dm-act" id="dm-next">Next instant</button></div>
    ${reveal}
  </section>`;
}

/* ---- 4. cohort toggle --------------------------------------------------- */
/* The published decomposition (results/final/apples_to_apples_42.json):
 *   apparent gain = best@43 - historical@42 = +0.023130
 *   genuine       = best@42 - historical@42 = +0.006658
 *   cohort effect = best@43 - best@42       = +0.016472   (71% of the apparent gain)
 * The trap is the MISMATCH: the new model scored on 43 participants, compared
 * against a baseline scored on 42. */
function cohortToggle() {
  const H42 = COHORT[42].hist, B42 = COHORT[42].best, B43 = COHORT[43].best;
  const on43 = state.cohort === 43;
  const modelScore = on43 ? B43 : B42;
  const gain = modelScore - H42;
  const genuine = B42 - H42;
  const cohortEffect = B43 - B42;
  const pct = (cohortEffect / (B43 - H42)) * 100;
  return `
  <section class="dm-card">
    <h3>One participant, 71% of an "improvement"</h3>
    <p class="dm-sub">Two eligibility rules exist in the codebase. They disagree about a single participant,
      who contributes exactly one scored observation. The baseline below is always scored on 42.
      Switch the rule used for the learned model and watch the comparison move.</p>
    <div class="dm-toggle">
      <span class="dm-rule">score the learned model on</span>
      <button class="${!on43 ? "on" : ""}" data-c="42">42 participants</button>
      <button class="${on43 ? "on" : ""}" data-c="43">43 participants</button>
      <span class="dm-rule">${esc(COHORT[state.cohort].label)}</span>
    </div>
    <div class="dm-nums">
      <div><span>historical baseline (42)</span><b>${H42.toFixed(6)}</b></div>
      <div class="${on43 ? "t-stop" : ""}"><span>learned model (${state.cohort})</span><b>${modelScore.toFixed(6)}</b></div>
      <div class="${on43 ? "t-stop" : ""}"><span>apparent gain</span><b>+${gain.toFixed(6)}</b></div>
    </div>
    <p class="dm-note">${on43
      ? `The comparison now reads <b>+${gain.toFixed(6)}</b> — but the two arms were scored on different cohorts.
         Only <b>+${genuine.toFixed(6)}</b> of it is the model; <b>+${cohortEffect.toFixed(6)}</b> is the extra
         participant, <b>${pct.toFixed(0)}%</b> of the apparent improvement. Because macro-F1 averages over
         participants, that person's single prediction scores either 1.0 or 0.0 and carries 1/43 = 0.0233 of the mean.
         The identity closes exactly: +${gain.toFixed(6)} = +${genuine.toFixed(6)} + ${cohortEffect.toFixed(6)}.`
      : `Scored on the same 42 participants as the baseline, the genuine model effect is <b>+${genuine.toFixed(6)}</b>,
         and its interval does not exclude zero. Switch to 43 to see what the mismatched comparison reports.
         Every result in this project uses the matched 42-participant cohort.`}</p>
  </section>`;
}

/* ---- view --------------------------------------------------------------- */
export function viewDemo(root) {
  root.innerHTML = `<div class="dm">
    <header class="dm-head">
      <h2>Review demonstration</h2>
      <p>Five beats: the question, the protocol, the result, what the result hides, and what we built.
         Every number below is read from a frozen artefact; nothing on this page is recomputed.</p>
    </header>
    ${resultCard()}${timeline()}${guessGame()}${cohortToggle()}
  </div>`;
  wire(root);
}
function wire(root) {
  const re = () => viewDemo(root);
  const pid = $("#dm-pid", root);
  if (pid) pid.addEventListener("change", (e) => {
    state.pid = e.target.value; state.k = 0; state.guess = null; state.revealed = false; re();
  });
  root.querySelectorAll(".dm-g").forEach((b) =>
    b.addEventListener("click", () => { state.guess = Number(b.dataset.g); re(); }));
  const rv = $("#dm-reveal", root);
  if (rv) rv.addEventListener("click", () => { state.revealed = true; re(); });
  const nx = $("#dm-next", root);
  if (nx) nx.addEventListener("click", () => {
    const p = withSeries().find((x) => x.id === state.pid) || withSeries()[0];
    state.k = (state.k + 1) % Math.max(1, scored(p).length);
    state.guess = null; state.revealed = false; re();
  });
  root.querySelectorAll(".dm-toggle button").forEach((b) =>
    b.addEventListener("click", () => { state.cohort = Number(b.dataset.c); re(); }));
}
