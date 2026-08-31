/**
 * MODE 1 -- My Digital Twin.
 *
 * The flow a reviewer walks through: write a check-in, watch the system say
 * what it understood and where each part came from, correct anything wrong,
 * then see the event join the history, the hypergraph update, comparable past
 * episodes come back with reasons, and either a pattern or an admission that
 * there is not enough history for one.
 *
 * Two things are always on screen and never softened: which classifier
 * produced the feeling, and whether the history is real or the fictional demo.
 */
import { CONTEXT_FIELDS, FIELD_LABELS, FIELD_OPTIONS, FieldSource,
         buildEventHypergraph, correctField, knownFields, unknownFields }
  from "../lib/twin.js";
import { ENGINE_INFO } from "../lib/emotion_engine.js";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const SOURCE_WORD = {
  [FieldSource.EXTRACTED]: "read from your text",
  [FieldSource.USER_REPORTED]: "you told it",
  [FieldSource.MODEL]: "predicted by the model",
  [FieldSource.INFERRED]: "estimated from your history",
  [FieldSource.CORRECTED]: "you corrected it",
  [FieldSource.UNKNOWN]: "not known",
};

const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined,
  { day: "numeric", month: "short", year: "numeric" });

/** The banner that must appear whenever the fictional history is loaded. */
export function syntheticBanner(twin) {
  if (!twin || !twin.isSynthetic) return "";
  return `<div class="msg warn"><b>Synthetic demonstration user — not real participant data.</b>
    This history belongs to a fictional person, written to demonstrate the system.
    Nothing here describes anyone.</div>`;
}

/**
 * Which engine actually ran. Three states, never blurred:
 *   A the research model produced this
 *   B it did not, and the reason is given
 *   C is transient and handled during the request, not here
 */
export function engineBanner(ev) {
  if (ev.backend === "transformer") {
    const timing = ev.inferenceMs != null
      ? ` Inference took ${Math.round(ev.inferenceMs)} ms`
        + (ev.roundTripMs ? `, ${Math.round(ev.roundTripMs)} ms including the network.` : ".")
      : "";
    return `<div class="msg ok"><b>Emotion identified using the RoBERTa GoEmotions model.</b>
      Running as an int8 ONNX build of the same fine-tune${timing}</div>`;
  }
  return `<div class="msg warn"><b>The research model was not used.</b>
    ${esc(ev.note || "A simplified word-list baseline produced this result instead.")}
    <br><span class="hint">This baseline scores macro-F1 0.098 against the model's
    0.493 on the same held-out data, so treat the label as indicative only.</span></div>`;
}

/**
 * GoEmotions is MULTI-LABEL: the scores are independent sigmoids and do not sum
 * to 1. Showing one label at "57%" invites reading it as a probability over
 * emotions, which it is not. So the primary is shown with its own score, and
 * anything else above the operating threshold is listed beside it.
 */
const ALSO_THRESHOLD = 0.15;   // the value tuned on the GoEmotions validation split

export function alsoDetected(ev) {
  const raw = ev.rawTop || [];
  if (ev.backend !== "transformer" || raw.length < 2) return "";
  const others = raw.slice(1).filter(([, s]) => s >= ALSO_THRESHOLD);
  const primary = raw[0];
  return `<p class="hint" style="margin:8px 0 0">
    The model scores each of its 28 emotions independently, so these are not
    percentages of one another. Strongest: <b>${esc(primary[0])}</b>
    (${(primary[1] * 100).toFixed(0)})${
      others.length
        ? `. Also above the ${ALSO_THRESHOLD} threshold: `
          + others.map(([l, sc]) => `${esc(l)} (${(sc * 100).toFixed(0)})`).join(", ")
        : `. Nothing else passed the ${ALSO_THRESHOLD} threshold.`}</p>`;
}


export function renderUnderstanding(ev, { engine }) {
  const known = knownFields(ev);
  const missing = unknownFields(ev);
  const info = ENGINE_INFO[ev.backend] || ENGINE_INFO.lexicon;

  const row = (f) => {
    const p = ev[f];
    const opts = FIELD_OPTIONS[f] || [];
    return `<tr>
      <th>${esc(FIELD_LABELS[f] || f)}</th>
      <td><b>${esc(p.value)}</b></td>
      <td class="src">${esc(SOURCE_WORD[p.source])}${
        p.confidence != null && p.source === FieldSource.MODEL
          // NOT a percentage. GoEmotions has a multi-label sigmoid head, so
          // this is that one label's independent score and the 28 scores do
          // not sum to 1. Calling it "% confident" invited exactly the wrong
          // reading, and made a perfectly ordinary 0.57 look like a weak result.
          ? ` · score ${p.confidence.toFixed(2)}` : ""}</td>
      <td class="ev">${p.evidence ? esc(p.evidence) : "—"}</td>
      <td><select class="fix" data-field="${f}">
        <option value="">change…</option>
        ${opts.map((o) => `<option value="${esc(o)}"${o === p.value ? " selected" : ""}>${esc(o)}</option>`).join("")}
      </select></td></tr>`;
  };

  return `
    <div class="panel">
      <header><h3>What it understood</h3>
        <span class="meta">every field says where it came from</span></header>
      <div class="gridwrap"><table class="grid">
        <thead><tr><th>Field</th><th>Value</th><th>Source</th><th>Evidence</th><th>Correct it</th></tr></thead>
        <tbody>${Object.keys(known).map(row).join("")}</tbody>
      </table></div>
      <div class="pad">
        ${missing.length ? `<p class="hint" style="margin:0 0 8px">
          Not mentioned, so left blank rather than guessed:
          <b>${missing.map((f) => esc(FIELD_LABELS[f] || f)).join(", ")}</b>.</p>` : ""}
        ${engineBanner(ev)}
        ${alsoDetected(ev)}
      </div>
      <div class="actions">
        <button class="b run" id="t-save">Add to my history</button>
        <button class="b" id="t-discard">Discard</button>
        <span class="hint">Correct anything wrong first — corrections are recorded as yours.</span>
      </div>
    </div>`;
}

export function renderSimilar(similar, insight) {
  return `
    <div class="panel">
      <header><h3>Comparable past episodes</h3>
        <span class="meta">${similar.length} found</span></header>
      ${similar.length ? `<div class="gridwrap"><table class="grid">
        <thead><tr><th>When</th><th>Feeling</th><th>Why it matched</th><th>What you wrote</th></tr></thead>
        <tbody>${similar.map((s) => `<tr>
          <td class="mono">${esc(fmtDate(s.timestamp))}</td>
          <td>${esc(s.emotion || "—")}</td>
          <td>${esc(s.explanation)}</td>
          <td class="quote">${esc(s.rawText)}</td>
        </tr>`).join("")}</tbody></table></div>`
        : `<div class="pad"><p class="empty" style="padding:18px">
             Nothing comparable in your history yet.</p></div>`}
    </div>

    <div class="panel">
      <header><h3>What your Digital Twin can say</h3>
        <span class="meta">${insight.sufficient ? "enough history" : "not enough history"}</span></header>
      <div class="verdict ${insight.sufficient ? "note" : "warn"}">
        <div class="vlabel">${insight.sufficient ? "Pattern insight" : "Still learning"}</div>
        <div class="vtext" style="font-size:16px;line-height:1.45">${esc(insight.statement)}</div>
      </div>
      <div class="pad">
        <ul class="bul">${(insight.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
        <p class="note" style="margin:8px 0 0">This is a description of your own recorded
          history. It is not a diagnosis, not a prediction, and not medical advice.</p>
      </div>
    </div>`;
}

export function renderHistory(twin) {
  const p = twin.profile();
  if (!p.nEvents) {
    return `<div class="panel"><div class="pad"><p class="empty">
      No history yet. Write a check-in above, or load the demonstration user.</p></div></div>`;
  }
  const g = buildEventHypergraph(twin.personId, twin.events);
  const emo = Object.entries(p.emotionCounts);

  return `
    <div class="panel">
      <header><h3>Your Digital Twin</h3>
        <span class="meta">${p.nEvents} episodes, ${fmtDate(p.firstEvent)} to ${fmtDate(p.lastEvent)}</span></header>
      <div class="gridwrap"><table class="grid"><tbody>
        <tr><th>Most recent feeling</th><td>${esc(p.recentEmotion || "—")}</td></tr>
        <tr><th>Feelings recorded</th><td>${emo.map(([k, n]) => `${esc(k)} ×${n}`).join(", ")}</td></tr>
        ${Object.entries(p.recurringContext).map(([f, vals]) => `<tr>
          <th>Recurring ${esc(FIELD_LABELS[f] || f).toLowerCase()}</th>
          <td>${vals.map((v) => `${esc(v.value)} ×${v.n}`).join(", ")}</td></tr>`).join("")}
      </tbody></table></div>
    </div>

    <div class="panel">
      <header><h3>Knowledge hypergraph</h3>
        <span class="meta">built from your stored episodes</span></header>
      <div class="pad">
        <p class="hint" style="margin-top:0">Each episode is ONE relation joining everything
          that co-occurred in it — not a set of pairwise links. That is what lets the twin ask
          "when these things happened <i>together</i>, what followed?"</p>
        <div class="gridwrap"><table class="grid"><tbody>
          <tr><th>Entities (vertices)</th><td class="mono">${g.summary.nVertices}</td></tr>
          <tr><th>Episodes (hyperedges)</th><td class="mono">${g.summary.nEdges}</td></tr>
          <tr><th>Average entities per episode</th><td class="mono">${g.summary.meanArity.toFixed(1)}</td></tr>
          <tr><th>Largest episode</th><td class="mono">${g.summary.maxArity} entities</td></tr>
          <tr><th>Entity types</th><td>${Object.entries(g.summary.verticesByType)
            .map(([t, n]) => `${esc(t)} ${n}`).join(", ")}</td></tr>
        </tbody></table></div>
      </div>
      <div class="pad">
        <details><summary>Combinations that recur (${g.recurring.length})</summary>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>Entity</th><th>with</th><th class="num">Episodes</th></tr></thead>
            <tbody>${g.recurring.slice(0, 15).map((r) => `<tr>
              <td>${esc(r.a.replace("=", ": "))}</td><td>${esc(r.b.replace("=", ": "))}</td>
              <td class="num mono">${r.n}</td></tr>`).join("")}</tbody>
          </table></div>
        </details>
        <details><summary>All episodes (${twin.events.length})</summary>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>When</th><th>Feeling</th><th>Context</th><th>Text</th></tr></thead>
            <tbody>${[...twin.events].reverse().map((e) => `<tr>
              <td class="mono">${esc(fmtDate(e.timestamp))}</td>
              <td>${esc(e.emotion?.value ?? "—")}</td>
              <td class="hint">${Object.entries(knownFields(e))
                .filter(([f]) => f !== "emotion")
                .map(([f, p2]) => `${esc(FIELD_LABELS[f] || f)}: ${esc(p2.value)}`).join(" · ")}</td>
              <td class="quote">${esc(e.rawText)}</td></tr>`).join("")}</tbody>
          </table></div>
        </details>
      </div>
    </div>`;
}
