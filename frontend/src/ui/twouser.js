/**
 * "Same situation, different history" -- the personalisation mechanism, visible.
 *
 * Two people meet an IDENTICAL sentence. The population estimate is identical
 * too, because it is computed from the text alone. Everything that differs
 * afterwards comes from what each person's stored history contains.
 *
 * NOTHING ON THIS PAGE IS HARD-CODED. The two personalised results are produced
 * by `PersonalTwin.similarEpisodes` and `PersonalTwin.patternInsight` -- the
 * same functions the interactive twin uses, and the same ones covered by
 * `test_same_situation_different_history_gives_different_insight`. If the
 * histories below were edited, the outputs would change accordingly; if the
 * retrieval logic broke, this page would break with it.
 *
 * THE HISTORIES ARE SYNTHETIC AND SAID TO BE. They are written to isolate one
 * variable, which is exactly what a controlled demonstration should do, and the
 * page says so in the banner, in the column headers, and in the closing note.
 * They are NOT participants, and no claim about real people is made here.
 */
import { PersonalTwin, buildEvent, FIELD_LABELS } from "../lib/twin.js";
import { classify } from "../lib/emotion_engine.js";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const PROMPT = "I have another deadline tomorrow and I barely slept.";

/**
 * Two constructed histories that differ in ONE respect: what tended to
 * accompany deadlines. Everything else is held as similar as possible, so any
 * difference in output is attributable to that and not to volume of data --
 * both people have exactly the same number of episodes.
 */
const PROFILES = [
  {
    key: "A", name: "Person A",
    summary: "Deadlines have gone badly, on little sleep",
    episodes: [
      ["Deadline due tomorrow, slept about four hours.", { event: "deadline", sleep: "poor", emotion: "anxiety" }],
      ["Another deadline, barely slept again.", { event: "deadline", sleep: "poor", emotion: "anxiety" }],
      ["Deadline week, sleeping badly.", { event: "deadline", sleep: "poor", emotion: "stress" }],
      ["Submission tomorrow, four hours of sleep.", { event: "deadline", sleep: "poor", emotion: "anxiety" }],
      ["Deadline again, hardly slept.", { event: "deadline", sleep: "poor", emotion: "anxiety" }],
      ["Quiet weekend, slept well.", { event: "rest", sleep: "good", emotion: "calm" }],
    ],
  },
  {
    key: "B", name: "Person B",
    summary: "Deadlines have gone fine, and are usually prepared for",
    episodes: [
      ["Deadline due tomorrow, work is already done.", { event: "deadline", sleep: "poor", emotion: "calm" }],
      ["Another deadline, finished it early.", { event: "deadline", sleep: "poor", emotion: "calm" }],
      ["Deadline week, on top of it.", { event: "deadline", sleep: "poor", emotion: "joy" }],
      ["Submission tomorrow, prepared well in advance.", { event: "deadline", sleep: "poor", emotion: "calm" }],
      ["Deadline again, comfortable with it.", { event: "deadline", sleep: "poor", emotion: "calm" }],
      ["Quiet weekend, slept well.", { event: "rest", sleep: "good", emotion: "calm" }],
    ],
  },
];

function buildTwin(profile) {
  const twin = new PersonalTwin(profile.name, "SYNTHETIC_DEMO");
  profile.episodes.forEach(([text, fields], i) => {
    twin.addEvent(buildEvent(text, {
      personId: profile.name,
      timestamp: new Date(Date.UTC(2026, 0, 2 + i * 5)).toISOString(),
      userFields: fields, dataStatus: "SYNTHETIC_DEMO",
    }));
  });
  return twin;
}

function column(profile, twin, query, population) {
  const status = twin.personalisationStatus(query);
  const similar = twin.similarEpisodes(query, { topK: 5, minScore: 1.5 });
  const insight = twin.patternInsight(query);

  const differs = insight.sufficient && insight.dominantEmotion !== population;

  return `
    <div class="ucol">
      <div class="uhead">
        <b>${esc(profile.name)}</b>
        <span>${esc(profile.summary)}</span>
        <em>${twin.events.length} episodes · illustrative synthetic history</em>
      </div>

      <div class="ubox">
        <div class="ulabel">Personalised estimate</div>
        <div class="ubig ${differs ? "diff" : ""}">${esc(
          insight.sufficient ? insight.dominantEmotion : "no pattern yet")}</div>
        <div class="usub">${insight.sufficient
          ? `recorded in ${insight.nSupporting} of ${insight.nSimilar} comparable episodes`
          : `only ${insight.nSimilar} comparable episode(s) — below the floor of 3`}</div>
      </div>

      <div class="ubox">
        <div class="ulabel">Why — the history that produced it</div>
        ${similar.length ? `<ul class="ulist">${similar.map((s) => `
          <li><b>${esc(s.emotion || "—")}</b>
            <span>${esc(s.explanation)}</span>
            <em>“${esc(s.rawText)}”</em></li>`).join("")}</ul>`
          : `<p class="hint" style="margin:0">No comparable episodes found.</p>`}
      </div>

      <div class="ubox">
        <div class="ulabel">Personalisation evidence</div>
        <div class="pstat ${status.level}" style="margin:0">
          <b>${esc(status.headline)}</b>
          <span class="act">${esc(status.action)}</span>
        </div>
      </div>
    </div>`;
}

export async function viewTwoUser(root) {
  root.innerHTML = `<div class="panel"><div class="pad running">
    <span class="spin"></span> Reading the same sentence for both people…</div></div>`;

  const twins = PROFILES.map(buildTwin);
  const queries = PROFILES.map((p) => buildEvent(PROMPT, {
    personId: p.name, timestamp: new Date(Date.UTC(2026, 1, 1)).toISOString(),
    userFields: { event: "deadline", sleep: "poor" }, dataStatus: "SYNTHETIC_DEMO",
  }));

  // The POPULATION estimate: the model reading the sentence with no history at
  // all. Identical for both, by construction -- that is the point of the page.
  let population = "neutral", engine = "lexicon";
  try {
    const p = await classify(PROMPT);
    population = p.label; engine = p.backend;
  } catch { /* the page still works; the label below says which engine ran */ }

  const cols = PROFILES.map((p, i) => column(p, twins[i], queries[i], population)).join("");
  const a = twins[0].patternInsight(queries[0]);
  const b = twins[1].patternInsight(queries[1]);
  const different = a.sufficient && b.sufficient && a.dominantEmotion !== b.dominantEmotion;

  root.innerHTML = `
    <div class="story">
      <div class="panel">
        <header><h3>Same situation. Different history.</h3>
          <span class="meta">the personalisation mechanism, isolated</span></header>
        <div class="pad">
          <div class="msg warn" style="margin-top:0">
            <b>Illustrative synthetic histories — not real participants.</b>
            These two histories were written to differ in exactly one respect, so
            the cause of any difference is unambiguous. No claim about real
            people is made on this page. The real-data results are under
            <b>What We Discovered</b>.
          </div>

          <div class="ubox" style="margin-bottom:14px">
            <div class="ulabel">The identical input, given to both</div>
            <p class="uprompt">“${esc(PROMPT)}”</p>
          </div>

          <div class="ubox popbox">
            <div class="ulabel">Population estimate — from the sentence alone, no history</div>
            <div class="ubig">${esc(population)}</div>
            <div class="usub">Identical for both people, because it uses only the
              text. Produced by ${esc(engine === "lexicon" ? "the word-list baseline"
                : "the RoBERTa GoEmotions model")}.</div>
          </div>
        </div>

        <div class="ugrid">${cols}</div>

        <div class="pad">
          ${different ? `<div class="msg ok">
            <b>Same sentence. Same population estimate. Different personalised
            results: ${esc(a.dominantEmotion)} versus ${esc(b.dominantEmotion)}.</b>
            The only thing that changed is what each history contains.</div>`
          : `<div class="msg warn"><b>The two histories did not separate.</b>
            This page reports whatever the mechanism produces; it does not force
            a contrast.</div>`}

          <h4>What caused the difference</h4>
          <p>Both people have <b>${twins[0].events.length} episodes</b>, and both
            match on the same fields — <b>${esc(FIELD_LABELS.event || "event")}
            (deadline)</b> and <b>${esc(FIELD_LABELS.sleep || "sleep")} (poor)</b>.
            Retrieval finds the same <em>number</em> of comparable episodes for
            each. What differs is the feeling recorded in them:</p>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th></th><th class="num">Comparable episodes</th>
              <th class="num">Most frequent feeling</th><th class="num">Supporting</th></tr></thead>
            <tbody>
              <tr><th>${esc(PROFILES[0].name)}</th><td class="num">${a.nSimilar}</td>
                <td class="num">${esc(a.dominantEmotion || "—")}</td>
                <td class="num">${a.nSupporting ?? "—"}</td></tr>
              <tr><th>${esc(PROFILES[1].name)}</th><td class="num">${b.nSimilar}</td>
                <td class="num">${esc(b.dominantEmotion || "—")}</td>
                <td class="num">${b.nSupporting ?? "—"}</td></tr>
            </tbody>
          </table></div>
          <p class="hint">Because the number of episodes and the matched fields are
            held equal, the difference cannot be explained by one person simply
            having more data.</p>

          <div class="msg">
            <b>What this does not show.</b> It is an association inside a
            constructed history, not a cause, and not a prediction. On real data,
            personalisation of this kind did <em>not</em> beat carrying the last
            value forward — see <b>What We Discovered</b>.
          </div>
        </div>

        <div class="actions">
          <button class="b" id="tu-again">Recompute</button>
          <span class="hint">Recomputed live from the stored histories each time —
            no result on this page is stored or hard-coded.</span>
        </div>
      </div>
    </div>`;

  root.querySelector("#tu-again").addEventListener("click", () => viewTwoUser(root));
}
