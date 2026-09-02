/**
 * "Two People" -- an experiment the visitor runs, not a case study they read.
 *
 * The visitor moves through stages and causes each result:
 *
 *   ask -> population estimate -> inspect the two histories -> predict again
 *       -> reveal -> why -> invert the histories and watch it flip
 *
 * NOTHING IS HARD-CODED. Both personalised results come from
 * `PersonalTwin.similarEpisodes` / `patternInsight`, and the population estimate
 * from the same `classify()` the live check-in uses. The inversion stage is the
 * proof: it re-runs the identical code over histories whose recorded feelings
 * have been swapped, and the conclusions swap with them.
 *
 * The synthetic labelling is contextual rather than a wall at the top -- a
 * banner nobody reads protects nobody. It appears on the histories themselves,
 * where the claim is actually made, and again in the closing tie-back to the
 * real result.
 */
import { PersonalTwin, buildEvent } from "../lib/twin.js";
import { classify } from "../lib/emotion_engine.js";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const PROMPT = "I have another deadline tomorrow and I barely slept.";

/**
 * Two histories differing in ONE respect: what tended to accompany deadlines.
 * Equal length, equal matched fields -- so the contrast cannot be a
 * data-volume artefact, and the page says so.
 */
const BASE = [
  { key: "A", name: "Person A", tone: "s1",
    summary: "Deadlines have gone badly, on little sleep",
    episodes: [
      ["Deadline due tomorrow, slept about four hours.", "anxiety"],
      ["Another deadline, barely slept again.", "anxiety"],
      ["Deadline week, sleeping badly.", "stress"],
      ["Submission tomorrow, four hours of sleep.", "anxiety"],
      ["Deadline again, hardly slept.", "anxiety"],
    ] },
  { key: "B", name: "Person B", tone: "s2",
    summary: "Deadlines have gone fine, and are usually prepared for",
    episodes: [
      ["Deadline due tomorrow, work is already done.", "calm"],
      ["Another deadline, finished it early.", "calm"],
      ["Deadline week, on top of it.", "joy"],
      ["Submission tomorrow, prepared well in advance.", "calm"],
      ["Deadline again, comfortable with it.", "calm"],
    ] },
];

const st = { stage: 0, population: null, engine: null, inverted: false, why: false };

function profiles() {
  if (!st.inverted) return BASE;
  // Swap the recorded feelings between the two people, nothing else.
  return [
    { ...BASE[0], episodes: BASE[0].episodes.map(([t], i) => [t, BASE[1].episodes[i][1]]) },
    { ...BASE[1], episodes: BASE[1].episodes.map(([t], i) => [t, BASE[0].episodes[i][1]]) },
  ];
}

function twinFor(p) {
  const twin = new PersonalTwin(p.name, "SYNTHETIC_DEMO");
  p.episodes.forEach(([text, emotion], i) => {
    twin.addEvent(buildEvent(text, {
      personId: p.name,
      timestamp: new Date(Date.UTC(2026, 0, 2 + i * 5)).toISOString(),
      userFields: { event: "deadline", sleep: "poor", emotion },
      dataStatus: "SYNTHETIC_DEMO",
    }));
  });
  return twin;
}

function queryFor(name) {
  return buildEvent(PROMPT, {
    personId: name, timestamp: new Date(Date.UTC(2026, 1, 1)).toISOString(),
    userFields: { event: "deadline", sleep: "poor" }, dataStatus: "SYNTHETIC_DEMO",
  });
}

function historyList(p, twin) {
  return `<ol class="tl">${twin.events.map((e, i) => `
    <li class="tl-i">
      <span class="tl-n">${i + 1}</span>
      <span class="tl-e ${p.tone}">${esc(e.emotion.value)}</span>
      <span class="tl-t">“${esc(e.rawText)}”</span>
    </li>`).join("")}</ol>`;
}

export async function viewTwoUser(root) {
  const ps = profiles();
  const twins = ps.map(twinFor);
  const insights = ps.map((p, i) => twins[i].patternInsight(queryFor(p.name)));
  const sims = ps.map((p, i) => twins[i].similarEpisodes(queryFor(p.name), { topK: 5, minScore: 1.5 }));

  const S = st.stage;
  const revealed = S >= 3;

  root.innerHTML = `
    <div class="exp">

      <section class="exp-q">
        <h2>Can the same situation mean different things to different people?</h2>
        <p class="exp-sent">“${esc(PROMPT)}”</p>
        ${S === 0 ? `<button class="cta" id="x-run">Run the experiment</button>
          <p class="exp-hint">One sentence, two people, one model.</p>` : ""}
      </section>

      ${S >= 1 ? `
      <section class="exp-step">
        <p class="exp-lab">Step 1 · No personal history</p>
        <div class="exp-pop">
          <div>
            <p class="exp-sub">The model reads the sentence on its own and returns</p>
            <p class="exp-big">${esc(st.population || "…")}</p>
            <p class="exp-sub">Same input, same answer, for anyone.
              ${st.engine ? `Produced by ${esc(st.engine === "lexicon"
                ? "the word-list baseline" : "the RoBERTa GoEmotions model")}.` : ""}</p>
          </div>
        </div>
        ${S === 1 ? `<button class="cta" id="x-hist">Now give it a history</button>` : ""}
      </section>` : ""}

      ${S >= 2 ? `
      <section class="exp-step">
        <p class="exp-lab">Step 2 · Two different histories</p>
        <p class="exp-sub">Both people have written five check-ins about deadlines on
          poor sleep. Read them — the difference is in what they recorded feeling.</p>
        <div class="exp-two">
          ${ps.map((p, i) => `
            <div class="exp-col">
              <h3 class="exp-name ${p.tone}">${esc(p.name)}</h3>
              <p class="exp-sum">${esc(p.summary)}</p>
              ${historyList(p, twins[i])}
              <p class="exp-syn">Illustrative synthetic history — not a real participant</p>
            </div>`).join("")}
        </div>
        ${S === 2 ? `<button class="cta" id="x-pred">Predict again, with history</button>
          <p class="exp-hint">Same sentence as before. What do you expect to change?</p>` : ""}
      </section>` : ""}

      ${revealed ? `
      <section class="exp-step">
        <p class="exp-lab">Step 3 · The same sentence, read through each history</p>
        <div class="exp-two exp-res">
          ${ps.map((p, i) => `
            <div class="exp-col">
              <p class="exp-sub">${esc(p.name)}</p>
              <p class="exp-big ${p.tone}">${esc(insights[i].dominantEmotion || "no pattern")}</p>
              <p class="exp-sub">${insights[i].sufficient
                ? `in ${insights[i].nSupporting} of ${insights[i].nSimilar} comparable episodes`
                : "not enough comparable history"}</p>
            </div>`).join("")}
        </div>
        <p class="exp-said">Population estimate was <b>${esc(st.population || "—")}</b>.
          Both personalised readings moved away from it, in opposite directions.</p>

        <button class="linkb2" id="x-why">${st.why ? "Hide the evidence" : "Why did the answers change?"}</button>
        ${st.why ? `
          <div class="exp-two exp-ev">
            ${ps.map((p, i) => `
              <div class="exp-col">
                <p class="exp-sub">${esc(p.name)} — episodes the system matched</p>
                <ul class="ev">${sims[i].map((s) => `
                  <li><b class="${p.tone}">${esc(s.emotion)}</b>
                    <span>${esc(s.explanation)}</span></li>`).join("")}</ul>
              </div>`).join("")}
          </div>
          <p class="exp-sub">Both have <b>${twins[0].events.length} episodes</b> and both match on
            the same fields, so retrieval returns the same <i>number</i> of comparable
            episodes for each. Only the recorded feeling differs.</p>` : ""}
      </section>

      <section class="exp-step">
        <p class="exp-lab">Step 4 · Is it really the history?</p>
        <p class="exp-sub">Swap what the two people recorded feeling, change nothing
          else, and re-run the same code.</p>
        <button class="cta ${st.inverted ? "on" : ""}" id="x-invert">
          ${st.inverted ? "Restore the original histories" : "Swap the two histories"}</button>
        ${st.inverted ? `<p class="exp-said">The conclusions swapped with them:
          <b class="s1">${esc(insights[0].dominantEmotion)}</b> and
          <b class="s2">${esc(insights[1].dominantEmotion)}</b>.
          Nothing is stored or hard-coded — the answer is computed from whatever the
          history contains.</p>` : ""}
      </section>

      <section class="exp-end">
        <p>This shows how personal history <i>can</i> change a reading. On the real
          held-out data, this kind of personalisation <b>did not</b> beat carrying the
          previous value forward.</p>
        <button class="linkb2" data-go2="discovered">See the real results →</button>
      </section>` : ""}
    </div>`;

  const on = (id, fn) => root.querySelector(id)?.addEventListener("click", fn);

  on("#x-run", async () => {
    const b = root.querySelector("#x-run");
    b.disabled = true; b.textContent = "Reading the sentence…";
    try {
      const p = await classify(PROMPT);
      st.population = p.label; st.engine = p.backend;
    } catch { st.population = "unavailable"; }
    st.stage = 1; viewTwoUser(root);
  });
  on("#x-hist", () => { st.stage = 2; viewTwoUser(root); });
  on("#x-pred", () => { st.stage = 3; viewTwoUser(root); });
  on("#x-why", () => { st.why = !st.why; viewTwoUser(root); });
  on("#x-invert", () => { st.inverted = !st.inverted; viewTwoUser(root); });
  root.querySelector("[data-go2]")?.addEventListener("click", (e) => {
    document.querySelector(`[data-tab="${e.target.dataset.go2}"]`)?.click();
  });

  // move focus to the newest step so keyboard users are not left behind
  if (S > 0) root.querySelector(".exp-step:last-of-type .exp-lab")
    ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
