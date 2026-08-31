/**
 * Result rendering, in layers.
 *
 * A reader meets the verdict in one line, then what it does and does not mean
 * in plain words, then the evidence, and only then the statistics. Nothing is
 * hidden — everything is on the page — but nobody has to read a probit
 * specification to learn whether drift was found.
 *
 * Every figure carries a title, labelled axes and a sentence saying what to
 * take from it. A figure that needs no interpretation is not drawn.
 */
import { drawCurves, drawForest, drawUsage, drawTimeline } from "../charts.js";

const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f3 = (v) => (v == null || !isFinite(v) ? "—" : v.toFixed(3));

const TONE = { DRIFT_DETECTED: "warn", NO_MEANINGFUL_DRIFT: "ok",
               INSUFFICIENT_EVIDENCE: "note", DATA_QUALITY_ISSUE: "stop" };

/** Plain-language pairs: what the verdict means, and what it must not be read as. */
function meaning(r) {
  switch (r.status) {
    case "DRIFT_DETECTED":
      return {
        means: "The link between the sensing measure and the answers is not the same in the "
             + "later period as in the earlier one. The same answer may not mean the same thing "
             + "at the end of the study as at the start.",
        notMeans: "This does not say anyone got better or worse, and it does not say why the "
             + "link changed. Added measurement noise produces the same signature as genuine "
             + "recalibration, and this method cannot tell them apart.",
      };
    case "NO_MEANINGFUL_DRIFT":
      return {
        means: "The data do not show a change in that link. The interval includes 1, which is "
             + "the value meaning 'identical in both periods'.",
        notMeans: "This is not proof that the scale is stable. A wide interval means the "
             + "evidence is absent, not that the question is settled. Look at how wide it is "
             + "before concluding anything.",
      };
    case "INSUFFICIENT_EVIDENCE":
      if (r.placebo && r.placebo.runnable === false)
        return {
          means: "The safety pre-check could not be run on this data, so the main analysis was "
               + "not run either. Splitting the earlier period in half needs twice as many "
               + "observations inside that period as the main comparison needs in each period.",
          notMeans: "This is not a finding of stability, and people passing the main screen does "
               + "not contradict it — the pre-check has a stricter requirement by design. More "
               + "observations per person in the earlier period would allow it to run.",
        };
      return {
        means: "There was not enough usable repeated measurement to estimate anything with an "
             + "interval, so no number was produced.",
        notMeans: "This is not a finding of stability, and it is not a failure of the software. "
             + "It is the correct output when the data cannot answer the question.",
      };
    default:
      return {
        means: "A data-quality check failed before the main analysis, so the result is withheld.",
        notMeans: "This is not a finding either way. The check exists to stop a number being "
             + "reported from data that cannot support one.",
      };
  }
}

function fig(id, title, caption, w = 640, h = 220) {
  return `<figure class="fig">
    <figcaption class="figtitle">${title}</figcaption>
    <canvas id="${id}" width="${w}" height="${h}" role="img" aria-label="${esc(title)}"></canvas>
    <figcaption class="figcap">${caption}</figcaption>
  </figure>`;
}

export function renderReport(root, r, ctx = {}) {
  const p = r.primary;
  const tone = TONE[r.status] || "note";
  const m = meaning(r);
  const h = ctx.header || {};
  const drawable = Boolean(ctx.byPid);

  root.innerHTML = `
    ${ctx.messages || ""}
    ${ctx.restored ? `<div class="msg"><b>Restored from history.</b> This is the result as it was
      computed earlier in this session. Figures that need the original data are not redrawn.</div>` : ""}

    <div class="panel result">
      <div class="verdict ${tone}">
        <div class="vlabel">Result</div>
        <div class="vtext">${esc(r.headline)}</div>
        ${p ? `<div class="vnum"><span class="mono">&rho;* = ${f3(p.rhoStar)}</span>
          <span class="mono dim">95% interval ${f3(p.ciLow)} to ${f3(p.ciHigh)}</span></div>` : ""}
      </div>
      <div class="pad">
        <p class="why">${esc(r.why || "")}</p>
        <div class="twocol">
          <div><h4>What this means</h4><p>${esc(m.means)}</p></div>
          <div><h4>What it does not mean</h4><p>${esc(m.notMeans)}</p></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <header><h3>The evidence</h3><span class="meta">how the result was reached</span></header>
      <div class="gridwrap"><table class="grid">
        <caption class="vh">Analysis inputs and screening outcome</caption>
        <tbody>
          <tr><th>Data</th><td>${esc(h.dataset || "—")}</td></tr>
          <tr><th>Sensing measure</th><td>${esc(h.feature || "—")}</td></tr>
          <tr><th>Periods compared</th><td>${esc(h.windows || "—")}</td></tr>
          <tr><th>People in the data</th><td class="mono">${r.nScreened ?? "—"}</td></tr>
          <tr><th>People with enough data to include</th><td class="mono">${r.nEligible ?? "—"}</td></tr>
          ${p ? `<tr><th>Resamples for the interval</th><td class="mono">${p.nResamples ?? "—"}</td></tr>` : ""}
          <tr><th>Run at</th><td class="mono">${esc(h.time || "—")}</td></tr>
        </tbody>
      </table></div>
      ${r.placebo ? `<div class="pad"><p class="hint" style="margin:0">
        <b>Pre-check:</b> before the main comparison, the earlier period is split in half and
        compared with itself, where the answer must be "no change". ${
          r.placebo.rejected
            ? "It <b>failed</b> here, so the main result is withheld: an estimator that finds "
              + "change where none can exist cannot be trusted on this data."
            : r.placebo.runnable === false
              ? "It could <b>not be run</b> here. Splitting the earlier period in half needs twice "
                + "the per-period minimum inside that period alone, so people can pass the main "
                + "screen and still leave the pre-check without enough data. The main analysis is "
                + "not run without it."
              : "It passed, so the main analysis was allowed to proceed."
        }</p></div>` : ""}
    </div>

    ${drawable ? `
    <div class="panel">
      <header><h3>Figures</h3><span class="meta">each with what to take from it</span></header>
      <div class="pad figs">
        ${fig("c-curves", "Figure 1. Fitted relationship, earlier vs later period",
              "Two curves for one person: how the expected answer rises with the sensing measure, "
            + "fitted separately in each period. If the later curve is flatter, the link weakened.")}
        ${p && p.perParticipant && p.perParticipant.length
          ? fig("c-forest", "Figure 2. One estimate per person, and the pooled result",
                "Each mark is one person's ratio, sorted. The solid line is the pooled estimate; "
              + "the dashed lines are its 95% interval. The thin vertical line at 1 marks "
              + "'no change'.")
          : ""}
        ${fig("c-usage", "Figure 3. Which answers people actually used",
              "How often each answer category was chosen in each period. Categories nobody uses "
            + "carry no information, which is why the screen requires at least two.")}
        ${fig("c-timeline", "Figure 4. One person's observations over time",
              "When this person's observations fall, and where the boundary between the two "
            + "periods sits. Gaps and clustering here are what the screen is checking for.")}
      </div>
    </div>` : ""}

    <div class="panel">
      <header><h3>Details</h3><span class="meta">for a technical reader</span></header>
      <div class="pad">
        <details><summary>Statistical detail</summary>
          <div class="gridwrap"><table class="grid"><tbody>
            <tr><th>Estimand</th><td>&rho;* = &beta;<sub>2</sub>/&beta;<sub>1</sub>, the ratio of
              within-period standardised ordinal probit slopes</td></tr>
            <tr><th>Pooling</th><td>mean of per-person log ratios</td></tr>
            <tr><th>Interval</th><td>percentile bootstrap over whole participants</td></tr>
            ${p ? `<tr><th>Lower bound on recalibration</th>
              <td class="mono">1 &minus; &rho;* = ${f3(1 - p.rhoStar)}</td></tr>` : ""}
            <tr><th>Not identified</th><td>&rho; itself, and the additive component</td></tr>
          </tbody></table></div>
        </details>
        ${r.exclusions && r.exclusions.length ? `
        <details><summary>Why ${r.exclusions.length} ${r.exclusions.length === 1 ? "person was" : "people were"} not included</summary>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>Person</th><th>Reason</th></tr></thead>
            <tbody>${r.exclusions.slice(0, 60).map((e) => `<tr>
              <td class="mono">${esc(e.pid)}</td><td>${esc(e.reason)}</td></tr>`).join("")}</tbody>
          </table></div>
          ${r.exclusions.length > 60 ? `<p class="hint">Showing the first 60.</p>` : ""}
        </details>` : ""}
        <details><summary>Limits that apply to every result here</summary>
          <ul class="bul">
            <li>This is about the link between a sensor and a report, not about anyone's wellbeing.</li>
            <li>&rho; is not identified; 1 &minus; &rho;* is a lower bound on recalibration.</li>
            <li>Scale direction cannot be checked from data and must come from a codebook.</li>
            <li>Added measurement noise mimics recalibration exactly.</li>
          </ul>
        </details>
      </div>
    </div>`;

  if (!drawable) return;
  const pid = ctx.focusPid || Object.keys(ctx.byPid)[0];
  const rows = ctx.byPid[pid] || [];
  try {
    const lr = ctx.personLogRatio ? ctx.personLogRatio(rows, r.K ?? 5) : null;
    drawCurves(document.getElementById("c-curves"), rows, lr, ctx.featureLabel);
    drawUsage(document.getElementById("c-usage"), ctx.byPid, r.K ?? 5);
    drawTimeline(document.getElementById("c-timeline"), rows, pid);
    if (p && p.perParticipant && p.perParticipant.length)
      drawForest(document.getElementById("c-forest"), p.perParticipant, p);
  } catch (e) {
    // A figure that cannot be drawn is omitted rather than faked, but the reason
    // is reported: swallowing it silently once hid a real argument-order bug.
    console.error("figure rendering failed:", e);
  }
}

/* ------------------------------------------------- precomputed real results */

/** Results computed offline. Rendered from the exported summary, never recomputed. */
export function renderPrecomputed(root, ds, meta) {
  if (!ds.runs || !ds.runs.length) {
    root.innerHTML = `
      <div class="panel">
        <div class="verdict stop">
          <div class="vlabel">Result</div>
          <div class="vtext">Data incompatible with this analysis</div>
        </div>
        <div class="pad">
          <p class="why">${esc((ds.exclusion_reasons || [])[0] || "This archive cannot support the analysis.")}</p>
          <h4>Why</h4>
          <ul class="bul">${(ds.notes || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
          <p class="note">This is a statement about the fit between this archive and this method.
            It is not a criticism of the study, which was designed to answer different questions.</p>
        </div>
      </div>`;
    return;
  }

  const primary = ds.runs.find((r) => r.primary) || ds.runs[0];
  const others = ds.runs.filter((r) => r !== primary);
  const tone = TONE[primary.status] || "note";

  const runTable = (rs) => `<div class="gridwrap"><table class="grid">
    <caption class="vh">Pre-specified analyses and their outcomes</caption>
    <thead><tr><th>Configuration</th><th class="num">Included</th><th>Result</th>
      <th class="num">&rho;*</th><th class="num">95% interval</th></tr></thead>
    <tbody>${rs.map((r) => `<tr>
      <td>${esc(r.label)}</td>
      <td class="num mono">${r.eligible} of ${r.screened}</td>
      <td>${esc(r.headline)}</td>
      <td class="num mono">${r.rho_star != null ? r.rho_star.toFixed(3) : "—"}</td>
      <td class="num mono">${r.ci_low != null ? `[${r.ci_low.toFixed(3)}, ${r.ci_high.toFixed(3)}]` : "—"}</td>
    </tr>`).join("")}</tbody></table></div>`;

  root.innerHTML = `
    <div class="panel result">
      <div class="verdict ${tone}">
        <div class="vlabel">Primary result — ${esc(ds.name)}</div>
        <div class="vtext">${esc(primary.headline)}</div>
        ${primary.rho_star != null ? `<div class="vnum">
          <span class="mono">&rho;* = ${primary.rho_star.toFixed(3)}</span>
          <span class="mono dim">95% interval ${primary.ci_low.toFixed(3)} to ${primary.ci_high.toFixed(3)}</span>
        </div>` : ""}
      </div>
      <div class="pad">
        <p class="why">${esc(primary.why || "")}</p>
        <p class="note"><b>The primary configuration was fixed before any result was seen.</b>
          It is reported here whatever it says, and it is not replaced by a secondary analysis
          that happened to produce a number.</p>
      </div>
    </div>

    <div class="panel">
      <header><h3>All pre-specified analyses</h3>
        <span class="meta">every one that was run, whatever it reported</span></header>
      ${runTable([primary, ...others])}
      <div class="pad"><p class="hint" style="margin:0">
        Rows S1–S4 are sensitivity analyses declared alongside the primary. Two of them produced
        an estimate and both report no detectable drift with wide intervals. They do not overturn
        the primary and are not presented as the headline.</p></div>
    </div>

    <div class="panel">
      <header><h3>Dataset</h3></header>
      <div class="gridwrap"><table class="grid"><tbody>
        <tr><th>People</th><td class="mono">${ds.participants ?? "—"}</td></tr>
        <tr><th>Aligned observations</th><td class="mono">${(ds.observations ?? 0).toLocaleString()}</td></tr>
        <tr><th>Median reports per person</th><td class="mono">${ds.median_obs_per_participant ?? "—"}</td></tr>
        <tr><th>Span</th><td class="mono">${ds.span_days ? Math.round(ds.span_days) + " days" : "—"}</td></tr>
        <tr><th>Response scale</th><td>${esc(ds.report_scale || "—")}</td></tr>
        <tr><th>Source</th><td>${esc(ds.citation || "—")}</td></tr>
        <tr><th>Computed</th><td>${esc(meta.generated_utc)} (seed ${meta.seed})</td></tr>
      </tbody></table></div>
      <div class="pad">
        <details><summary>Audit notes</summary>
          <ul class="bul">${(ds.notes || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
        </details>
        <details><summary>Why people were excluded from the primary analysis</summary>
          <div class="gridwrap"><table class="grid">
            <thead><tr><th>Reason</th><th class="num">Times it applied</th></tr></thead>
            <tbody>${(primary.exclusion_reasons || []).map((e) => `<tr>
              <td>${esc(e.reason)}</td><td class="num mono">${e.participants}</td></tr>`).join("")}</tbody>
          </table></div>
          <p class="hint">One person can fail several checks, so these total more than the
            number excluded.</p>
        </details>
        <p class="note" style="margin-bottom:0">${esc(meta.privacy)}.</p>
      </div>
    </div>`;
}
