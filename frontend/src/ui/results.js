/** Analysis report. Sequential headed sections and tables, as in statistical
 *  software output — not tiles. Every number is computed at run time. */
import { STATUS, expectedCategory } from "../lib/estimator.js";
import { drawCurves, drawForest, drawTimeline, drawUsage } from "../charts.js";

const STATUS_UI = {
  [STATUS.DRIFT]:        { cls: "drift",  label: "Drift detected" },
  [STATUS.STABLE]:       { cls: "stable", label: "No meaningful drift" },
  [STATUS.INSUFFICIENT]: { cls: "insuf",  label: "Insufficient evidence" },
  [STATUS.QUALITY]:      { cls: "qual",   label: "Data quality issue" },
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");

function interpretation(r) {
  const p = r.primary;
  if (r.status === STATUS.DRIFT)
    return `The feature predicts the report ${num(p.rhoStar)} times as strongly in the comparison
      window as in the baseline window, and the 95% interval excludes 1. The same feature value now
      corresponds to a different reported number. Because the estimator is attenuated by
      construction, ${(100 * p.lowerBound).toFixed(1)}% is a lower bound on the change, not a point
      estimate — the true change is at least this large.`;
  if (r.status === STATUS.STABLE)
    return `The interval [${num(p.ciLow)}, ${num(p.ciHigh)}] includes 1, so these data do not
      support a claim that the relationship changed. This is not evidence of stability: an interval
      of this width would also fail to detect a small real change.`;
  if (r.status === STATUS.QUALITY)
    return `The negative control rejected. Splitting the baseline window in half — where no change
      can exist by construction — still produced a detection. Something in these data violates the
      method's assumptions, so the primary analysis was not run.`;
  return `The data did not clear the pre-specified screen, so no estimate was produced. Reporting a
    value here would report noise.`;
}

function limitations(r, ctx) {
  const out = [
    `The method identifies a ratio of association strengths between two windows. It cannot
     distinguish a change in how the participant reports from a change in how the feature relates
     to the underlying construct.`,
    `The additive component of any shift is not identifiable and is never estimated.`,
  ];
  if (ctx.scaleUnverified)
    out.push(`Scale direction was assumed ascending and could not be verified from the file. If
      larger values mean less of the construct, the sign of this result is inverted.`);
  if (r.primary && r.primary.nUsed < 20)
    out.push(`${r.primary.nUsed} participants contributed. Intervals at this size are wide and a
      change smaller than roughly 15% would likely go undetected.`);
  if (r.status === STATUS.INSUFFICIENT)
    out.push(`To reach a conclusion the data would need at least
      ${r.thresholds.MIN_REPORTS_PER_EPOCH} observations per participant per window and at least
      ${r.thresholds.MIN_PARTICIPANTS_FOR_CI} eligible participants.`);
  return out;
}

export function renderReport(root, r, ctx = {}) {
  const ui = STATUS_UI[r.status], p = r.primary;
  const excl = (r.screened || []).filter((s) => !s.eligible);
  const hdr = ctx.header || {};

  root.innerHTML = `
  <div class="panel report">
    <div class="pad">
      <h2>Analysis result</h2>
      ${ctx.messages || ""}

      <div class="rsec">
        <table class="kv"><tbody>
          <tr><td>Dataset</td><td>${esc(hdr.dataset || "—")}</td></tr>
          <tr><td>Feature analysed</td><td>${esc(hdr.feature || "—")}</td></tr>
          <tr><td>Participants in cohort</td><td class="num">${r.nScreened}</td></tr>
          <tr><td>Window definition</td><td>${esc(hdr.windows || "Halves of each participant's own observation span")}</td></tr>
          <tr><td>Run at</td><td class="num">${esc(hdr.time || "")}</td></tr>
        </tbody></table>
      </div>

      <div class="rsec">
        <h3>Status</h3>
        <div class="statusline ${ui.cls}">
          <div class="status ${ui.cls}">${ui.label}</div>
          <div>${esc(r.why)}</div>
        </div>
      </div>

      <div class="rsec">
        <h3>Evidence</h3>
        <div class="steps" style="margin-bottom:10px">
          <span class="done">ingest</span><span class="done">validate</span>
          <span class="done">window split</span><span class="done">eligibility</span>
          <span class="${r.placebo ? "done" : "skip"}">negative control</span>
          <span class="${p ? "done" : "skip"}">estimate</span>
          <span class="${p ? "done" : "skip"}">bootstrap</span>
        </div>
        <table class="kv"><tbody>
          <tr><td>Participants screened</td><td class="num">${r.nScreened}</td></tr>
          <tr><td>Passed eligibility</td><td class="num">${r.nEligible}</td></tr>
          ${r.placebo ? `<tr><td>Negative control (baseline split-half)</td><td class="num">${
            r.placebo.runnable
              ? `ρ* = ${num(r.placebo.rhoStar)}, CI [${num(r.placebo.ciLow)}, ${num(r.placebo.ciHigh)}] — ${
                  r.placebo.rejected ? "REJECTS" : "does not reject"}`
              : "not runnable"}</td></tr>` : ""}
          ${p ? `<tr><td>Participants contributing an estimate</td><td class="num">${p.nUsed}</td></tr>` : ""}
          <tr><td>Bootstrap resamples (unit: participant)</td><td class="num">${r.bootstrap}</td></tr>
        </tbody></table>
        ${excl.length ? `
          <p class="note" style="margin:10px 0 6px">Excluded participants (${excl.length}), each with its reason:</p>
          <div class="gridwrap"><table>
            <thead><tr><th>Participant</th><th>Reason for exclusion</th></tr></thead>
            <tbody>${excl.map((s) => `<tr><td class="num">${esc(s.pid)}</td><td>${esc(s.reasons.join("; "))}</td></tr>`).join("")}</tbody>
          </table></div>` : `<p class="note" style="margin-top:8px">No participants were excluded.</p>`}
      </div>

      <div class="rsec">
        <h3>Visualization</h3>
        <div class="figs">
          <figure><div class="figtitle">Fig. 1 — Fitted response function by window</div>
            <canvas id="c-curves"></canvas>
            <figcaption id="cap-curves"></figcaption></figure>
          <figure><div class="figtitle">Fig. 2 — Response category frequencies</div>
            <canvas id="c-usage"></canvas>
            <figcaption>Distribution of reported categories in each window, across the cohort.</figcaption></figure>
          <figure><div class="figtitle">Fig. 3 — Observation series, selected participant</div>
            <canvas id="c-timeline"></canvas>
            <figcaption id="cap-timeline"></figcaption></figure>
          ${p ? `<figure><div class="figtitle">Fig. 4 — Per-participant estimates</div>
            <canvas id="c-forest"></canvas>
            <figcaption>Each mark is one participant. Solid line, cohort estimate; dashed lines,
              95% interval; thin line at 1 is no change.</figcaption></figure>`
          : `<figure><div class="figtitle">Fig. 4 — Per-participant estimates</div>
            <div class="empty">No estimate was produced, so there is nothing to plot.</div></figure>`}
        </div>
      </div>

      <div class="rsec">
        <h3>Interpretation</h3>
        <p style="font-size:13px">${interpretation(r)}</p>
      </div>

      <div class="rsec">
        <h3>Limitations</h3>
        <ul style="margin:0;padding-left:18px;font-size:12.5px;line-height:1.6">
          ${limitations(r, ctx).map((l) => `<li>${l}</li>`).join("")}
        </ul>
      </div>

      <div class="rsec" style="margin-bottom:0">
        <h3>Technical details</h3>
        <div class="scroll"><table>
          <thead><tr><th>Quantity</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Estimate ρ*</td><td class="num">${p ? num(p.rhoStar, 4) : "not produced"}</td></tr>
            <tr><td>95% confidence interval</td><td class="num">${p ? `[${num(p.ciLow, 4)}, ${num(p.ciHigh, 4)}]` : "—"}</td></tr>
            <tr><td>Interval excludes 1</td><td class="num">${p ? (p.excludesNull ? "yes" : "no") : "—"}</td></tr>
            <tr><td>Median per-participant ρ*</td><td class="num">${p ? num(p.medianRhoStar, 4) : "—"}</td></tr>
            <tr><td>Change, lower bound (1 − ρ*)</td><td class="num">${p ? (100 * p.lowerBound).toFixed(2) + "%" : "—"}</td></tr>
            <tr><td>Sample size (participants used)</td><td class="num">${p ? p.nUsed : 0}</td></tr>
            <tr><td>Response categories K</td><td class="num">${r.K}</td></tr>
            <tr><td>Model</td><td>Ordinal probit, P(R≤k|x) = Φ(c<sub>k</sub> − β<sub>e</sub>·x); ρ* = β₂/β₁</td></tr>
            <tr><td>Standardisation</td><td>Within each window separately</td></tr>
            <tr><td>Eligibility: min observations per window</td><td class="num">${r.thresholds.MIN_REPORTS_PER_EPOCH}</td></tr>
            <tr><td>Eligibility: Var ratio bounds</td><td class="num">[${r.thresholds.VAR_RATIO_LO}, ${r.thresholds.VAR_RATIO_HI}]</td></tr>
            <tr><td>Eligibility: min |β|</td><td class="num">${r.thresholds.MIN_ABS_BETA}</td></tr>
            <tr><td>Random seed</td><td class="num">${r.seed}</td></tr>
            <tr><td>Compute time</td><td class="num">${r.wallMs ?? r.elapsedMs} ms</td></tr>
            <tr><td>Engine</td><td>${esc(r.engineName || "In-browser")} — cross-validated against the Python reference (tolerance 1e-3)</td></tr>
          </tbody>
        </table></div>
      </div>
    </div>
  </div>`;

  // --- plots, drawn from computed values
  const byPid = ctx.byPid;
  if (!byPid) return;
  drawUsage(root.querySelector("#c-usage"), byPid, r.K);
  const focusPid = ctx.focusPid && byPid[ctx.focusPid] ? ctx.focusPid : Object.keys(byPid)[0];
  const rows = byPid[focusPid];
  drawTimeline(root.querySelector("#c-timeline"), rows, ctx.featureLabel);
  root.querySelector("#cap-timeline").textContent =
    `Participant ${focusPid}: ${rows.length} observations. The dashed line is the boundary between ` +
    `the baseline and comparison windows for this participant.`;
  const lr = ctx.personLogRatio(rows, r.K);
  const cap = root.querySelector("#cap-curves");
  if (lr.fits?.[0]?.converged && lr.fits?.[1]?.converged) {
    drawCurves(root.querySelector("#c-curves"), lr.fits, expectedCategory, r.K);
    cap.innerHTML = `Participant ${esc(focusPid)}: β₁ = ${num(lr.fits[0].beta)},
      β₂ = ${num(lr.fits[1].beta)}, ρ* = ${num(Math.exp(lr.value))}. A flatter comparison curve means
      the same feature value now corresponds to a different report.`;
  } else {
    cap.textContent = `Participant ${focusPid} did not yield two fitted curves: ${lr.reason}`;
  }
  if (p) drawForest(root.querySelector("#c-forest"), p.perParticipant, p);
}
