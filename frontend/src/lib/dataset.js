/**
 * Dataset ingestion and validation.
 *
 * Nothing reaches the estimator without passing `validate`. A dataset that
 * cannot support the analysis produces a stated reason, never a result.
 * This mirrors aedt/schemas.py::validate_long_frame and the eligibility
 * screen's data-shape requirements.
 */
import { sd } from "./stats.js";
import { THRESHOLDS } from "./estimator.js";

/** Minimal RFC-4180-ish CSV parser: quoted fields, embedded commas/newlines. */
export function parseCsv(text) {
  const rows = [];
  let row = [], field = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
      else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((v) => v !== "")) rows.push(row);
  if (!rows.length) return { header: [], rows: [] };
  const header = rows[0].map((h) => h.trim());
  return { header, rows: rows.slice(1) };
}

export function toRecords(header, rows) {
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, (r[i] ?? "").trim()])));
}

/** Heuristic column suggestions. The user always confirms; nothing is assumed. */
export function suggestColumns(header, records) {
  const low = header.map((h) => h.toLowerCase());
  const pick = (...pats) => {
    for (const p of pats) {
      const i = low.findIndex((h) => h.includes(p));
      if (i >= 0) return header[i];
    }
    return "";
  };
  const numericCols = header.filter((h) => {
    const vals = records.slice(0, 300).map((r) => Number(r[h])).filter(Number.isFinite);
    return vals.length > records.slice(0, 300).length * 0.7;
  });
  const smallInt = header.filter((h) => {
    const vals = records.map((r) => Number(r[h])).filter(Number.isFinite);
    if (vals.length < records.length * 0.7) return false;
    const u = new Set(vals);
    return u.size >= 2 && u.size <= 11 && [...u].every((v) => Number.isInteger(v));
  });
  const participant = pick("participant", "subject", "pid", "uid", "user", "id");
  const time = pick("timestamp", "time", "date", "occasion", "day");
  const report = smallInt.find((c) => c !== participant && c !== time) ||
                 pick("stress", "report", "rating", "score", "response");
  // the sensor must not be a column already spoken for, nor the ordinal report
  const taken = new Set([participant, time, report]);
  const sensor = numericCols.find((c) => !taken.has(c) && !smallInt.includes(c)) ||
                 numericCols.find((c) => !taken.has(c)) || "";
  return { participant, time, report, sensor, numericCols, smallInt };
}

export const CHECK = { PASS: "pass", WARN: "warn", FAIL: "fail" };

/**
 * The validation layer. Returns an ordered list of named checks plus, when
 * everything required passes, the analysis-ready {byPid, K}.
 */
export function validate(records, cols, opts = {}) {
  const th = { ...THRESHOLDS, ...(opts.thresholds || {}) };
  const checks = [];
  const add = (name, status, detail) => checks.push({ name, status, detail });
  const fail = () => checks.some((c) => c.status === CHECK.FAIL);

  // ---- required columns -------------------------------------------------
  const missing = ["participant", "time", "report", "sensor"]
    .filter((k) => !cols[k] || !(cols[k] in (records[0] || {})));
  if (missing.length) {
    add("Required columns", CHECK.FAIL,
        `Not mapped or not present: ${missing.join(", ")}. Every analysis needs a ` +
        `participant identifier, a time column, an ordinal report and a numeric sensor.`);
    return { checks, ready: false };
  }
  add("Required columns", CHECK.PASS,
      `participant=${cols.participant}, time=${cols.time}, report=${cols.report}, sensor=${cols.sensor}`);

  // ---- parse ------------------------------------------------------------
  const parsed = [];
  let badTime = 0, badReport = 0, badSensor = 0;
  for (const r of records) {
    const pid = String(r[cols.participant] ?? "").trim();
    if (!pid) continue;
    const rawT = r[cols.time];
    let t = Number(rawT);
    if (!Number.isFinite(t)) { const d = Date.parse(rawT); t = Number.isFinite(d) ? d : NaN; }
    const rep = Number(r[cols.report]);
    const sen = Number(r[cols.sensor]);
    if (!Number.isFinite(t)) { badTime++; continue; }
    if (!Number.isFinite(rep)) { badReport++; continue; }
    if (!Number.isFinite(sen)) { badSensor++; continue; }
    parsed.push({ pid, t, report: rep, sensor: sen });
  }
  const dropped = badTime + badReport + badSensor;
  add("Parseable rows", dropped === 0 ? CHECK.PASS : CHECK.WARN,
      `${parsed.length} of ${records.length} rows usable` +
      (dropped ? ` — dropped ${badTime} unparseable timestamps, ${badReport} non-numeric ` +
                 `reports, ${badSensor} non-numeric sensor values. Nothing is imputed.` : ""));
  if (!parsed.length) { add("Analysis possible", CHECK.FAIL, "No usable rows."); return { checks, ready: false }; }

  // ---- ordinal report ---------------------------------------------------
  const vals = parsed.map((r) => r.report);
  const uniq = [...new Set(vals)].sort((a, b) => a - b);
  const allInt = uniq.every((v) => Number.isInteger(v));
  if (!allInt || uniq.length > 11) {
    add("Ordinal report", CHECK.FAIL,
        `The report column has ${uniq.length} distinct value(s)` +
        (allInt ? "" : " and is not integer-valued") +
        `. This method models an ORDINAL response with a small number of ordered ` +
        `categories; a continuous measure cannot be analysed by thresholding it here, ` +
        `because that would invent a discretisation the respondent never used.`);
    return { checks, ready: false };
  }
  const minV = uniq[0], K = uniq[uniq.length - 1];
  if (minV < 1) {
    add("Ordinal report", CHECK.FAIL,
        `Values run ${minV}..${K}. Categories must be 1-based (1..K). A 0 usually ` +
        `encodes "not answered" — confirm before recoding, and do not guess.`);
    return { checks, ready: false };
  }
  add("Ordinal report", CHECK.PASS, `${uniq.length} categories used, range 1..${K}`);
  add("Scale direction", CHECK.WARN,
      `Assumed ASCENDING: a larger value means MORE of the construct. This cannot be ` +
      `verified from the data — if your instrument is reversed, the sign of the result ` +
      `flips. Confirm against your codebook.`);

  // ---- repeated measures + epochs --------------------------------------
  const byPid = {};
  for (const r of parsed) (byPid[r.pid] ||= []).push(r);
  const pids = Object.keys(byPid);
  add("Participants", pids.length >= th.MIN_PARTICIPANTS_FOR_CI ? CHECK.PASS : CHECK.FAIL,
      `${pids.length} participant(s); at least ${th.MIN_PARTICIPANTS_FOR_CI} are required ` +
      `for a participant-clustered interval`);

  let dupes = 0;
  for (const pid of pids) {
    byPid[pid].sort((a, b) => a.t - b.t);
    const seen = new Set();
    byPid[pid] = byPid[pid].filter((r) => {
      if (seen.has(r.t)) { dupes++; return false; }
      seen.add(r.t); return true;
    });
    // epochs = halves of THIS participant's own span (frozen rule)
    const lo = byPid[pid][0].t, hi = byPid[pid][byPid[pid].length - 1].t;
    const mid = lo + (hi - lo) / 2;
    byPid[pid] = byPid[pid].map((r) => ({ ...r, epoch: r.t > mid ? 1 : 0 }));
  }
  if (dupes) add("Duplicate occasions", CHECK.WARN,
                 `${dupes} duplicate (participant, time) row(s) removed, first kept`);

  const counts = pids.map((p) => byPid[p].length);
  const enough = pids.filter((p) =>
    byPid[p].filter((r) => r.epoch === 0).length >= th.MIN_REPORTS_PER_EPOCH &&
    byPid[p].filter((r) => r.epoch === 1).length >= th.MIN_REPORTS_PER_EPOCH);
  const med = counts.slice().sort((a, b) => a - b)[Math.floor(counts.length / 2)];
  add("Repeated observations",
      enough.length >= th.MIN_PARTICIPANTS_FOR_CI ? CHECK.PASS : CHECK.FAIL,
      `${enough.length} of ${pids.length} participants have at least ` +
      `${th.MIN_REPORTS_PER_EPOCH} observations in BOTH halves of their own span ` +
      `(median ${med} observations per participant overall)`);

  const span = Math.max(...pids.map((p) => {
    const rs = byPid[p]; return rs[rs.length - 1].t - rs[0].t;
  }));
  add("Time coverage", span > 0 ? CHECK.PASS : CHECK.FAIL,
      span > 0 ? `Longest participant span covers ${span.toLocaleString()} time units`
               : "All timestamps identical — no longitudinal structure at all");

  // Tests variation only. The participant count is the "Participants" check's job; testing it
  // again here produced a failure whose own message said everything was fine.
  const sensorSd = pids.filter((p) => sd(byPid[p].map((r) => r.sensor)) > 0);
  const flat = pids.length - sensorSd.length;
  add("Sensor variation",
      flat === 0 ? CHECK.PASS
                 : sensorSd.length >= th.MIN_PARTICIPANTS_FOR_CI ? CHECK.WARN : CHECK.FAIL,
      flat === 0
        ? `All ${pids.length} participants show variation in the sensor column`
        : `${flat} of ${pids.length} participants have a constant sensor value and will be ` +
          `excluded; ${sensorSd.length} remain`);

  return { checks, ready: !fail(), byPid, K,
           summary: { nRows: parsed.length, nParticipants: pids.length, K,
                      medianPerParticipant: med, nWithEnough: enough.length } };
}
