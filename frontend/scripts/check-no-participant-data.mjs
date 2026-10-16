#!/usr/bin/env node
/**
 * Build guard: refuse to build the application with participant-level research
 * records in frontend/src/data/.
 *
 * Runs automatically before `npm run build` (the "prebuild" script), so a local
 * or hosted build fails instead of bundling such records into public assets.
 *
 * Two rules:
 *   1. No file in frontend/src/data/ may be one of the participant-derived files
 *      that earlier versions of this application shipped. They are recognised by
 *      their git blob SHA-1 (a hash of the file, which reveals nothing about it).
 *   2. Any people-level demonstration file (participants.json, trajectories.json)
 *      must carry `"_synthetic": true` at the top level, as written by
 *      scripts/make_synthetic_demo_data.py.
 *
 * Exit 0 when both rules hold, 1 otherwise. Prints file names and reasons only.
 */
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "data");

// git blob SHA-1 of the participant-derived files removed from the application.
const FORBIDDEN_BLOBS = new Map([
  ["74acce2ca5e4bc6782f38583cafced93a95fe3e2", "participant summary statistics (removed)"],
  ["6c464c45e4169a8fc04b4586c14a39e25dd10a03", "participant report series and predictions (removed)"],
]);
const PEOPLE_LEVEL = new Set(["participants.json", "trajectories.json"]);

const gitBlobSha1 = (buf) =>
  createHash("sha1").update(`blob ${buf.length}\0`).update(buf).digest("hex");

const problems = [];
for (const name of readdirSync(DATA)) {
  if (!name.endsWith(".json")) continue;
  const buf = readFileSync(join(DATA, name));
  // Hash both as stored and with CRLF normalised to LF, so a Windows checkout
  // (core.autocrlf) is recognised too.
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  const hit = [gitBlobSha1(buf), gitBlobSha1(lf)].find((s) => FORBIDDEN_BLOBS.has(s));
  if (hit) {
    problems.push(`${name}: is a removed participant-derived file (${FORBIDDEN_BLOBS.get(hit)})`);
    continue;
  }
  if (PEOPLE_LEVEL.has(name)) {
    let doc;
    try {
      doc = JSON.parse(buf.toString("utf8"));
    } catch {
      problems.push(`${name}: not valid JSON`);
      continue;
    }
    if (!doc || doc._synthetic !== true) {
      problems.push(`${name}: people-level file without "_synthetic": true; ` +
        "regenerate it with scripts/make_synthetic_demo_data.py");
    }
  }
}

if (problems.length) {
  console.error("Refusing to build: participant-level data check failed.");
  for (const p of problems) console.error(`  - ${p}`);
  console.error("See DATA_PROVENANCE.md. No participant-level research record may ship with this application.");
  process.exit(1);
}
console.log("Participant-data check passed: frontend/src/data/ holds no participant-level research record.");
