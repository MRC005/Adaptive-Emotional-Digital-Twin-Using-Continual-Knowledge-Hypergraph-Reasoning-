/**
 * Panel Demo readiness.
 *
 * The demonstration must never degrade silently. This checks, on demand, every
 * dependency the live flow needs and reports what is actually true -- including
 * which inference engine would run right now, so a cached-model failure shows
 * up here rather than in front of an audience.
 *
 * It runs no fake checks: each row queries the real subsystem.
 */
import { apiUrl, probeHealth, ENGINE_INFO } from "../lib/emotion_engine.js";
import T from "../data/trajectories.json";
import F from "../data/findings.json";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Is the in-browser model already cached? Checks the real Cache Storage. */
async function browserModelCached() {
  try {
    if (!("caches" in window)) return { ok: false, detail: "Cache Storage unavailable" };
    const names = await caches.keys();
    for (const n of names) {
      const c = await caches.open(n);
      const reqs = await c.keys();
      // Must be the WEIGHTS. Matching config.json reported "cached" while the
      // 125 MB download was still pending -- exactly the false green this page
      // exists to prevent.
      for (const r of reqs) {
        if (!/\.onnx(_data)?$/i.test(new URL(r.url).pathname)) continue;
        const res = await c.match(r);
        const len = Number(res?.headers?.get("content-length") || 0);
        const name = new URL(r.url).pathname.split("/").pop();
        if (len && len < 5e6) continue;           // a stub, not the model
        return { ok: true, detail: len ? `${name}, ${(len / 1e6).toFixed(0)} MB cached`
                                       : `${name} cached` };
      }
    }
    return { ok: false, detail: "weights not cached — the first check-in will "
                              + "download ~125 MB. Use Pre-warm below." };
  } catch (e) {
    return { ok: false, detail: `could not inspect cache: ${e.message}` };
  }
}

export async function viewReadiness(root) {
  root.innerHTML = `<div class="panel"><div class="pad running">
    <span class="spin"></span> Checking every dependency the demonstration needs…</div></div>`;

  const checks = [];
  const push = (name, ok, detail, critical = true) =>
    checks.push({ name, ok, detail, critical });

  // 1. real research artefacts must be present, or the story pages are empty
  push("Research findings loaded",
       Boolean(F?.headline?.models?.T_twin),
       F?.headline ? `${F.cohort.participants} participants, ${F.cohort.prediction_pairs.toLocaleString()} pairs`
                   : "findings.json missing");

  push("Real trajectories loaded",
       Array.isArray(T?.participants) && T.participants.length >= 2,
       T?.participants ? `${T.participants.length} held-out participants`
                       : "trajectories.json missing");

  // 2. which engine would actually run right now
  const cached = await browserModelCached();
  push("In-browser model cached", cached.ok, cached.detail);

  const base = apiUrl();
  if (base) {
    const h = await probeHealth(6000);
    push("Backend service",
         h.reachable && h.modelReady,
         h.modelReady ? `ready — ${h.model || "model loaded"}`
           : h.loading ? "waking up — retry in a moment"
           : h.reachable ? `reachable, model ${h.reason || "unavailable"}`
           : "not reachable (the demo does not require it)",
         false);
  } else {
    push("Backend service", true, "not configured — browser inference is used", false);
  }

  // 3. no silent fallback: say which engine the next check-in would use
  let engine = "browser";
  try { engine = localStorage.getItem("aedt.engine") || "browser"; } catch { /* private mode */ }
  const isReal = engine !== "lexicon";
  push("Selected engine is the research model", isReal,
       isReal ? `${ENGINE_INFO[engine]?.name || engine}`
              : "WORD-LIST BASELINE selected — switch before presenting");

  // 4. storage for the interactive twin
  let store = false;
  try { localStorage.setItem("aedt.__probe", "1"); localStorage.removeItem("aedt.__probe"); store = true; }
  catch { store = false; }
  push("Local history storage", store,
       store ? "available" : "blocked (private window?) — history will not persist", false);

  const criticalFails = checks.filter((c) => c.critical && !c.ok);
  const ready = criticalFails.length === 0;

  root.innerHTML = `
    <div class="panel">
      <header><h3>Panel demo readiness</h3>
        <span class="meta">checked just now, nothing cached</span></header>
      <div class="pad">
        <div class="msg ${ready ? "ok" : "warn"}">
          <b>${ready ? "READY FOR PANEL DEMO" : "NOT READY — " + criticalFails.length + " blocking issue(s)"}</b>
          ${ready ? " Every dependency the live flow needs is present."
                  : " Fix the items marked below before presenting."}
        </div>
        <div class="gridwrap"><table class="grid">
          <thead><tr><th></th><th>Check</th><th>Detail</th></tr></thead>
          <tbody>${checks.map((c) => `<tr>
            <td class="num">${c.ok ? "✓" : c.critical ? "✗" : "!"}</td>
            <td><b>${esc(c.name)}</b></td>
            <td class="dim">${esc(c.detail)}</td></tr>`).join("")}</tbody>
        </table></div>
        <p class="hint">The <b>Twin Evolution</b> and <b>What We Discovered</b> pages
          read from committed result files and work with no network at all. Only the
          live check-in needs a model.</p>
      </div>
      <div class="actions">
        <button class="b run" id="rd-warm">Pre-warm the model now</button>
        <button class="b" id="rd-again">Re-check</button>
        <span class="hint" id="rd-msg"></span>
      </div>
    </div>`;

  root.querySelector("#rd-again").addEventListener("click", () => viewReadiness(root));
  root.querySelector("#rd-warm").addEventListener("click", async () => {
    const msg = root.querySelector("#rd-msg");
    msg.textContent = "Downloading and warming the model… this is the slow step, done now rather than live.";
    try {
      const { classify } = await import("../lib/emotion_engine.js");
      const t0 = performance.now();
      const r = await classify("Warming the model before the demonstration.");
      msg.textContent = `Done in ${(performance.now() - t0) / 1000 | 0}s — engine reported: ${r.backend}.`;
    } catch (e) {
      msg.textContent = `Warm-up failed: ${e.message}`;
    }
  });
}
