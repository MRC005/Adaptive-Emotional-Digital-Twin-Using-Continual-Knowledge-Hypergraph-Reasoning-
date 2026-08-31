/**
 * Analysis engine abstraction.
 *
 * The product talks to an ENGINE, never to an implementation. Today the only
 * active engine is `local`, which runs the validated JavaScript port of the
 * estimator in the browser. A `remote` engine that forwards the identical
 * request to a Python FastAPI service is implemented below. Enabling it needs
 * `window.AEDT_API_URL` AND `window.AEDT_ENGINE = "remote"`, and a service that
 * implements POST /api/analyze — no product code changes.
 *
 * Both engines take the same input ({byPid, K, options}) and return the same
 * result shape, so swapping them cannot change what the UI renders.
 *
 * WHY THE LOCAL ENGINE IS TRUSTWORTHY: it is not a re-derivation. It is a port
 * of aedt/models/ordinal.py and aedt/estimators/slope_ratio.py, and
 * tests/regression/test_js_python_agreement.py runs fixed cases through BOTH
 * implementations and fails if the fitted slopes diverge by more than 1e-3.
 * Measured agreement at the time of writing: 1.8e-5.
 */
import { runPipeline } from "./estimator.js";

export const ENGINES = {
  local: {
    id: "local",
    name: "In-browser engine",
    detail: "Validated JavaScript port of the Python estimator. Nothing leaves your machine.",
    available: () => true,
    async run({ byPid, K, options }) {
      // yield a frame so the UI can paint its progress state before we block
      await new Promise((r) => setTimeout(r, 0));
      return runPipeline(byPid, K, options);
    },
  },
  remote: {
    id: "remote",
    name: "Python analysis service",
    detail: "Forwards the same request to the reference Python implementation.",
    // Requires BOTH a configured URL and an explicit opt-in. Presence of a URL is not
    // enough: the currently deployed service exposes a bounded demo endpoint
    // (/api/demo/run, fixed scenarios) and does NOT implement the general /api/analyze
    // contract below, so selecting it automatically would fail every analysis. The
    // opt-in is also a privacy boundary - an uploaded CSV is participant data and must
    // not leave the machine unless someone deliberately asks for that.
    available: () => Boolean(window.AEDT_API_URL) && window.AEDT_ENGINE === "remote",
    async run({ byPid, K, options }) {
      const base = String(window.AEDT_API_URL).replace(/\/$/, "");
      const ctl = new AbortController();
      const to = setTimeout(() => ctl.abort(), 90000);
      try {
        const res = await fetch(base + "/api/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          signal: ctl.signal, body: JSON.stringify({ byPid, K, options }),
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        return await res.json();
      } finally { clearTimeout(to); }
    },
  },
};

/**
 * The engine actually in use. Local unless someone has deliberately opted in to the
 * remote service, which is off by default for the reasons given above.
 */
export function activeEngine() {
  return ENGINES.remote.available() ? ENGINES.remote : ENGINES.local;
}

export async function analyze(payload) {
  const engine = activeEngine();
  const t0 = performance.now();
  const result = await engine.run(payload);
  return { ...result, engine: engine.id, engineName: engine.name,
           wallMs: Math.round(performance.now() - t0) };
}
