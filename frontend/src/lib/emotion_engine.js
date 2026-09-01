/**
 * Which classifier produced a label, and saying so.
 *
 * Two backends, and the difference is never hidden:
 *
 *   "transformer"  RoBERTa-base fine-tuned on GoEmotions, running in the
 *                  Python service. Held-out macro-F1 0.4925 on the GoEmotions
 *                  test split (scripts/eval_emotion_model.py).
 *   "lexicon"      A word list running in this tab. Macro-F1 0.0979 on the
 *                  same split. It is a BASELINE and the interface says so.
 *
 * The Transformer is off unless someone turns it on, for the same reason the
 * Review 2 remote engine is: sending a sentence about your health to a server
 * should be a decision, not a default. `window.AEDT_API_URL` alone is not
 * enough; `useTransformer` must be set explicitly by the user through the
 * interface, and the choice is remembered per device.
 *
 * A failed call falls back to the lexicon and SAYS it fell back. It never
 * silently degrades and keeps the "transformer" label.
 */
import { lexiconEmotion } from "./twin.js";
import { BROWSER_MODEL, classifyInBrowser, isLoaded, loadModel }
  from "./browser_model.js";

const KEY = "aedt.engine";
const LEGACY_KEY = "aedt.useTransformer";   // pre-three-engine setting

export const ENGINES = { LEXICON: "lexicon", BROWSER: "browser", SERVICE: "service" };

export const ENGINE_INFO = {
  lexicon: {
    name: "Word-list baseline",
    short: "Baseline",
    detail: "A word list running in this tab. Instant, but it scores macro-F1 "
          + "0.098 against the model's 0.493 on the same held-out data. It is a "
          + "baseline, not the model.",
  },
  browser: {
    name: "RoBERTa in your browser",
    short: "RoBERTa (on device)",
    detail: `The real 124.7M-parameter model, running here. One-time download of `
          + `about ${BROWSER_MODEL.approxMB} MB, then roughly 20 ms per check-in. `
          + `Your text never leaves this device, and there is no server to be `
          + `asleep or unreachable.`,
  },
  service: {
    name: "RoBERTa via the analysis service",
    short: "RoBERTa (service)",
    detail: "The same model in the Python service. Nothing to download, but your "
          + "check-in sentence is sent to the server, and a free-tier instance "
          + "may need to wake up first.",
  },
};

export function apiUrl() {
  const built = import.meta.env?.VITE_API_BASE_URL || "";
  const runtime = typeof window !== "undefined" && window.AEDT_API_URL
    ? String(window.AEDT_API_URL) : "";
  const u = runtime || built;
  return u ? String(u).replace(/\/$/, "") : "";
}

export function serviceConfigured() { return Boolean(apiUrl()); }

/** Kept for callers that only ask "is a real model available at all?". */
export function transformerConfigured() { return true; }   // the browser one always is

/**
 * The default is the REAL MODEL, in the browser.
 *
 * It used to be the word list, which is how a product whose whole point is a
 * Transformer ended up showing baseline results to everyone who never opened
 * the settings. The download is 125 MB and is never silent: the size is stated
 * before it starts, progress is shown while it runs, and it is cached
 * afterwards. Anyone who would rather not can switch to the baseline, which is
 * then labelled a baseline everywhere it appears.
 */
export function getEngine() {
  try {
    const v = localStorage.getItem(KEY);
    if (v && Object.values(ENGINES).includes(v)) return v;
    // migrate the old boolean rather than silently resetting someone's choice
    if (localStorage.getItem(LEGACY_KEY) === "1") return ENGINES.SERVICE;
  } catch { /* storage unavailable; fall through to the default */ }
  return ENGINES.BROWSER;
}

export function setEngine(v) {
  try { localStorage.setItem(KEY, v); } catch { /* not fatal */ }
}

/** True when the browser model's weights are already in this tab. */
export function browserModelReady() { return isLoaded(); }

/** Start the download explicitly. Never called without the user asking. */
export function warmBrowserModel(onProgress) { return loadModel(onProgress); }

/** What actually happened, so the interface never has to guess. */
export const STATE = {
  MODEL: "model",         // A: the research model produced this
  FALLBACK: "fallback",   // B: it did not, and we say so
  WAKING: "waking",       // C: the service is starting; not a failure yet
};

/** Distinguish "still starting" from "genuinely broken". */
function classifyFailure(err, elapsedMs) {
  const m = String(err && err.message || err);
  if (m.includes("aborted") || m.includes("signal") || elapsedMs > 20000)
    return "timeout";                 // Render cold start looks exactly like this
  if (m.includes("Failed to fetch") || m.includes("NetworkError"))
    return "unreachable";             // CORS or DNS or offline
  if (/HTTP 5\d\d/.test(m)) return "server";
  return "other";
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Ask the backend whether the model is loaded.
 * Cheap, so it is used to detect a cold start before committing to a fallback.
 */
export async function probeHealth(timeoutMs = 8000) {
  const base = apiUrl();
  if (!base) return { reachable: false, modelReady: false };
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, { signal: ctl.signal });
    if (!res.ok) return { reachable: true, modelReady: false };
    const d = await res.json();
    const status = d?.model?.status;
    return { reachable: true, modelReady: status === "loaded",
             loading: status === "loading",     // starting, not broken
             reason: d?.model?.reason, model: d?.model?.name };
  } catch {
    return { reachable: false, modelReady: false };
  } finally { clearTimeout(t); }
}

/**
 * Classify one sentence, with a BOUNDED retry across a cold start.
 *
 * The previous version made one attempt and fell back to the word list on any
 * failure, which on Render's free tier means the first request of the day
 * always got the baseline while the interface said nothing useful. Now:
 *
 *   attempt 1  ->  if it looks like a cold start, report WAKING via onState
 *   wait, retry (bounded: MAX_ATTEMPTS, growing delay)
 *   success    ->  STATE.MODEL
 *   exhausted  ->  STATE.FALLBACK, with the reason attached
 *
 * `onState` is called as the situation changes so the UI can show progress
 * rather than freezing. The function NEVER returns a lexicon answer labelled
 * as the model: `state` and `backend` always describe what really ran.
 */
const MAX_ATTEMPTS = 3;
const ATTEMPT_TIMEOUT_MS = 30000;
const RETRY_DELAYS_MS = [2000, 4000];

/** GoEmotions label -> check-in label. Mirrors aedt/emotion/detect.py exactly. */
const TO_CHECKIN = {
  nervousness: "anxiety", fear: "anxiety", embarrassment: "anxiety",
  sadness: "sadness", grief: "sadness", disappointment: "sadness", remorse: "sadness",
  anger: "anger", annoyance: "anger", disgust: "anger", disapproval: "anger",
  joy: "joy", excitement: "joy", amusement: "joy", love: "joy",
  optimism: "joy", pride: "joy", admiration: "joy",
  gratitude: "gratitude", relief: "calm", approval: "calm", caring: "calm",
  desire: "calm",
  confusion: "confusion", curiosity: "confusion",
  // "realization" was mapped to confusion only to give all 28 labels a target.
  // Realisation is arguably the OPPOSITE of confusion, and the mapping produced
  // a real failure. Coverage is not a semantic justification; unmapped labels
  // surface as neutral with the raw label shown beside them.
  surprise: "confusion", neutral: "neutral",
};

/** Collapse the 28 raw labels onto the check-in taxonomy, taking the max. */
function collapse(raw) {
  const out = {};
  for (const [label, score] of raw) {
    const k = TO_CHECKIN[label] || "neutral";
    out[k] = Math.max(out[k] || 0, score);
  }
  return Object.entries(out).sort((a, b) => b[1] - a[1]);
}

/**
 * Classify one sentence with whichever engine the user chose.
 *
 * The returned `backend` and `state` always describe what ACTUALLY ran, never
 * what was requested. A lexicon answer is never labelled as a model.
 */
export async function classify(text, { onState = () => {} } = {}) {
  const engine = getEngine();

  if (engine === ENGINES.LEXICON) {
    return { ...lexiconEmotion(text), model: "lexicon-v1",
             backend: "lexicon", state: STATE.FALLBACK, chosen: true };
  }

  // ---- in-browser model: no network call per check-in, no server to fail ----
  if (engine === ENGINES.BROWSER) {
    try {
      if (!isLoaded()) {
        onState({ state: STATE.WAKING, phase: "download",
                  reason: `downloading the model (about ${BROWSER_MODEL.approxMB} MB, once)` });
        await loadModel((p) => onState({ state: STATE.WAKING, phase: "download",
                                         pct: p.pct, reason: "downloading the model" }));
      }
      const { raw, inferenceMs } = await classifyInBrowser(text);
      const ranked = collapse(raw);
      return {
        label: ranked[0][0], score: ranked[0][1], backend: "transformer",
        model: `${BROWSER_MODEL.id} (in browser)`, rawLabel: raw[0][0],
        distribution: ranked.slice(0, 6), rawTop: raw.slice(0, 6),
        inferenceMs, state: STATE.MODEL, ranOnDevice: true,
      };
    } catch (e) {
      const reason = `the in-browser model could not be loaded (${e.message})`;
      onState({ state: STATE.FALLBACK, reason });
      return { ...lexiconEmotion(text), model: "lexicon-v1", backend: "lexicon",
               state: STATE.FALLBACK,
               note: `${reason.charAt(0).toUpperCase()}${reason.slice(1)}. `
                   + "A simplified word-list baseline produced this result instead." };
    }
  }

  // ---- the Python service, with a bounded retry across a cold start ----
  const base = apiUrl();
  if (!base) {
    return { ...lexiconEmotion(text), model: "lexicon-v1", backend: "lexicon",
             state: STATE.FALLBACK,
             note: "No analysis service is configured. A simplified word-list "
                 + "baseline produced this result instead." };
  }
  let lastReason = "";

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ATTEMPT_TIMEOUT_MS);
    const t0 = performance.now();
    try {
      const res = await fetch(`${base}/api/emotion`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: ctl.signal, body: JSON.stringify({ text }),
      });
      if (res.status === 503) {
        lastReason = "the service is running but its model is not loaded yet";
        onState({ state: STATE.WAKING, attempt, reason: lastReason });
        if (attempt < MAX_ATTEMPTS) { await sleep(RETRY_DELAYS_MS[attempt - 1]); continue; }
        break;
      }
      if (res.status === 404) {
        // the deployed build predates this endpoint: retrying cannot help
        lastReason = "the service does not provide the emotion endpoint "
                   + "(its deployed build is older than this app)";
        break;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      const p = d.prediction || {};
      if (p.backend !== "transformer") {
        lastReason = "the service answered without running the model";
        break;
      }
      return {
        label: p.emotion, score: p.confidence, backend: "transformer",
        model: p.model, rawLabel: p.raw_label, ambiguous: p.ambiguous,
        distribution: p.distribution, rawTop: p.raw_top,
        inferenceMs: p.inference_ms ?? d.inference_ms, state: STATE.MODEL,
        roundTripMs: Math.round(performance.now() - t0),
      };
    } catch (e) {
      const kind = classifyFailure(e, performance.now() - t0);
      lastReason = { timeout: "the service did not answer in time",
                     unreachable: "the service could not be reached",
                     server: "the service returned an error" }[kind]
                   || `unexpected error (${e.message})`;
      if ((kind === "timeout" || kind === "server") && attempt < MAX_ATTEMPTS) {
        onState({ state: STATE.WAKING, attempt, reason: lastReason });
        await sleep(RETRY_DELAYS_MS[attempt - 1]);
        continue;
      }
      break;
    } finally { clearTimeout(timer); }
  }

  onState({ state: STATE.FALLBACK, reason: lastReason });
  return { ...lexiconEmotion(text), model: "lexicon-v1", backend: "lexicon",
           state: STATE.FALLBACK,
           note: `${lastReason.charAt(0).toUpperCase()}${lastReason.slice(1)}. `
               + "A simplified word-list baseline produced this result instead. "
               + "You can switch to the in-browser model, which needs no server." };
}
