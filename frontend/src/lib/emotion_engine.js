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

const KEY = "aedt.useTransformer";

export const ENGINE_INFO = {
  transformer: {
    name: "RoBERTa (GoEmotions)",
    detail: "124.7M-parameter Transformer in the Python service. "
          + "Held-out macro-F1 0.4925.",
  },
  lexicon: {
    name: "Word-list baseline",
    detail: "Runs in this tab, nothing leaves your device. "
          + "Held-out macro-F1 0.0979 — a baseline, not the model.",
  },
};

/**
 * The backend URL, resolved at BUILD time by Vite.
 *
 * It used to come only from `window.AEDT_API_URL`, set by `frontend/config.js`
 * — a file `index.html` never loaded. So the variable was always undefined in
 * the built bundle, the Transformer was permanently unavailable, and the
 * interface silently offered the word list. That is fixed by baking the value
 * in at build time; the window global still works as a runtime override for
 * anyone who wants one.
 */
export function apiUrl() {
  const built = import.meta.env?.VITE_API_BASE_URL || "";
  const runtime = typeof window !== "undefined" && window.AEDT_API_URL
    ? String(window.AEDT_API_URL) : "";
  const u = runtime || built;
  return u ? String(u).replace(/\/$/, "") : "";
}

export function transformerConfigured() { return Boolean(apiUrl()); }

export function useTransformer() {
  if (!transformerConfigured()) return false;
  try { return localStorage.getItem(KEY) === "1"; } catch { return false; }
}

export function setUseTransformer(on) {
  try { localStorage.setItem(KEY, on ? "1" : "0"); } catch { /* not fatal */ }
}

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
    return { reachable: true, modelReady: d?.model?.status === "loaded",
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

export async function classify(text, { onState = () => {} } = {}) {
  if (!useTransformer()) {
    return { ...lexiconEmotion(text), model: "lexicon-v1",
             state: STATE.FALLBACK, chosen: true };
  }
  const base = apiUrl();
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
        // the service is up but the model is not loaded: a real, reported state
        lastReason = "the service is running but its model is not loaded";
        onState({ state: STATE.WAKING, attempt, reason: lastReason });
        if (attempt < MAX_ATTEMPTS) { await sleep(RETRY_DELAYS_MS[attempt - 1]); continue; }
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
        inferenceMs: d.inference_ms, state: STATE.MODEL,
        roundTripMs: Math.round(performance.now() - t0),
      };
    } catch (e) {
      const kind = classifyFailure(e, performance.now() - t0);
      lastReason = { timeout: "the service did not answer in time",
                     unreachable: "the service could not be reached",
                     server: "the service returned an error" }[kind]
                   || `unexpected error (${e.message})`;
      if (kind === "timeout" && attempt < MAX_ATTEMPTS) {
        // A free-tier instance sleeps and takes ~30-60s to wake. That is not
        // a failure yet, and calling it one is what made the app look broken.
        onState({ state: STATE.WAKING, attempt, reason: lastReason });
        await sleep(RETRY_DELAYS_MS[attempt - 1]);
        continue;
      }
      if (attempt < MAX_ATTEMPTS && kind === "server") {
        onState({ state: STATE.WAKING, attempt, reason: lastReason });
        await sleep(RETRY_DELAYS_MS[attempt - 1]);
        continue;
      }
      break;
    } finally { clearTimeout(timer); }
  }

  onState({ state: STATE.FALLBACK, reason: lastReason });
  return { ...lexiconEmotion(text), model: "lexicon-v1", state: STATE.FALLBACK,
           // The banner already says the model was not used; this supplies the
           // reason only, so the two do not read as a stutter.
           note: `${lastReason.charAt(0).toUpperCase()}${lastReason.slice(1)}. `
               + "A simplified word-list baseline produced this result instead." };
}
