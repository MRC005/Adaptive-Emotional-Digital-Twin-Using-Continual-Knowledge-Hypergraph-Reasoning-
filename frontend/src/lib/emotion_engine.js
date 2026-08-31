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

export function apiUrl() {
  const u = typeof window !== "undefined" ? window.AEDT_API_URL : "";
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

/**
 * Classify one sentence.
 * Returns {label, score, backend, model, rawLabel, note} — `backend` is always
 * the engine that actually produced the answer, not the one that was asked.
 */
export async function classify(text) {
  if (!useTransformer()) return { ...lexiconEmotion(text), model: "lexicon-v1" };

  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 45000);   // Render cold start
  try {
    const res = await fetch(`${apiUrl()}/api/emotion`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: ctl.signal, body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    const p = d.prediction || {};
    if (p.backend !== "transformer") {
      return { ...lexiconEmotion(text), model: "lexicon-v1",
               note: "The service answered with its own lexicon fallback, so "
                   + "the Transformer did not run." };
    }
    return { label: p.emotion, score: p.confidence, backend: "transformer",
             model: p.model, rawLabel: p.raw_label, ambiguous: p.ambiguous };
  } catch (e) {
    return { ...lexiconEmotion(text), model: "lexicon-v1",
             note: `The analysis service could not be reached (${e.message}), so `
                 + "the word-list baseline was used instead." };
  } finally {
    clearTimeout(timer);
  }
}
