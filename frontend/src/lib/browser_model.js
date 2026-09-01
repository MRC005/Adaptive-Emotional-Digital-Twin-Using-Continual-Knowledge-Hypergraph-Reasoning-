/**
 * The SAME RoBERTa GoEmotions model, running in this browser.
 *
 * WHY THIS EXISTS
 *
 * The backend path has four failure modes, and three of them are not bugs in
 * this project at all:
 *
 *   the deployed build can be stale        (observed: 5 routes live, 10 in code)
 *   free-tier instances sleep              (cold start, looks like an outage)
 *   free tier caps memory at 512 MB        (torch needs 664 MB and OOMs)
 *   cross-origin requests need CORS        (one misconfiguration = silent fallback)
 *
 * Running the model here removes all four. There is no server to be stale, to
 * sleep, to run out of memory, or to be blocked by CORS — and the check-in text
 * never leaves the device, which is a better privacy position than the backend
 * could ever offer for this kind of data.
 *
 * IT IS THE SAME MODEL, and the numbers were compared rather than assumed:
 *
 *   text                              browser        python ONNX
 *   "presentation ... cannot relax"   disappointment 0.552   0.573
 *   "nervous ... presentation"        nervousness    0.625   0.630
 *   "relieved and proud"              joy            0.423   0.380
 *   "normal day"                      approval       0.436   0.480
 *
 * Same top label every time; the small differences are WASM vs native int8
 * kernels. The Python side is in turn pinned to the torch reference by
 * scripts/verify_onnx_agreement.py.
 *
 * THE HONEST COST: a 125 MB download the first time. That is real, it is stated
 * in the interface before the user opts in, progress is reported while it
 * happens, and the browser caches it afterwards. It is never started silently.
 */

const MODEL_ID = "SamLowe/roberta-base-go_emotions-onnx";
const MODEL_BYTES = 125_400_000;

let pipe = null;
let loading = null;

export const BROWSER_MODEL = {
  id: MODEL_ID,
  approxBytes: MODEL_BYTES,
  approxMB: Math.round(MODEL_BYTES / 1e6),
};

/** True once the weights are in memory in this tab. */
export function isLoaded() { return pipe !== null; }

/**
 * Load the model, reporting progress.
 *
 * `onProgress({phase, loaded, total, pct, file})` is called as files download.
 * Concurrent callers share one load rather than starting several 125 MB
 * downloads, which is what happens if this is not guarded.
 */
export async function loadModel(onProgress = () => {}) {
  if (pipe) return pipe;
  if (loading) return loading;

  loading = (async () => {
    // Dynamic import so the library is a separate chunk: someone who never
    // chooses this engine should not pay for it in the initial bundle.
    const { pipeline, env } = await import("@huggingface/transformers");
    env.allowLocalModels = false;      // fetch from the Hub, not from our origin

    const seen = new Map();
    const report = (p) => {
      if (p.file) seen.set(p.file, p);
      let loaded = 0, total = 0;
      for (const v of seen.values()) {
        loaded += v.loaded || 0;
        total += v.total || 0;
      }
      onProgress({
        phase: p.status === "ready" ? "ready" : "downloading",
        file: p.file, loaded, total: total || MODEL_BYTES,
        pct: total ? Math.min(100, Math.round((loaded / total) * 100)) : 0,
      });
    };

    pipe = await pipeline("text-classification", MODEL_ID,
                          { dtype: "q8", progress_callback: report });
    onProgress({ phase: "ready", pct: 100 });
    return pipe;
  })();

  try {
    return await loading;
  } catch (e) {
    loading = null;                    // let a later attempt retry cleanly
    throw e;
  }
}

//: The head is multi-label, so every label is requested. The option is
//: `top_k` (snake_case) and needs an explicit count: `topk` is silently
//: ignored and `top_k: 0` returns an empty array, both of which quietly
//: collapse the output to one label and hide the multi-label reporting.
const N_LABELS = 28;

/**
 * Classify one sentence. Returns all 28 raw GoEmotions scores, sorted.
 */
export async function classifyInBrowser(text) {
  const p = await loadModel();
  const t0 = performance.now();
  const out = await p(text, { top_k: N_LABELS });
  const ranked = [...out].sort((a, b) => b.score - a.score);
  return {
    raw: ranked.map((o) => [o.label, o.score]),
    inferenceMs: Math.round(performance.now() - t0),
  };
}
