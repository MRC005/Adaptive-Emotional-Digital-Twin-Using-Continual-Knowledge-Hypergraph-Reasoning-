/**
 * Does the in-browser model agree with the Python ONNX path?
 *
 * The interface says "RoBERTa GoEmotions" when the browser engine runs. That
 * claim is only honest if the browser produces what the reference produces, so
 * it is measured rather than assumed -- the same discipline as
 * scripts/verify_onnx_agreement.py, which in turn pins the Python ONNX path to
 * the torch reference. Chained, that gives: browser == python ONNX == torch.
 *
 *   node scripts/verify_browser_agreement.mjs        (run from frontend/)
 *
 * Exits non-zero if a top label changes or a score moves more than TOL.
 */
import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;
const TOL = 0.08;   // WASM int8 vs native int8 kernels differ slightly

// label + score measured on the Python ONNX path (aedt/emotion/onnx_detect.py)
const EXPECTED = [
  ["My presentation is tomorrow. Even though I prepared, I keep thinking something will go wrong. I slept only four hours and I cannot relax.",
   "disappointment", 0.5728],
  ["I am nervous because I have an important presentation tomorrow.", "nervousness", 0.6300],
  ["I finally completed my work and I feel relieved and proud.", "joy", 0.3800],
  ["Today was a normal day. Nothing particularly good or bad happened.", "approval", 0.4800],
];

const pipe = await pipeline("text-classification",
  "SamLowe/roberta-base-go_emotions-onnx", { dtype: "q8" });

let failures = 0;
console.log(`${"text".padEnd(46)} ${"expected".padEnd(18)} ${"got".padEnd(18)} delta`);
for (const [text, wantLabel, wantScore] of EXPECTED) {
  const out = await pipe(text, { top_k: 28 });
  const top = [...out].sort((a, b) => b.score - a.score)[0];
  const delta = Math.abs(top.score - wantScore);
  const ok = top.label === wantLabel && delta <= TOL;
  if (!ok) failures++;
  console.log(`${text.slice(0, 44).padEnd(46)} ${`${wantLabel} ${wantScore.toFixed(3)}`.padEnd(18)}`
    + `${`${top.label} ${top.score.toFixed(3)}`.padEnd(18)} ${delta.toFixed(4)}  ${ok ? "ok" : "MISMATCH"}`);
}
console.log(`\ntolerance ${TOL}; ${failures ? `${failures} MISMATCH(ES)` : "all agree"}`);
if (failures) {
  console.error("The browser model no longer matches the reference. The interface "
    + "must stop claiming RoBERTa produced these results.");
}
process.exit(failures ? 1 : 0);
