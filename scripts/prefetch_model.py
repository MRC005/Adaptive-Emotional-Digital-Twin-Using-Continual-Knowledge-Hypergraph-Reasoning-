#!/usr/bin/env python3
"""Download the emotion model at BUILD time, not at first request.

Measured: a cold start that also downloads takes ~250 s; loading the same
artefact from disk takes ~1 s. Running this in the build step moves that cost
off the request path entirely and out of the health-check window.

Exits 0 even on failure: a missing model must not fail the whole deploy, since
the API's other endpoints are still useful and /health reports the model as
unavailable with a reason.
"""
from __future__ import annotations

import sys
import time

def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
        from aedt.emotion.onnx_detect import ONNX_FILE, ONNX_REPO
        t0 = time.time()
        for f in (ONNX_FILE, "onnx/tokenizer.json"):
            p = hf_hub_download(ONNX_REPO, f)
            print(f"  prefetched {f} -> {p}")
        print(f"model prefetched in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"WARNING: prefetch failed ({type(exc).__name__}: {exc}). "
              "The service will still start; /health will report the model as "
              "unavailable and the browser will say so.", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
