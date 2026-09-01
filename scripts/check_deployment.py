#!/usr/bin/env python3
"""Is the deployed backend actually running the current code?

Written after the deployed service served 5 routes while main had 10 for three
consecutive pushes, with nothing visibly wrong: /health returned "ok" the whole
time. A build can be stale and healthy at once, and only a route-level
comparison catches it.

    python3 scripts/check_deployment.py [URL]

Exit 0 when the deployment matches, 1 when it is stale or the model is not
loaded, 2 when it cannot be reached.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT = "https://aedt-api.onrender.com"


def fetch(url: str, timeout: int = 90):
    with urllib.request.urlopen(url, timeout=timeout) as f:
        return json.load(f)


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    from backend.app import app
    local = {r.path for r in app.routes if hasattr(r, "methods")}

    print(f"checking {base}")
    try:
        spec = fetch(f"{base}/openapi.json")
    except Exception as exc:
        print(f"  UNREACHABLE: {type(exc).__name__}: {exc}")
        return 2

    deployed = set(spec.get("paths", {}))
    missing = {p for p in local if p.startswith(("/api", "/health"))
               or p == "/"} - deployed
    print(f"  routes deployed : {len(deployed)}")
    print(f"  routes in code  : {len(local)}")

    ok = True
    if missing:
        ok = False
        print("  STALE BUILD. Missing from the deployment:")
        for p in sorted(missing):
            print(f"    {p}")
        print("  Fix: Render dashboard -> the service -> Manual Deploy ->")
        print("       Deploy latest commit, then enable Auto-Deploy.")

    try:
        h = fetch(f"{base}/health")
    except Exception as exc:
        print(f"  /health unreachable: {exc}")
        return 2
    model = h.get("model")
    if model is None:
        ok = False
        print("  /health does not report the model: this is the OLD health route.")
    else:
        print(f"  model           : {model.get('status')} ({model.get('name')})")
        if model.get("status") != "loaded":
            ok = False
            print(f"    reason: {model.get('reason', 'not stated')}")

    print("  VERDICT:", "deployment matches the code" if ok else "DEPLOYMENT IS NOT CURRENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
