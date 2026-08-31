#!/usr/bin/env python3
"""Inline the Vite build into ONE self-contained HTML file.

Produces a file that opens from disk with no server and no network, which is the
form to use for a live demonstration on an unfamiliar machine or a room with bad
wifi. The analysis is unchanged: it is the same bundle, just inlined.

    npm --prefix frontend run build
    python3 scripts/build_single_file.py dist/aedt-standalone.html
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")


def build(out_path: Path) -> Path:
    index = DIST / "index.html"
    if not index.exists():
        sys.exit(f"{index} not found. Run: npm --prefix frontend run build")

    css = Path(glob.glob(str(DIST / "assets" / "*.css"))[0]).read_text(encoding="utf-8")
    js = Path(glob.glob(str(DIST / "assets" / "*.js"))[0]).read_text(encoding="utf-8")
    body = index.read_text(encoding="utf-8").split("<body>", 1)[1].split("</body>", 1)[0]

    # charset first, so the page decodes correctly even if a host serves it without one
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>AEDT — Longitudinal Drift Analysis</title>\n"
        f'<link rel="stylesheet" href="{FONTS}">\n'
        f"<style>\n{css}\n</style>\n</head>\n<body>\n{body.strip()}\n"
        f'<script type="module">\n{js}\n</script>\n</body>\n</html>\n'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DIST / "aedt-standalone.html"
    written = build(ROOT / target if not Path(target).is_absolute() else target)
    print(f"{written}  ({written.stat().st_size:,} bytes)")
