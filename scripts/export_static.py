"""Build the clickable live page (docs/live) from finished runs: the real Retrodict web page, reading
saved results instead of a server. GitHub Pages serves it; no key, no model calls.

  python scripts/export_static.py                 # runs in examples/ -> docs/live/
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.web import run_summary  # noqa: E402

STATIC = ROOT / "app" / "static"
NOTE = ('<div class="actions"><p class="static-note">A finished run, read only. '
        '<a href="../">Back to the project</a></p></div>')


def main(src=ROOT / "examples", dst=ROOT / "docs" / "live"):
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "static").mkdir(parents=True)
    (dst / "api" / "runs").mkdir(parents=True)
    for f in ("app.js", "style.css"):
        shutil.copy(STATIC / f, dst / "static" / f)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace("<html", '<html data-static="1"', 1)
    a, b = html.index('<div class="actions">'), html.index("</header>")
    html = html[:a] + NOTE + "\n" + html[b:]
    (dst / "index.html").write_text(html, encoding="utf-8")
    runs = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        s = run_summary(d)
        if not s:
            continue
        runs.append(s)
        shutil.copytree(d, dst / "files" / d.name)
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        (dst / "api" / "runs" / f"{d.name}.json").write_text(
            json.dumps({"name": d.name, "result": res, "files": files}), encoding="utf-8")
    (dst / "api" / "runs.json").write_text(json.dumps(sorted(runs, key=lambda r: -r["updated"])), encoding="utf-8")
    print(f"Saved {dst} ({len(runs)} run(s)): open docs/live/index.html through GitHub Pages")


if __name__ == "__main__":
    main()
