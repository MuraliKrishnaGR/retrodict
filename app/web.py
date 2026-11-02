"""The Retrodict web page: upload or take a photo, watch the investigation live, read the timeline.

  python scripts/serve.py          -> http://127.0.0.1:8000
  python scripts/serve.py --lan    -> also reachable from a phone on the same Wi-Fi

Runs happen one at a time (SAM uses the CPU). The same photo is never paid for twice: a finished run
is reused, and every model call is cached on disk anyway. Uploaded photos stay on this computer."""
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.cost import BudgetExceeded
from app.errors import explain
from app.pipeline import run_photo
from app.rewind import make_rewind

STATIC = Path(__file__).parent / "static"
UPLOADS = config.ROOT / "data" / "uploads"
MAX_UPLOAD_MB = 25
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

app = FastAPI(title="Retrodict")
JOBS: dict = {}
RUN_LOCK = asyncio.Lock()
_models = None


def get_models():
    """One set of models for the whole server, so SAM loads once. Tests replace this."""
    global _models
    if _models is None:
        from app.clients.models import Models
        _models = Models()
    return _models


def out_root() -> Path:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUT_DIR


def run_summary(run_dir: Path) -> dict | None:
    p = run_dir / "result.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {"name": run_dir.name, "summary": r.get("summary", ""), "events": len(r.get("events", [])),
            "updated": p.stat().st_mtime, "rewind": (run_dir / "rewind.mp4").exists()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/runs")
def list_runs():
    runs = [s for d in out_root().iterdir() if d.is_dir() and (s := run_summary(d))]
    return sorted(runs, key=lambda r: -r["updated"])


@app.get("/api/runs/{name}")
def get_run(name: str):
    d = out_root() / name
    if "/" in name or "\\" in name or ".." in name or not (d / "result.json").exists():
        raise HTTPException(404, "No such run")
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    return JSONResponse({"name": name, "result": res, "files": files})


@app.post("/api/run")
async def start_run(photo: UploadFile = File(...)):
    ext = IMAGE_TYPES.get(photo.content_type or "")
    if ext is None:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP photo.")
    data = await photo.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The photo is larger than {MAX_UPLOAD_MB} MB.")
    digest = hashlib.sha256(data).hexdigest()[:12]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{digest}{ext}"
    if not path.exists():
        path.write_bytes(data)
    name = f"photo_{digest}"
    job = {"id": uuid.uuid4().hex[:12], "name": name, "events": [], "done": False}
    JOBS[job["id"]] = job
    if run_summary(out_root() / name):                 # this exact photo was already reconstructed
        job["events"].append({"kind": "cached", "data": {"name": name}})
        job["events"].append({"kind": "done", "data": {"name": name}})
        job["done"] = True
    else:
        asyncio.create_task(_run_job(job, path, out_root() / name))
    return {"job": job["id"], "name": name}


async def _run_job(job: dict, path: Path, out_dir: Path):
    def progress(kind, data):
        job["events"].append({"kind": kind, "data": data, "t": round(time.time(), 2)})

    if RUN_LOCK.locked():
        progress("queued", {"message": "Another photo is being reconstructed; this one starts next."})
    async with RUN_LOCK:
        try:
            res = await run_photo(get_models(), path, out_dir, progress=progress)
            if res.get("events"):
                progress("step_started", {"step": "rewind"})
                clip = await asyncio.to_thread(make_rewind, out_dir)
                progress("step_done", {"step": "rewind", "detail": f"{clip['seconds']}s clip", "seconds": 0})
            progress("done", {"name": job["name"]})
        except BudgetExceeded as e:
            progress("error", {"message": f"Stopped by the spend limit: {e}"})
        except Exception as e:                          # shown on the page in plain words
            progress("error", {"message": f"{type(e).__name__}: {str(e)[:200]}", "fix": explain(e)})
        finally:
            job["done"] = True


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")

    async def stream():
        sent = 0
        while True:
            while sent < len(job["events"]):
                yield f"data: {json.dumps(job['events'][sent])}\n\n"
                sent += 1
            if job["done"] and sent >= len(job["events"]):
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/files", StaticFiles(directory=str(out_root())), name="files")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
