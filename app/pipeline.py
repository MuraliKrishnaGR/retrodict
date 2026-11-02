"""The whole Retrodict run for one photo, with a step-by-step trace for the UI and the demo.

  perceive     -> object kinds (Muse Spark Open) + one mask per object (SAM 3.1)
  read         -> current state of each object from its own close-up (Muse Spark Open)
  investigate  -> the agent zooms, searches with SAM and checks trace clocks, then gives a backward
                  timeline in which every event cites visible evidence; trace clocks fix the times
                  (falls back to a single reasoning call if the agent fails)
  verify       -> independent check of each piece of evidence on sharp close-ups; contradicted events are
                  rejected, and a trace clock only dates an event if the checker confirms the trace
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.agent import Scene, investigate
from app.perceive import perceive
from app.quality import check_result
from app.read_state import read_state
from app.reason import reason, rewrite_summary
from app.imaging import prepare_image
from app.verify import verify


class Trace:
    def __init__(self):
        self.steps = []

    def step(self, name: str, detail: str, started: float):
        self.steps.append({"step": name, "detail": detail, "seconds": round(time.time() - started, 1)})


async def run_photo(models, photo_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    trace = Trace()

    t = time.time()
    scene = await perceive(models, photo_path, out_dir)
    trace.step("perceive", f"{len(scene['objects'])} objects masked", t)

    result = {"photo": str(photo_path), "size": scene["size"], "objects": scene["objects"],
              "states": {}, "summary": "", "events": [], "rejected": [], "moves": [], "trace": trace.steps}
    if not scene["objects"]:
        result["summary"] = "No objects were found, so there is nothing to reason from."
        return _save(out_dir, result)

    img, jpeg = prepare_image(photo_path)            # identical to what SAM saw in perceive
    drawn = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"],
              "mask": np.array(Image.open(out_dir / o["mask_png"])) > 127} for o in scene["objects"]]
    objects = [dict(o) for o in scene["objects"]]

    t = time.time()
    scene_ctx = Scene(photo_path, out_dir, img, jpeg, objects, drawn, {})
    states = await read_state(models, scene_ctx.sheet_jpeg(), objects)
    scene_ctx.states = states
    n_traces = sum(len(s["traces"]) for s in states.values())
    trace.step("read", f"{len(states)} objects read, {n_traces} physical traces seen", t)

    t = time.time()
    try:
        found = await investigate(models, scene_ctx)
    except Exception as e:                          # a broken agent answer must not sink the run
        found = {"finished": False, "moves": [], "error": f"{type(e).__name__}: {str(e)[:200]}"}
    result["moves"] = found["moves"]
    if found["finished"]:
        timeline = {"summary": found["summary"], "events": found["events"]}
        tools = [m["move"] for m in found["moves"] if m["move"] != "finish"]
        detail = f"{len(tools)} tool move(s){' (' + ', '.join(tools) + ')' if tools else ''}"
    else:
        timeline = await reason(models, scene_ctx.sheet_jpeg(), states, scene_ctx.labels)
        detail = "agent gave no answer, used single reasoning call" + (f" ({found['error']})" if found.get("error") else "")
    clocked = sum(1 for e in timeline["events"] if e["clock"] != "no_clock")
    trace.step("investigate", f"{detail}; {len(timeline['events'])} events, {clocked} dated by a trace clock", t)
    result.update(summary=timeline["summary"], objects=objects, states=states)

    t = time.time()
    cited = list(dict.fromkeys(ev["object_id"] for e in timeline["events"] for ev in e["evidence"]))
    image = scene_ctx.evidence_jpeg(cited) if cited else scene_ctx.sheet_jpeg()
    checked = await verify(models, image, timeline["events"], scene_ctx.labels)
    trace.step("verify", f"{checked['claims']} claims checked, {len(checked['rejected'])} events rejected", t)
    for i, e in enumerate(checked["events"], 1):     # no gaps after rejections: e1, e2, e3...
        e["id"] = f"e{i}"
    result["events"], result["rejected"] = checked["events"], checked["rejected"]
    if checked["rejected"]:
        result["summary"] = await rewrite_summary(models, checked["events"], checked["rejected"],
                                                  states, scene_ctx.labels)
    return _save(out_dir, result)


def _save(out_dir: Path, result: dict) -> dict:
    result["quality_issues"] = check_result(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
