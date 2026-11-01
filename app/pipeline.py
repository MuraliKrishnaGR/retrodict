"""The whole Retrodict run for one photo, with a step-by-step trace for the UI and the demo.

  perceive  -> object kinds (Muse Spark Open) + one mask per object (SAM 3.1)
  read      -> current state of each object from its own close-up (Muse Spark Open)
  reason    -> backward timeline, every event citing visible evidence; trace clocks fix the times
  verify    -> independent check of each piece of evidence; contradicted events are rejected
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.imaging import contact_sheet, to_jpeg
from app.perceive import perceive
from app.quality import check_result
from app.read_state import read_state
from app.reason import reason
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
              "states": {}, "summary": "", "events": [], "rejected": [], "trace": trace.steps}
    if not scene["objects"]:
        result["summary"] = "No objects were found, so there is nothing to reason from."
        return _save(out_dir, result)

    img = Image.open(out_dir / "input.jpg").convert("RGB")
    drawn = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"],
              "mask": np.array(Image.open(out_dir / o["mask_png"])) > 127} for o in scene["objects"]]
    sheet = contact_sheet(img, drawn)
    sheet.save(out_dir / "sheet.jpg", quality=88)
    sheet_jpeg = to_jpeg(sheet)
    labels = {o["id"]: o["label"] for o in scene["objects"]}

    t = time.time()
    states = await read_state(models, sheet_jpeg, scene["objects"])
    n_traces = sum(len(s["traces"]) for s in states.values())
    trace.step("read", f"{len(states)} objects read, {n_traces} physical traces seen", t)
    result["states"] = states

    t = time.time()
    timeline = await reason(models, sheet_jpeg, states, labels)
    clocked = sum(1 for e in timeline["events"] if e["clock"] != "no_clock")
    trace.step("reason", f"{len(timeline['events'])} events, {clocked} dated by a trace clock", t)
    result["summary"] = timeline["summary"]

    t = time.time()
    checked = await verify(models, sheet_jpeg, timeline["events"], labels)
    trace.step("verify", f"{checked['claims']} claims checked, {len(checked['rejected'])} events rejected", t)
    result["events"], result["rejected"] = checked["events"], checked["rejected"]
    return _save(out_dir, result)


def _save(out_dir: Path, result: dict) -> dict:
    result["quality_issues"] = check_result(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
