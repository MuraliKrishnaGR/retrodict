"""The whole Retrodict run for one photo, with a step-by-step trace for the UI and the demo.

  perceive     -> object kinds (Muse Spark Open) + one mask per object (SAM 3.1)
  propose      -> one reading of the whole photo: what most likely happened, and what each claim rests on
  gather       -> SAM 3.1 searches for what every claim rests on; new finds become objects
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
from app.order import order_by_rules
from app.propose import gather, propose
from app.second_look import second_look
from app.perceive import perceive
from app.quality import check_result
from app.read_state import name_objects, read_state
from app.reason import propagate_order, reason, rewrite_summary
from app.errors import save_details
from app.imaging import prepare_image
from app.verify import verify


def mostly_contradicted(e: dict) -> bool:
    """Left out of the summary when at least as many of its facts were contradicted as confirmed (run 5:
    'flour': 1 yes, 1 no). One wrong detail among several confirmed facts keeps it in (run 8: 'milk poured
    from the pan into the mug' had 2 yes and 1 'no' on small drips, and the drink vanished from the story)."""
    v = [ev.get("verdict") for ev in e["evidence"]]
    return v.count("no") > 0 and v.count("no") >= v.count("yes") + v.count("partly")


GATHERED_MOVES = 5     # the evidence is gathered before the agent starts, so it needs fewer moves

LOW_DETAIL_MP = 3.0    # phone cameras take 12+ MP; messaging apps shrink photos to about 1-2 MP


def photo_note(photo_path) -> str:
    """Tell people when the photo itself limits what can be seen (run on the milk photo: a 1.4 MP
    messaging-app copy, in which the sugar spoon inside the jar is barely visible)."""
    with Image.open(photo_path) as im:
        mp = im.width * im.height / 1e6
    if mp < LOW_DETAIL_MP:
        return (f"Low-detail photo ({mp:.1f} MP): small clues may be missed. "
                "The original from the phone's camera keeps much more detail than a copy sent through a chat app.")
    return ""


class Trace:
    def __init__(self, progress=None):
        self.steps = []
        self.progress = progress or (lambda kind, data: None)

    def start(self, name: str):
        self.progress("step_started", {"step": name})
        return time.time()

    def step(self, name: str, detail: str, started: float):
        s = {"step": name, "detail": detail, "seconds": round(time.time() - started, 1)}
        self.steps.append(s)
        self.progress("step_done", s)


async def run_photo(models, photo_path, out_dir, progress=None) -> dict:
    """progress(kind, data), if given, is told about every step and every agent move as it happens
    (the web page shows the investigation live)."""
    out_dir = Path(out_dir)
    trace = Trace(progress)

    t = trace.start("perceive")
    scene = await perceive(models, photo_path, out_dir)
    trace.step("perceive", f"{len(scene['objects'])} objects masked", t)

    result = {"photo": str(photo_path), "size": scene["size"], "photo_note": photo_note(photo_path),
              "objects": scene["objects"],
              "states": {}, "summary": "", "events": [], "rejected": [], "moves": [], "trace": trace.steps}
    if not scene["objects"]:
        result["summary"] = "No objects were found, so there is nothing to reason from."
        return _save(out_dir, result)

    img, jpeg = prepare_image(photo_path)            # identical to what SAM saw in perceive
    drawn = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"],
              "mask": np.array(Image.open(out_dir / o["mask_png"])) > 127} for o in scene["objects"]]
    objects = [dict(o) for o in scene["objects"]]

    scene_ctx = Scene(photo_path, out_dir, img, jpeg, objects, drawn, {})

    t = trace.start("propose")
    try:                                            # propose: one reading of the whole photo...
        claims = await propose(models, jpeg)
    except Exception as e:
        save_details(e)
        claims = []
    trace.step("propose", f"{len(claims)} claims", t)

    gathered = None
    if claims:                                      # ...gather: SAM 3.1 finds what each claim rests on
        t = trace.start("gather")
        gathered = await gather(models, scene_ctx, claims, progress=trace.progress)
        n = sum(1 for c in claims if c["found"])
        trace.step("gather", f"{len(gathered['searches'])} searches, {len(gathered['new'])} new objects; "
                             f"evidence found for {n} of {len(claims)} claims", t)
    result["proposals"] = [c["text"] for c in claims]
    result["claims"] = claims
    result["gather"] = gathered["searches"] if gathered else []

    t = trace.start("read")
    states = await read_state(models, objects, scene_ctx.detail_jpeg)
    scene_ctx.states = states
    name_objects(objects, states, drawn)               # "peel 3" -> "orange peel", from its own close-up
    scene_ctx._sheet = None
    n_traces = sum(len(s["traces"]) for s in states.values())
    trace.step("read", f"{len(states)} objects read and named, {n_traces} physical traces seen", t)

    t = trace.start("investigate")
    try:                                            # ...then prove: the agent decides each claim
        if gathered:
            found = await investigate(models, scene_ctx, max_moves=GATHERED_MOVES, progress=trace.progress,
                                      proposals=claims, min_searches=0)
        else:
            found = await investigate(models, scene_ctx, progress=trace.progress)
    except Exception as e:                          # a broken agent answer must not sink the run
        save_details(e)                             # ...but keep the exact failing line for fixing it
        found = {"finished": False, "moves": [], "error": f"{type(e).__name__}: {str(e)[:200]}"}
    result["moves"] = (gathered["moves"] if gathered else []) + found["moves"]
    if found["finished"]:
        timeline = {"summary": found["summary"], "events": found["events"]}
        tools = [m["move"] for m in found["moves"] if m["move"] != "finish"]
        detail = f"{len(tools)} tool move(s){' (' + ', '.join(tools) + ')' if tools else ''}"
    else:
        timeline = await reason(models, scene_ctx.sheet_jpeg(), states, scene_ctx.labels)
        detail = "agent gave no answer, used single reasoning call" + (f" ({found['error']})" if found.get("error") else "")
    clocked = sum(1 for e in timeline["events"] if e["clock"] != "no_clock")
    trace.step("investigate", f"{detail}; {len(timeline['events'])} events, {clocked} dated by a trace clock", t)

    t = trace.start("second_look")                  # a last look for steps the story missed
    try:
        extra = await second_look(models, scene_ctx, timeline["events"], progress=trace.progress)
    except Exception as e:
        save_details(e)
        extra = {"events": [], "unused": [], "searched": []}
    timeline["events"] += extra["events"]
    for i, e in enumerate(timeline["events"], 1):   # unique ids for the checker
        e["id"] = f"e{i}"
    trace.step("second_look", f"{len(extra['unused'])} unused objects looked at, "
                              f"{len(extra['events'])} missed step(s) added", t)
    result.update(summary=timeline["summary"], objects=objects, states=states)

    t = trace.start("verify")
    cited = list(dict.fromkeys(ev["object_id"] for e in timeline["events"] for ev in e["evidence"]))
    image = scene_ctx.evidence_jpeg(cited) if cited else scene_ctx.sheet_jpeg()
    checked = await verify(models, image, timeline["events"], scene_ctx.labels)
    trace.step("verify", f"{checked['claims']} claims checked, {len(checked['rejected'])} events rejected", t)
    checked["events"] = order_by_rules(checked["events"])   # physical order: peel, cut, into the bowl, mix
    propagate_order(checked["events"])              # surviving clocks tighten the steps that follow them
    for i, e in enumerate(checked["events"], 1):     # no gaps after rejections: e1, e2, e3...
        e["id"] = f"e{i}"
    result["events"], result["rejected"] = checked["events"], checked["rejected"]
    doubtful = [e for e in checked["events"] if mostly_contradicted(e)]
    added = any(e.get("source") == "second look" for e in checked["events"])
    if checked["rejected"] or doubtful or added:    # the story told to people uses only what held up
        sure = [e for e in checked["events"] if not mostly_contradicted(e)]
        result["summary"] = await rewrite_summary(models, sure, checked["rejected"] + doubtful,
                                                  states, scene_ctx.labels)
    return _save(out_dir, result)


def _save(out_dir: Path, result: dict) -> dict:
    result["quality_issues"] = check_result(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
