"""The investigator agent: Muse Spark Open works like a detective with tools, instead of answering once.

Each turn it looks at the evidence so far and picks ONE move:
  zoom   <object>  -> a close-up from the ORIGINAL full-resolution photo (real extra detail)
  search <noun>    -> SAM 3.1 looks for something not yet found ("spoon", "crumbs", "wet ring")
  clock  <trace>   -> the physical rule for how long that trace lasts
  finish           -> the timeline: every event cites visible evidence and says how to rewind it
It has a small budget of moves; if it never produces a valid answer, the pipeline falls back to a
single reasoning call. Every move is recorded so the UI can replay the investigation."""
import numpy as np
from PIL import Image, ImageOps

from app.clocks import CLOCKS, NO_CLOCK, clock_for, clock_table_text
from app.imaging import boosted_pair, contact_sheet, evidence_sheet, save_mask_png, to_jpeg
from app.perceive import IOU_DUPLICATE, mask_iou
from app.reason import ground_events, plain_text, states_text
from app.schemas import UNDO_KINDS, AgentStep, Timeline

MAX_MOVES = 4             # tool moves before it must finish (cost and time cap)
MAX_NEW_PER_SEARCH = 3
ZOOM_ABOVE = 0.8          # a zoom also shows the space above the object, where steam rises
SEARCH_MIN_SCORE = 0.5

AGENT_PROMPT = """You are an investigator reconstructing what most likely happened BEFORE this photo was taken.
The image shows: {image_note}

What is known so far, per object (visible NOW):
{states}

Trace clocks for traces seen so far (minutes BEFORE the photo):
{clocks}

Your investigation so far:
{history}

Tools (one per turn). Use a tool only if it could change your answer:
- "zoom" + "object_id": see a sharper close-up of one object from the full-resolution photo. Usually the
  most useful move: zoom on the objects your main events rely on before you finish.
- "search" + "phrase": look for something not yet found, as a plain noun (e.g. "spoon", "crumbs",
  "wet ring", "tea bag", "wrapper"). New finds become objects you can cite.
- "clock" + "trace": the timing rule for a trace you can SEE on an object (rules already listed above
  need no lookup). Traces: {all_traces}
- "finish": give the final answer.
You have {moves_left} tool move(s) left.{must_finish}

Time rules:
- Only a trace you can SEE gives a minimum time. A MISSING trace proves nothing: no visible steam does not
  mean a drink is old (steam is often invisible in photos). Without a seen trace, minutes_before_min is 0.

When you finish:
- "summary": 1 or 2 plain sentences, at most 40 words, saying what most likely happened (not a list of
  objects); no object ids, no underscores.
- "events": 0 to 6 events, earliest first. Each cites 1 to 3 pieces of evidence: an object id AND a fact
  visible NOW. Time is a RANGE (minutes_before_min to minutes_before_max). If a trace dates it, set "trace"
  and stay inside its clock; otherwise "trace" is null and the range is honest and wide.
- "undo": how a rewind of the REAL photo should show the scene before this event. One of:
  "remove" (the object was put there by this event), "add_steam" (the drink was still steaming),
  "refill" (some was eaten or drunk), "move" (the object was moved), "mark" (cannot be shown honestly).
- Lower "confidence" for weak evidence; add "alternatives" when another explanation fits; people are "someone".
Return ONLY JSON: {{"thought": "...", "action": "zoom|search|clock|finish", "object_id": null, "phrase": null,
"trace": null, "summary": "", "events": [{{"id": "e1", "description": "...", "minutes_before_min": 0,
"minutes_before_max": 30, "evidence": [{{"object_id": "o1", "observation": "..."}}], "trace": null,
"confidence": 0.6, "alternatives": [], "undo": "mark"}}]}}"""


class Scene:
    """Everything the agent can look at and add to."""

    def __init__(self, photo_path, out_dir, img, jpeg, objects, drawn, states):
        self.photo_path, self.out_dir = photo_path, out_dir
        self.img, self.jpeg = img, jpeg                  # prepared image (what SAM saw) and its bytes
        self.objects, self.drawn, self.states = objects, drawn, states
        self._sheet = None

    @property
    def labels(self):
        return {o["id"]: o["label"] for o in self.objects}

    def sheet_jpeg(self) -> bytes:
        if self._sheet is None:
            sheet = contact_sheet(self.img, self.drawn)
            sheet.save(self.out_dir / "sheet.jpg", quality=88)
            self._sheet = to_jpeg(sheet)
        return self._sheet

    def zoom_jpeg(self, object_id: str, margin: float = 0.25) -> bytes:
        """Close-up from the original photo at full resolution, not the shrunken copy."""
        o = next(o for o in self.objects if o["id"] == object_id)
        full = ImageOps.exif_transpose(Image.open(self.photo_path)).convert("RGB")
        sx, sy = full.width / self.img.width, full.height / self.img.height
        l, t, r, b = o["bbox"]
        mx, my, up = (r - l) * margin, (b - t) * margin, (b - t) * ZOOM_ABOVE
        box = (max(0, int((l - mx) * sx)), max(0, int((t - up) * sy)),
               min(full.width, int((r + mx) * sx)), min(full.height, int((b + my) * sy)))
        crop = full.crop(box)
        crop.thumbnail((768, 768))
        return to_jpeg(boosted_pair(crop))

    def evidence_jpeg(self, object_ids: list) -> bytes:
        """Sharp, contrast-boosted close-ups of the cited objects for the verifier (saved as evidence.jpg)."""
        full = ImageOps.exif_transpose(Image.open(self.photo_path)).convert("RGB")
        scale = (full.width / self.img.width, full.height / self.img.height)
        objs = [o for oid in object_ids for o in self.objects if o["id"] == oid]
        sheet = evidence_sheet(full, scale, objs)
        sheet.save(self.out_dir / "evidence.jpg", quality=88)
        return to_jpeg(sheet)

    def add_found(self, phrase: str, items: list) -> list:
        """Add SAM finds that are confident and not already known. Returns the new object ids."""
        new_ids = []
        for it in sorted(items, key=lambda x: -x.get("score", 0)):
            if len(new_ids) >= MAX_NEW_PER_SEARCH or it.get("score", 0) < SEARCH_MIN_SCORE:
                continue
            if it["mask"].sum() < 50 or any(mask_iou(it["mask"], d["mask"]) >= IOU_DUPLICATE for d in self.drawn):
                continue
            oid = f"o{len(self.objects) + 1}"
            mask_path = self.out_dir / "masks" / f"{oid}.png"
            save_mask_png(it["mask"], mask_path)
            label = phrase
            self.objects.append({"id": oid, "label": label, "bbox": it["bbox"], "area": int(it["mask"].sum()),
                                 "mask_png": mask_path.relative_to(self.out_dir).as_posix(),
                                 "score": round(float(it.get("score", 0)), 4), "concept": phrase, "found_by": "agent"})
            self.drawn.append({"id": oid, "label": label, "mask": it["mask"], "bbox": it["bbox"]})
            self.states[oid] = {"object_id": oid, "what": phrase, "traces": [], "confidence": round(float(it["score"]), 2),
                                "observations": [f"found by searching for '{phrase}'"]}
            new_ids.append(oid)
        if new_ids:
            self._sheet = None                          # the sheet now shows the new objects
        return new_ids


async def investigate(models, scene: Scene, max_moves: int = MAX_MOVES) -> dict:
    """Runs the tool loop. Returns {'summary','events','moves','finished'}; events are grounded and clocked."""
    history, moves, checked = [], [], set()
    image, image_note = scene.sheet_jpeg(), "the scene with numbered objects on top and close-ups below."
    all_traces = ", ".join(sorted(list(CLOCKS) + list(NO_CLOCK)))

    for turn in range(max_moves + 1):
        left = max_moves - turn
        seen = {t for s in scene.states.values() for t in s["traces"]} | checked
        prompt = AGENT_PROMPT.format(
            image_note=image_note, states=states_text(scene.states, scene.labels),
            clocks=clock_table_text(seen), history="\n".join(history) or "(nothing yet)",
            all_traces=all_traces, moves_left=left,
            must_finish="" if left else " You MUST finish now.")
        step = await models.spark_json(prompt, AgentStep, jpeg=image, tag=f"agent_{turn}")
        if left == 0 and step.action != "finish":
            break                                       # out of moves without an answer: caller falls back
        action = step.action
        image, image_note = scene.sheet_jpeg(), "the scene with numbered objects on top and close-ups below."

        if action == "zoom" and step.object_id in scene.labels:
            image = scene.zoom_jpeg(step.object_id)
            image_note = (f"a sharp close-up of {step.object_id} ({scene.labels[step.object_id]}) from the full "
                          "photo, with the space above it (steam rises). LEFT: as photographed. RIGHT: the same "
                          "crop with contrast boosted to reveal faint steam, wet shine or residue; glare also looks "
                          "brighter there, so count steam only if it rises from the drink as a soft haze.")
            result = f"close-up of {step.object_id} is shown in the image now"
        elif action == "search" and (step.phrase or "").strip():
            phrase = " ".join(step.phrase.strip().lower().split())[:40]
            items = await models.sam_segment(scene.jpeg, scene.img.width, scene.img.height, phrase)
            new_ids = scene.add_found(phrase, items)
            result = (f"found {', '.join(f'{i} ({phrase})' for i in new_ids)}" if new_ids
                      else f"no new '{phrase}' found")
        elif action == "clock" and (clock_for(step.trace) or step.trace in NO_CLOCK) and step.trace not in seen:
            result = f"{step.trace}: not seen on any object, so its clock does not apply here"
        elif action == "clock" and clock_for(step.trace):
            c = clock_for(step.trace)
            checked.add(step.trace)
            result = f"{step.trace}: {c.min_minutes:g} to {c.max_minutes:g} min ({c.rule})"
        elif action == "clock" and step.trace in NO_CLOCK:
            result = f"{step.trace}: no clock, it shows what happened, not when"
        elif action == "finish":
            timeline = Timeline(summary=step.summary, events=step.events)
            events = ground_events(timeline, set(scene.labels))
            for e in events:
                e["description"] = plain_text(e["description"], scene.states, scene.labels)
                if e.get("undo") not in UNDO_KINDS:
                    e["undo"] = "mark"
            moves.append({"move": "finish", "thought": step.thought, "result": f"{len(events)} events"})
            return {"summary": plain_text(step.summary, scene.states, scene.labels), "events": events,
                    "moves": moves, "finished": True}
        else:
            result = f"could not use {action} with those arguments"

        moves.append({"move": action, "thought": step.thought, "result": result,
                      "object_id": step.object_id, "phrase": step.phrase, "trace": step.trace})
        history.append(f"{turn + 1}. {action} -> {result}")
    return {"summary": "", "events": [], "moves": moves, "finished": False}
