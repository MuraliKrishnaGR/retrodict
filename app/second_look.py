"""A second look for what the story missed.

After the investigator finishes, SAM 3.1 looks for a few common leftovers that tell of a step (a glass, a
cap, a cloth, a wrapper), and every object no event uses is gathered. ONE Muse Spark Open call sees sharp
close-ups of just those objects, with the story so far, and adds an event only if one of them shows a
step the story is missing (water poured into a glass that nobody mentioned). Its events go through the
same independent check as all the others."""
from app.reason import ground_events, plain_text
from app.schemas import UNDO_KINDS, Timeline

LEFTOVERS = ["drinking glass", "steel tumbler", "bottle cap", "cloth", "wrapper"]
MAX_UNUSED = 6
STAGE_SHARE = 0.15           # a tray or table this big is the stage, not a clue

SECOND_PROMPT = """You are checking a reconstruction of what happened BEFORE this photo was taken.
The story so far (earliest first):
{story}

The image shows sharp close-ups of things in the photo that NO event in the story uses:
{unused}

For each of them, decide: does it show a step that happened before the photo and that the story is
MISSING (for example: water in a glass means something was poured into it; a cap lying apart means a
bottle was opened)? Add an event only for a step the photo shows evidence of; repeat nothing already in the
story. Most often the answer is no new events.
Each new event cites 1 or 2 of these objects with ONE short fact visible NOW. Time: minutes_before_min 0
and an honest minutes_before_max, "trace" null unless a listed trace is visible. "undo": "remove" (the
object was put there), "refill", "move" or "mark". People are "someone".
Return ONLY JSON: {{"summary": "", "events": [{{"id": "n1", "description": "...", "minutes_before_min": 0,
"minutes_before_max": 30, "evidence": [{{"object_id": "o1", "observation": "..."}}], "trace": null,
"confidence": 0.5, "alternatives": [], "undo": "mark", "undo_object": "o1"}}]}}"""


def unused_objects(scene, events: list) -> list:
    used = {ev["object_id"] for e in events for ev in e["evidence"]}
    total = scene.img.width * scene.img.height
    return [o["id"] for o in scene.objects
            if o["id"] not in used and o.get("area", 0) < STAGE_SHARE * total][:MAX_UNUSED]


async def second_look(models, scene, events: list, progress=None) -> dict:
    """Returns {'events': new grounded events, 'searched': [...], 'unused': [...]}."""
    searched = []
    for phrase in LEFTOVERS:
        items = await models.sam_segment(scene.jpeg, scene.img.width, scene.img.height, phrase)
        new = scene.add_found(phrase, items, max_new=1)
        searched.append({"phrase": phrase, "new": new})
        if progress and new:
            progress("move", {"move": "search", "phrase": phrase, "thought": "second look for missed steps",
                              "result": f"found {new[0]} ({phrase})"})
    unused = unused_objects(scene, events)
    if not unused:
        return {"events": [], "searched": searched, "unused": []}
    story = "\n".join(f"- {e['description']}" for e in events) or "- (nothing yet)"
    lines = "\n".join(f"- {oid} ({scene.labels[oid]}): {scene.states.get(oid, {}).get('what', '')}"
                      for oid in unused)
    reply = await models.spark_json(SECOND_PROMPT.format(story=story, unused=lines), Timeline,
                                    jpeg=scene.evidence_jpeg(unused), tag="second_look")
    reply.events = reply.events[:3]
    found = ground_events(reply, set(unused))
    for e in found:
        e["description"] = plain_text(e["description"], scene.states, scene.labels)
        e["source"] = "second look"
        if e.get("undo") not in UNDO_KINDS:
            e["undo"] = "mark"
    return {"events": found, "searched": searched, "unused": unused}
