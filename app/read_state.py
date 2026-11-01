"""Pipeline step 2: read each object's CURRENT state from its own close-up crop."""
from app.clocks import CLOCKS, NO_CLOCK
from app.schemas import StateReport

TRACES = ", ".join(sorted(list(CLOCKS) + list(NO_CLOCK)))

STATE_PROMPT = """This image has two parts: on top, a photo of a scene with numbered objects; below, a close-up crop
of each object labelled with its id. You are a careful investigator. For EACH object id listed, describe
only what is VISIBLE NOW. Do not guess what happened.

Objects: {objects}

For each object give:
- "what": a short description that tells it apart (colour, material, lid, contents).
- "observations": 2 to 5 short visible facts (fill level, open or closed, wet or dry, steam, crumbs,
  wrapper torn, position, residue). Write "unclear" rather than guess.
- "traces": any of these physical traces you can actually SEE on it (exact words, can be empty):
  {traces}
- "confidence": 0 to 1, how clearly the crop shows its state.
Return ONLY JSON: {{"objects": [{{"object_id": "o1", "what": "...", "observations": ["..."],
"traces": [], "confidence": 0.8}}]}}"""


async def read_state(models, sheet_jpeg: bytes, objects: list) -> dict:
    """objects: SceneObject dicts. Returns {object_id: ObjectState dict}; unknown ids are dropped."""
    listing = "; ".join(f"{o['id']} = {o['label']}" for o in objects)
    report = await models.spark_json(STATE_PROMPT.format(objects=listing, traces=TRACES),
                                     StateReport, jpeg=sheet_jpeg, tag="read_state")
    valid = {o["id"] for o in objects}
    states = {s.object_id: s.model_dump() for s in report.objects if s.object_id in valid}
    for o in objects:                       # every object gets an entry, even if the model skipped it
        states.setdefault(o["id"], {"object_id": o["id"], "what": o["label"], "observations": ["unclear"],
                                    "traces": [], "confidence": 0.0})
    return states
