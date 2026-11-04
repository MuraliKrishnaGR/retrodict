"""Pipeline step 2: read each object's CURRENT state from its own full-resolution close-up."""
import asyncio
import re

from app.clocks import traces_guide
from app.schemas import ObjectState

TRACES = traces_guide()

STATE_PROMPT = """This image is a sharp close-up of ONE object, {oid} ({label}), cut from the full-size photo
of a scene, in six views, each labelled in its corner: as photographed; shadows lifted (faint steam, film or
drips; glare looks brighter there too); glare dimmed (what reflections hide on or behind steel, glass or
plastic); and three views of the object ALONE on black: a whiteness map (white things bright, coloured things
dark) and the two light edits that show the most detail in this object. Shapes INSIDE a jar or glass (a
scoop standing in powder, a film on a drink, drips) show best in the views of the object alone. Other objects in the scene, for
context only: {others}. You are a careful investigator: describe {oid} only, and only what is VISIBLE NOW.
Do not guess what happened.

Give:
- "name": 1 to 3 words, the specific everyday name of what {oid} is, as a person would say it: "banana
  peel", "orange peel", "apple pieces", "steel tumbler", "bottle cap", "fruit bowl". Never a vague one
  ("peel", "object", "item") when the close-up shows which kind it is.
  When the kind cannot be told for sure from the photo (a brown powder may be sugar, coffee or cocoa; a
  milky drink may be tea or coffee; a clear liquid may be water or something else), name it by how it
  looks ("brown powder", "milky brown drink") and put the likely kinds in "what" ("most likely coffee or
  sugar"). Never state one of them as fact.
- "what": a short description that tells it apart (colour, material, lid, contents).
- "observations": 2 to 5 short visible facts (fill level, open or closed, wet or dry, steam, crumbs,
  wrapper torn, position, residue). Write "unclear" rather than guess. For a cup, glass or pot, always
  say: how far the level is below the rim, whether the inside wall ABOVE the level is clean or has a film
  or drips, and what is left inside an empty one. For a spoon or utensil, say whether it is wet or dirty.
  For a see-through container, say what is inside it, including any spoon or scoop standing in it.
  Never leave out a shape you cannot name: describe it ("something white stuck in the brown powder") and
  give its most likely identity from context, marked as likely ("most likely a spoon or scoop").
  A round, shallow thing lying next to an open jar, bottle or pot may be its LID, upside down: flat, with
  a seal, liner, ridges or thread inside and no real depth. Say so, and do not read a drink, skin or film
  on it unless liquid clearly fills it.
- "traces": any of these physical traces you can actually SEE on it (exact names, can be empty). Each is
  defined by what it looks like; only use a name when the crop really shows that:
{traces}
- "confidence": 0 to 1, how clearly the crop shows its state.
Return ONLY JSON: {{"object_id": "{oid}", "name": "...", "what": "...", "observations": ["..."], "traces": [], "confidence": 0.8}}"""


async def read_state(models, objects: list, image_for) -> dict:
    """Each object is read from its OWN full-resolution close-up, one call per object, in parallel.
    One shared sheet shrank every object to a 256-320 px tile, and small clues were lost (the white
    sugar spoon inside the jar, the film above the milk). image_for(object_id) -> JPEG bytes.
    Returns {object_id: ObjectState dict}; a failed read leaves that object 'unclear', never the run."""
    async def one(o):
        others = "; ".join(f"{x['id']} = {x['label']}" for x in objects if x["id"] != o["id"]) or "none"
        prompt = STATE_PROMPT.format(oid=o["id"], label=o["label"], others=others, traces=TRACES)
        try:
            st = await models.spark_json(prompt, ObjectState, jpeg=image_for(o["id"]), tag=f"read_{o['id']}")
        except Exception:
            return None
        return dict(st.model_dump(), object_id=o["id"])

    results = await asyncio.gather(*(one(o) for o in objects))
    states = {}
    for o, st in zip(objects, results):
        states[o["id"]] = st or {"object_id": o["id"], "what": o["label"], "observations": ["unclear"],
                                 "traces": [], "confidence": 0.0}
    return states


VAGUE = {"object", "item", "thing", "stuff", "unclear", "unknown", "something"}


def name_objects(objects: list, states: dict, drawn: list = ()) -> dict:
    """Give each object the specific name its own close-up showed ("peel 3" becomes "orange peel"), numbered
    only when several share it. The first name is kept as 'first_label'. Returns {id: new label}."""
    names = {}
    for o in objects:
        n = " ".join(str(states.get(o["id"], {}).get("name") or "").lower().replace("_", " ").split())[:30]
        names[o["id"]] = n if n and n not in VAGUE and not re.fullmatch(r"o\d+", n) else ""
    counts, seen, out = {}, {}, {}
    for oid, n in names.items():
        if n:
            counts[n] = counts.get(n, 0) + 1
    for o in objects:
        n = names[o["id"]]
        if not n:
            continue
        seen[n] = seen.get(n, 0) + 1
        label = n if counts[n] == 1 else f"{n} {seen[n]}"
        o.setdefault("first_label", o["label"])
        o["label"] = out[o["id"]] = label
    for d in drawn:
        if d["id"] in out:
            d["label"] = out[d["id"]]
    return out
