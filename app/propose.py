"""Pipeline stages 2 and 3: PROPOSE what happened, then GATHER the evidence for every claim.

propose -> one Muse Spark Open reading of the whole photo: what most likely happened, and for each claim
           the things in the photo it rests on, described by how they LOOK ("yellow banana peel",
           "steel tumbler", "black bottle cap").
gather  -> SAM 3.1 searches the photo for every one of those things (local, free). New finds become
           objects, so they are read and can be cited; finds that are already objects are linked.

The investigator then starts with each claim already tied to the objects it rests on, so its moves go to
deciding (zooming where a detail matters), not to hunting."""
from pydantic import BaseModel, Field, field_validator

from app.clocks import traces_guide

PROPOSE_PROMPT = """You are shown one photo. Reconstruct what most likely happened in this scene BEFORE the
photo was taken. Look closely: drinks, food, containers, utensils, peels, scraps, caps, crumbs, spills,
steam, levels, wrappers, cut surfaces.

Physical traces you may use (what each looks like, and what it is confused with):
{traces}

List the events oldest first, one action per event, with specific names ("the banana was peeled", not
"fruit was prepared"). Only include events the photo gives evidence for; say "most likely" where you infer.
Name a substance only as surely as the photo shows it: "brown powder (most likely coffee or sugar)", "a
milky drink (tea or coffee)"; a round flat thing beside an open jar may be its lid lying upside down.
For each event, "look_for": 1 to 3 things IN THE PHOTO the event rests on, each a short phrase of how it
LOOKS (colour, material, noun) so a segmenter can find it: "yellow banana peel", "green orange peel",
"pale apple pieces", "steel tumbler", "black bottle cap", "metal spoon". Include the container something
went into, and leftovers (peels, cores, caps, scraps).
Return JSON only:
{{"summary": "...", "events": [{{"description": "...", "look_for": ["..."]}}]}}"""

MAX_PHRASES = 10          # SAM searches in the gather stage (local CPU time)
MAX_NEW_OBJECTS = 8       # new objects it may add (each is read by its own call)
NEW_PER_PHRASE = 2


class Proposal(BaseModel):
    description: str
    look_for: list[str] = Field(default_factory=list)

    @field_validator("look_for", mode="before")
    @classmethod
    def _clean(cls, v):
        if isinstance(v, str):
            v = [v]
        out = []
        for p in v or []:
            p = " ".join(str(p).strip().lower().split())[:40]
            if p and p not in out:
                out.append(p)
        return out[:3]


class Proposals(BaseModel):
    summary: str = ""
    events: list[Proposal] = []


async def propose(models, jpeg: bytes) -> list:
    """[{'text', 'look_for'}], oldest first."""
    p = await models.spark_json(PROPOSE_PROMPT.format(traces=traces_guide()), Proposals, jpeg=jpeg, tag="propose")
    return [{"text": e.description.strip(), "look_for": e.look_for} for e in p.events if e.description.strip()]


def _shorter(phrase: str) -> str:
    """A fallback when the full look finds nothing: 'yellow banana peel' -> 'banana peel', 'tumbler' -> ''."""
    words = phrase.split()
    return " ".join(words[-2:]) if len(words) >= 3 else (words[-1] if len(words) == 2 else "")


async def gather(models, scene, claims: list, progress=None) -> dict:
    """SAM 3.1 searches for what every claim rests on. Adds each claim's 'found' (object ids).
    Returns {'claims', 'searches', 'new', 'moves'}."""
    phrases = []
    for c in claims:
        for p in c["look_for"]:
            if p not in phrases:
                phrases.append(p)
    phrases = phrases[:MAX_PHRASES]
    hits, searches, new_total, moves = {}, [], [], []
    for phrase in phrases:
        items = await models.sam_segment(scene.jpeg, scene.img.width, scene.img.height, phrase)
        used = phrase
        if not items and _shorter(phrase):
            used = _shorter(phrase)
            items = await models.sam_segment(scene.jpeg, scene.img.width, scene.img.height, used)
        known = []
        room = max(0, min(NEW_PER_PHRASE, MAX_NEW_OBJECTS - len(new_total)))
        new = scene.add_found(phrase, items, known=known, max_new=room)
        new_total += new
        hits[phrase] = list(dict.fromkeys(new + known))
        s = {"phrase": phrase, "searched_as": used, "new": new, "known": known}
        searches.append(s)
        found = ", ".join(f"{i} ({scene.labels[i]})" for i in hits[phrase])
        moves.append({"move": "search", "phrase": phrase, "object_id": None, "trace": None,
                      "thought": "looking for what a claim rests on",
                      "result": f"found {found}" if found else f"no '{phrase}' found"})
        if progress:
            progress("move", moves[-1])
    for c in claims:
        c["found"] = list(dict.fromkeys(i for p in c["look_for"] for i in hits.get(p, [])))
    return {"claims": claims, "searches": searches, "new": new_total, "moves": moves}
