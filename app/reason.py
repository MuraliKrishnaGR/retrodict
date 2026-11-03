"""Pipeline step 3: reason BACKWARD from the visible traces to a timeline of earlier events.
Every event must cite visible evidence; trace clocks then fix or narrow its time range."""
import json
import re

from app.clocks import apply_clock, clock_for, clock_table_text
from app.schemas import TRACE_NAMES, Summary, Timeline

REASON_PROMPT = """You are a careful investigator reasoning BACKWARD from physical traces to what most likely
happened before this photo was taken. The image shows the scene with numbered objects and close-ups.

What is visible now, per object (from an earlier careful look):
{states}

Trace clocks for the traces seen in this photo (physical rules; minutes BEFORE the photo):
{clocks}

Rules:
- Every event must cite 1 to 3 pieces of evidence: an object id from the list AND ONE short fact that is
  visible in the photo NOW (not a guess about the past).
- Connect the objects: a container with leftovers of the same food or liquid that is now in another
  container most likely held it before (it was poured or served from it).
- If an event is dated by one of the traces above, set "trace" to that trace name and keep its time range
  inside the trace clock. Otherwise set "trace" to null, minutes_before_min to 0, and an honest maximum.
- A MISSING trace proves nothing: no visible steam does not mean a drink is old.
- Give time as a RANGE: minutes_before_min to minutes_before_max. Never a single exact time.
- Lower "confidence" when evidence is weak. Add "alternatives" when another explanation fits.
- Refer to people only as "someone". Do not invent objects. 1 to 6 events, earliest first.
- "undo": how a rewind of the REAL photo should show the scene before this event. One of: "remove" (the
  object was put there by this event), "add_steam" (the drink was still steaming), "refill" (some was eaten
  or drunk), "move" (the object was moved), "mark" (cannot be shown honestly with pixels). "undo_object": the
  id of the object that change applies to (the brownie that was eaten, not the mug the crumbs lie next to).
- If the scene shows no sign of recent activity, return few or no events.
- "summary": 1 or 2 plain sentences (at most 40 words) for a person who has never seen this system. Say what most likely
  happened, or that nothing shows recent activity. Mention only what IS visible: never list traces or
  things that are absent. Do not use object ids or words with underscores; name objects by what they are.
Return ONLY JSON: {{"summary": "...", "events": [{{"id": "e1", "description": "...",
"minutes_before_min": 0, "minutes_before_max": 30, "evidence": [{{"object_id": "o1",
"observation": "..."}}], "trace": null, "confidence": 0.6, "alternatives": [], "undo": "mark",
"undo_object": "o1"}}]}}"""


def states_text(states: dict, labels: dict) -> str:
    return "\n".join(f"- {oid} ({labels.get(oid, '?')}): {s['what']}. "
                     f"Visible: {'; '.join(s['observations'])}. Traces: {', '.join(s['traces']) or 'none'}."
                     for oid, s in states.items())


def ground_events(timeline: Timeline, valid_ids: set) -> list:
    """Keep only evidence that points at real objects, drop events left with none, apply the clocks.
    The model's own order (earliest first) is kept: it reasons about sequence (milk poured, THEN drunk),
    which sorting by time ranges would destroy when ranges are wide. Only if that order contradicts the
    ranges (a later event surely longer ago than an earlier one) are events sorted by range instead."""
    events = []
    for e in timeline.events:
        ev = [x.model_dump() for x in e.evidence if x.object_id in valid_ids and x.observation.strip()]
        if not ev:
            continue
        lo, hi, status = apply_clock(e.minutes_before_min, e.minutes_before_max, e.trace)
        if status == "no_clock":
            lo = 0.0    # nothing visible sets a minimum time: it could have happened just before the photo
        d = e.model_dump()
        c = clock_for(e.trace)
        d.update(evidence=ev, minutes_before_min=lo, minutes_before_max=hi, clock=status,
                 clock_rule=(c.rule if c else None),   # the physics behind the range, shown to people
                 model_range=[e.minutes_before_min, e.minutes_before_max])
        events.append(d)
    if not order_is_consistent(events):
        events.sort(key=lambda d: (-d["minutes_before_max"], -d["minutes_before_min"]))
    for i, d in enumerate(events, 1):
        d["id"] = f"e{i}"
    return events


MAX_WORDS = 60   # same cap the quality gate uses


def order_is_consistent(events: list) -> bool:
    """events earliest first: no later event may be certainly longer ago than an earlier one."""
    return not any(events[j]["minutes_before_min"] > events[i]["minutes_before_max"]
                   for i in range(len(events)) for j in range(i + 1, len(events)))


def _objects(e: dict) -> set:
    """Objects it cites, and the foods it names: steps on the same banana are linked even when each cites
    a different piece of it (the peel, the slices)."""
    from app.order import links
    return links(e)


DATED_CLOCKS = {"clock_agrees", "narrowed_by_clock", "replaced_by_clock"}


def is_dated(e: dict) -> bool:
    """True when the event's time comes from a confirmed physical clue (its own clock, or its place in order
    next to an event that has one). Otherwise its numbers are only the model's guess and are not shown."""
    return e["dated"] if "dated" in e else e.get("clock") in DATED_CLOCKS


def propagate_order(events: list) -> list:
    """Events that share an object happened in the order given (poured INTO the glass, then drunk FROM it),
    so a dated step dates the steps around it: a later step cannot be longer ago than an earlier one, and an
    earlier step is at least as long ago as a later one. Only DATED times are spread: spreading the model's
    guesses would dress a guess up as evidence. Run AFTER the verifier, so only clocks that survived the
    trace check count. Events on different objects are left alone: their order is a guess."""
    for e in events:
        e["dated"] = is_dated(e)
    for _ in range(len(events)):                     # repeat so a clock reaches along a chain of steps
        changed = False
        for i, a in enumerate(events):
            for b in events[i + 1:]:
                if not (_objects(a) & _objects(b)):
                    continue
                # a is dated, b came after it: b is at most as long ago as a (b keeps any known minimum;
                # clues that contradict each other are left as they are rather than forced together)
                if a["dated"] and b["minutes_before_min"] <= a["minutes_before_max"] and \
                        (not b["dated"] or a["minutes_before_max"] < b["minutes_before_max"]):
                    b.update(minutes_before_max=a["minutes_before_max"], dated=True,
                             bounded_by=f"happened after: {a['description']}")
                    changed = True
                if b["dated"] and b["minutes_before_min"] > a["minutes_before_min"] and \
                        (not a["dated"] or b["minutes_before_min"] <= a["minutes_before_max"]):
                    a["minutes_before_min"] = b["minutes_before_min"]       # undated: a known minimum only
                    a["bounded_by"] = f"happened before: {b['description']}"
                    changed = True
        if not changed:
            break
    return events


def plain_text(text: str, states: dict, labels: dict) -> str:
    """Safety net for anything shown to people: object ids become object names, trace names become
    plain words, and the text is kept to at most two sentences."""
    names = {oid: (s.get("what") or labels.get(oid) or oid) for oid, s in states.items()}

    def name_for(m):
        n = names.get(m.group(0))
        return f"the {n}" if n else m.group(0)
    text = re.sub(r"\bo\d+\b", name_for, text or "")
    for t in TRACE_NAMES:
        text = re.sub(rf"\b{t}\b", t.replace("_", " "), text)
    text = re.sub(r"\bthe the\b", "the", text, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:2]).strip()
    if len(out.split()) > MAX_WORDS and len(sentences[0].split()) <= MAX_WORDS:
        out = sentences[0].strip()
    return out


async def reason(models, sheet_jpeg: bytes, states: dict, labels: dict) -> dict:
    seen = {t for s in states.values() for t in s["traces"]}
    prompt = REASON_PROMPT.format(states=states_text(states, labels), clocks=clock_table_text(seen))
    timeline = await models.spark_json(prompt, Timeline, jpeg=sheet_jpeg, tag="reason")
    events = ground_events(timeline, set(states))
    for e in events:
        e["description"] = plain_text(e["description"], states, labels)
    return {"summary": plain_text(timeline.summary, states, labels), "events": events}


def to_json(x) -> str:
    return json.dumps(x, indent=2)


SUMMARY_PROMPT = """Write what most likely happened before a photo was taken, for a person who has never seen
this system: 1 or 2 plain sentences, at most 40 words. Use ONLY these confirmed events:
{kept}
An independent check contradicted these (fully or in part), so do not mention them at all:
{rejected}
Refer to people only as "someone". Return ONLY JSON: {{"summary": "..."}}"""


async def rewrite_summary(models, events: list, rejected: list, states: dict, labels: dict) -> str:
    """The first summary was written before the verifier ran. If it rejected an event, the summary may
    still tell that story (run 3: "whipped foam in a bowl"), so it is rewritten from what survived."""
    if not events:
        return "Nothing in the photo could be confirmed as a sign of recent activity."
    kept = "\n".join(f"- {e['description']}" for e in events)
    gone = "\n".join(f"- {e['description']}" for e in rejected)
    out = await models.spark_json(SUMMARY_PROMPT.format(kept=kept, rejected=gone), Summary, tag="summary")
    return plain_text(out.summary, states, labels)
