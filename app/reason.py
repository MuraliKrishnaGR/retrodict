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
- Every event must cite 1 to 3 pieces of evidence: an object id from the list AND a short fact that is
  visible in the photo NOW (not a guess about the past).
- If an event is dated by one of the traces above, set "trace" to that trace name and keep its time range
  inside the trace clock. Otherwise set "trace" to null, minutes_before_min to 0, and an honest maximum.
- A MISSING trace proves nothing: no visible steam does not mean a drink is old.
- Give time as a RANGE: minutes_before_min to minutes_before_max. Never a single exact time.
- Lower "confidence" when evidence is weak. Add "alternatives" when another explanation fits.
- Refer to people only as "someone". Do not invent objects. 1 to 6 events, earliest first.
- "undo": how a rewind of the REAL photo should show the scene before this event. One of: "remove" (the
  object was put there by this event), "add_steam" (the drink was still steaming), "refill" (some was eaten
  or drunk), "move" (the object was moved), "mark" (cannot be shown honestly with pixels).
- If the scene shows no sign of recent activity, return few or no events.
- "summary": 1 or 2 plain sentences (at most 40 words) for a person who has never seen this system. Say what most likely
  happened, or that nothing shows recent activity. Mention only what IS visible: never list traces or
  things that are absent. Do not use object ids or words with underscores; name objects by what they are.
Return ONLY JSON: {{"summary": "...", "events": [{{"id": "e1", "description": "...",
"minutes_before_min": 0, "minutes_before_max": 30, "evidence": [{{"object_id": "o1",
"observation": "..."}}], "trace": null, "confidence": 0.6, "alternatives": [], "undo": "mark"}}]}}"""


def states_text(states: dict, labels: dict) -> str:
    return "\n".join(f"- {oid} ({labels.get(oid, '?')}): {s['what']}. "
                     f"Visible: {'; '.join(s['observations'])}. Traces: {', '.join(s['traces']) or 'none'}."
                     for oid, s in states.items())


def ground_events(timeline: Timeline, valid_ids: set) -> list:
    """Keep only evidence that points at real objects, drop events left with none, apply the clocks,
    and order earliest first (largest minutes_before first)."""
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
    events.sort(key=lambda d: (-d["minutes_before_max"], -d["minutes_before_min"]))
    for i, d in enumerate(events, 1):
        d["id"] = f"e{i}"
    return events


MAX_WORDS = 60   # same cap the quality gate uses


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
An independent check found NO support for these, so do not mention them at all:
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
