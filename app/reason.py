"""Pipeline step 3: reason BACKWARD from the visible traces to a timeline of earlier events.
Every event must cite visible evidence; trace clocks then fix or narrow its time range."""
import json
import re

from app.clocks import apply_clock, clock_table_text
from app.schemas import TRACE_NAMES, Timeline

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
  inside the trace clock. Otherwise set "trace" to null and give an honest, wide range.
- Give time as a RANGE: minutes_before_min to minutes_before_max. Never a single exact time.
- Lower "confidence" when evidence is weak. Add "alternatives" when another explanation fits.
- Refer to people only as "someone". Do not invent objects. 1 to 6 events, earliest first.
- If the scene shows no sign of recent activity, return few or no events.
- "summary": 1 or 2 plain sentences for a person who has never seen this system. Say what most likely
  happened, or that nothing shows recent activity. Mention only what IS visible: never list traces or
  things that are absent. Do not use object ids or words with underscores; name objects by what they are.
Return ONLY JSON: {{"summary": "...", "events": [{{"id": "e1", "description": "...",
"minutes_before_min": 0, "minutes_before_max": 30, "evidence": [{{"object_id": "o1",
"observation": "..."}}], "trace": null, "confidence": 0.6, "alternatives": []}}]}}"""


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
        d = e.model_dump()
        d.update(evidence=ev, minutes_before_min=lo, minutes_before_max=hi, clock=status,
                 model_range=[e.minutes_before_min, e.minutes_before_max])
        events.append(d)
    events.sort(key=lambda d: (-d["minutes_before_max"], -d["minutes_before_min"]))
    for i, d in enumerate(events, 1):
        d["id"] = f"e{i}"
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
    return " ".join(sentences[:2]).strip()


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
