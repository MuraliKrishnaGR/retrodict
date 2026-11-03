"""The strongest simple alternative to Retrodict: ONE Muse Spark Open call with the photo and a good prompt.

No object finding, no per-object reading, no tools, no checker, no clocks. If the full system cannot beat
this on scenes it was not tuned on, the extra machinery is not earning its place. The prompt is written to
be a fair, capable competitor (same task, same honesty rules), not a straw man."""
from pydantic import BaseModel, Field

from app.clocks import traces_guide

BASELINE_PROMPT = """You are shown one photo. Reconstruct what most likely happened in this scene BEFORE the
photo was taken. Look closely: drinks, food, containers, utensils, crumbs, spills, steam, levels, wrappers.

Physical traces you may use (what each looks like, and what it is confused with):
{traces}

List the events oldest first. Each event: one plain sentence saying what happened, and a time range in
minutes before the photo (minutes_before_min to minutes_before_max). Only include events the photo gives
evidence for. Say "most likely" where you are inferring. Return JSON only:
{{"summary": "...", "events": [{{"description": "...", "minutes_before_min": 0, "minutes_before_max": 30}}]}}"""


class BaselineEvent(BaseModel):
    description: str
    minutes_before_min: float = Field(default=0, ge=0)
    minutes_before_max: float = Field(default=60, ge=0)


class BaselineTimeline(BaseModel):
    summary: str = ""
    events: list[BaselineEvent] = []


async def run_baseline(models, jpeg: bytes) -> dict:
    tl = await models.spark_json(BASELINE_PROMPT.format(traces=traces_guide()), BaselineTimeline, jpeg=jpeg,
                                 tag="baseline")
    events = []
    for i, e in enumerate(tl.events, 1):
        lo, hi = sorted((e.minutes_before_min, e.minutes_before_max))
        # same rule as the full system: a time needs a confirmed clue, and nothing here confirms anything
        events.append({"id": f"e{i}", "description": e.description.strip(), "minutes_before_min": lo,
                       "minutes_before_max": hi, "status": "unchecked", "clock": "no_clock", "evidence": []})
    return {"kind": "baseline: one Muse Spark Open call", "summary": tl.summary, "events": events}
