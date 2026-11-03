"""Data shapes for day 1. More are added on later days."""
from pydantic import BaseModel, Field, field_validator


class ObjectName(BaseModel):
    name: str = Field(description="plain generic noun SAM can find, e.g. 'mug', 'jar', 'packet'")
    count: int = Field(default=1, ge=1, le=10, description="how many of these are visible")
    desc: str = ""        # distinguishing details: colour, material, lid, contents
    why: str = ""         # what its state might reveal


class ObjectList(BaseModel):
    objects: list[ObjectName]


class SceneObject(BaseModel):
    id: str               # 'o1', 'o2', ...
    label: str            # 'mug'
    bbox: list[float]     # [left, top, right, bottom] in pixels of the prepared image
    mask_png: str         # path to the mask PNG
    area: int             # mask pixels
    score: float = 0.0    # SAM confidence
    concept: str = ""     # the phrase SAM was given


# ---------------- Step 2: state, timeline, verification ----------------
from typing import Literal, Optional   # noqa: E402

TRACE_NAMES = {
    "steam", "liquid_film", "ice_whole", "ice_partly_melted", "condensation",
    "wet_spill", "wet_ring", "dry_ring_stain", "damp_cloth",
    "crumbs", "open_package", "partially_eaten", "lid_off", "utensil_used",
    "door_or_drawer_open", "displaced_furniture", "residue", "level_dropped",
    "fresh_cut", "browned_cut",
}


# How the rewind can show the scene BEFORE an event, using only the real photo:
UNDO_KINDS = {
    "remove",      # the object was put there by the event: erase it and repair the background
    "add_steam",   # the drink was still hot: draw soft steam above it
    "refill",      # some was eaten or drunk: show it fuller
    "move",        # the object was moved: show where it was
    "mark",        # cannot be shown honestly with pixels: outline and label only
}


def _norm_trace(t):
    t = str(t or "").strip().lower().replace(" ", "_").replace("-", "_")
    return t if t in TRACE_NAMES else None


class ObjectState(BaseModel):
    object_id: str
    name: str = ""                                   # e.g. "banana peel": the specific everyday name
    what: str = ""                                   # e.g. "white ceramic mug"
    observations: list[str] = []                     # only what is visible now
    traces: list[str] = []                           # only names from TRACE_NAMES are kept
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("traces", mode="before")
    @classmethod
    def _known_traces(cls, v):   # an unknown trace name is dropped, not an error
        return [t for t in (_norm_trace(x) for x in (v or [])) if t]


class StateReport(BaseModel):
    objects: list[ObjectState]


class Evidence(BaseModel):
    object_id: str
    observation: str                                 # a fact visible in the photo NOW


class Event(BaseModel):
    id: str
    description: str                                 # what happened, e.g. "tea was poured into the mug"
    minutes_before_min: float = Field(ge=0)
    minutes_before_max: float = Field(ge=0)
    evidence: list[Evidence]
    trace: Optional[str] = None                      # the physical trace that dates it, if any
    confidence: float = Field(default=0.5, ge=0, le=1)
    alternatives: list[str] = []
    undo: str = "mark"                               # how the rewind shows the scene BEFORE this event
    undo_object: Optional[str] = None                # the object the rewind changes (e.g. the brownie, not the mug)

    @field_validator("trace", mode="before")
    @classmethod
    def _known_trace(cls, v):
        return _norm_trace(v)

    @field_validator("undo", mode="before")
    @classmethod
    def _known_undo(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in UNDO_KINDS else "mark"


class Timeline(BaseModel):
    summary: str = ""
    events: list[Event]


class Summary(BaseModel):
    summary: str = ""


class ClaimCheck(BaseModel):
    claim_id: str
    verdict: Literal["yes", "partly", "no", "unclear"]
    note: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("yes", "partly", "no") else "unclear"


class VerifyReport(BaseModel):
    checks: list[ClaimCheck]


class AgentStep(BaseModel):
    """One move of the investigator: look closer, search for something, check a clock, or finish."""
    thought: str = ""                                 # one short sentence: why this move
    action: Literal["zoom", "search", "clock", "finish"]
    object_id: Optional[str] = None                   # for zoom; for search: look inside/around this object
    phrase: Optional[str] = None                      # for search: a plain noun, e.g. "spoon"
    trace: Optional[str] = None                       # for clock
    summary: str = ""                                 # for finish
    events: list[Event] = []                          # for finish

    @field_validator("action", mode="before")
    @classmethod
    def _norm_action(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("zoom", "search", "clock", "finish") else "finish"
