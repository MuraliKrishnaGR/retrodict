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
    "door_or_drawer_open", "displaced_furniture",
}


def _norm_trace(t):
    t = str(t or "").strip().lower().replace(" ", "_").replace("-", "_")
    return t if t in TRACE_NAMES else None


class ObjectState(BaseModel):
    object_id: str
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

    @field_validator("trace", mode="before")
    @classmethod
    def _known_trace(cls, v):
        return _norm_trace(v)


class Timeline(BaseModel):
    summary: str = ""
    events: list[Event]


class ClaimCheck(BaseModel):
    claim_id: str
    verdict: Literal["yes", "no", "unclear"]
    note: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("yes", "no") else "unclear"


class VerifyReport(BaseModel):
    checks: list[ClaimCheck]
