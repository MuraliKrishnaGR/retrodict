"""Data shapes for day 1. More are added on later days."""
from pydantic import BaseModel, Field


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
