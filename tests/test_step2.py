"""Offline tests for step 2: trace clocks, grounding, verification and the whole pipeline."""
import asyncio
import json

import numpy as np
from PIL import Image

from app.clocks import apply_clock
from app.pipeline import run_photo
from app.reason import ground_events
from app.schemas import Timeline
from app.verify import apply_checks, build_claims


def test_clock_rules():
    assert apply_clock(2, 8, "steam") == (2, 8, "clock_agrees")
    assert apply_clock(0, 60, "steam") == (0, 10, "narrowed_by_clock")
    assert apply_clock(120, 300, "steam") == (0, 10, "replaced_by_clock")    # model contradicted physics
    assert apply_clock(300, 10, "steam") == (10, 10, "narrowed_by_clock")    # swapped range repaired, then clamped
    assert apply_clock(5, 500, "crumbs") == (5, 500, "no_clock")
    assert apply_clock(5, 500, None) == (5, 500, "no_clock")


def _timeline(events):
    return Timeline.model_validate({"summary": "s", "events": events})


def test_ground_events_drops_bad_evidence_applies_clocks_and_orders():
    tl = _timeline([
        {"id": "x", "description": "tea poured", "minutes_before_min": 0, "minutes_before_max": 60,
         "trace": "steam", "evidence": [{"object_id": "o1", "observation": "steam rising"}]},
        {"id": "y", "description": "ghost object", "minutes_before_min": 0, "minutes_before_max": 5,
         "evidence": [{"object_id": "o9", "observation": "made up"}]},
        {"id": "z", "description": "packet opened", "minutes_before_min": 10, "minutes_before_max": 120,
         "trace": "open_package", "evidence": [{"object_id": "o2", "observation": "wrapper torn"},
                                               {"object_id": "o7", "observation": "invented"}]},
    ])
    events = ground_events(tl, {"o1", "o2"})
    assert [e["description"] for e in events] == ["packet opened", "tea poured"]   # earliest first
    assert [e["id"] for e in events] == ["e1", "e2"]
    assert events[0]["evidence"] == [{"object_id": "o2", "observation": "wrapper torn"}]
    assert events[1]["minutes_before_max"] == 10 and events[1]["clock"] == "narrowed_by_clock"
    assert events[1]["model_range"] == [0, 60]


def test_verifier_rejects_and_penalises():
    events = [
        {"id": "e1", "description": "a", "confidence": 0.8,
         "evidence": [{"object_id": "o1", "observation": "mug half full"}]},
        {"id": "e2", "description": "b", "confidence": 0.8,
         "evidence": [{"object_id": "o2", "observation": "packet torn"}]},
        {"id": "e3", "description": "c", "confidence": 0.8,
         "evidence": [{"object_id": "o1", "observation": "x"}, {"object_id": "o2", "observation": "y"}]},
    ]
    claims = build_claims(events, {"o1": "mug", "o2": "packet"})
    assert [c["claim_id"] for c in claims] == ["c1", "c2", "c3", "c4"]
    assert "o1 (mug): mug half full" == claims[0]["text"]
    kept, rejected = apply_checks(events, claims, {"c1": ("yes", ""), "c2": ("no", "sealed"),
                                                   "c3": ("yes", ""), "c4": ("unclear", "")})
    assert [e["id"] for e in kept] == ["e1", "e3"] and [e["id"] for e in rejected] == ["e2"]
    assert kept[0]["status"] == "verified" and kept[0]["confidence"] == 0.8
    assert kept[1]["status"] == "partly_verified" and kept[1]["confidence"] == 0.6
    assert rejected[0]["evidence"][0]["note"] == "sealed"


class StoryModels:
    """Fake models for a tea scene: SAM finds a mug and a packet; Spark answers per pipeline step."""
    def __init__(self):
        self.tags = []

    async def sam_segment(self, jpeg, w, h, concept):
        m = np.zeros((h, w), bool)
        boxes = {"mug": (10, 10, 40, 40), "packet": (50, 20, 90, 60)}
        if concept not in boxes:
            return []
        l, t, r, b = boxes[concept]
        m[t:b, l:r] = True
        return [{"object_id": "0", "bbox": [l, t, r - 1, b - 1], "score": 0.95, "mask": m}]

    async def spark_json(self, prompt, schema, jpeg=None, tag=""):
        self.tags.append(tag)
        assert jpeg, "every step must look at the image"
        replies = {
            "list_objects": {"objects": [{"name": "mug"}, {"name": "packet"}]},
            "read_state": {"objects": [
                {"object_id": "o1", "what": "white mug", "observations": ["half full", "no steam"],
                 "traces": ["liquid film"], "confidence": 0.8},
                {"object_id": "o2", "what": "biscuit packet", "observations": ["torn open", "crumbs"],
                 "traces": ["open_package", "crumbs", "glitter"], "confidence": 0.9},
                {"object_id": "o9", "what": "imaginary", "observations": [], "traces": []}]},
            "reason": {"summary": "Someone had tea and biscuits.", "events": [
                {"id": "a", "description": "tea was poured", "minutes_before_min": 5, "minutes_before_max": 200,
                 "trace": "liquid_film", "evidence": [{"object_id": "o1", "observation": "skin on the tea"}],
                 "confidence": 0.7},
                {"id": "b", "description": "packet was opened", "minutes_before_min": 0, "minutes_before_max": 30,
                 "trace": None, "evidence": [{"object_id": "o2", "observation": "wrapper torn"}],
                 "confidence": 0.8},
                {"id": "c", "description": "sugar was added", "minutes_before_min": 0, "minutes_before_max": 30,
                 "evidence": [{"object_id": "o1", "observation": "sugar spoon in mug"}], "confidence": 0.5}]},
        }
        if tag == "verify":
            verdict = {"skin on the tea": "yes", "wrapper torn": "yes", "sugar spoon in mug": "no"}
            checks = []
            for line in prompt.splitlines():
                if line.startswith("- c"):
                    cid, text = line[2:].split(": ", 1)
                    obs = text.split("): ", 1)[1]
                    checks.append({"claim_id": cid, "verdict": verdict[obs]})
            return schema.model_validate({"checks": checks})
        return schema.model_validate(replies[tag])


def test_full_pipeline(tmp_path):
    photo = tmp_path / "tea.jpg"
    Image.new("RGB", (100, 80), (180, 170, 160)).save(photo)
    models = StoryModels()
    res = asyncio.run(run_photo(models, photo, tmp_path / "out"))

    assert models.tags == ["list_objects", "read_state", "reason", "verify"]   # 4 model calls per photo
    assert [s["step"] for s in res["trace"]] == ["perceive", "read", "reason", "verify"]
    assert set(res["states"]) == {"o1", "o2"}                                  # imaginary o9 dropped
    assert res["states"]["o1"]["traces"] == ["liquid_film"]
    assert res["states"]["o2"]["traces"] == ["open_package", "crumbs"]        # unknown 'glitter' dropped

    descs = [e["description"] for e in res["events"]]
    assert descs == ["tea was poured", "packet was opened"]                   # earliest first
    tea = res["events"][0]
    assert (tea["minutes_before_min"], tea["minutes_before_max"]) == (10, 90)  # liquid_film clock
    assert tea["clock"] == "narrowed_by_clock" and tea["status"] == "verified"
    assert [e["description"] for e in res["rejected"]] == ["sugar was added"]  # caught by the verifier

    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    assert saved["summary"] == "Someone had tea and biscuits."
    assert (tmp_path / "out" / "sheet.jpg").exists()


def test_pipeline_with_no_objects(tmp_path):
    class Empty(StoryModels):
        async def sam_segment(self, *a):
            return []
    photo = tmp_path / "wall.jpg"
    Image.new("RGB", (60, 40), (250, 250, 250)).save(photo)
    models = Empty()
    res = asyncio.run(run_photo(models, photo, tmp_path / "out"))
    assert res["events"] == [] and "No objects" in res["summary"] and models.tags == ["list_objects"]


def test_clock_table_only_lists_seen_traces():
    from app.clocks import clock_table_text
    assert "wet_ring" not in clock_table_text({"steam"}) and "steam" in clock_table_text({"steam"})
    assert "crumbs" in clock_table_text({"crumbs"}) and "steam" not in clock_table_text({"crumbs"})
    assert clock_table_text(set()) == "(no physical traces with known timing were seen)"


def test_plain_text_for_people():
    from app.reason import plain_text
    states = {"o1": {"what": "large ceramic jar"}, "o2": {"what": "rusk packet"}}
    out = plain_text("o1 lid is closed and o2 shows no wet_ring or liquid_film. Third sentence. Fourth.",
                     states, {"o1": "jar 1", "o2": "packet"})
    assert out == "the large ceramic jar lid is closed and the rusk packet shows no wet ring or liquid film. Third sentence."
    assert "o10" in plain_text("o10 stays", states, {})          # unknown ids are left alone


def test_quality_gate_catches_the_bad_summary():
    from app.quality import check_result
    bad = {"objects": [{"id": "o1"}], "states": {}, "events": [], "summary":
           "No time-sensitive traces are visible. o1 lid is closed with no condensation/steam. The absence "
           "of steam, liquid_film, ice, wet_spill, wet_ring or crumbs means there is no reliable evidence."}
    issues = check_result(bad)
    assert any("object id" in i for i in issues)
    assert any("underscore" in i for i in issues)
    assert any("absent" in i for i in issues)
    good = {"objects": [{"id": "o1"}], "states": {}, "summary": "Tea was poured a few minutes ago and is still steaming.",
            "events": [{"id": "e1", "description": "tea was poured", "minutes_before_min": 0,
                        "minutes_before_max": 10, "status": "verified",
                        "evidence": [{"object_id": "o1", "observation": "steam rising"}]}]}
    assert check_result(good) == []


def test_pipeline_saves_quality_result(tmp_path):
    photo = tmp_path / "tea.jpg"
    Image.new("RGB", (100, 80), (180, 170, 160)).save(photo)
    res = asyncio.run(run_photo(StoryModels(), photo, tmp_path / "out"))
    assert res["quality_issues"] == []
    assert json.loads((tmp_path / "out" / "result.json").read_text())["quality_issues"] == []
