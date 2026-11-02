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
    """Fake models for a tea scene: SAM finds a mug and a packet (and a spoon, if searched);
    Spark answers per pipeline step. `agent` scripts the investigator's moves, turn by turn."""
    def __init__(self, agent=None, trace_verdict="yes"):
        self.tags, self.images, self.prompts = [], {}, {}
        self.agent = agent
        self.trace_verdict = trace_verdict          # what the verifier says when asked to confirm a trace

    STORY = {"summary": "Someone had tea and biscuits.", "events": [
                {"id": "a", "description": "tea was poured", "minutes_before_min": 5, "minutes_before_max": 200,
                 "trace": "liquid_film", "evidence": [{"object_id": "o1", "observation": "skin on the tea"}],
                 "confidence": 0.7},
                {"id": "b", "description": "packet was opened", "minutes_before_min": 0, "minutes_before_max": 30,
                 "trace": None, "evidence": [{"object_id": "o2", "observation": "wrapper torn"}],
                 "confidence": 0.8},
                {"id": "c", "description": "sugar was added", "minutes_before_min": 0, "minutes_before_max": 30,
                 "evidence": [{"object_id": "o1", "observation": "sugar spoon in mug"}], "confidence": 0.5}]}

    async def sam_segment(self, jpeg, w, h, concept):
        m = np.zeros((h, w), bool)
        boxes = {"mug": (10, 10, 40, 40), "packet": (50, 20, 90, 60), "spoon": (12, 50, 40, 58)}
        if concept not in boxes:
            return []
        l, t, r, b = boxes[concept]
        m[t:b, l:r] = True
        return [{"object_id": "0", "bbox": [l, t, r - 1, b - 1], "score": 0.95, "mask": m}]

    async def spark_json(self, prompt, schema, jpeg=None, tag=""):
        self.tags.append(tag)
        self.images[tag] = jpeg
        self.prompts[tag] = prompt
        if tag == "summary":                            # text only: rewrites the story from what survived
            assert "sugar was added" in prompt.split("do not mention them")[1]
            return schema.model_validate({"summary": "Someone poured tea and opened a packet of biscuits."})
        assert jpeg, "every step must look at the image"
        if tag.startswith("agent_"):
            script = self.agent if self.agent is not None else [{"action": "finish", **self.STORY}]
            return schema.model_validate(script[int(tag.split("_")[1])])
        replies = {
            "list_objects": {"objects": [{"name": "mug"}, {"name": "packet"}]},
            "read_state": {"objects": [
                {"object_id": "o1", "what": "white mug", "observations": ["half full", "no steam"],
                 "traces": ["liquid film"], "confidence": 0.8},
                {"object_id": "o2", "what": "biscuit packet", "observations": ["torn open", "crumbs"],
                 "traces": ["open_package", "crumbs", "glitter"], "confidence": 0.9},
                {"object_id": "o9", "what": "imaginary", "observations": [], "traces": []}]},
            "reason": self.STORY,
        }
        if tag == "verify":
            verdict = {"skin on the tea": "yes", "wrapper torn": "yes", "sugar spoon in mug": "no",
                       "used spoon lying on the table": "yes"}
            checks = []
            for line in prompt.splitlines():
                if line.startswith("- c"):
                    cid, text = line[2:].split(": ", 1)
                    obs = text.split("): ", 1)[1]
                    v = self.trace_verdict if obs.startswith("shows ") else verdict[obs]
                    checks.append({"claim_id": cid, "verdict": v})
            return schema.model_validate({"checks": checks})
        return schema.model_validate(replies[tag])


def test_full_pipeline(tmp_path):
    photo = tmp_path / "tea.jpg"
    Image.new("RGB", (100, 80), (180, 170, 160)).save(photo)
    models = StoryModels()
    res = asyncio.run(run_photo(models, photo, tmp_path / "out"))

    assert models.tags == ["list_objects", "read_state", "agent_0", "verify", "summary"]  # agent finished at once;
                                                    # the verifier rejected an event, so the summary is rewritten
    assert [s["step"] for s in res["trace"]] == ["perceive", "read", "investigate", "verify"]
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
    assert saved["summary"] == "Someone poured tea and opened a packet of biscuits."   # no rejected story in it
    assert [e["id"] for e in saved["events"]] == ["e1", "e2"]
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


def test_contact_sheet_keeps_true_colours():
    from app.imaging import contact_sheet, mask_outline
    m = np.zeros((40, 40), bool); m[10:30, 10:30] = True
    edge = mask_outline(m, 3)
    assert edge[10, 20] and edge[12, 20] and not edge[13, 20] and not edge[20, 20]   # 3-px inner edge only
    img = Image.new("RGB", (40, 40), (120, 80, 40))                                   # brown object
    sheet = np.array(contact_sheet(img, [{"id": "o1", "label": "jar", "mask": m, "bbox": [10, 10, 29, 29]}]))
    centre = sheet[20, 20].astype(int)
    assert abs(centre[0] - 120) < 25 and abs(centre[1] - 80) < 25 and abs(centre[2] - 40) < 25  # still brown


import pytest  # noqa: E402

TRUE_COLOURS = {"white": (245, 245, 245), "black": (15, 15, 15), "brown": (120, 80, 40),
                "yellow": (240, 200, 30), "red": (200, 30, 30), "green": (40, 160, 60),
                "blue": (40, 70, 200), "tea": (150, 95, 45), "milk": (235, 225, 200)}


@pytest.mark.parametrize("name", sorted(TRUE_COLOURS))
def test_contact_sheet_keeps_every_colour(name):
    """Every object keeps its own colour on the sheet: no channel moves by more than the 12% tint allows,
    whichever of the 8 mask colours it gets (we try one object per mask colour)."""
    from app.imaging import COLORS, contact_sheet
    rgb = TRUE_COLOURS[name]
    img = Image.new("RGB", (8 * 40, 40), rgb)
    objs = []
    for i in range(len(COLORS)):
        m = np.zeros((40, 8 * 40), bool); m[5:35, i * 40 + 5:i * 40 + 35] = True
        objs.append({"id": f"o{i + 1}", "label": "x", "mask": m, "bbox": [i * 40 + 5, 5, i * 40 + 34, 34]})
    sheet = np.array(contact_sheet(img, objs)).astype(int)
    for i in range(len(COLORS)):
        shift = np.abs(sheet[20, i * 40 + 20] - np.array(rgb)).max()
        assert shift <= 0.12 * 255 + 3, f"{name} under mask colour {i}: shifted by {shift}"


def _photo(tmp_path, name="tea.jpg", size=(100, 80)):
    photo = tmp_path / name
    Image.new("RGB", size, (180, 170, 160)).save(photo)
    return photo


def test_agent_zooms_searches_and_cites_what_it_found(tmp_path):
    script = [
        {"thought": "is there liquid?", "action": "zoom", "object_id": "o1"},
        {"thought": "a used spoon would date the stirring", "action": "search", "phrase": "Spoon"},
        {"thought": "how long does a skin take?", "action": "clock", "trace": "liquid_film"},
        {"thought": "enough evidence", "action": "finish", "summary": "Someone had tea; o1 cooled.",
         "events": [{"id": "x", "description": "tea was stirred with o3", "minutes_before_min": 5,
                     "minutes_before_max": 60, "trace": "liquid_film", "undo": "Remove",
                     "evidence": [{"object_id": "o3", "observation": "used spoon lying on the table"}]}]},
    ]
    photo = _photo(tmp_path, size=(2000, 1600))     # shrunk to 1024 for SAM; zoom must use the original
    models = StoryModels(agent=script)
    res = asyncio.run(run_photo(models, photo, tmp_path / "out"))

    assert models.tags == ["list_objects", "read_state", "agent_0", "agent_1", "agent_2", "agent_3", "verify"]
    assert [m["move"] for m in res["moves"]] == ["zoom", "search", "clock", "finish"]
    assert models.images["agent_1"] != models.images["agent_0"]          # turn after zoom sees the close-up
    zoom = Image.open(__import__("io").BytesIO(models.images["agent_1"]))
    half = (zoom.width - 8) // 2                    # as photographed | contrast boosted, side by side
    assert 80 < half < 100                          # o1 is 30 px wide in the 1024 copy (+ margin ~45 px);
                                                    # from the 2000 px original it is about twice that
    assert zoom.height > half                       # the space above the mug is included (steam rises)
    assert res["moves"][1]["result"] == "found o3 (spoon)"                # SAM found it, it became o3
    assert res["moves"][2]["result"].startswith("liquid_film: 10 to 90 min")
    assert [o["id"] for o in res["objects"]] == ["o1", "o2", "o3"] and res["objects"][2]["found_by"] == "agent"
    e = res["events"][0]
    assert e["undo"] == "remove" and e["status"] == "verified"
    assert e["description"] == "tea was stirred with the spoon"           # ids cleaned for people
    assert res["summary"] == "Someone had tea; the white mug cooled."
    assert "investigate" in [s["step"] for s in res["trace"]] and res["quality_issues"] == []


def test_agent_search_skips_things_already_known(tmp_path):
    script = [{"action": "search", "phrase": "mug"},
              {"action": "finish", **StoryModels.STORY}]
    res = asyncio.run(run_photo(StoryModels(agent=script), _photo(tmp_path), tmp_path / "out"))
    assert res["moves"][0]["result"] == "no new 'mug' found" and len(res["objects"]) == 2


def test_agent_out_of_moves_falls_back_to_single_reasoning(tmp_path):
    script = [{"action": "zoom", "object_id": "o1"}] * 6          # never finishes
    models = StoryModels(agent=script)
    res = asyncio.run(run_photo(models, _photo(tmp_path), tmp_path / "out"))
    assert models.tags[-3:] == ["reason", "verify", "summary"] and "agent_4" in models.tags and "agent_5" not in models.tags
    assert [e["description"] for e in res["events"]] == ["tea was poured", "packet was opened"]
    assert "single reasoning call" in res["trace"][2]["detail"]


def test_agent_crash_falls_back(tmp_path):
    class Broken(StoryModels):
        async def spark_json(self, prompt, schema, jpeg=None, tag=""):
            if tag.startswith("agent_"):
                self.tags.append(tag)
                raise ValueError("model returned nonsense twice")
            return await super().spark_json(prompt, schema, jpeg, tag)
    res = asyncio.run(run_photo(Broken(), _photo(tmp_path), tmp_path / "out"))
    assert len(res["events"]) == 2 and "ValueError" in res["trace"][2]["detail"]


def test_bad_moves_are_reported_not_fatal(tmp_path):
    script = [{"action": "zoom", "object_id": "o42"}, {"action": "clock", "trace": "rainbow"},
              {"action": "finish", **StoryModels.STORY}]
    res = asyncio.run(run_photo(StoryModels(agent=script), _photo(tmp_path), tmp_path / "out"))
    assert [m["result"] for m in res["moves"][:2]] == ["could not use zoom with those arguments",
                                                        "could not use clock with those arguments"]
    assert len(res["events"]) == 2


def test_no_clock_means_no_minimum_time():
    """Real run (milk photo): the model gave 'at least 30 min' to events that happened ~2 min before.
    Nothing visible set those minimums, so without a seen trace the range must start at 0."""
    from app.schemas import Timeline
    tl = Timeline.model_validate({"summary": "s", "events": [
        {"id": "a", "description": "powder was scooped", "minutes_before_min": 30, "minutes_before_max": 120,
         "evidence": [{"object_id": "o1", "observation": "spoon on lid"}], "trace": None},
        {"id": "b", "description": "tea was poured", "minutes_before_min": 20, "minutes_before_max": 60,
         "evidence": [{"object_id": "o1", "observation": "skin on the tea"}], "trace": "liquid_film"}]})
    ev = {e["description"]: e for e in ground_events(tl, {"o1"})}
    assert (ev["powder was scooped"]["minutes_before_min"], ev["powder was scooped"]["minutes_before_max"]) == (0, 120)
    assert ev["powder was scooped"]["model_range"] == [30, 120]          # the model's guess is kept for review
    assert (ev["tea was poured"]["minutes_before_min"], ev["tea was poured"]["minutes_before_max"]) == (20, 60)


def test_clock_lookup_for_a_trace_nobody_sees_is_refused(tmp_path):
    script = [{"action": "clock", "trace": "steam"}, {"action": "clock", "trace": "liquid_film"},
              {"action": "finish", **StoryModels.STORY}]
    res = asyncio.run(run_photo(StoryModels(agent=script), _photo(tmp_path), tmp_path / "out"))
    assert res["moves"][0]["result"] == "steam: not seen on any object, so its clock does not apply here"
    assert res["moves"][1]["result"].startswith("liquid_film: 10 to 90 min")   # seen on the mug: allowed


def test_long_summary_is_cut_to_its_first_sentence():
    from app.reason import plain_text
    long2 = "Someone made a milky drink and ate part of a brownie. " + "word " * 60 + "end."
    out = plain_text(long2, {}, {})
    assert out == "Someone made a milky drink and ate part of a brownie."


def test_residue_is_a_known_trace_without_a_clock():
    from app.clocks import NO_CLOCK
    from app.schemas import ObjectState
    assert "residue" in NO_CLOCK
    assert ObjectState(object_id="o1", traces=["residue", "sparkles"]).traces == ["residue"]


def test_boosted_pair_reveals_faint_haze():
    """A faint haze (steam) 6 levels brighter than the dark counter behind it becomes clearly visible,
    even when the crop also holds white tiles and black shadows (a plain contrast stretch does nothing then)."""
    from app.imaging import boosted_pair
    a = np.full((60, 80, 3), 40, np.uint8); a[10:30, 30:50] = 46
    a[:5, :] = 255; a[55:, :] = 0                                    # white tiles, black shadow
    pair = np.array(boosted_pair(Image.fromarray(a))).astype(int)
    left, right = pair[:, :80], pair[:, 88:]
    assert left[20, 40, 0] - left[40, 10, 0] == 6
    assert right[20, 40, 0] - right[40, 10, 0] > 60


def test_clock_only_dates_an_event_when_the_verifier_confirms_the_trace(tmp_path):
    """Run 2 on the milk photo: bubbles were read as a skin (liquid_film) and its clock moved a drink
    made 3 minutes earlier to '10 to 90 min'. Now the checker must confirm the trace by itself."""
    ok = asyncio.run(run_photo(StoryModels(), _photo(tmp_path), tmp_path / "a"))
    tea = ok["events"][0]
    assert (tea["minutes_before_min"], tea["minutes_before_max"]) == (10, 90)
    assert tea["trace_check"]["verdict"] == "yes" and tea["clock_rule"]

    models = StoryModels(trace_verdict="no")
    res = asyncio.run(run_photo(models, _photo(tmp_path, "b.jpg"), tmp_path / "b"))
    tea = next(e for e in res["events"] if e["description"] == "tea was poured")
    assert tea["clock"] == "clock_not_applied" and tea["clock_rule"] is None
    assert (tea["minutes_before_min"], tea["minutes_before_max"]) == (0, 200)   # back to the model's honest max
    assert tea["trace_check"]["verdict"] == "no"
    assert tea["status"] == "verified"                     # the visible facts still hold; only the clock went
    claims = [l for l in models.prompts["verify"].splitlines() if l.startswith("- c")]
    assert any("shows a matte, wrinkled skin" in l for l in claims)


def test_verifier_sees_sharp_close_ups_of_the_cited_objects(tmp_path):
    models = StoryModels()
    asyncio.run(run_photo(models, _photo(tmp_path, size=(2000, 1600)), tmp_path / "out"))
    ev = Image.open(__import__("io").BytesIO(models.images["verify"]))
    assert ev.width > 700                                  # two columns of as-photographed | boosted pairs
    assert (tmp_path / "out" / "evidence.jpg").exists()
