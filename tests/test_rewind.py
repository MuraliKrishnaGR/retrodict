"""The rewind clip edits only the real photo: removed objects fade to a ghost, steam is drawn above a drink,
marked events change no pixels of the scene itself."""
import json

import numpy as np
from PIL import Image

from app.rewind import Rewinder, make_rewind, short, when_text


def _run_dir(tmp_path, events):
    d = tmp_path / "run"
    (d / "masks").mkdir(parents=True)
    img = np.full((120, 200, 3), 30, np.uint8)
    img[60:100, 40:80] = (200, 200, 200)                   # o1: a mug
    img[70:95, 120:170] = (150, 40, 160)                   # o2: a packet
    Image.fromarray(img).save(d / "input.jpg", quality=95)
    objects = []
    for oid, (t, b, l, r) in {"o1": (60, 100, 40, 80), "o2": (70, 95, 120, 170)}.items():
        m = np.zeros((120, 200), np.uint8); m[t:b, l:r] = 255
        Image.fromarray(m).save(d / "masks" / f"{oid}.png")
        objects.append({"id": oid, "label": oid, "bbox": [l, t, r, b], "mask_png": f"masks/{oid}.png"})
    (d / "result.json").write_text(json.dumps({"objects": objects, "events": events, "summary": "s"}))
    return d


def _ev(eid, oid, undo, lo=0, hi=30, desc=None):
    desc = desc or ("the mug was put on the counter" if undo == "remove" else "something happened")
    return {"id": eid, "description": desc, "minutes_before_min": lo, "minutes_before_max": hi, "undo": undo,
            "evidence": [{"object_id": oid, "observation": "x"}]}


def test_states_remove_fades_object_and_mark_changes_nothing(tmp_path):
    d = _run_dir(tmp_path, [_ev("e1", "o1", "remove"), _ev("e2", "o2", "mark")])
    states = Rewinder(d).states()
    assert len(states) == 3                                              # now, -e2, -e2-e1
    photo = states[0].astype(int)
    assert np.abs(states[1].astype(int) - photo).max() == 0             # 'mark' draws nothing on the scene
    mug_now, mug_before = photo[65:95, 45:75].mean(), states[2][65:95, 45:75].astype(int).mean()
    assert mug_before < mug_now - 80                                     # the mug faded towards the dark counter
    assert mug_before > 35                                               # ...but a faint ghost stays


def test_steam_is_drawn_above_the_drink_only(tmp_path):
    d = _run_dir(tmp_path, [_ev("e1", "o1", "add_steam", 0, 10)])
    before = Rewinder(d).states()[1].astype(int)
    now = np.array(Image.open(d / "input.jpg")).astype(int)
    above, beside = (before - now)[30:58, 50:70].mean(), (before - now)[30:58, 150:190].mean()
    assert above > 5 and abs(beside) < 2


def test_video_and_stills_are_written(tmp_path):
    d = _run_dir(tmp_path, [_ev("e1", "o1", "remove"), _ev("e2", "o2", "refill")])
    info = make_rewind(d)
    assert (d / "rewind.mp4").stat().st_size > 1000
    assert info["stills"] == ["rewind_0.jpg", "rewind_1.jpg", "rewind_2.jpg"]
    assert [s["kind"] for s in info["steps"]] == ["refill", "remove"]   # latest event is undone first
    assert 3.5 < info["seconds"] < 4.5                                   # intro + 2 events + outro


def test_events_without_a_mask_are_skipped(tmp_path):
    d = _run_dir(tmp_path, [_ev("e1", "o9", "remove"), _ev("e2", "o1", "move")])
    assert [s["event"]["id"] for s in Rewinder(d).steps] == ["e2"]


def test_captions_are_short_and_times_honest():
    assert short("Someone tore open the purple and brown brownie packet, exposing the layered brownie inside") \
        == "Someone tore open the purple and brown brownie packet"
    assert short("Tea was poured.") == "Tea was poured"
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 30}) == "up to 30 min before"
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 240}) == "up to 4.0 h before"
    assert when_text({"minutes_before_min": 3, "minutes_before_max": 10}) == "3 to 10 min before"


def test_rewind_changes_the_object_the_event_is_about(tmp_path):
    """Run 4: 'someone ate part of the brownie' cited the crumbs by the mug first, and the mug faded out."""
    ate = _ev("e1", "o1", "remove", desc="Someone ate part of the brownie, dropping crumbs next to the mug")
    ate["evidence"].append({"object_id": "o2", "observation": "packet torn open"})
    d = _run_dir(tmp_path, [ate])
    res = json.loads((d / "result.json").read_text())
    res["states"] = {"o1": {"what": "steel mug with milk"}, "o2": {"what": "torn brownie packet"}}
    (d / "result.json").write_text(json.dumps(res))
    assert Rewinder(d).steps[0]["oid"] == "o2"                    # the brownie, found by its name
    res["events"][0]["undo_object"] = "o1"                       # the model's own choice wins when it is real
    (d / "result.json").write_text(json.dumps(res))
    assert Rewinder(d).steps[0]["oid"] == "o1"


def test_long_captions_are_cut_at_a_semicolon_too():
    assert short("Someone scooped granules from the jar to make the drink now in the mug; the jar still holds "
                 "the rest of the granules") == "Someone scooped granules from the jar to make the drink now in the mug"


def test_undo_is_the_opposite_of_what_the_event_says():
    from app.rewind import undo_kind
    assert undo_kind({"description": "Someone ate part of the brownie", "undo": "remove"}) == "refill"
    assert undo_kind({"description": "Some milk was drunk from the glass", "undo": "remove"}) == "refill"
    assert undo_kind({"description": "Milk was poured into the glass", "undo": "remove"}) == "remove"
    assert undo_kind({"description": "The packet was torn open", "undo": "remove"}) == "mark"
    assert undo_kind({"description": "The drink was still steaming", "undo": "add_steam"}) == "add_steam"
