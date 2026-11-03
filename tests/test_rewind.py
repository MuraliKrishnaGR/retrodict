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
    rw = Rewinder(d)
    states = rw.states()
    assert len(states) == 3                                              # now, -e2, -e2-e1
    photo = states[0].astype(int)
    assert np.abs(states[1].astype(int) - photo).max() == 0             # 'mark' draws nothing on the scene
    mug_now, mug_before = photo[65:95, 45:75].mean(), states[2][65:95, 45:75].astype(int).mean()
    assert mug_before < mug_now - 80                                     # the mug faded towards the dark counter
    if rw.steps[1]["eraser"] == "opencv":
        assert mug_before > 35                                           # rough fill: a faint ghost stays
    else:
        assert abs(mug_before - 30) < 6                                  # clean texture fill: the counter again


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
    assert 8.5 < info["seconds"] < 9.5       # intro + two spotlight steps + outro


def test_captions_say_what_it_was_like_before(tmp_path):
    """The picture goes backward, so the caption must too (2 Nov: event captions over a reversing picture
    confused the viewer)."""
    d = _run_dir(tmp_path, [_ev("e1", "o1", "remove"), _ev("e2", "o2", "refill")])
    r = Rewinder(d)
    texts = [r._before_text(s) for s in r.steps]
    assert texts[0].startswith("Before: more was in the") and texts[0].endswith("(can't be drawn)")
    assert texts[1].startswith("Before: the") and texts[1].endswith("was not here yet")


def test_events_without_a_mask_are_skipped(tmp_path):
    d = _run_dir(tmp_path, [_ev("e1", "o9", "remove"), _ev("e2", "o1", "move")])
    assert [s["event"]["id"] for s in Rewinder(d).steps] == ["e2"]


def test_captions_are_short_and_times_honest():
    assert short("Someone tore open the purple and brown brownie packet, exposing the layered brownie inside") \
        == "Someone tore open the purple and brown brownie packet"
    assert short("Tea was poured.") == "Tea was poured"
    dated = {"clock": "clock_agrees"}
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 30, **dated}) == "up to 30 min before"
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 240, **dated}) == "up to 4.0 h before"
    assert when_text({"minutes_before_min": 3, "minutes_before_max": 10, **dated}) == "3 to 10 min before"
    # no clue in the photo: the model's guess is not shown as a time
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 60, "clock": "no_clock"}) == "time unknown"
    assert when_text({"minutes_before_min": 0, "minutes_before_max": 60, "clock": "clock_not_applied"}) == "time unknown"
    assert when_text({"minutes_before_min": 5, "minutes_before_max": 60, "dated": False}) == "at least 5 min before"


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
    assert undo_kind({"description": "Milk was poured into the glass", "undo": "remove"}) == "empty"
    assert undo_kind({"description": "The frothy milk was poured into the mug", "undo": "refill"}) == "empty"
    assert undo_kind({"description": "The glass was put on the counter", "undo": "remove"}) == "remove"
    assert undo_kind({"description": "The packet was torn open", "undo": "remove"}) == "mark"
    assert undo_kind({"description": "The drink was still steaming", "undo": "add_steam"}) == "add_steam"


def test_eraser_fills_the_object_and_its_reflection_and_never_a_neighbour():
    """The hole includes the object's reflection below it, and a neighbouring object inside that reach is
    protected. OpenCV only: no AI model other than Meta's runs (hackathon rule)."""
    from app import eraser
    img = np.full((120, 200, 3), 30, np.uint8)
    img[30:70, 80:110] = 220                                            # a bright glass
    img[75:90, 90:120] = (200, 40, 40)                                  # a red packet just below it
    obj = np.zeros((120, 200), bool); obj[30:70, 80:110] = True
    nb = np.zeros((120, 200), bool); nb[75:90, 90:120] = True
    out, method = eraser.erase(img, obj, keep=nb)
    assert method in ("patchmatch", "opencv")                          # classic algorithms only
    assert out[50, 95].mean() < 120                                     # the glass is gone
    assert tuple(out[85, 100]) == (200, 40, 40)                         # the packet below is untouched
    assert tuple(out[10, 10]) == (30, 30, 30)                           # far away: untouched


def test_poured_drink_is_erased_from_the_cup_and_left_behind_crumbs_go_too(tmp_path):
    """Run 9: 'frothy milk was poured into the mug' -> before it, the mug was empty (contents erased, the mug
    stays); 'opened the brownie ... leaving brown crumbs' -> the crumbs were not there yet."""
    d = tmp_path / "run"
    (d / "masks").mkdir(parents=True)
    img = np.full((120, 200, 3), 30, np.uint8)
    img[40:100, 40:90] = (90, 90, 95)                                 # steel mug body
    img[47:60, 48:82] = (235, 225, 210)                               # milk seen from above, inside the rim
    img[105:110, 120:126] = (110, 70, 40)                             # crumbs
    Image.fromarray(img).save(d / "input.jpg", quality=95)
    masks = {"o4": (40, 100, 40, 90), "o6": (104, 111, 119, 127), "o5": (60, 90, 140, 190)}
    objects = []
    for oid, (t, b, l, r) in masks.items():
        m = np.zeros((120, 200), np.uint8); m[t:b, l:r] = 255
        Image.fromarray(m).save(d / "masks" / f"{oid}.png")
        objects.append({"id": oid, "label": {"o4": "mug", "o6": "brown crumbs", "o5": "packet"}[oid],
                        "bbox": [l, t, r, b], "mask_png": f"masks/{oid}.png"})
    events = [_ev("e1", "o4", "refill", desc="The frothy milk was poured into the mug"),
              dict(_ev("e2", "o5", "mark", desc="The brownie packet was opened, leaving brown crumbs near the mug"),
                   evidence=[{"object_id": "o5", "observation": "torn"}, {"object_id": "o6", "observation": "crumbs"}])]
    (d / "result.json").write_text(json.dumps({"objects": objects, "events": events, "summary": "s"}))
    rw = Rewinder(d)
    assert [(s["kind"], s["erase"]) for s in rw.steps] == [("mark", ["o6"]), ("empty", [])]
    states = rw.states()
    before_crumbs, before_milk = states[1].astype(int), states[2].astype(int)
    assert abs(before_crumbs[107, 123] - 30).max() < 40                # crumbs gone: counter colour again
    assert before_milk[48, 65].mean() < 200                            # milk gone from the mug's opening
    assert abs(before_milk[85, 65] - img[85, 65].astype(int)).max() < 15   # the mug body itself stays


def test_no_non_meta_model_code_in_the_app():
    """Hackathon rule (confirmed 3 Nov): every AI model called at runtime is a Meta model."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    code = " ".join(p.read_text(encoding="utf-8").lower() for p in list((root / "app").rglob("*.py")) +
                    list((root / "scripts").glob("*.py")))
    for name in ("big-lama", "qwen", "gradio_client", "stable-diffusion", "diffusers"):
        assert name not in code, name


def test_patchmatch_copies_the_background_never_a_neighbour():
    """A striped desk continues its stripes through the hole; the red packet next to it is never a source."""
    pm = __import__("pytest").importorskip("patchmatch")
    if not pm.patchmatch_available:
        __import__("pytest").skip("PatchMatch library not built here")
    from app import eraser
    img = np.zeros((120, 200, 3), np.uint8)
    img[:, :] = (60, 60, 60)
    img[::8] = (160, 160, 160)                                          # desk with light stripes
    img[40:70, 80:110] = (230, 230, 230)                                # a bright cup
    img[40:70, 115:140] = (220, 30, 30)                                 # a red packet beside it
    cup = np.zeros((120, 200), bool); cup[40:70, 80:110] = True
    red = np.zeros((120, 200), bool); red[40:70, 115:140] = True
    out, method = eraser.erase(img, cup, keep=red, grow_down=False)
    assert method == "patchmatch"
    patch = out[45:65, 85:105].astype(int)
    assert patch.max() < 200 and (patch[..., 0] - patch[..., 1]).max() < 40   # no cup left, no red copied in
    assert tuple(out[50, 120]) == (220, 30, 30)                         # the packet itself is untouched


def test_inside_of_a_cup_uses_the_smooth_fill():
    from app import eraser
    img = np.full((60, 60, 3), 90, np.uint8); img[20:40, 20:40] = 240
    m = np.zeros((60, 60), bool); m[20:40, 20:40] = True
    _, method = eraser.erase(img, m, grow_down=False, smooth=True)
    assert method == "opencv"
