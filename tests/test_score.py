"""Scoring a run against the answer key and the V-JEPA 2 witness (offline)."""
import json
from pathlib import Path

from app import score

KEY = {"video": "v.mp4", "video_seconds": 120, "photo_delay_min": [0, 1], "events": [
    {"id": "pour", "what": "milk poured", "seconds": [10, 20], "match": r"\bpour", "witness": ["add_into"]},
    {"id": "sugar", "what": "sugar added", "seconds": [30, 35], "match": r"\b(sugar|powder|scoop\w*)\b", "witness": ["add_into"]},
    {"id": "lid", "what": "spoon on lid", "seconds": [60, 62], "match": r"\blid\b", "witness": ["put_down"]},
    {"id": "bite", "what": "bite eaten", "seconds": [90, 100], "match": r"\b(bite|eaten)\b", "witness": []}]}


def ev(i, text, lo, hi, status="verified", clock="clock_agrees"):
    return {"id": f"e{i}", "description": text, "minutes_before_min": lo, "minutes_before_max": hi, "status": status,
            "clock": clock}


def test_true_minutes_counts_back_from_the_end_plus_photo_delay():
    assert score.true_minutes(KEY, KEY["events"][0]) == (1.67, 2.83)


def test_each_event_used_once_and_specific_events_placed_first():
    events = [ev(1, "The spoon scooped powder and was left on the lid", 0, 20), ev(2, "Sugar was scooped from the jar", 0, 60)]
    m = score.match_events(KEY, events)
    assert m["sugar"]["id"] == "e2" and m["lid"]["id"] == "e1"


def test_score_counts_found_covered_order_and_unsupported():
    events = [ev(1, "Sugar scooped in", 0, 60), ev(2, "Milk poured into the glass", 0, 1.0),
              ev(3, "Milk was frothed", 0, 40), ev(4, "A rejected guess about a lid", 0, 5, status="rejected")]
    s = score.score_run({"events": events}, KEY)
    assert (s["found"], s["total"], s["dated"], s["covered"]) == (2, 4, 2, 1)   # the 0-1 min pour misses 1.67-2.83
    assert [r["key"] for r in s["retro"]] == ["sugar", "pour", None]
    assert s["order_wrong"] == ["pour happened before sugar"] and s["order_pairs"] == 1
    assert [u["event"] for u in s["unsupported"]] == ["e3"]              # rejected events are not scored


def test_witness_kinds_from_labels():
    assert score.witness_kinds("Pouring [something] into [something]") == {"add_into"}
    assert score.witness_kinds("Putting [something] into [something]") == {"add_into"}
    assert score.witness_kinds("Putting [something] next to [something]") == {"put_down"}
    assert score.witness_kinds("Pushing [something] so it spins") == {"stir"}
    assert score.witness_kinds("Showing that [something] is inside [something]") == set()


def test_witness_check_seen_unseen_cant_see_and_false_alarms():
    win = lambda s, label, p: {"start": s, "end": s + 3, "actions": [[label, p]]}
    watched = {"windows": [win(12, "Pouring [something] into [something]", 0.8),
                           win(31, "Pouring [something] into [something]", 0.2),      # too unsure
                           win(45, "Sprinkling [something] onto [something]", 0.6),   # nothing real then
                           win(60, "Showing that [something] is inside [something]", 0.9)]}
    w = score.witness_check(KEY, watched)
    seen = {r["id"]: r.get("seen") for r in w["events"]}
    assert seen == {"pour": True, "sugar": False, "lid": False, "bite": None}
    assert (w["seen"], w["seeable"], w["action_windows"], w["stray_windows"]) == (1, 3, 2, 1)


def test_milk_key_is_valid():
    key = score.load_key(Path(__file__).resolve().parent.parent / "eval/keys/milk_01.json")
    assert key["split"] == "dev" and len(key["events"]) == 8
    assert all(e["seconds"][0] <= e["seconds"][1] <= key["video_seconds"] for e in key["events"])


def test_plain_action_words():
    assert score.plain_action("Pouring [something] into [something]") == "pouring one thing into another"
    assert score.plain_action("Pushing [something] so it spins") == "pushing something so it spins"
    assert score.plain_action("Putting [something similar to other things that are already on the table]") == \
        "putting something like what is already on the table"


def test_guessed_times_are_not_scored_as_right():
    s = score.score_run({"events": [ev(1, "Milk poured", 0, 60, clock="no_clock")]}, KEY)
    assert (s["found"], s["dated"], s["covered"]) == (1, 0, 0) and s["median_width_min"] is None
    assert s["events"][0]["range_min"] is None and s["real_span_min"] == [0.33, 2.83]


def test_cutting_words_v_jepa_uses():
    assert score.witness_kinds("Poking a hole into [something soft]") == {"cut"}
    assert score.witness_kinds("Bending [something] until it breaks") == {"cut"}
    assert score.witness_kinds("Piling [something] up") == {"add_into"}
