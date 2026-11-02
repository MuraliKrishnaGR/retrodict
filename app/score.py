"""Score a Retrodict timeline against what really happened.

Two sources of truth, kept apart:
- the answer key: the person's notes from the real video, as eval/keys/<video>.json (committed, so the key is on record before the runs it scores), written
  before scoring (times in seconds from the start of the video);
- a witness: V-JEPA 2 (Meta's video model) watching the same video (app/watch.py). It only knows hand
  actions, so it confirms some key events independently and is silent on others (eating, drinking).

What is measured, all from the photo's point of view (minutes before the photo):
- found: key events Retrodict describes (matched by the key's patterns, one Retrodict event per key event);
- covered: of those, how many have a time range that contains the real time;
- width: how wide those ranges are, next to the real time (a range of "up to 60 min" for something
  2 minutes ago is right but not useful);
- order: pairs of found events in the right order;
- unsupported: Retrodict events that match nothing in the key."""
import json
import re
from pathlib import Path

# V-JEPA 2 (Something-Something v2) labels -> the kinds of action they show. Labels about the camera or
# about holding still ("Showing that [something] is inside [something]") show no action and are left out.
WITNESS_KINDS = [
    (r"^(Pouring|Trying to pour|Sprinkling|Putting \[something\] into|Stuffing|Spilling)", {"add_into"}),
    (r"scoop", {"add_into", "stir"}),
    (r"so (that )?it spins|Spinning|Stirring", {"stir"}),
    (r"^(Putting|Laying|Placing|Letting)", {"put_down"}),
    (r"^(Picking \[something\] up|Lifting)", {"pick_up"}),
    (r"^(Opening|Tearing|Unfolding|Peeling)", {"open"}),
]
WITNESS_MIN_PROB = 0.3
WITNESS_SLACK_S = 2.0


def load_key(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def true_minutes(key: dict, ev: dict) -> tuple:
    """When a key event happened, in minutes before the photo: (earliest, latest)."""
    end = key["video_seconds"]
    lo_delay, hi_delay = key.get("photo_delay_min", [0, 0])
    return (round((end - ev["seconds"][1]) / 60 + lo_delay, 2), round((end - ev["seconds"][0]) / 60 + hi_delay, 2))


def match_events(key: dict, events: list) -> dict:
    """{key event id: Retrodict event}. Each Retrodict event is used once; events that fit fewer key
    events are placed first, then the earliest fitting key event in the key's order."""
    fits = {e["id"]: [k["id"] for k in key["events"] if re.search(k["match"], e["description"], re.I)] for e in events}
    by_id = {e["id"]: e for e in events}
    out = {}
    for eid in sorted(fits, key=lambda i: (len(fits[i]) or 99, i)):
        for kid in fits[eid]:
            if kid not in out:
                out[kid] = by_id[eid]
                break
    return out


def score_run(result: dict, key: dict) -> dict:
    events = [e for e in result.get("events", []) if e.get("status") != "rejected"]
    matched = match_events(key, events)
    rows = []
    for k in key["events"]:
        t_lo, t_hi = true_minutes(key, k)
        e = matched.get(k["id"])
        row = {"id": k["id"], "what": k["what"], "true_min": [t_lo, t_hi], "found": e is not None}
        if e:
            lo, hi = e["minutes_before_min"], e["minutes_before_max"]
            row.update(event=e["id"], text=e["description"], range_min=[lo, hi],
                       covered=lo <= t_hi and hi >= t_lo, width=round(hi - lo, 1))
        rows.append(row)
    found = [r for r in rows if r["found"]]
    # order: Retrodict lists events oldest first; compare every pair of found events with the real order
    pairs = agree = 0
    wrong = []
    pos = {r["event"]: i for i, r in enumerate(sorted(found, key=lambda r: int(r["event"].lstrip("e"))))}
    start = {k["id"]: k["seconds"][0] for k in key["events"]}
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            if start[a["id"]] == start[b["id"]]:
                continue
            pairs += 1
            real = start[a["id"]] < start[b["id"]]
            if (pos[a["event"]] < pos[b["event"]]) == real:
                agree += 1
            else:
                first, then = (a, b) if real else (b, a)
                wrong.append(f"{first['id']} happened before {then['id']}")
    used = {r["event"] for r in found}
    unsupported = [{"event": e["id"], "text": e["description"]} for e in events if e["id"] not in used]
    covered = [r for r in found if r["covered"]]
    return {
        "events": rows,
        "found": len(found), "total": len(rows),
        "covered": len(covered),
        "median_width_min": _median([r["width"] for r in found]),
        "median_true_min": _median([sum(r["true_min"]) / 2 for r in found]),
        "order_agree": agree, "order_pairs": pairs, "order_wrong": wrong,
        "unsupported": unsupported,
    }


def witness_kinds(label: str) -> set:
    kinds = set()
    for pattern, ks in WITNESS_KINDS:
        if re.search(pattern, label, re.I):
            kinds |= ks
            if pattern.startswith("^(Pouring"):
                break                      # "Putting [something] into" is adding, not putting down
    return kinds


def witness_check(key: dict, watched: dict, min_prob: float = WITNESS_MIN_PROB, slack: float = WITNESS_SLACK_S) -> dict:
    """Does V-JEPA 2's top action, in the windows around each key event, show that kind of action?"""
    windows = [(w["start"], w["end"], witness_kinds(w["actions"][0][0]), w["actions"][0])
               for w in watched["windows"] if w["actions"][0][1] >= min_prob]
    rows = []
    for k in key["events"]:
        if not k.get("witness"):
            rows.append({"id": k["id"], "what": k["what"], "can_see": False})
            continue
        s, e = k["seconds"]
        near = [w for w in windows if w[0] <= e + slack and w[1] >= s - slack and w[2] & set(k["witness"])]
        best = max(near, key=lambda w: w[3][1]) if near else None
        rows.append({"id": k["id"], "what": k["what"], "can_see": True, "seen": bool(near),
                     "label": best[3][0] if best else None, "prob": best[3][1] if best else None,
                     "at": [best[0], best[1]] if best else None})
    # how often the witness reports an action where the key has none (its false alarms)
    acting = [w for w in windows if w[2]]
    spans = [(k["seconds"][0] - slack, k["seconds"][1] + slack) for k in key["events"]]
    stray = [w for w in acting if not any(w[0] <= b and w[1] >= a for a, b in spans)]
    seeable = [r for r in rows if r["can_see"]]
    return {"events": rows, "seen": sum(r["seen"] for r in seeable), "seeable": len(seeable),
            "action_windows": len(acting), "stray_windows": len(stray),
            "stray": [{"at": [w[0], w[1]], "label": w[3][0], "prob": w[3][1]} for w in stray]}


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2, 2)
