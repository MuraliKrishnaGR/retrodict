"""Score a Retrodict timeline against what really happened.

Two sources of truth, kept apart:
- the answer key: the person's notes from the real video, as eval/keys/<video>.json (committed, so the key is on record before the runs it scores), written
  before scoring (times in seconds from the start of the video);
- a witness: V-JEPA 2 (Meta's video model) watching the same video (app/watch.py). It only knows hand
  actions, so it confirms some key events independently and is silent on others (eating, drinking).

What is measured, all from the photo's point of view (minutes before the photo):
- found: key events Retrodict describes (matched by the key's patterns, one Retrodict event per key event);
- dated / covered: how many found events got a time from a clue in the photo, and how many of those
  times contain the real time (the model's guesses are not times and are not scored as right);
- width: how wide those times are (a range of "up to 60 min" for something 2 minutes ago is right but
  not useful);
- order: pairs of found events in the right order;
- unsupported: Retrodict events that match nothing in the key."""
import json
import re
from pathlib import Path

from app.reason import is_dated

# V-JEPA 2 (Something-Something v2) labels -> the kinds of action they show. Labels about the camera or
# about holding still ("Showing that [something] is inside [something]") show no action and are left out.
WITNESS_KINDS = [
    (r"^(Pouring|Trying to pour|Sprinkling|Putting \[something\] into|Stuffing|Spilling)", {"add_into"}),
    (r"scoop", {"add_into", "stir"}),
    (r"so (that )?it spins|Spinning|Stirring", {"stir"}),
    (r"^(Putting|Laying|Placing|Letting)", {"put_down"}),
    (r"^(Picking \[something\] up|Lifting)", {"pick_up"}),
    (r"^(Opening|Tearing|Unfolding|Peeling)", {"open"}),
    # it has no word for cutting: a knife through fruit reads as these (salad_01, 3 Nov)
    (r"^(Poking a hole into|Bending \[something\] until it breaks|Tearing|Cutting|Breaking)", {"cut"}),
    (r"^Piling", {"add_into"}),
    (r"^Taking \[one of many", {"pick_up"}),
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


def load_judged(run_dir, result: dict):
    """The blind judge's reading of this run, or None if there is none or it was made for an older run
    (a re-run replaces the events, so a stale reading must never be applied to them)."""
    p = Path(run_dir) / "judge.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    now = [e["description"] for e in result.get("events", []) if e.get("status") != "rejected"]
    return j if j.get("claims_text") == now else None


def judged_matches(key: dict, events: list, judged: dict) -> dict:
    """{key event id: event} from a judge's {event id: [key ids it covers]}. One event may cover several
    real events ("banana, apple and orange were chopped"); each real event takes the first event covering it."""
    by_id = {e["id"]: e for e in events}
    out = {}
    for e in events:
        for kid in judged.get("covers", {}).get(e["id"], []):
            if kid not in out and kid in {k["id"] for k in key["events"]}:
                out[kid] = by_id[e["id"]]
    return out


def score_run(result: dict, key: dict, judged: dict | None = None) -> dict:
    """Without `judged`, events are matched by the key's word patterns; with it, by a blind judge's
    reading (scripts/judge.py), which handles wording the patterns miss."""
    events = [e for e in result.get("events", []) if e.get("status") != "rejected"]
    matched = judged_matches(key, events, judged) if judged else match_events(key, events)
    rows = []
    for k in key["events"]:
        t_lo, t_hi = true_minutes(key, k)
        e = matched.get(k["id"])
        row = {"id": k["id"], "what": k["what"], "true_min": [t_lo, t_hi], "found": e is not None}
        if e:
            lo, hi = e["minutes_before_min"], e["minutes_before_max"]
            dated = is_dated(e)                    # a time counts only if a clue in the photo gave one
            row.update(event=e["id"], text=e["description"], dated=dated, range_min=[lo, hi] if dated else None,
                       covered=dated and lo <= t_hi and hi >= t_lo, width=round(hi - lo, 1) if dated else None)
            est = None if dated else e.get("estimate")       # the reasoned time, scored apart from proven ones
            row.update(estimate=est, est_covered=bool(est) and est[0] <= t_hi and est[1] >= t_lo)
        rows.append(row)
    found = [r for r in rows if r["found"]]
    # order: Retrodict lists events oldest first; compare every pair of found events with the real order
    pairs = agree = 0
    wrong = []
    pos = {r["event"]: i for i, r in enumerate(sorted(found, key=lambda r: int(r["event"].lstrip("e"))))}
    start = {k["id"]: k["seconds"][0] for k in key["events"]}
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            if start[a["id"]] == start[b["id"]] or a["event"] == b["event"]:
                continue
            pairs += 1
            real = start[a["id"]] < start[b["id"]]
            if (pos[a["event"]] < pos[b["event"]]) == real:
                agree += 1
            else:
                first, then = (a, b) if real else (b, a)
                wrong.append(f"{first['id']} happened before {then['id']}")
    used = {r["event"] for r in found}
    wrong_ids = set((judged or {}).get("wrong", []))
    unsupported = [{"event": e["id"], "text": e["description"], "wrong": e["id"] in wrong_ids}
                   for e in events if e["id"] not in used]
    dated = [r for r in found if r["dated"]]
    keys_of = {}
    for r in found:
        keys_of.setdefault(r["event"], []).append(r["id"])
    return {
        "matched_by": "blind judge" if judged else "word patterns",
        "events": rows,
        "retro": [{"event": e["id"], "text": e["description"], "key": (keys_of.get(e["id"]) or [None])[0],
                   "keys": keys_of.get(e["id"], []), "wrong": e["id"] in wrong_ids} for e in events],
        "false_claims": [{"event": e["id"], "text": e["description"]} for e in events if e["id"] in wrong_ids],
        "found": len(found), "total": len(rows),
        "dated": len(dated), "covered": sum(r["covered"] for r in dated),
        "median_width_min": _median([r["width"] for r in dated]),
        "estimated": sum(1 for r in found if r.get("estimate")),
        "est_covered": sum(1 for r in found if r.get("est_covered")),
        "real_span_min": [min(r["true_min"][0] for r in rows), max(r["true_min"][1] for r in rows)],
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
                     "label": best[3][0] if best else None, "plain": plain_action(best[3][0]) if best else None,
                     "prob": best[3][1] if best else None, "at": [best[0], best[1]] if best else None})
    # how often the witness reports an action where the key has none (its false alarms)
    acting = [w for w in windows if w[2]]
    spans = [(k["seconds"][0] - slack, k["seconds"][1] + slack) for k in key["events"]]
    stray = [w for w in acting if not any(w[0] <= b and w[1] >= a for a, b in spans)]
    seeable = [r for r in rows if r["can_see"]]
    return {"events": rows, "seen": sum(r["seen"] for r in seeable), "seeable": len(seeable),
            "action_windows": len(acting), "stray_windows": len(stray),
            "stray": [{"at": [w[0], w[1]], "label": w[3][0], "prob": w[3][1]} for w in stray]}


def plain_action(label: str) -> str:
    """V-JEPA 2 never names objects (it learned actions from videos whose labels say 'something'), so its
    labels read 'Pouring [something] into [something]'. For people: 'pouring one thing into another'."""
    text = label.replace("[something similar to other things that are already on the table]",
                         "something like what is already on the table")
    if text.count("[something]") == 2:
        text = text.replace("[something]", "one thing", 1).replace("[something]", "another", 1)
    text = text.replace("[something]", "something")
    return text[:1].lower() + text[1:]


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2, 2)
