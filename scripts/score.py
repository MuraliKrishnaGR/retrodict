"""Score a Retrodict run against the answer key from the real video, and V-JEPA 2's view of that video.
Free: no model calls.

  python scripts/score.py out/milk_photo_original eval/keys/milk_01.json
  python scripts/score.py out/milk_photo_original eval/keys/milk_01.json --witness out/watch_milk_01/actions.json

Saves out/<run>/score.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import score                          # noqa: E402


def mins(lo, hi):
    return f"{lo:g}-{hi:g} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("key")
    ap.add_argument("--witness", help="actions.json from scripts/watch_video.py")
    a = ap.parse_args()
    run = Path(a.run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    key = score.load_key(a.key)
    s = score.score_run(result, key)

    print(f"Scene {key['video']} ({key.get('split', '?')} scene). Times are minutes before the photo.\n")
    for r in s["events"]:
        real = mins(*r["true_min"])
        if not r["found"]:
            print(f"  MISSED   {r['what']}  (really {real})")
        else:
            if not r["dated"]:
                when = "no time given (no clue in the photo)"
            else:
                when = f"{mins(*r['range_min'])}, " + ("in range" if r["covered"] else "OUT OF RANGE")
            print(f"  FOUND    {r['what']}  (really {real}; Retrodict {r['event']}: {when})")
    for u in s["unsupported"]:
        print(f"  NOT IN KEY  {u['event']}: {u['text']}")

    lo, hi = s["real_span_min"]
    print(f"\nFound {s['found']} of {s['total']} real events. They really happened {lo:g}-{hi:g} min before the photo.")
    if s["dated"]:
        print(f"Times: {s['dated']} of {s['found']} found events were dated by a clue; {s['covered']} of those contain "
              f"the real time; typical width {s['median_width_min']:g} min.")
    else:
        print("Times: no clue in the photo dated any of them, so Retrodict gave no times (it says 'time unknown').")
    print(f"Order: {s['order_agree']} of {s['order_pairs']} pairs right"
          + (f" (wrong: {'; '.join(s['order_wrong'])})" if s["order_wrong"] else "") + ".")
    print(f"Events not in the key: {len(s['unsupported'])}.")

    out = {"key": a.key, "run": str(run), "scene": {"video": key["video"], "split": key.get("split", ""),
                                                   "video_seconds": key["video_seconds"]}, "score": s}
    if a.witness:
        w = score.witness_check(key, json.loads(Path(a.witness).read_text(encoding="utf-8")))
        out["witness"] = w
        print("\nV-JEPA 2 watching the real video (a second witness, separate from the notes):")
        for r in w["events"]:
            if not r["can_see"]:
                print(f"  CAN'T SEE  {r['what']}  (no such action in its vocabulary)")
            elif r["seen"]:
                print(f"  SAW        {r['what']}  as \"{r['plain']}\" at {r['at'][0]:g}-{r['at'][1]:g} s ({r['prob']:.2f})")
            else:
                print(f"  DIDN'T SEE {r['what']}")
        print(f"Saw {w['seen']} of the {w['seeable']} events it has words for. "
              f"{w['stray_windows']} of its {w['action_windows']} action windows match no real event (false alarms).")

    (run / "score.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {run / 'score.json'}")


if __name__ == "__main__":
    main()
