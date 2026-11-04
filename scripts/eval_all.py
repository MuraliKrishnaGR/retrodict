"""Score every scene in eval/keys against the saved runs, Retrodict and the one-call baseline side by side.
Free: reads result.json and judge.json, makes no model calls. Judges can reproduce the results table.

  python scripts/eval_all.py               # runs in out/ (falls back to examples/)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import score  # noqa: E402


def find_run(*parts):
    for base in (ROOT / "out", ROOT / "examples"):
        d = base.joinpath(*parts)
        if (d / "result.json").exists():
            return d
    return None


def row(run_dir, key):
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    judged = score.load_judged(run_dir, result)
    s = score.score_run(result, key, judged)
    wrong = len(s["false_claims"]) if judged else "?"
    times = f"{s['covered']} of {s['dated']}" if s["dated"] else "none"
    order = f"{s['order_agree']} of {s['order_pairs']}" if s["order_pairs"] else "-"
    return [f"{s['found']} of {s['total']}", str(wrong), order, times, s["matched_by"]]


def main():
    head = ["scene", "split", "system", "found", "wrong", "order (pairs)", "times right", "matched by"]
    rows = []
    for kp in sorted((ROOT / "eval" / "keys").glob("*.json")):
        key = score.load_key(kp)
        for photo in key.get("photos", []):
            stem = Path(photo).stem
            for system, parts in (("Retrodict", (stem,)), ("one call", ("baselines", stem))):
                d = find_run(*parts)
                if d:
                    rows.append([key["video"], key.get("split", ""), system] + row(d, key))
                else:
                    rows.append([key["video"], key.get("split", ""), system, "not run", "", "", "", ""])
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head, ["-" * w for w in widths]] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print("\n'wrong' needs the blind judge (scripts/judge.py); '?' means that run has no valid judge.json.")


if __name__ == "__main__":
    main()
