"""Make (or remake) the rewind clip for a finished run, with no model calls.

  python scripts/rewind.py out/milk_photo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rewind import make_rewind  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/rewind.py out/<photo name>")
    run_dir = Path(sys.argv[1])
    if not (run_dir / "result.json").exists():
        raise SystemExit(f"No result.json in {run_dir}. Run: python scripts/run.py <photo> first.")
    info = make_rewind(run_dir)
    print(f"Saved {run_dir / info['video']} ({info['seconds']}s, {info['frames']} frames)")
    for s in info["steps"]:
        print(f"  undo {s['event_id']} on {s['object_id']}: {s['kind']}")
    print(f"Stills for the web page: {', '.join(info['stills'])}")


if __name__ == "__main__":
    main()
