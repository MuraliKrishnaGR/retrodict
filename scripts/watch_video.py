"""Let V-JEPA 2 (Meta's video model, open weights, free, runs on CPU) watch a real video and list the hand
actions it sees, with times. This checks Retrodict against what really happened; Retrodict itself only
ever gets the photo.

  python scripts/watch_video.py data/videos/milk_01.mp4
  python scripts/watch_video.py data/videos/milk_01.mp4 --window 3 --stride 1.5

The first run downloads the model (about 1.3 GB, once). No token needed for this model.
Saves out/watch_<video>/actions.json (every window's top 3 actions) and prints the merged timeline."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, watch                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--window", type=float, default=3.0, help="seconds the model sees at once")
    ap.add_argument("--stride", type=float, default=1.5, help="seconds between windows")
    ap.add_argument("--min-prob", type=float, default=0.2, help="hide windows the model is unsure about")
    a = ap.parse_args()
    video = Path(a.video)
    if not video.exists():
        raise SystemExit(f"No video at {video}")

    print(f"Reading {video.name}...", flush=True)
    frames, times, duration = watch.read_frames(video)
    print(f"  {duration:.0f} s, {len(frames)} frames kept", flush=True)
    print(f"Loading V-JEPA 2 ({watch.MODEL_ID})...", flush=True)
    model = watch.load()
    t0 = time.time()

    def progress(k, total, row):
        left = (time.time() - t0) / k * (total - k)
        label, p = row["actions"][0]
        print(f"  [{k}/{total}] {row['start']:6.1f}-{row['end']:6.1f} s  {label} ({p:.2f})  ~{left / 60:.1f} min left",
              flush=True)

    rows = watch.label_windows(model, frames, times, duration, a.window, a.stride, progress=progress)
    segs = watch.segments(rows, duration, a.min_prob)

    out = config.OUT_DIR / f"watch_{video.stem}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "actions.json").write_text(json.dumps(
        {"video": str(video), "model": watch.MODEL_ID, "duration": round(duration, 2), "window": a.window,
         "stride": a.stride, "windows": rows, "segments": segs}, indent=2), encoding="utf-8")

    print(f"\nWhat V-JEPA 2 saw ({len(segs)} segments, unsure windows hidden). Times from the start, "
          f"and counted back from the end like notes.txt:")
    for s in segs:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f} s   ({s['before_end'][0] / 60:.1f}-{s['before_end'][1] / 60:.1f} min "
              f"before end)   {s['action']}  (best {s['best']:.2f})")
    print(f"\nTook {(time.time() - t0) / 60:.1f} min. Saved {out / 'actions.json'}")


if __name__ == "__main__":
    main()
