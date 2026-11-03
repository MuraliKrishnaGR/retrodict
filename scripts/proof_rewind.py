"""Proof rewind: the photo Retrodict read, side by side with the real video played backward (free).

  python scripts/proof_rewind.py out/salad_photo eval/keys/salad_01.json
  python scripts/proof_rewind.py out/salad_photo eval/keys/salad_01.json --video data/videos/salad_01.mp4

Uses the blind judge's matches if out/<run>/judge.json exists (scripts/judge.py). Saves out/<run>/proof_rewind.mp4."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                       # noqa: E402
from app.proof import ProofRewind            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("key")
    ap.add_argument("--video")
    a = ap.parse_args()
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    video = Path(a.video) if a.video else config.ROOT / "data" / "videos" / key["video"]
    if not video.exists():
        raise SystemExit(f"No video at {video}")
    print("Reading the video and building the clip (a minute or two)...", flush=True)
    info = ProofRewind(a.run, a.key, video).write()
    print(f"Saved {info['video']} ({info['seconds']} s)")


if __name__ == "__main__":
    main()
