"""A contact sheet of a video: one frame every few seconds, each stamped with its time, for writing an
answer key (what really happened, and when) before any run on the photo. Free, no model calls.

  python scripts/frames.py data/videos/tea_01.mp4            # every 3 s -> out/frames_tea_01.jpg
  python scripts/frames.py data/videos/tea_01.mp4 --every 5
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    r = iio.get_reader(a.video, "ffmpeg")
    fps = float(r.get_meta_data().get("fps") or 30)
    step = max(1, int(round(a.every * fps)))
    tiles, f = [], font(22)
    for i, frame in enumerate(r):
        if i % step:
            continue
        im = Image.fromarray(frame)
        im.thumbnail((320, 320))
        d = ImageDraw.Draw(im)
        t = i / fps
        d.rectangle([0, 0, 92, 30], fill=(0, 0, 0))
        d.text((6, 3), f"{int(t // 60)}:{int(t % 60):02d}", font=f, fill=(255, 220, 0))
        tiles.append(im)
    r.close()
    if not tiles:
        sys.exit("no frames read")
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (w * a.cols, h * rows))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % a.cols) * w, (k // a.cols) * h))
    out = ROOT / "out" / f"frames_{Path(a.video).stem}.jpg"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out, quality=82)
    print(f"Saved {out} ({len(tiles)} frames, one every {a.every:g} s)")


if __name__ == "__main__":
    main()
