"""Try the microscope search by hand, with no model calls (SAM only, free, ~30-60 s on CPU):
SAM looks for <phrase> on the whole photo, then inside the full-resolution close-up of one object.

  python scripts/microscope.py out/milk_photo_original o1 spoon

Saves out/<run>/micro_<object>_<phrase>.jpg: the close-up with every find outlined and scored."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                       # noqa: E402
from app.agent import Scene                   # noqa: E402
from app.cache import DiskCache               # noqa: E402
from app.clients.sam_local import LocalSAM    # noqa: E402
from app.imaging import COLORS, mask_outline, prepare_image  # noqa: E402


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: python scripts/microscope.py out/<run> <object id> <phrase>")
    run_dir, oid, phrase = Path(sys.argv[1]), sys.argv[2], " ".join(sys.argv[3:]).lower()
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    img, jpeg = prepare_image(res["photo"])
    objects = [dict(o) for o in res["objects"]]
    if oid not in {o["id"] for o in objects}:
        raise SystemExit(f"No {oid} in this run. Objects: {', '.join(o['id'] + ' ' + o['label'] for o in objects)}")
    drawn = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"],
              "mask": np.array(Image.open(run_dir / o["mask_png"])) > 127} for o in objects]
    scene = Scene(res["photo"], run_dir, img, jpeg, objects, drawn, res.get("states", {}))
    sam = LocalSAM(cache=DiskCache(config.CACHE_DIR))

    whole = sam.segment(jpeg, phrase)
    print(f"Whole photo ({img.width}x{img.height}): {len(whole)} '{phrase}' found"
          + (f", best score {max(i['score'] for i in whole):.2f}" if whole else ""))

    crop_jpeg, box, (cw, ch) = scene.micro_crop(oid)
    found = sam.segment(crop_jpeg, phrase)
    print(f"Inside {oid} close-up ({cw}x{ch} from the original photo): {len(found)} found")
    crop = Image.open(__import__("io").BytesIO(crop_jpeg)).convert("RGB")
    out = np.array(crop)
    for i, it in enumerate(found):
        out[mask_outline(it["mask"], 3)] = COLORS[i % len(COLORS)]
        print(f"  #{i + 1}: score {it['score']:.2f}, box {it['bbox']}, {int(it['mask'].sum())} px")
    pic = Image.fromarray(out)
    d = ImageDraw.Draw(pic)
    for i, it in enumerate(found):
        l, t = it["bbox"][:2]
        d.text((l + 3, max(0, t - 12)), f"#{i + 1} {it['score']:.2f}", fill=COLORS[i % len(COLORS)])
    path = run_dir / f"micro_{oid}_{phrase.replace(' ', '_')}.jpg"
    pic.save(path, quality=90)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
