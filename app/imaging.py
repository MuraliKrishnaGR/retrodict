"""Image prep and drawing helpers."""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from app import config

COLORS = [(230, 57, 70), (42, 157, 143), (69, 123, 157), (244, 162, 97),
          (155, 93, 229), (0, 180, 216), (233, 196, 106), (106, 153, 78)]


def prepare_image(path, max_side: int = config.MAX_IMAGE_SIDE):
    """Fix phone rotation, drop EXIF (location etc.), shrink. Returns (PIL image, jpeg bytes)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return img, buf.getvalue()


def save_mask_png(mask: np.ndarray, path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def overlay(img: Image.Image, objects: list) -> Image.Image:
    """objects: [{'id','label','mask','bbox'}] -> image with tinted masks and labels."""
    base = np.array(img).astype(np.float32)
    for i, o in enumerate(objects):
        c = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        base[o["mask"]] = base[o["mask"]] * 0.45 + c * 0.55
    out = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, o in enumerate(objects):
        l, t, r, b = o["bbox"]
        col = COLORS[i % len(COLORS)]
        d.rectangle([l, t, r, b], outline=col, width=2)
        tag = f"{o['id']} {o['label']}"
        d.rectangle([l, max(0, t - 14), l + 7 * len(tag) + 6, t], fill=col)
        d.text((l + 3, max(0, t - 13)), tag, fill=(255, 255, 255))
    return out
