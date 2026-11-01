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


def mask_outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Pixels on the inside edge of a mask (mask minus the mask shrunk by `width` pixels)."""
    inner = mask.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    return mask & ~inner


def overlay(img: Image.Image, objects: list, tint: float = 0.55, outline: int = 0) -> Image.Image:
    """objects: [{'id','label','mask','bbox'}] -> image with tinted masks and labels.
    tint is how strongly the mask colour covers the object; outline > 0 also draws its edge."""
    base = np.array(img).astype(np.float32)
    for i, o in enumerate(objects):
        c = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        if tint > 0:
            base[o["mask"]] = base[o["mask"]] * (1 - tint) + c * tint
        if outline > 0:
            base[mask_outline(o["mask"], outline)] = c
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


def to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def contact_sheet(img: Image.Image, objects: list, tile: int = 256, cols: int = 4,
                  margin: float = 0.15) -> Image.Image:
    """One image the model can read: the scene with numbered masks on top, and below it a close-up
    crop of every object labelled with its id. objects: [{'id','label','mask','bbox'}]."""
    top = overlay(img, objects, tint=0.12, outline=3)   # light tint + edge: the model sees true colours
    scale = min(1.0, 1024 / top.width)
    top = top.resize((int(top.width * scale), int(top.height * scale)))
    rows = (len(objects) + cols - 1) // cols
    sheet = Image.new("RGB", (max(top.width, cols * tile), top.height + rows * (tile + 22)), (255, 255, 255))
    sheet.paste(top, (0, 0))
    d = ImageDraw.Draw(sheet)
    for i, o in enumerate(objects):
        l, t, r, b = o["bbox"]
        mx, my = (r - l) * margin, (b - t) * margin
        box = (max(0, int(l - mx)), max(0, int(t - my)), min(img.width, int(r + mx)), min(img.height, int(b + my)))
        crop = img.crop(box)
        crop.thumbnail((tile, tile))
        x = (i % cols) * tile
        y = top.height + (i // cols) * (tile + 22)
        d.rectangle([x, y, x + tile - 1, y + 20], fill=COLORS[i % len(COLORS)])
        d.text((x + 4, y + 4), f"{o['id']}  {o['label']}", fill=(255, 255, 255))
        sheet.paste(crop, (x + (tile - crop.width) // 2, y + 22))
    return sheet
