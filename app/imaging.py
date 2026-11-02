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
        ty = t - 14 if t >= 14 else min(b, out.height - 14)   # no room above (top edge): label goes below
        ty = max(0, ty)
        d.rectangle([l, ty, l + 7 * len(tag) + 6, ty + 14], fill=col)
        d.text((l + 3, ty + 1), tag, fill=(255, 255, 255))
    return out


def boosted_pair(crop: Image.Image, gap: int = 8) -> Image.Image:
    """The crop as photographed (left) beside the same crop with its histogram equalised (right).
    Faint steam over a dark background, wet shine and thin residue are often invisible at normal
    contrast. Equalising (not a plain stretch) works even when the crop already has white and black."""
    crop = crop.convert("RGB")
    strong = ImageOps.equalize(crop)
    pair = Image.new("RGB", (crop.width * 2 + gap, crop.height), (255, 255, 255))
    pair.paste(crop, (0, 0))
    pair.paste(strong, (crop.width + gap, 0))
    return pair


def evidence_sheet(full: Image.Image, scale: tuple, objects: list, tile: int = 360, cols: int = 2,
                   margin: float = 0.25, above: float = 0.8) -> Image.Image:
    """Sharp close-ups for the verifier, cut from the ORIGINAL photo: each object as photographed beside a
    contrast-boosted copy, with the space above it. objects: [{'id','label','bbox'}] in prepared-image
    pixels; scale = (sx, sy) from prepared image to original."""
    sx, sy = scale
    cells = []
    for o in objects:
        l, t, r, b = o["bbox"]
        mx, my, up = (r - l) * margin, (b - t) * margin, (b - t) * above
        box = (max(0, int((l - mx) * sx)), max(0, int((t - up) * sy)),
               min(full.width, int((r + mx) * sx)), min(full.height, int((b + my) * sy)))
        crop = full.crop(box)
        crop.thumbnail((tile, tile))
        cells.append((o, boosted_pair(crop)))
    cw = 2 * tile + 8
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + (cols - 1) * 12, rows * (tile + 22)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (o, pair) in enumerate(cells):
        x, y = (i % cols) * (cw + 12), (i // cols) * (tile + 22)
        d.rectangle([x, y, x + cw - 1, y + 20], fill=COLORS[i % len(COLORS)])
        d.text((x + 4, y + 4), f"{o['id']}  {o['label']}", fill=(255, 255, 255))
        sheet.paste(pair, (x, y + 22))
    return sheet


def to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


ABOVE = 0.5     # extra height kept above an object in close-ups (steam rises out of the box)


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
        up = (b - t) * ABOVE                      # steam rises: keep the space above each object in view
        box = (max(0, int(l - mx)), max(0, int(t - up)), min(img.width, int(r + mx)), min(img.height, int(b + my)))
        crop = img.crop(box)
        crop.thumbnail((tile, tile))
        x = (i % cols) * tile
        y = top.height + (i // cols) * (tile + 22)
        d.rectangle([x, y, x + tile - 1, y + 20], fill=COLORS[i % len(COLORS)])
        d.text((x + 4, y + 4), f"{o['id']}  {o['label']}", fill=(255, 255, 255))
        sheet.paste(crop, (x + (tile - crop.width) // 2, y + 22))
    return sheet
