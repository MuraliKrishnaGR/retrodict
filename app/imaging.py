"""Image prep and drawing helpers."""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

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


GLARE_GAMMA = 2.5   # "dim the light": bright areas darken but keep their differences


def three_views(crop: Image.Image, gap: int = 8) -> Image.Image:
    """Three copies side by side:
    LEFT   as photographed;
    MIDDLE shadows lifted (histogram equalised): faint steam over a dark counter, film, thin residue.
           Equalising works even when the crop already has white and black, unlike a plain stretch;
    RIGHT  glare dimmed (Murali, 2 Nov: "if there is glare, dim the light to see the hidden things"):
           shiny steel, glass and plastic hide what is behind or on them under bright reflections that
           are rarely pure white, so darkening the highlights brings that detail back."""
    crop = crop.convert("RGB")
    lifted = ImageOps.equalize(crop)
    dimmed = Image.fromarray((255 * (np.asarray(crop, np.float32) / 255) ** GLARE_GAMMA).astype(np.uint8))
    out = Image.new("RGB", (crop.width * 3 + gap * 2, crop.height), (255, 255, 255))
    for i, im in enumerate((crop, lifted, dimmed)):
        out.paste(im, (i * (crop.width + gap), 0))
    return out


def inside_view(crop: Image.Image, mask: np.ndarray) -> Image.Image:
    """Only the object itself, as a WHITENESS map (Murali, 2 Nov: "mask the interior", "light editing is
    necessary"). Everything outside its mask is black. Each pixel shows its weakest colour channel, so white
    things (a white spoon, milk film, foam, steam) stay bright while coloured things (brown sugar, an orange
    label) go dark; contrast is stretched over the object's OWN pixels, not the white tiles around it."""
    a = np.asarray(crop.convert("RGB"), np.float32).min(axis=2) / 255
    out = np.zeros_like(a)
    if mask.any():
        vals = a[mask]
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        out[mask] = np.clip((vals - lo) / max(hi - lo, 1e-3), 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8)).convert("RGB")


def _stretch(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """0..1 over the object's own pixels; outside the object is black."""
    out = np.zeros_like(x, np.float32)
    if mask.any():
        vals = x[mask]
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        out[mask] = np.clip((vals - lo) / max(hi - lo, 1e-3), 0, 1)
    return out


def view_bank(crop: Image.Image) -> dict:
    """Ways to look at the same pixels (Murali, 2 Nov: "you know how it can be viewed in various ways").
    Each brings out something different: colour channels split white from brown and orange, local
    contrast and fine detail show faint edges (a spoon handle behind plastic), saturation separates printed
    labels from what is behind them."""
    c = np.asarray(crop.convert("RGB"))
    f = c.astype(np.float32) / 255
    gray = f.mean(-1)
    views = {"blue channel": f[..., 2], "red channel": f[..., 0],
             "colourless parts (low saturation)": 1 - (f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1e-3)}
    try:
        import cv2
        lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB)
        views["local contrast"] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0]) / 255.0
        views["fine detail"] = np.clip(0.5 + 3 * (gray - cv2.GaussianBlur(gray, (0, 0), 12)), 0, 1)
    except ImportError:
        pass
    return views


def detail_score(v: np.ndarray, mask: np.ndarray) -> float:
    """How much edge detail a view shows inside the object (a light median filter first, so JPEG noise
    does not count as detail)."""
    d = np.asarray(Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
                   np.float32) / 255
    gy, gx = np.gradient(d)
    return float(np.hypot(gx, gy)[mask].mean()) if mask.any() else 0.0


def best_views(crop: Image.Image, mask: np.ndarray, k: int = 2) -> list:
    """The k views from the bank that show the most detail inside THIS object, skipping near-duplicates."""
    scored = []
    for name, v in view_bank(crop).items():
        s = _stretch(v, mask)
        scored.append((detail_score(s, mask), name, s))
    scored.sort(key=lambda t: -t[0])
    def similar(x, y):                    # near-duplicate views (or a flat one) add nothing
        a, b = x[mask], y[mask]
        if a.std() < 1e-6 or b.std() < 1e-6:
            return True
        return abs(float(np.corrcoef(a, b)[0, 1])) >= 0.9

    picked = []
    for sc, name, s in scored:
        if mask.sum() > 2 and not any(similar(s, p[2]) for p in picked):
            picked.append((sc, name, s))
        if len(picked) == k:
            break
    return [(name, s) for _, name, s in picked]


def _label(tile: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, min(tile.width, 7 * len(text) + 8), 14], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 220, 0))
    return tile


def object_views(crop: Image.Image, mask: np.ndarray, gap: int = 8) -> Image.Image:
    """3 x 2 grid, each tile labelled: as photographed | shadows lifted | glare dimmed /
    object alone: whiteness | object alone: the two views from the bank that reveal most in THIS object."""
    three = three_views(crop, gap)
    w, h = crop.width, crop.height
    tiles = [(three.crop((i * (w + gap), 0, i * (w + gap) + w, h)), name)
             for i, name in enumerate(("as photographed", "shadows lifted", "glare dimmed"))]
    tiles.append((inside_view(crop, mask), "object alone: whiteness"))
    for name, s in best_views(crop, mask):
        tiles.append((Image.fromarray((s * 255).astype(np.uint8)).convert("RGB"), f"object alone: {name}"))
    out = Image.new("RGB", (3 * w + 2 * gap, 2 * h + gap), (255, 255, 255))
    for i, (t, name) in enumerate(tiles):
        out.paste(_label(t.copy(), name), ((i % 3) * (w + gap), (i // 3) * (h + gap)))
    return out


def evidence_sheet(full: Image.Image, scale: tuple, objects: list, tile: int = 300, cols: int = 2,
                   margin: float = 0.25, above: float = 0.8, overview: Image.Image | None = None) -> Image.Image:
    """Sharp close-ups for the verifier, cut from the ORIGINAL photo: each object as photographed beside a
    contrast-boosted copy, with the space above it. objects: [{'id','label','bbox'}] in prepared-image
    pixels; scale = (sx, sy) from prepared image to original."""
    sx, sy = scale
    cells = []
    for o in objects:
        crop = full.crop(crop_box(o["bbox"], (sx, sy), full.size, margin, above))
        crop.thumbnail((tile, tile))
        cells.append((o, three_views(crop)))
    cw = 3 * tile + 16
    rows = (len(cells) + cols - 1) // cols
    width = cols * cw + (cols - 1) * 12
    top = 0
    if overview is not None:                        # the whole scene first, so positions can be checked
        overview = overview.copy()
        overview.thumbnail((width, cw))
        top = overview.height + 12
    sheet = Image.new("RGB", (width, top + rows * (tile + 22)), (255, 255, 255))
    if overview is not None:
        sheet.paste(overview, ((width - overview.width) // 2, 0))
    d = ImageDraw.Draw(sheet)
    for i, (o, pair) in enumerate(cells):
        x, y = (i % cols) * (cw + 12), top + (i // cols) * (tile + 22)
        d.rectangle([x, y, x + cw - 1, y + 20], fill=COLORS[i % len(COLORS)])
        d.text((x + 4, y + 4), f"{o['id']}  {o['label']}", fill=(255, 255, 255))
        sheet.paste(pair, (x, y + 22))
    return sheet


def to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


ABOVE = 0.5     # extra height kept above an object in close-ups (steam rises out of the box)
MIN_SIDE = 0.22  # a close-up covers at least this share of the photo, so thin objects (a spoon) keep context


def crop_box(bbox, scale, size, margin=0.15, above=ABOVE, min_side=MIN_SIDE) -> tuple:
    """Box in ORIGINAL-photo pixels around an object given in prepared-image pixels: a margin, extra room
    above (steam), and never thinner than min_side of the photo (run 4: the spoon's close-up was a 30-px
    strip, read as "black-handled spoon, bowl not visible")."""
    sx, sy = scale
    W, H = size
    l, t, r, b = bbox
    mx, my, up = (r - l) * margin, (b - t) * margin, (b - t) * above
    x0, y0 = max(0.0, (l - mx) * sx), max(0.0, (t - up) * sy)      # margins are simply cut at the edge
    x1, y1 = min(float(W), (r + mx) * sx), min(float(H), (b + my) * sy)
    need = min(min_side * min(W, H), W, H)

    def widen(a0, a1, limit):                                         # only a too-small box is widened,
        if a1 - a0 >= need:                                           # and it stays inside the photo
            return a0, a1
        c = (a0 + a1) / 2
        a0, a1 = c - need / 2, c + need / 2
        if a0 < 0:
            a0, a1 = 0.0, need
        if a1 > limit:
            a0, a1 = limit - need, float(limit)
        return a0, a1
    x0, x1 = widen(x0, x1, W)
    y0, y1 = widen(y0, y1, H)
    return int(x0), int(y0), int(x1), int(y1)


def contact_sheet(img: Image.Image, objects: list, tile: int = 320, cols: int = 4,
                  margin: float = 0.15, full: Image.Image | None = None) -> Image.Image:
    """One image the model can read: the scene with numbered masks on top, and below it a close-up
    crop of every object labelled with its id. objects: [{'id','label','mask','bbox'}].
    Close-ups are cut from `full` (the original photo) when given: more real detail per object."""
    src = full if full is not None else img
    scale = (src.width / img.width, src.height / img.height)
    top = overlay(img, objects, tint=0.12, outline=3)   # light tint + edge: the model sees true colours
    top = top.resize((int(top.width * min(1.0, 1024 / top.width)), int(top.height * min(1.0, 1024 / top.width))))
    rows = (len(objects) + cols - 1) // cols
    sheet = Image.new("RGB", (max(top.width, cols * tile), top.height + rows * (tile + 22)), (255, 255, 255))
    sheet.paste(top, (0, 0))
    d = ImageDraw.Draw(sheet)
    for i, o in enumerate(objects):
        box = crop_box(o["bbox"], scale, src.size, margin)  # room above for steam; thin objects get context
        crop = src.crop(box)
        crop.thumbnail((tile, tile))
        x = (i % cols) * tile
        y = top.height + (i // cols) * (tile + 22)
        d.rectangle([x, y, x + tile - 1, y + 20], fill=COLORS[i % len(COLORS)])
        d.text((x + 4, y + 4), f"{o['id']}  {o['label']}", fill=(255, 255, 255))
        sheet.paste(crop, (x + (tile - crop.width) // 2, y + 22))
    return sheet
