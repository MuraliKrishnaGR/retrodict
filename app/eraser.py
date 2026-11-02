"""Realistic object removal for the rewind, with LaMa (open source, Apache-2.0, runs on a CPU in seconds).

When an event says an object ARRIVED (the glass was put down), the frame before it should show the counter
and wall where it stood, not a grey smear. LaMa fills a masked area from the rest of the photo; it is a
helper tool, not a source of facts: SAM decides what to remove and the timeline decides when.

Best-quality settings for a CPU:
- only a crop around the object goes through the model (2.2x its box, at most 1024 px), so detail is kept;
- the mask is grown a little, and grown further downward, so shadows and reflections on a glossy counter
  go with the object instead of floating in mid-air;
- the result is pasted back with a soft edge.
The model (~200 MB) downloads once into .cache/models. If torch or the download is unavailable, the
rewind falls back to OpenCV's simple fill, so a clip is always made."""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app import config

MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"   # a wrong download is refused
MAX_SIDE = 1024
CONTEXT = 2.2                  # crop = this many times the object's box
GROW = 9                       # px of mask growth all round
SHADOW = 0.5                   # extra growth downward, as a share of the object's height: on a glossy counter
                               # the reflection reaches about half the object's height (tested 0.18/0.35/0.5)

_model = None


def model_path() -> Path:
    return config.CACHE_DIR / "models" / "big-lama.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    """The LaMa model, downloaded once and checked; None if it cannot be used here."""
    global _model
    if _model is not None:
        return _model
    try:
        import torch
    except ImportError:
        return None
    path = model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print("Downloading the LaMa eraser model (~200 MB, once)...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    if MODEL_SHA256 and _sha256(path) != MODEL_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError("The LaMa model file did not match its checksum and was deleted. Run again.")
    _model = torch.jit.load(str(path), map_location="cpu").eval()
    return _model


def removal_mask(mask: np.ndarray, grow_down: bool = True) -> np.ndarray:
    """Grow the object's mask a little all round, and further downward (shadow, reflection). The contents
    of a cup are not grown downward: their 'reflection' is the cup itself."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * GROW + 1))
    grown = np.array(m) > 127
    ys, xs = np.nonzero(mask)
    if len(ys) and grow_down:
        drop = int((ys.max() - ys.min()) * SHADOW)
        shifted = np.zeros_like(grown)
        for d in range(1, drop + 1, max(1, drop // 6)):
            shifted[d:] |= grown[:-d]
        grown |= shifted
    return grown


def _crop_box(mask: np.ndarray):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    bh, bw = (ys.max() - ys.min() + 1) * CONTEXT, (xs.max() - xs.min() + 1) * CONTEXT
    side_h, side_w = max(bh, 64), max(bw, 64)
    y0, x0 = int(max(0, cy - side_h / 2)), int(max(0, cx - side_w / 2))
    y1, x1 = int(min(h, cy + side_h / 2)), int(min(w, cx + side_w / 2))
    return x0, y0, x1, y1


def lama_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Run LaMa on a crop around `hole` (bool HxW) and return the whole image with the hole filled."""
    import torch
    model = load()
    x0, y0, x1, y1 = _crop_box(hole)
    crop, m = img[y0:y1, x0:x1], hole[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, MAX_SIDE / max(ch, cw))
    if scale < 1:
        size = (max(8, int(cw * scale)), max(8, int(ch * scale)))
        crop_s = np.array(Image.fromarray(crop).resize(size, Image.LANCZOS))
        m_s = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    else:
        crop_s, m_s = crop, m
    h, w = m_s.shape
    H, W = -(-h // 8) * 8, -(-w // 8) * 8                       # the model wants sides divisible by 8
    im = np.pad(crop_s.astype(np.float32) / 255, ((0, H - h), (0, W - w), (0, 0)), mode="symmetric")
    mk = np.pad(m_s.astype(np.float32), ((0, H - h), (0, W - w)), mode="symmetric")
    with torch.inference_mode():
        out = model(torch.from_numpy(im.transpose(2, 0, 1))[None], torch.from_numpy(mk)[None, None])
    res = np.clip(out[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:h, :w]
    if scale < 1:
        res = np.array(Image.fromarray(res).resize((cw, ch), Image.LANCZOS))
    soft = np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255
    merged = crop * (1 - soft) + res * soft
    full = img.copy()
    full[y0:y1, x0:x1] = merged.astype(np.uint8)
    return full


def erase(img: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None, grow_down: bool = True) -> tuple:
    """Remove the object: (image, method). LaMa when available, else OpenCV's fill.
    `keep`: pixels of other objects, which the grown mask must never cover."""
    hole = removal_mask(mask, grow_down)
    if keep is not None:
        hole &= ~keep
    try:
        if load() is not None:
            return lama_fill(img, hole), "lama"
    except Exception as e:                       # never lose the clip over the eraser
        print(f"LaMa eraser unavailable ({type(e).__name__}: {str(e)[:120]}); using a simple fill.")
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), "opencv"
