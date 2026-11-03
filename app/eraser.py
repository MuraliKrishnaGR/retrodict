"""Object removal for the rewind with classic algorithms, no AI model (hackathon rule: every AI model called
at runtime must be a Meta model).

- PatchMatch (Barnes et al. 2009, the algorithm behind "content-aware fill"; PyPatchMatch, MIT): fills the
  hole by copying matching patches from around it, so a desk or a wall continues with its own texture.
  Other objects are never used as a source (a spoon's hole is not filled with bits of the jar).
- OpenCV's Telea fill: a smooth fill, used for the inside of a cup (a smooth dark inside reads as empty;
  copied patches there looked like the cup's outside), and whenever PatchMatch is not installed.
SAM 3.1's masks decide what to remove and the timeline decides when; this only fills the hole.

- the mask is grown a little, and grown further downward, so shadows and reflections on a glossy counter
  go with the object instead of floating in mid-air;
- other objects' pixels are protected and never filled."""
import numpy as np
from PIL import Image, ImageFilter

GROW = 9                       # px of mask growth all round
PATCH = 5                      # PatchMatch patch radius (tested 3 and 7 on the milk photo: 5 is clean and fast)
CONTEXT = 2.5                  # PatchMatch works on a crop this many times the hole's box (speed)
SHADOW = 0.5                   # extra growth downward, as a share of the object's height: on a glossy counter
                               # the reflection reaches about half the object's height


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


def _crop(hole: np.ndarray):
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    hh = max(64, (ys.max() - ys.min() + 1) * CONTEXT) / 2
    hw = max(64, (xs.max() - xs.min() + 1) * CONTEXT) / 2
    return int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh))


def patchmatch_fill(img: np.ndarray, hole: np.ndarray, keep: np.ndarray | None):
    """PatchMatch on a crop around the hole; None if the library is not installed."""
    try:
        import patchmatch
    except ImportError:
        return None
    if not getattr(patchmatch, "patchmatch_available", False):
        return None
    x0, y0, x1, y1 = _crop(hole)
    crop, h = np.ascontiguousarray(img[y0:y1, x0:x1]), hole[y0:y1, x0:x1]
    no_source = None if keep is None else ((keep[y0:y1, x0:x1] & ~h).astype(np.uint8) * 255)
    out = patchmatch.inpaint(crop, h.astype(np.uint8) * 255, global_mask=no_source, patch_size=PATCH)
    full = img.copy()
    full[y0:y1, x0:x1] = out
    return full


def erase(img: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None, grow_down: bool = True,
          smooth: bool = False) -> tuple:
    """Remove the object: (image, method). `keep`: pixels of other objects, never filled and never copied.
    `smooth`: the inside of a cup (Telea's smooth fill)."""
    import cv2
    hole = removal_mask(mask, grow_down)
    if keep is not None:
        hole &= ~keep
    if not smooth and hole.any():
        try:
            filled = patchmatch_fill(img, hole, keep)
            if filled is not None:
                return filled, "patchmatch"
        except Exception as e:                       # never lose the clip over the fill
            print(f"PatchMatch fill failed ({type(e).__name__}); using a smooth fill.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), "opencv"
