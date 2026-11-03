"""Object removal for the rewind with OpenCV's inpainting (a classic algorithm, not an AI model).

Hackathon rule: every AI model called at runtime must be a Meta model, so no learned inpainter is used.
SAM 3.1's masks decide what to remove and the timeline decides when; this only fills the hole from the
surrounding pixels. The fill is rough on large areas, so the rewind keeps a faint ghost of what was
removed (honest: "not here yet") instead of pretending to show the exact scene behind it.

- the mask is grown a little, and grown further downward, so shadows and reflections on a glossy counter
  go with the object instead of floating in mid-air;
- other objects' pixels are protected and never filled."""
import numpy as np
from PIL import Image, ImageFilter

GROW = 9                       # px of mask growth all round
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


def erase(img: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None, grow_down: bool = True) -> tuple:
    """Remove the object: (image, method). `keep`: pixels of other objects, which are never filled."""
    import cv2
    hole = removal_mask(mask, grow_down)
    if keep is not None:
        hole &= ~keep
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), "opencv"
