"""The rewind clip: the REAL photo, walked backward through the reconstructed events.

No image generator is involved: every frame is the photo itself, edited only in ways we can defend.
  remove     -> the object fades out over its own surroundings (inpainted), leaving a faint outline
  add_steam  -> soft rising steam is drawn above the drink, and labelled as an illustration
  refill     -> the object is highlighted with "some was eaten or drunk"; the missing part is NOT drawn
  move, mark -> the object is highlighted and the event is captioned; nothing is drawn
The clip says on screen that it is a best guess from the photo, not proof.

Output in the run folder: rewind.mp4 (H.264, plays in browsers) and rewind_0.jpg ... rewind_N.jpg,
one still per state (0 = the photo, N = the earliest scene), which the web page uses for its slider."""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.eraser import erase, removal_mask
from app.reason import is_dated

FPS = 24
INTRO, PER_EVENT, OUTRO = 2.2, 2.4, 2.0      # seconds: long enough to read each caption
PER_UNDRAWN = 1.6                             # a step that cannot be drawn changes nothing to watch: shorter
BAR_H = 104                                   # caption band at the bottom: three lines

# What the scene was like BEFORE the event: the picture goes backward, so the caption must too
# (2 Nov: captions that described the event while the picture undid it confused the viewer).
BEFORE = {"remove": "Before: the {o} was not here yet",
          "move": "Before: the {o} was somewhere else",
          "empty": "Before: the {o} was empty",
          "refill": "Before: more was in the {o}",
          "mark": "Before: the {o} looked different",
          "add_steam": "Before: the drink in the {o} was steaming"}
TAG = {"remove": "erased", "move": "erased here", "empty": "drink erased", "refill": "can't be drawn",
       "mark": "can't be drawn", "add_steam": "steam drawn (illustration)"}
DRAWN = {"remove", "move", "empty", "add_steam"}
GHOST = 0.72        # a removed object keeps 28% of itself: honest ("not here yet") and hides inpaint smear
ACCENT = (255, 196, 0)


def font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def when_text(e: dict) -> str:
    """Times come only from clues in the photo; without one the model's guess is not shown as a time."""
    lo, hi = e["minutes_before_min"], e["minutes_before_max"]

    def fmt(m):
        return f"{m:g} min" if m < 120 else f"{m / 60:.1f} h"
    if not is_dated(e):
        return f"at least {fmt(lo)} before" if lo > 0 else "time unknown"
    if lo == 0:
        return f"up to {fmt(hi)} before"
    return f"{lo:g} to {hi:g} min before" if hi < 120 else f"{fmt(lo)} to {fmt(hi)} before"


def short(text: str, limit: int = 70) -> str:
    """Captions must be read in one second: keep the main clause of a long description."""
    text = text.strip().rstrip(".")
    for sep in ("; ", ", "):
        if len(text) > limit and sep in text:
            text = text.split(sep)[0]
    return text


CONSUMED = re.compile(r"\b(ate|eaten|eat|eats|bite|bitten|drank|drunk|drink|sipped|sip|finished|used up)\b")
ARRIVED = re.compile(r"\b(put|puts|placed|place|poured|pour|set|left|leaves|brought|served|dropped|added|"
                     r"made|prepared|filled|moved|carried|took out)\b")


POURED_INTO = re.compile(r"\b(poured|pour|pouring|filled|fill|made|prepared|served|added)\b.*\b(into|in)\b")
CONTAINER = re.compile(r"\b(mug|glass|cup|tumbler|bowl|plate|beaker)\b")
LEFT_BEHIND = re.compile(r"\b(leaving|dropping|dropped|spilling|spilled|scattering|scattered)\b((?:\W+\w+){1,4})")
LEFTOVER_MAX = 0.02            # only small things (crumbs, drips) count as left behind: never the jar a spoon
                               # was "left resting on" (run 9)


def ease(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def rewind_glitch(img: np.ndarray, strength: float, t: float) -> np.ndarray:
    """Tape-rewind look while a step is undone: a few bands slide sideways and a bright tracking line
    rolls up the picture. Strength 0..1."""
    if strength <= 0.01:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(int(t * 24))
    for _ in range(3):
        y = int(rng.integers(0, h - 12))
        band = int(rng.integers(4, 12))
        out[y:y + band] = np.roll(out[y:y + band], int(rng.integers(-18, 18) * strength), axis=1)
    y = int((1 - (t * 1.7) % 1) * h)
    line = slice(max(0, y - 2), min(h, y + 2))
    out[line] = np.clip(out[line].astype(np.float32) * (1 - 0.5 * strength) + 255 * 0.5 * strength, 0, 255).astype(np.uint8)
    return out


def undo_kind(e: dict) -> str:
    """The rewind must show the OPPOSITE of the event. A model's undo can contradict its own description
    (run 4: 'someone ate part of the brownie' with undo 'remove', which made the brownie vanish, but the
    brownie was there before; the opposite of eating is a whole brownie). Code keeps them consistent."""
    kind, text = e.get("undo", "mark"), e.get("description", "").lower()
    if CONSUMED.search(text) and kind == "remove":
        return "refill"                          # it was there before, only fuller
    if kind == "remove" and not ARRIVED.search(text):
        return "mark"                            # nothing says the object arrived in this event
    if kind in ("refill", "mark", "remove") and POURED_INTO.search(text) and CONTAINER.search(text):
        return "empty"                           # the drink was poured INTO the cup: before it, the cup was empty
    return kind


def contents_mask(img: np.ndarray, container: np.ndarray) -> np.ndarray:
    """What is inside a cup or glass seen from above: the surface around the middle of the opening (about
    a sixth of the way down the cup), grown over pixels of the same colour. Not the bright rim or the
    shiny inner wall (run 9: 'far from the cup's colour' also took the steel rim and the handle)."""
    import cv2
    ys, xs = np.nonzero(container)
    if not len(ys):
        return np.zeros_like(container)
    top, h = ys.min(), ys.max() - ys.min()
    row = top + int(h * 0.16)
    cols = np.nonzero(container[row])[0]
    if not len(cols):
        return np.zeros_like(container)
    cy, cx = row, int(cols.mean())
    r = max(2, h // 40)
    seed = np.median(img[cy - r:cy + r + 1, cx - r:cx + r + 1].reshape(-1, 3).astype(np.float32), axis=0)
    opening = container.copy()
    opening[top + int(h * 0.40):] = False
    near = opening & (np.abs(img.astype(np.float32) - seed).max(-1) < 38)
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(near)
    if n <= 1 or lab[cy, cx] == 0:
        return np.zeros_like(container)
    out = lab == lab[cy, cx]
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)       # a bright highlight on the drink is drink too
    return cv2.dilate(filled, np.ones((3, 3), np.uint8)).astype(bool) & container


def outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    grown = np.array(m.filter(ImageFilter.MaxFilter(2 * width + 1))) > 127
    return grown & ~mask


def steam_layer(mask: np.ndarray, t: float) -> np.ndarray:
    """Alpha (0..1) of soft wisps rising from the top of the object; t in seconds animates them."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    top, left, right = ys.min(), xs.min(), xs.max()
    cx, span = (left + right) / 2, (right - left) * 0.28
    height = int(min(top, (right - left) * 0.9))
    alpha = np.zeros((h, w), np.float32)
    if height < 10:
        return alpha
    yy, xx = np.mgrid[top - height:top, 0:w].astype(np.float32)
    rise = (top - yy) / height                                    # 0 at the rim, 1 at the top
    for k, phase in enumerate((0.0, 2.1, 4.2)):
        x0 = cx + (k - 1) * span * 0.6 + np.sin(rise * 5 + t * 2.2 + phase) * span * 0.35 * (0.3 + rise)
        wisp = np.exp(-((xx - x0) ** 2) / (2 * (6 + 14 * rise) ** 2))
        alpha[top - height:top] += wisp * (1 - rise) ** 1.5 * 0.42 * (0.6 + 0.4 * np.sin(t * 3 + phase + rise * 4))
    return np.clip(alpha, 0, 0.6)


def _label(draw, xy, text, f, fill=(255, 255, 255), bg=(0, 0, 0, 170)):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=f)
    draw.rounded_rectangle([l - 6, t - 4, r + 6, b + 4], radius=6, fill=bg)
    draw.text((x, y), text, font=f, fill=fill)


def _wrap(draw, text, f, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = w
    lines.append(line)
    if len(lines) > 2:
        lines = lines[:2]
        while draw.textlength(lines[1] + "…", font=f) > width and lines[1]:
            lines[1] = lines[1].rsplit(" ", 1)[0] if " " in lines[1] else lines[1][:-1]
        lines[1] += "…"
    return lines


class Rewinder:
    """Builds the states and frames from a finished run folder."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.res = json.loads((self.dir / "result.json").read_text(encoding="utf-8"))
        self.photo = np.array(Image.open(self.dir / "input.jpg").convert("RGB"))
        h, w = self.photo.shape[:2]
        self.h, self.w = h - h % 2, w - w % 2                     # H.264 needs even sizes
        self.photo = self.photo[:self.h, :self.w]
        self.masks = {}
        for o in self.res["objects"]:
            p = self.dir / o["mask_png"]
            if p.exists():
                self.masks[o["id"]] = (np.array(Image.open(p)) > 127)[:self.h, :self.w]
        self.labels = {o["id"]: o["label"] for o in self.res["objects"]}
        self.steps = self._steps()
        self.f_big, self.f_small = font(max(24, self.w // 38)), font(max(16, self.w // 60))

    def _steps(self) -> list:
        """Latest event first (the photo is 'now'); each step knows its object and how to undo it."""
        steps = []
        for e in reversed(self.res.get("events", [])):
            oid = self._object_for(e)
            if oid is None:
                continue
            kind = undo_kind(e)
            if kind == "add_steam" and not self.masks[oid].any():
                kind = "mark"
            erase_ids = [oid] if kind in ("remove", "move") else []
            erase_ids += self._left_behind(e, oid)          # crumbs "left" by eating came with this event
            step = {"event": e, "oid": oid, "kind": kind, "erase": erase_ids}
            steps.append(step)
        return steps

    def _left_behind(self, e: dict, oid: str) -> list:
        """Cited objects the event says it left behind ('...leaving brown crumbs near the mug')."""
        m = LEFT_BEHIND.search(e.get("description", "").lower())
        if not m:
            return []
        after = set(re.findall(r"[a-z]{3,}", m.group(2)))
        out = []
        for ev in e["evidence"]:
            cid = ev["object_id"]
            if cid == oid or cid not in self.masks or cid in out:
                continue
            small = self.masks[cid].sum() < LEFTOVER_MAX * self.masks[cid].size
            if small and set(re.findall(r"[a-z]{3,}", self.labels.get(cid, ""))) & after:
                out.append(cid)
        return out

    def _object_for(self, e: dict):
        """The object the rewind changes: the model's undo_object if real; otherwise the cited object whose
        name shares the most words with the event (run 4: 'ate part of the brownie' cited the crumbs by
        the mug first, and the MUG faded out)."""
        if e.get("undo_object") in self.masks:
            return e["undo_object"]
        cited = [ev["object_id"] for ev in e["evidence"] if ev["object_id"] in self.masks]
        if not cited:
            return None
        words = re.findall(r"[a-z]+", e["description"].lower())

        def first_mention(oid):          # the event's main object is named first ("ate the brownie ... mug")
            name = self.res.get("states", {}).get(oid, {}).get("what") or self.labels.get(oid, "")
            names = set(re.findall(r"[a-z]{3,}", name.lower())) | set(re.findall(r"[a-z]{3,}", self.labels.get(oid, "")))
            return next((i for i, w in enumerate(words) if w in names), len(words))
        return min(cited, key=lambda oid: (first_mention(oid), cited.index(oid)))

    def _before_text(self, step: dict) -> str:
        name = re.sub(r"\s*\d+$", "", self.labels.get(step["oid"], "object")) or "object"   # "peel 1" -> "peel"
        text = BEFORE.get(step["kind"], BEFORE["mark"]).format(o=name)
        left = [self.labels.get(c, "") for c in step["erase"] if c != step["oid"]]
        if left:
            text += f", and no {' or '.join(left)}"
        if step["kind"] not in DRAWN and not left:
            text += " (can't be drawn)"
        return text

    def states(self) -> list:
        """Image of the scene after undoing 0, 1, ..., N events (steam is drawn at a fixed moment)."""
        cur = self.photo.copy()
        out = [cur.copy()]
        for s in self.steps:
            cur = self._apply(cur, s, 1.0, t=0.0)
            out.append(cur.copy())
        return out

    def _apply(self, base: np.ndarray, step: dict, a: float, t: float) -> np.ndarray:
        """Undo one event on `base` with strength a (0..1)."""
        m = self.masks[step["oid"]]
        img = base.astype(np.float32)
        if step["erase"] or step["kind"] == "empty":
            if "fill" not in step:
                self._prepare_fill(base, step)
            if step["hole"].any():
                # a clean texture fill needs no ghost; over a rough smooth fill a faint ghost stays (honest)
                ghost = 1.0 if step["kind"] == "empty" or step["eraser"] == "patchmatch" else GHOST
                soft = np.array(Image.fromarray(step["hole"].astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(3)))
                soft = soft.astype(np.float32) / 255.0
                if step["kind"] == "empty" and 0 < a < 1:
                    # the drink drains: its surface sinks from the top of the cup's opening to the bottom
                    ys = np.nonzero(step["hole"])[0]
                    level = ys.min() + (ys.max() - ys.min() + 1) * ease(a)
                    rows = np.arange(soft.shape[0], dtype=np.float32)[:, None]
                    soft = soft * np.clip((level - rows) / 4 + 0.5, 0, 1)
                    k = soft[..., None] * ghost
                else:
                    flying = step["kind"] in ("remove", "move") and 0 < a < 1
                    k = soft[..., None] * (min(1.0, a * 4) if flying else a) * ghost
                img = img * (1 - k) + step["fill"] * k
            if step["kind"] in ("remove", "move") and 0 < a < 1 and step["oid"] in step["erase"]:
                img = self._fly(img, base, m, a)            # the real object, cut out by SAM, carried away
            if step["kind"] in ("remove", "move") and step["eraser"] != "patchmatch":
                edge = outline(m, 2)
                img[edge] = img[edge] * (1 - 0.45 * a) + np.array([255, 255, 255]) * 0.45 * a   # faint ghost line
        elif step["kind"] == "add_steam":
            alpha = steam_layer(m, t)[..., None] * a
            img = img * (1 - alpha) + 245 * alpha
        return np.clip(img, 0, 255).astype(np.uint8)

    def _fly(self, img: np.ndarray, base: np.ndarray, m: np.ndarray, a: float) -> np.ndarray:
        """Lift the object's own pixels (SAM's cut-out from the photo) and carry them out of the nearest
        edge of the picture, with a soft shadow: the object leaving, played backward. Nothing is drawn
        that is not in the photo."""
        ys, xs = np.nonzero(m)
        if not len(ys):
            return img
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = m.shape
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        edges = {(-1, 0): cx, (1, 0): w - cx, (0, -1): cy, (0, 1): h - cy}
        (vx, vy), near = min(edges.items(), key=lambda kv: kv[1])
        e = ease(a)
        dist = near + max(x1 - x0, y1 - y0)
        dx, dy = vx * dist * e, vy * dist * e - 18 * np.sin(np.pi * e)       # a small lift as it goes
        scale = 1 + 0.12 * np.sin(np.pi * min(1.0, e * 1.5))
        sprite = Image.fromarray(base[y0:y1, x0:x1].astype(np.uint8))
        alpha = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
        sw, sh = max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))
        sprite, alpha = sprite.resize((sw, sh), Image.LANCZOS), alpha.resize((sw, sh), Image.LANCZOS)
        out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        px, py = int(cx + dx - sw / 2), int(cy + dy - sh / 2)
        shade = alpha.filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.4 * min(1.0, e * 6)))
        out.paste(Image.new("RGB", (sw, sh), (0, 0, 0)), (px + 6, py + 14), shade)
        out.paste(sprite, (px, py), alpha)
        return np.asarray(out).astype(np.float32)

    def _prepare_fill(self, base: np.ndarray, step: dict):
        """Erase, in one pass, everything this event brought: the object itself (remove, move), the drink
        poured into a cup (empty) and what it left behind (crumbs). Other objects are protected."""
        hole = np.zeros_like(self.masks[step["oid"]])
        for oid in step["erase"]:
            hole |= self.masks[oid]
        if step["kind"] == "empty":
            hole |= contents_mask(base, self.masks[step["oid"]])
        keep = np.zeros_like(hole)
        for oid, om in self.masks.items():
            if oid not in step["erase"] and not (step["kind"] == "empty" and oid == step["oid"]):
                keep |= om
        if step["kind"] == "empty":                  # the cup's inside may be refilled, but never its rim
            cup = self.masks[step["oid"]]
            inner = np.array(Image.fromarray(cup.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(13))) > 127
            keep = (keep & ~cup) | (cup & ~inner)
        if not hole.any():
            step.update(fill=base.astype(np.float32), hole=hole, eraser="none")
            return
        filled, step["eraser"] = erase(base, hole, keep, grow_down=step["kind"] != "empty",
                                       smooth=step["kind"] == "empty" and not step["erase"])
        step["fill"] = filled.astype(np.float32)
        step["hole"] = removal_mask(hole, grow_down=step["kind"] != "empty") & ~keep

    def _decorate(self, img: np.ndarray, step, k: int, pulse: float, back: float) -> Image.Image:
        """Time cue, highlight, caption band and the honesty line."""
        f = img.astype(np.float32)
        grey = f.mean(axis=2, keepdims=True)
        f = f * (1 - 0.35 * back) + grey * 0.35 * back                # older -> less colour
        f[..., 2] += 10 * back                                         # and a little cooler
        if step is not None and pulse > 0 and step["kind"] != "remove":
            edge = outline(self.masks[step["oid"]], 3)
            f[edge] = f[edge] * (1 - pulse) + np.array(ACCENT) * pulse
        frame = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        d.rectangle([0, self.h - BAR_H, self.w, self.h], fill=(0, 0, 0, 185))
        n = len(self.steps)
        if step is None and k == 0:
            top, main, sub = "NOW", "This is the photo.", "Going back one step at a time, latest first."
        elif step is None:
            top, main, sub = "AS FAR BACK AS THE CLUES GO", "The earliest moment the photo points to.", ""
        else:
            e = step["event"]
            top = f"STEP BACK {k} OF {n}" + (f"  ·  {when_text(e).upper()}" if is_dated(e) else "")
            main = self._before_text(step)
            sub = "Because: " + short(e["description"], 90)[:1].lower() + short(e["description"], 90)[1:]
            tag = TAG.get(step["kind"])
            if tag:
                m = self.masks[step["oid"]]
                ys, xs = np.nonzero(m)
                y = int(ys.min()) - 30
                y = y if y >= 44 else max(44, int(ys.min()) + 10)   # keep clear of the top badges and scrubber
                tw = d.textlength(tag, font=self.f_small)
                x = int(min(max(6, int(xs.min())), self.w - tw - 14))   # never off the right edge
                _label(d, (x, y), tag, self.f_small, fill=ACCENT)
        y0 = self.h - BAR_H
        d.text((16, y0 + 10), top, font=self.f_small, fill=ACCENT)
        d.text((16, y0 + 32), _wrap(d, main, self.f_big, self.w - 32)[0], font=self.f_big, fill=(255, 255, 255))
        if sub:
            d.text((16, y0 + 70), _wrap(d, sub, self.f_small, self.w - 32)[0], font=self.f_small, fill=(200, 200, 200))
        _label(d, (38, 10), "REWIND  -  best guess from the photo, not proof", self.f_small)
        for x0 in (8, 17):                       # a drawn rewind sign: fonts on some computers lack the glyph
            d.polygon([(x0 + 9, 12), (x0 + 9, 26), (x0, 19)], fill=ACCENT)
        n = max(1, len(self.steps))
        x0, x1, y = self.w - 220, self.w - 14, 22                   # scrubber: now on the right
        d.line([x0, y, x1, y], fill=(255, 255, 255, 160), width=2)
        for i in range(n + 1):
            x = x1 - (x1 - x0) * i / n
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255, 200))
        x = x1 - (x1 - x0) * min(k, n) / n
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT)
        return Image.alpha_composite(frame, over).convert("RGB")

    def frames(self):
        """Intro on the photo, then one event at a time (highlight, then undo), then a hold on the
        earliest scene. Steam, once added, keeps rising in every earlier moment."""
        n = len(self.steps)
        cur, steaming = self.photo.copy(), []

        def with_steam(img, t):
            for st in steaming:
                img = self._apply(img, st, 1.0, t)
            return img

        for i in range(int(INTRO * FPS)):
            yield self._decorate(cur, None, 0, 0, 0)
        for k, step in enumerate(self.steps):
            drawn = step["kind"] in DRAWN or len(step["erase"]) > 0
            frames = int((PER_EVENT if drawn else PER_UNDRAWN) * FPS)
            for i in range(frames):
                p = i / (frames - 1)
                a = min(1.0, max(0.0, (p - 0.25) / 0.5))              # highlight first, then undo
                a = a * a * (3 - 2 * a)
                t = INTRO + k * PER_EVENT + i / FPS
                img = with_steam(self._apply(cur, step, a, t), t)
                pulse = 0.9 * (1 - abs(2 * p - 1) ** 2) if step["kind"] != "remove" else 0
                img = rewind_glitch(img, 0.8 * np.sin(np.pi * a), t)
                yield self._decorate(img, step, k + 1, pulse, (k + a) / max(1, n))
            if step["kind"] == "add_steam":
                steaming.append(step)
            else:
                cur = self._apply(cur, step, 1.0, t=0.0)
        for i in range(int(OUTRO * FPS)):
            t = INTRO + n * PER_EVENT + i / FPS
            yield self._decorate(with_steam(cur, t), None, n + 1, 0, 1.0 if n else 0)

    def write(self) -> dict:
        import imageio.v2 as imageio
        out = self.dir / "rewind.mp4"
        with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=2,
                                pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"]) as wr:
            count = 0
            for fr in self.frames():
                wr.append_data(np.asarray(fr))
                count += 1
        stills = []
        for i, s in enumerate(self.states()):
            p = self.dir / f"rewind_{i}.jpg"
            Image.fromarray(s).save(p, quality=90)
            stills.append(p.name)
        return {"video": out.name, "frames": count, "seconds": round(count / FPS, 2), "stills": stills,
                "steps": [{"event_id": s["event"]["id"], "object_id": s["oid"], "kind": s["kind"]}
                          for s in self.steps]}


def make_rewind(run_dir) -> dict:
    return Rewinder(run_dir).write()
