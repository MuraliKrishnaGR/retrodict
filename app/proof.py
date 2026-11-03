"""Proof rewind: what Retrodict read from ONE photo, side by side with the real video of that moment,
played backward.

Left: the photo. For each real event (latest first) the picture zooms into the object Retrodict cited as
evidence, outlined by SAM 3.1, with Retrodict's own sentence. Right: the real video of that event, run
backward. A badge says whether Retrodict found it. Nothing is generated: both sides are real footage.
The video is used only here, to show the result; Retrodict itself never sees it."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.rewind import ACCENT, ease, font, outline, short

W, H = 1280, 720
PANEL = (616, 400)                 # each side
TOP = 110
FPS = 15
INTRO, STEP, OUTRO = 3.0, 4.0, 4.0
GREEN, RED, GREY = (46, 160, 90), (200, 60, 50), (150, 150, 150)


def _fit(img: Image.Image, box=PANEL) -> Image.Image:
    """Letterbox into the panel."""
    im = img.copy()
    im.thumbnail(box, Image.LANCZOS)
    out = Image.new("RGB", box, (12, 12, 12))
    out.paste(im, ((box[0] - im.width) // 2, (box[1] - im.height) // 2))
    return out


def _zoom(photo: Image.Image, mask: np.ndarray | None, z: float) -> Image.Image:
    """The photo, zoomed by z (0 = whole photo, 1 = close on the mask's box), the mask outlined."""
    pw, ph = photo.size
    img = np.asarray(photo).copy()
    if mask is None or not mask.any():
        return _fit(photo)
    edge = outline(mask, 3)
    img[edge] = ACCENT
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bw = max(xs.max() - xs.min(), (ys.max() - ys.min()) * PANEL[0] / PANEL[1]) * 1.9
    bw = min(pw, max(bw, pw * 0.25))
    bh = bw * PANEL[1] / PANEL[0]
    w, h = pw + (bw - pw) * z, ph + (bh - ph) * z
    if h > ph:
        h, w = ph, ph * PANEL[0] / PANEL[1]
    x0 = min(max(0, cx - w / 2), pw - w) if z > 0 else 0
    y0 = min(max(0, cy - h / 2), ph - h) if z > 0 else 0
    crop = Image.fromarray(img).crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
    return _fit(crop)


def _text(d, xy, text, f, fill=(255, 255, 255), width=None):
    if width:
        while d.textlength(text, font=f) > width and len(text) > 4:
            text = text[:-2].rstrip() + "…" if not text.endswith("…") else text[:-2].rstrip() + "…"
    d.text(xy, text, font=f, fill=fill)


def _badge(d, xy, text, color, f):
    tw = d.textlength(text, font=f)
    x, y = xy
    d.rounded_rectangle([x, y, x + tw + 20, y + f.size + 12], radius=8, fill=color)
    d.text((x + 10, y + 5), text, font=f, fill=(255, 255, 255))


class ProofRewind:
    def __init__(self, run_dir, key_path, video_path):
        self.dir = Path(run_dir)
        self.res = json.loads((self.dir / "result.json").read_text(encoding="utf-8"))
        self.key = json.loads(Path(key_path).read_text(encoding="utf-8"))
        from app.score import load_judged, score_run
        judged = load_judged(self.dir, self.res)
        self.score = score_run(self.res, self.key, judged)      # the blind judge's matches when available
        self.video = Path(video_path)
        self.photo = Image.open(self.dir / "input.jpg").convert("RGB")
        self.masks = {}
        for o in self.res.get("objects", []):
            p = self.dir / o["mask_png"]
            if p.exists():
                m = np.array(Image.open(p).convert("L")) > 127
                self.masks[o["id"]] = m[:self.photo.height, :self.photo.width]
        self.events = {e["id"]: e for e in self.res.get("events", [])}
        self.f_big, self.f_mid, self.f_small = font(30), font(22), font(17)

    def align_to(self, frame: Image.Image) -> int:
        """Turn the photo (and its masks) so it faces the same way as the video: the phone was often held
        the other way round for the photo. Picks 0 or 180 degrees by which looks more like the last frame."""
        def small(im):
            a = np.asarray(im.convert("L").resize((64, 48)), np.float32)
            return (a - a.mean()) / (a.std() + 1e-6)
        target = small(frame)
        best = max((0, 180), key=lambda r: float((small(self.photo.rotate(r)) * target).mean()))
        if best:
            self.photo = self.photo.rotate(180)
            self.masks = {k: m[::-1, ::-1].copy() for k, m in self.masks.items()}
        return best

    def steps(self) -> list:
        """Real events, latest first, each with Retrodict's matching event (or None)."""
        rows = {r["id"]: r for r in self.score["events"]}
        out = []
        for k in reversed(self.key["events"]):
            r = rows.get(k["id"], {})
            e = self.events.get(r.get("event")) if r.get("found") else None
            mask = None
            if e:
                ids = [ev["object_id"] for ev in e.get("evidence", []) if ev["object_id"] in self.masks]
                big = 0.15 * self.photo.width * self.photo.height     # a tray or table is the stage, not the clue
                small = [i for i in ids if self.masks[i].sum() < big]
                ids = small or ids[:1]
                if ids:
                    mask = np.zeros_like(self.masks[ids[0]])
                    for i in ids:
                        mask |= self.masks[i]
            out.append({"key": k, "event": e, "mask": mask})
        return out

    def _video_frames(self, wanted: dict) -> dict:
        """{frame index: PIL image} for the indices wanted, read in one pass."""
        import imageio.v2 as iio
        r = iio.get_reader(str(self.video), "ffmpeg")
        fps = float(r.get_meta_data().get("fps") or 30)
        need = {int(round(t * fps)): None for t in wanted}
        last = max(need)
        got = {}
        for i, fr in enumerate(r):
            if i in need:
                im = Image.fromarray(fr)
                im.thumbnail((PANEL[0] * 2, PANEL[1] * 2))
                got[i] = im
            if i >= last:
                break
        r.close()
        keys = sorted(got)
        return {t: got.get(int(round(t * fps))) or (got[keys[-1]] if keys else Image.new("RGB", PANEL))
                for t in wanted}

    def frames(self):
        steps = self.steps()
        dur = self.key["video_seconds"]
        # which video moments each part shows: each event played BACKWARD, from its end to its start
        plan = [("intro", None, [dur - 0.2] * int(INTRO * FPS))]
        for st in steps:
            s0, s1 = st["key"]["seconds"]
            n = int(STEP * FPS)
            plan.append(("step", st, [s1 - (s1 - s0) * i / (n - 1) for i in range(n)]))
        plan.append(("outro", None, [steps[-1]["key"]["seconds"][0] if steps else 0] * int(OUTRO * FPS)))
        times = {t for _, _, ts in plan for t in ts}
        vid = self._video_frames(times)
        if self.align_to(vid[dur - 0.2]):           # the photo was turned: rebuild the clue outlines on it
            fresh = {st["key"]["id"]: st["mask"] for st in self.steps()}
            for _, st, _ in plan:
                if st is not None:
                    st["mask"] = fresh[st["key"]["id"]]
        n_steps = len(steps)
        for k, (kind, st, ts) in enumerate(plan):
            for i, t in enumerate(ts):
                yield self._frame(kind, st, k, n_steps, i / max(1, len(ts) - 1), t, vid[t])

    def _frame(self, kind, st, k, n, p, t, vframe) -> Image.Image:
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        d = ImageDraw.Draw(canvas)
        lx, rx = 16, W - 16 - PANEL[0]
        d.text((lx, 18), "PROOF REWIND", font=self.f_big, fill=ACCENT)
        d.text((lx + 270, 26), "one photo  vs  the real video, played backward", font=self.f_mid, fill=(220, 220, 220))
        d.text((lx, TOP - 30), "WHAT RETRODICT READ FROM THE PHOTO", font=self.f_small, fill=(200, 200, 200))
        d.text((rx, TOP - 30), "WHAT REALLY HAPPENED (VIDEO, BACKWARD)", font=self.f_small, fill=(200, 200, 200))
        mm, ss = divmod(int(t), 60)
        if kind == "intro":
            left = _fit(self.photo)
            cap = "This photo is all Retrodict sees. The video is only used to check it."
            sub = ""
        elif kind == "outro":
            left = _fit(self.photo)
            s = self.score
            cap = f"From one photo, Retrodict found {s['found']} of {s['total']} real events."
            made = len(s.get("false_claims", [])) if s.get("matched_by") == "blind judge" else len(s.get("unsupported", []))
            sub = f"Claims that were wrong: {made}.  Best guess from the photo, not proof."
        else:
            found = st["event"] is not None
            z = ease(min(1.0, p / 0.45)) if found else 0
            left = _zoom(self.photo, st["mask"], z)
            if not found:
                left = Image.blend(left, Image.new("RGB", left.size, (0, 0, 0)), 0.45)
            cap = st["key"]["what"][:1].upper() + st["key"]["what"][1:]
            sub = ("Retrodict: " + short(st["event"]["description"], 95)) if found else "Retrodict missed this."
        canvas.paste(left, (lx, TOP))
        right = _fit(vframe)
        canvas.paste(right, (rx, TOP))
        d = ImageDraw.Draw(canvas)
        if kind == "step":
            found = st["event"] is not None
            _badge(d, (lx + 10, TOP + 10), "FOUND" if found else "MISSED", GREEN if found else RED, self.f_mid)
            _badge(d, (rx + 10, TOP + 10), f"      {mm}:{ss:02d}", (0, 0, 0), self.f_mid)
            for x0 in (rx + 20, rx + 32):             # drawn rewind sign: some fonts lack the glyph
                d.polygon([(x0 + 11, TOP + 18), (x0 + 11, TOP + 36), (x0, TOP + 27)], fill=(255, 255, 255))
            d.text((lx, TOP + PANEL[1] + 24), f"STEP BACK {k} OF {n}", font=self.f_small, fill=ACCENT)
        y = TOP + PANEL[1] + 50
        _text(d, (lx, y), cap, self.f_big, width=W - 32)
        if sub:
            _text(d, (lx, y + 46), sub, self.f_mid, fill=(200, 200, 200), width=W - 32)
        # progress: now on the right, the start of the video on the left
        bar_y = H - 22
        d.line([lx, bar_y, W - 16, bar_y], fill=(90, 90, 90), width=3)
        x = lx + (W - 32) * (t / max(1, self.key["video_seconds"]))
        d.ellipse([x - 7, bar_y - 7, x + 7, bar_y + 7], fill=ACCENT)
        return canvas

    def write(self, out_path=None) -> dict:
        import imageio.v2 as iio
        out = Path(out_path or self.dir / "proof_rewind.mp4")
        count = 0
        with iio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=2, pixelformat="yuv420p",
                            ffmpeg_params=["-preset", "veryfast"]) as wr:
            for fr in self.frames():
                wr.append_data(np.asarray(fr))
                count += 1
        return {"video": str(out), "seconds": round(count / FPS, 1), "frames": count}
