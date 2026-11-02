"""Watch a real video with V-JEPA 2 (Meta, open weights) and list the hand actions it sees, with times.

This is for checking Retrodict, not part of it: Retrodict only ever gets the photo. The video is the
recording of what really happened before that photo, so a Meta video model watching it gives an
independent "what happened" to compare Retrodict's timeline with, next to the human notes.

The model is V-JEPA 2 fine-tuned on Something-Something v2: 174 everyday hand actions written with
"something" for the objects ("Pouring something into something", "Putting something on a surface").
It looks at 16 frames at a time, so the video is walked in short overlapping windows; neighbouring
windows with the same top action are merged into one segment. Times are given from the start and,
like the notes, counted back from the end of the video (the photo is taken right after it ends)."""
import os

import numpy as np
from PIL import Image

MODEL_ID = os.getenv("VJEPA_MODEL_ID", "facebook/vjepa2-vitl-fpc16-256-ssv2")
CROP = 256
SHORT = int(CROP * 256 / 224)              # the model's own preprocessing: short side 292, centre crop 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SAMPLE_FPS = 8.0                           # frames kept per second of video (enough for 16 per window)


def read_frames(path, fps: float = SAMPLE_FPS, short: int = SHORT):
    """Frames at about `fps`, resized so the short side is `short`: (list of HxWx3 uint8, times in s, duration)."""
    import imageio.v2 as iio
    reader = iio.get_reader(str(path), "ffmpeg")
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps") or 30.0)
    step = max(1, round(src_fps / fps))
    frames, times, n = [], [], 0
    for n, frame in enumerate(reader):
        if n % step:
            continue
        im = Image.fromarray(frame)
        s = short / min(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        frames.append(np.asarray(im))
        times.append(n / src_fps)
    reader.close()
    duration = float(meta.get("duration") or (n + 1) / src_fps)
    return frames, times, duration


def clip_array(frames, crop: int = CROP) -> np.ndarray:
    """Centre-crop and normalise frames into the model's input: (T, 3, crop, crop) float32."""
    out = []
    for f in frames:
        h, w = f.shape[:2]
        y, x = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
        c = f[y:y + crop, x:x + crop].astype(np.float32) / 255
        if c.shape[:2] != (crop, crop):                     # tiny video: pad rather than fail
            c = np.pad(c, ((0, crop - c.shape[0]), (0, crop - c.shape[1]), (0, 0)), mode="edge")
        out.append(((c - MEAN) / STD).transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


def windows(duration: float, window: float, stride: float):
    """(start, end) seconds of each window; the last one ends at the end of the video."""
    if duration <= window:
        return [(0.0, duration)]
    starts = list(np.arange(0.0, duration - window, stride)) + [duration - window]
    return [(round(float(s), 2), round(float(s) + window, 2)) for s in starts]


def pick(times, start: float, end: float, n: int):
    """Indices of n frames spread evenly over [start, end]."""
    idx = [i for i, t in enumerate(times) if start <= t <= end] or [min(range(len(times)), key=lambda i: abs(times[i] - start))]
    return [idx[round(k * (len(idx) - 1) / max(1, n - 1))] for k in range(n)]


def load(model_id: str = MODEL_ID):
    from transformers import VJEPA2ForVideoClassification
    model = VJEPA2ForVideoClassification.from_pretrained(model_id, token=os.getenv("HF_TOKEN") or None)
    return model.eval()


def label_windows(model, frames, times, duration, window: float = 3.0, stride: float = 1.5, top: int = 3,
                  progress=None):
    """Top actions for each window: [{start, end, actions: [(label, prob), ...]}]."""
    import torch
    n = int(model.config.frames_per_clip)
    labels = model.config.id2label
    rows, spans = [], windows(duration, window, stride)
    for k, (s, e) in enumerate(spans):
        x = torch.from_numpy(clip_array([frames[i] for i in pick(times, s, e, n)]))[None]
        with torch.inference_mode():
            p = torch.softmax(model(pixel_values_videos=x).logits[0].float(), -1)
        vals, ids = p.topk(min(top, p.numel()))
        rows.append({"start": s, "end": e,
                     "actions": [(labels[int(i)], round(float(v), 3)) for v, i in zip(vals, ids)]})
        if progress:
            progress(k + 1, len(spans), rows[-1])
    return rows


def segments(rows, duration: float, min_prob: float = 0.2):
    """Merge neighbouring windows whose top action is the same; drop unsure ones (top below min_prob)."""
    segs = []
    for r in rows:
        label, prob = r["actions"][0]
        if prob < min_prob:
            continue
        if segs and segs[-1]["action"] == label and r["start"] <= segs[-1]["end"]:
            segs[-1]["end"] = r["end"]
            segs[-1]["best"] = max(segs[-1]["best"], prob)
            segs[-1]["windows"] += 1
        else:
            segs.append({"action": label, "start": r["start"], "end": r["end"], "best": prob, "windows": 1})
    for s in segs:
        s["before_end"] = [round(duration - s["end"], 1), round(duration - s["start"], 1)]
    return segs
