"""V-JEPA 2 video check: frame reading, preprocessing, windows and merging, with a tiny random model (offline)."""
import numpy as np
import pytest

from app import watch


def _video(path, seconds=4, fps=10, size=(320, 240)):
    import imageio.v2 as iio
    w = iio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=8)
    for i in range(seconds * fps):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        f[:, (i * 7) % size[0]:] = 200
        w.append_data(f)
    w.close()


def test_read_and_clip(tmp_path):
    p = tmp_path / "v.mp4"
    _video(p)
    frames, times, duration = watch.read_frames(p, fps=5)
    assert 3.5 < duration < 4.5 and 18 <= len(frames) <= 21
    assert min(frames[0].shape[:2]) == watch.SHORT
    x = watch.clip_array(frames[:16])
    assert x.shape == (16, 3, watch.CROP, watch.CROP) and x.dtype == np.float32


def test_windows_cover_the_end():
    w = watch.windows(10.0, 3.0, 1.5)
    assert w[0] == (0.0, 3.0) and w[-1] == (7.0, 10.0)
    assert watch.windows(2.0, 3.0, 1.5) == [(0.0, 2.0)]


def test_pick_spreads_frames():
    times = [i * 0.125 for i in range(80)]
    idx = watch.pick(times, 1.0, 4.0, 16)
    assert len(idx) == 16 and times[idx[0]] == 1.0 and times[idx[-1]] == 4.0 and idx == sorted(idx)


def test_segments_merge_and_hide_unsure():
    rows = [{"start": 0, "end": 3, "actions": [("Pouring something into something", 0.6)]},
            {"start": 1.5, "end": 4.5, "actions": [("Pouring something into something", 0.7)]},
            {"start": 3, "end": 6, "actions": [("Holding something", 0.1)]},
            {"start": 4.5, "end": 7.5, "actions": [("Putting something on a surface", 0.5)]}]
    s = watch.segments(rows, duration=10, min_prob=0.2)
    assert [x["action"] for x in s] == ["Pouring something into something", "Putting something on a surface"]
    assert s[0]["end"] == 4.5 and s[0]["best"] == 0.7 and s[0]["before_end"] == [5.5, 10]


def test_tiny_model_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.VJEPA2Config(crop_size=watch.CROP, frames_per_clip=16, hidden_size=32, num_attention_heads=2,
                          num_hidden_layers=1, num_pooler_layers=1, pred_hidden_size=32,
                          pred_num_attention_heads=2, pred_num_hidden_layers=1,
                          id2label={0: "Pouring something into something", 1: "Holding something"},
                          label2id={"Pouring something into something": 0, "Holding something": 1})
    torch.manual_seed(0)
    model = tf.VJEPA2ForVideoClassification(cfg).eval()
    p = tmp_path / "v.mp4"
    _video(p, seconds=5)
    frames, times, duration = watch.read_frames(p)
    rows = watch.label_windows(model, frames, times, duration, window=3, stride=1.5)
    assert len(rows) == 3 and all(len(r["actions"]) == 2 for r in rows)
    assert abs(sum(p for _, p in rows[0]["actions"]) - 1) < 1e-3
