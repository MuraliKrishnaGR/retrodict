"""Offline tests: no key, no network, no model download, no cost."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.cache import DiskCache, make_key
from app.clients.llm_client import LLMClient, extract_json, strip_reasoning
from app.clients.sam_local import LocalSAM
from app.cost import BudgetExceeded, SpendLedger
from app.perceive import perceive
from app.schemas import ObjectList


def test_extract_json_handles_fences_and_reasoning():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<think>maybe {"x": 0}</think>{"a": 2}') == {"a": 2}
    assert strip_reasoning("<think>\nlong\n</think>\nanswer") == "answer"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    k = make_key("x", b"bytes", {"b": 2})
    assert c.get_json("llm", k) is None
    c.put_json("llm", k, {"text": "hi"})
    assert c.get_json("llm", k) == {"text": "hi"}
    assert make_key("x", b"bytes", {"b": 2}) == k and make_key("y") != k


def test_spend_guard_blocks(tmp_path):
    led = SpendLedger(tmp_path / "spend.json", max_total=0.10, max_run=0.05)
    led.check(0.03)
    led.add(0.03, "a")
    with pytest.raises(BudgetExceeded):
        led.check(0.03)                       # run would hit 0.06 > 0.05
    led2 = SpendLedger(tmp_path / "spend.json", max_total=0.04, max_run=1)
    assert abs(led2.total - 0.03) < 1e-9      # total persists across runs
    with pytest.raises(BudgetExceeded):
        led2.check(0.02)


class FakeOpenAI:
    """Mimics client.chat.completions.create; counts paid calls."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls += 1
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200))


def test_llm_json_caches_and_retries(tmp_path):
    fake = FakeOpenAI(["not json", '<think>hmm</think>{"objects":[{"name":"mug","why":"x"}]}'])
    led = SpendLedger(tmp_path / "s.json", 10, 10)
    llm = LLMClient(client=fake, cache=DiskCache(tmp_path / "c"), ledger=led)
    r = asyncio.run(llm.json("list", ObjectList, jpeg=b"img"))
    assert r.objects[0].name == "mug" and fake.calls == 2   # bad reply -> one retry
    r2 = asyncio.run(llm.json("list", ObjectList, jpeg=b"img"))
    assert r2.objects[0].name == "mug" and fake.calls == 2  # second time: cache, no new call
    assert led.calls == 2 and led.total > 0


class FakeSamProcessor:
    """Same interface as transformers Sam3Processor, but tiny."""
    def __call__(self, images=None, text=None, return_tensors=None):
        if images is not None:
            return {"pixel_values": np.zeros((1, 3, 8, 8))}
        return {"input_ids": np.array([[len(text)]]), "attention_mask": np.ones((1, 1))}

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        h, w = target_sizes[0]
        if outputs["concept_len"] != 3:                    # only "mug" is in the picture
            return [{"masks": np.zeros((0, h, w)), "boxes": np.zeros((0, 4)), "scores": np.zeros(0)}]
        m = np.zeros((2, h, w), dtype=np.int64)
        m[0, 5:10, 5:10] = 1
        m[1, 1:3, 1:3] = 1
        return [{"masks": m, "boxes": np.array([[5, 5, 9, 9], [1, 1, 2, 2]], float),
                 "scores": np.array([0.6, 0.9])}]


class FakeSamModel:
    def __init__(self):
        self.vision_calls = 0
        self.decode_calls = 0

    def get_vision_features(self, pixel_values):
        self.vision_calls += 1
        return "vision"

    def __call__(self, vision_embeds, input_ids, attention_mask):
        self.decode_calls += 1
        return {"concept_len": int(input_ids[0][0])}


def test_local_sam_reuses_vision_and_caches(tmp_path):
    img = Image.new("RGB", (20, 16), (10, 20, 30))
    import io
    buf = io.BytesIO(); img.save(buf, format="JPEG"); jpeg = buf.getvalue()
    model = FakeSamModel()
    sam = LocalSAM(cache=DiskCache(tmp_path), model=model, processor=FakeSamProcessor())
    items = sam.segment(jpeg, "mug")
    assert [it["score"] for it in items] == [0.9, 0.6]          # sorted by confidence
    assert items[0]["mask"].dtype == bool and items[0]["mask"].shape == (16, 20)
    assert sam.segment(jpeg, "plate") == []
    assert model.vision_calls == 1 and model.decode_calls == 2  # image encoded once
    again = sam.segment(jpeg, "mug")                            # from disk cache
    assert model.decode_calls == 2 and sam.cache_hits == 1
    assert np.array_equal(again[0]["mask"], items[0]["mask"])


def _box_mask(h, w, l, t, r, b):
    m = np.zeros((h, w), bool)
    m[t:b, l:r] = True
    return m


class ReplayModels:
    """Replays the real run on test1.jpeg: SAM matched every jar for both 'brown jar' and
    'ceramic jar' (scores copied from that run), found nothing for 'rusk bag', but finds 'bag'."""
    def __init__(self, listing):
        self.listing = listing
        self.asked = []

    async def spark_json(self, prompt, schema, jpeg=None, tag=""):
        return schema.model_validate({"objects": self.listing})

    async def sam_segment(self, jpeg, w, h, concept):
        self.asked.append(concept)
        brown, black, pot = (_box_mask(h, w, 49, 18, 62, 37), _box_mask(h, w, 32, 42, 49, 64),
                             _box_mask(h, w, 68, 16, 98, 47))
        table = {
            "brown jar":   [(brown, 0.8939), (black, 0.5634), (pot, 0.5238)],
            "ceramic jar": [(pot, 0.8802), (brown, 0.7945), (black, 0.7507)],
            "jar":         [(pot, 0.91), (brown, 0.88), (black, 0.86)],
            "bag":         [(_box_mask(h, w, 0, 18, 42, 55), 0.82)],
        }
        return [{"object_id": str(i), "bbox": [float(v) for v in (m.nonzero()[1].min(), m.nonzero()[0].min(),
                                                                      m.nonzero()[1].max(), m.nonzero()[0].max())],
                 "score": sc, "mask": m} for i, (m, sc) in enumerate(table.get(concept, []))]


def _run(tmp_path, listing):
    photo = tmp_path / "scene.jpg"
    Image.new("RGB", (100, 80), (200, 200, 200)).save(photo)
    models = ReplayModels(listing)
    return asyncio.run(perceive(models, photo, tmp_path / "out")), models


def test_perceive_no_double_counting_with_specific_names(tmp_path):
    # the exact listing from the real run: each jar must be claimed once, by its best-scoring name
    res, models = _run(tmp_path, [{"name": "rusk bag"}, {"name": "brown jar"},
                                  {"name": "black jar"}, {"name": "ceramic jar"}])
    where = {o["label"]: o["bbox"][:2] for o in res["objects"]}
    assert where == {"brown jar": [49.0, 18.0],    # each name lands on the right jar,
                     "ceramic jar": [68.0, 16.0],  # including the one found via a fallback word
                     "black jar": [32.0, 42.0],
                     "rusk bag": [0.0, 18.0]}
    assert "bag" in models.asked and "jar" in models.asked


def test_perceive_generic_names_with_counts(tmp_path):
    res, _ = _run(tmp_path, [{"name": "Jar", "count": 3, "desc": "brown, black, ceramic"},
                             {"name": "packet"}, {"name": "jar"}])
    labels = [o["label"] for o in res["objects"]]
    assert sorted(labels) == ["jar 1", "jar 2", "jar 3"]          # merged 'Jar'+'jar', no packet found
    assert res["concepts"][0] == {"name": "jar", "count": 4, "desc": "brown, black, ceramic"}
    saved = json.loads((tmp_path / "out" / "objects.json").read_text())
    assert saved["objects"][0]["mask_png"].startswith("masks/") and (tmp_path / "out" / "overlay.png").exists()


def test_perceive_keeps_confident_uncounted_object(tmp_path):
    # the second real run: Spark said "jar x2" and "pot x1"; SAM finds nothing for "pot",
    # but "jar" finds all three jars, the pot included, with high scores. Keep all three.
    res, _ = _run(tmp_path, [{"name": "packet"}, {"name": "jar", "count": 2}, {"name": "pot"}])
    labels = sorted(o["label"] for o in res["objects"])
    assert labels == ["jar 1", "jar 2", "jar 3"]
    assert res["not_found"] == ["packet", "pot"]


def test_low_confidence_extras_are_dropped(tmp_path):
    from app.perceive import assign_masks
    m1, m2 = _box_mask(10, 10, 0, 0, 5, 5), _box_mask(10, 10, 5, 5, 10, 10)
    kept = assign_masks([{"concept": "jar", "score": 0.9, "mask": m1, "bbox": [0, 0, 4, 4]},
                         {"concept": "jar", "score": 0.6, "mask": m2, "bbox": [5, 5, 9, 9]}],
                        {"jar": 1}, min_area=1)
    assert len(kept) == 1


def test_set_key_replaces_only_key_line():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("set_key", Path(__file__).parent.parent / "scripts" / "set_key.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.upsert_key("MODEL_API_KEY=abc\nLLM_API_KEY=old\nMAX_RUN_USD=1.00\n", "NEWKEY1234567890123456")
    assert out.splitlines() == ["LLM_API_KEY=NEWKEY1234567890123456", "MAX_RUN_USD=1.00"]


def test_error_hints():
    from app.errors import explain
    e = Exception("bad"); e.status_code = 401
    assert "use_host" in explain(e)
    e2 = Exception("model x not found"); e2.status_code = 404
    assert "find_model" in explain(e2)


def test_unexpected_error_saves_details_and_names_no_assistant(tmp_path, monkeypatch):
    from app import config
    from app.errors import explain
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    try:
        raise ValueError("y1 must be greater than or equal to y0")
    except ValueError as e:
        msg = explain(e)
    assert "claude" not in msg.lower() and "last_error.txt" in msg
    saved = (tmp_path / "last_error.txt").read_text(encoding="utf-8")
    assert "Traceback" in saved and "y1 must be greater" in saved


def test_event_carries_the_clock_rule_behind_its_range():
    from app.reason import ground_events
    from app.schemas import Timeline
    tl = Timeline.model_validate({"summary": "s", "events": [
        {"id": "x", "description": "tea was poured", "minutes_before_min": 0, "minutes_before_max": 60,
         "evidence": [{"object_id": "mug_1", "observation": "steam rising"}], "trace": "steam"},
        {"id": "y", "description": "packet was opened", "minutes_before_min": 0, "minutes_before_max": 60,
         "evidence": [{"object_id": "packet_1", "observation": "torn top"}], "trace": None}]})
    ev = {e["description"]: e for e in ground_events(tl, {"mug_1", "packet_1"})}
    assert "steam" in ev["tea was poured"]["clock_rule"]
    assert ev["tea was poured"]["minutes_before_max"] == 10
    assert ev["packet was opened"]["clock_rule"] is None


def test_box_from_sam_stays_inside_photo_for_object_at_top_edge():
    import numpy as np
    from app.clients.sam_local import clean_box
    m = np.zeros((100, 200), bool); m[0:30, 50:90] = True          # a jar touching the top edge
    assert clean_box([49.6, -0.4, 90.2, 30.1], m) == [50.0, 0.0, 90.0, 30.0]
    empty = np.zeros((100, 200), bool)                              # no mask: clamp SAM's box
    assert clean_box([-5, -3, 250, 40], empty) == [0.0, 0.0, 200.0, 40.0]
    assert clean_box([60, 50, 10, 20], empty) == [10.0, 20.0, 60.0, 50.0]


def test_overlay_draws_labels_for_objects_at_the_top_edge():
    import numpy as np
    from PIL import Image
    from app.imaging import overlay
    img = Image.new("RGB", (200, 100), (40, 40, 40))
    m = np.zeros((100, 200), bool); m[0:30, 50:90] = True
    out = overlay(img, [{"id": "o1", "label": "jar", "mask": m, "bbox": [50.0, 0.0, 90.0, 30.0]}],
                  tint=0.12, outline=3)
    assert out.size == (200, 100)


def test_sam_works_from_any_thread_without_gradients(tmp_path):
    """Run 6: the microscope search ran SAM in a new worker thread, where torch still tracked gradients,
    and crashed with "Can't call numpy() on Tensor that requires grad". Runs where torch is installed."""
    import threading
    torch = __import__("pytest").importorskip("torch")
    import io

    class GradModel(FakeSamModel):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def __call__(self, vision_embeds, input_ids, attention_mask):
            self.decode_calls += 1
            return {"scores": self.w * 0.9}               # requires grad unless grad is switched off

    class GradProcessor(FakeSamProcessor):
        def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
            h, w = target_sizes[0]
            s = float(outputs["scores"].numpy()[0])      # the call that crashed
            m = np.zeros((h, w), bool); m[2:6, 2:6] = True
            return [{"masks": [m], "boxes": [[2, 2, 6, 6]], "scores": [s]}]

    buf = io.BytesIO(); Image.new("RGB", (20, 16)).save(buf, format="JPEG")
    sam = LocalSAM(cache=DiskCache(tmp_path), model=GradModel(), processor=GradProcessor())
    out = {}
    t = threading.Thread(target=lambda: out.update(items=sam.segment(buf.getvalue(), "spoon")))
    t.start(); t.join()
    assert out["items"][0]["score"] == 0.9
