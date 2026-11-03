"""The single-call baseline: same output shape as a full run, and its guessed times are not dated."""
import asyncio

from app.baseline import BaselineTimeline, run_baseline
from app.score import score_run

KEY = {"video": "v.mp4", "video_seconds": 120, "photo_delay_min": [0, 1], "events": [
    {"id": "pour", "what": "milk poured", "seconds": [10, 20], "match": r"\bpour", "witness": []},
    {"id": "drink", "what": "some drunk", "seconds": [60, 70], "match": r"\b(drank|drunk|sip\w*)\b", "witness": []}]}


class FakeModels:
    def __init__(self, data):
        self.data, self.prompts = data, []

    async def spark_json(self, prompt, schema, jpeg=None, tag="llm"):
        self.prompts.append((prompt, tag))
        return schema.model_validate(self.data)


def test_baseline_result_scores_like_a_run():
    m = FakeModels({"summary": "milk", "events": [
        {"description": "Milk was most likely poured into the glass", "minutes_before_min": 0, "minutes_before_max": 20},
        {"description": "Some was drunk", "minutes_before_min": 10, "minutes_before_max": 5}]})
    res = asyncio.run(run_baseline(m, b"jpeg"))
    assert [e["id"] for e in res["events"]] == ["e1", "e2"]
    assert res["events"][1]["minutes_before_min"] == 5           # a reversed range is put the right way round
    assert m.prompts[0][1] == "baseline" and "level_dropped" in m.prompts[0][0]
    s = score_run(res, KEY)
    assert (s["found"], s["dated"], s["order_agree"], s["order_pairs"]) == (2, 0, 1, 1)


def test_empty_answer_is_allowed():
    assert BaselineTimeline.model_validate({}).events == []
