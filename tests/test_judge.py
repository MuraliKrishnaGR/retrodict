"""The blind judge: claims are anonymous, one claim may cover several real events, and the score uses it."""
import asyncio

from app.judge import Judgement, judge, judge_prompt
from app.score import score_run

KEY = {"video": "v.mp4", "video_seconds": 300, "photo_delay_min": [1, 1], "events": [
    {"id": "banana", "what": "banana sliced", "seconds": [10, 40], "match": "zzz", "witness": []},
    {"id": "apple", "what": "apple cut", "seconds": [50, 90], "match": "zzz", "witness": []},
    {"id": "bowl", "what": "fruit put into the bowl", "seconds": [200, 230], "match": "zzz", "witness": []}]}
RES = {"events": [
    {"id": "e1", "description": "Banana and apple were chopped", "minutes_before_min": 0, "minutes_before_max": 9},
    {"id": "e2", "description": "Fruit went into the bowl", "minutes_before_min": 0, "minutes_before_max": 9},
    {"id": "e3", "description": "The cup was knocked over", "minutes_before_min": 0, "minutes_before_max": 9},
    {"id": "e4", "description": "Rejected", "status": "rejected", "minutes_before_min": 0, "minutes_before_max": 9}]}


class FakeModels:
    def __init__(self):
        self.prompt = ""

    async def spark_json(self, prompt, schema, jpeg=None, tag="llm"):
        self.prompt = prompt
        return Judgement.model_validate({"claims": [
            {"claim": "A", "covers": ["banana", "apple"], "note": "both named"},
            {"claim": "B", "covers": ["bowl", "made_up_id"]},
            {"claim": "C", "covers": [], "wrong": True, "note": "the cup stood upright"}]})


def test_prompt_is_blind_and_skips_rejected_events():
    prompt, labels = judge_prompt(KEY, [e for e in RES["events"] if e.get("status") != "rejected"])
    assert labels == {"A": "e1", "B": "e2", "C": "e3"}
    assert "Retrodict" not in prompt and "baseline" not in prompt.lower()


def test_judged_score_lets_one_claim_cover_several_events():
    m = FakeModels()
    j = asyncio.run(judge(m, KEY, RES))
    assert j["covers"] == {"e1": ["banana", "apple"], "e2": ["bowl"], "e3": []} and j["wrong"] == ["e3"]
    s = score_run(RES, KEY, j)
    assert (s["found"], s["total"], s["matched_by"]) == (3, 3, "blind judge")
    assert s["order_pairs"] == 2 and s["order_agree"] == 2           # banana/apple share e1: not an order pair
    assert [f["event"] for f in s["false_claims"]] == ["e3"]
    assert [r["keys"] for r in s["retro"]] == [["banana", "apple"], ["bowl"], []]


def test_a_stale_judge_reading_is_never_applied(tmp_path):
    import json
    from app.score import load_judged
    j = asyncio.run(judge(FakeModels(), KEY, RES))
    (tmp_path / "judge.json").write_text(json.dumps(j))
    assert load_judged(tmp_path, RES) is not None
    rerun = {"events": [dict(RES["events"][0], description="Something else happened")]}
    assert load_judged(tmp_path, rerun) is None
