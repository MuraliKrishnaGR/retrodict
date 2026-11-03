"""A blind judge for scoring: which real events does each claimed event describe?

Word patterns miss paraphrases ("bottle was opened" vs "opened the bottle") and catch false friends
("mixed fruit" is not "mixing"). The judge is Muse Spark Open, shown only the real events (from the answer
key) and a list of claims labelled A, B, C... It is never told which system wrote the claims, so Retrodict
and the one-call baseline are judged by the same rules. Its reading is saved next to the run so a person
can check every match."""
from pydantic import BaseModel

JUDGE_PROMPT = """You are scoring a reconstruction of what happened in a scene, against what really happened.

REAL EVENTS (from a video of the scene):
{real}

CLAIMS made by a system that saw only a photo taken afterwards:
{claims}

For each claim, list the ids of the real events it correctly describes. Rules:
- A claim covers a real event only if it states that action on that object (or a clear synonym:
  "chopped" = "cut", "mandarin" or "citrus" = "orange", "lid removed" = "bottle opened").
- One claim may cover several real events ("banana, apple and orange were chopped" covers all three cuttings).
- A vague claim ("fruit was cut") covers a specific real event only if that fruit is named or the claim
  clearly includes every fruit ("all the fruit").
- Leftovers or states ("knife left on the board") cover nothing unless they state one of the real actions.
- Mark a claim as wrong if it states something that contradicts the real events or did not happen.
Return JSON only:
{{"claims": [{{"claim": "A", "covers": ["event_id", ...], "wrong": false, "note": "one short reason"}}]}}"""


class ClaimJudgement(BaseModel):
    claim: str
    covers: list[str] = []
    wrong: bool = False
    note: str = ""


class Judgement(BaseModel):
    claims: list[ClaimJudgement] = []


def letters(n: int) -> list:
    return [chr(ord("A") + i) if i < 26 else f"Z{i}" for i in range(n)]


def judge_prompt(key: dict, events: list) -> tuple:
    real = "\n".join(f"- {k['id']}: {k['what']}" for k in key["events"])
    labs = letters(len(events))
    claims = "\n".join(f"- {lab}: {e['description']}" for lab, e in zip(labs, events))
    return JUDGE_PROMPT.format(real=real, claims=claims), dict(zip(labs, [e["id"] for e in events]))


async def judge(models, key: dict, result: dict) -> dict:
    events = [e for e in result.get("events", []) if e.get("status") != "rejected"]
    prompt, label_to_id = judge_prompt(key, events)
    j = await models.spark_json(prompt, Judgement, tag="judge")
    valid = {k["id"] for k in key["events"]}
    covers, wrong, notes = {}, [], {}
    for c in j.claims:
        eid = label_to_id.get(c.claim.strip().upper())
        if not eid:
            continue
        covers[eid] = [k for k in c.covers if k in valid]
        if c.wrong:
            wrong.append(eid)
        notes[eid] = c.note
    return {"judge": "Muse Spark Open, blind (claims labelled A, B, C; system not named)",
            "claims_text": [e["description"] for e in events],
            "covers": covers, "wrong": wrong, "notes": notes}
