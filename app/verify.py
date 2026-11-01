"""Pipeline step 4: an independent check of every piece of evidence.

The checker sees the close-ups and a list of plain claims ("o2: the mug is half full"), but NOT the
story they support, so it cannot be talked into agreeing. Evidence it rejects is removed; an event
left with no confirmed evidence is moved to 'rejected', so the UI can show what the verifier caught."""
from app.schemas import VerifyReport

VERIFY_PROMPT = """You are checking facts about a photo, independently. The image shows a scene with numbered
objects on top and a close-up of each object below. For EACH claim, look only at the image and answer:
"yes" if the image clearly shows it, "no" if the image clearly contradicts it, "unclear" if you cannot tell.
Be strict: a claim about the past ("was poured", "someone ate") is "unclear" unless the image shows it.

Claims:
{claims}

Return ONLY JSON: {{"checks": [{{"claim_id": "c1", "verdict": "yes", "note": "short reason"}}]}}"""

UNCLEAR_PENALTY = 0.75   # each unclear piece of evidence lowers confidence


def build_claims(events: list, labels: dict) -> list:
    claims = []
    for e in events:
        for i, ev in enumerate(e["evidence"]):
            claims.append({"claim_id": f"c{len(claims) + 1}", "event_id": e["id"], "index": i,
                           "text": f"{ev['object_id']} ({labels.get(ev['object_id'], '?')}): {ev['observation']}"})
    return claims


def apply_checks(events: list, claims: list, verdicts: dict) -> tuple:
    """Returns (kept_events, rejected_events). verdicts: {claim_id: (verdict, note)}."""
    by_event = {}
    for c in claims:
        by_event.setdefault(c["event_id"], []).append(c)
    kept, rejected = [], []
    for e in events:
        results = []
        for c in by_event.get(e["id"], []):
            verdict, note = verdicts.get(c["claim_id"], ("unclear", "not checked"))
            results.append({**e["evidence"][c["index"]], "verdict": verdict, "note": note})
        confirmed = [r for r in results if r["verdict"] == "yes"]
        unclear = [r for r in results if r["verdict"] == "unclear"]
        e = dict(e, evidence=results)
        if not confirmed and not unclear:
            rejected.append(dict(e, status="rejected", reason="every piece of evidence was contradicted"))
            continue
        e["confidence"] = round(e["confidence"] * (UNCLEAR_PENALTY ** len(unclear)), 3)
        e["status"] = "verified" if confirmed and not unclear else ("partly_verified" if confirmed else "unverified")
        kept.append(e)
    return kept, rejected


async def verify(models, sheet_jpeg: bytes, events: list, labels: dict) -> dict:
    claims = build_claims(events, labels)
    if not claims:
        return {"events": [], "rejected": [], "claims": 0}
    text = "\n".join(f"- {c['claim_id']}: {c['text']}" for c in claims)
    report = await models.spark_json(VERIFY_PROMPT.format(claims=text), VerifyReport, jpeg=sheet_jpeg, tag="verify")
    verdicts = {c.claim_id: (c.verdict, c.note) for c in report.checks}
    kept, rejected = apply_checks(events, claims, verdicts)
    return {"events": kept, "rejected": rejected, "claims": len(claims)}
