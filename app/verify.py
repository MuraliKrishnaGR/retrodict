"""Pipeline step 4: an independent check of every piece of evidence.

The checker sees the close-ups and a list of plain claims ("o2: the mug is half full"), but NOT the
story they support, so it cannot be talked into agreeing. Evidence it rejects is removed; an event
left with no confirmed evidence is moved to 'rejected', so the UI can show what the verifier caught.

Clock gate: a trace clock may date an event only if the checker also confirms, by itself, that the trace
is really there (run 2 on the milk photo: small bubbles were read as a skin, and the skin's clock then
moved a 3-minute-old drink to "10 to 90 min"). An unconfirmed trace loses its clock."""
from app.clocks import LOOKS
from app.schemas import VerifyReport

VERIFY_PROMPT = """You are checking facts about a photo, independently. On top, the whole scene with numbered
objects (use it for positions). Below, sharp close-ups of the objects named in the claims, each labelled with
its id, in three copies: as photographed, shadows lifted (faint steam, film; glare looks brighter there too),
and glare dimmed (what reflections hide). For EACH claim, look only at the image and answer:
"yes" if the image clearly shows it, "no" if the image clearly contradicts it, "unclear" if you cannot tell,
"partly" if its MAIN fact is shown but a side detail is wrong (say which in the note).
Be strict: a claim about the past ("was poured", "someone ate") is "unclear" unless the image shows it.
If a claim joins several facts, judge its MAIN visible fact: "no" only when the image contradicts that main
fact (e.g. "a bowl full of thick contents" when the bowl is empty but smeared); a wrong side detail (which
edge a handle points to) is "partly", never "no". Something you cannot MAKE OUT at this size (small drips,
crumbs, splashes, a faint film) is "unclear": not seeing a small thing is not the same as seeing it absent.

Claims:
{claims}

Return ONLY JSON: {{"checks": [{{"claim_id": "c1", "verdict": "yes", "note": "short reason"}}]}}"""

UNCLEAR_PENALTY = 0.75   # each unclear piece of evidence lowers confidence
CONTRADICTED_PENALTY = 0.5   # each contradicted piece halves it, and the event is marked "contested"


def build_claims(events: list, labels: dict) -> list:
    claims = []
    for e in events:
        for i, ev in enumerate(e["evidence"]):
            claims.append({"claim_id": f"c{len(claims) + 1}", "event_id": e["id"], "index": i,
                           "text": f"{ev['object_id']} ({labels.get(ev['object_id'], '?')}): {ev['observation']}"})
        if e.get("clock", "no_clock") != "no_clock" and e.get("trace") in LOOKS:
            oid = e["evidence"][0]["object_id"]
            claims.append({"claim_id": f"c{len(claims) + 1}", "event_id": e["id"], "index": None, "kind": "trace",
                           "text": f"{oid} ({labels.get(oid, '?')}): shows {LOOKS[e['trace']]}"})
    return claims


def _drop_clock(e: dict, verdict: str, note: str) -> dict:
    """The trace was not confirmed: its clock may not date the event. Back to 'no minimum time'."""
    hi = max(e.get("model_range", [0, e["minutes_before_max"]])[1], e["minutes_before_max"])
    return dict(e, minutes_before_min=0.0, minutes_before_max=hi, clock="clock_not_applied", clock_rule=None,
                trace_check={"trace": e.get("trace"), "verdict": verdict, "note": note})


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
            if c.get("kind") == "trace":
                if verdict == "yes":
                    e = dict(e, trace_check={"trace": e.get("trace"), "verdict": verdict, "note": note})
                else:
                    e = _drop_clock(e, verdict, note)
                continue
            results.append({**e["evidence"][c["index"]], "verdict": verdict, "note": note})
        confirmed = [r for r in results if r["verdict"] in ("yes", "partly")]
        unclear = [r for r in results if r["verdict"] in ("unclear", "partly")]   # a wrong detail costs like doubt
        e = dict(e, evidence=results)
        if not confirmed and not unclear:
            rejected.append(dict(e, status="rejected", reason="every piece of evidence was contradicted"))
            continue
        contradicted = [r for r in results if r["verdict"] == "no"]
        e["confidence"] = round(e["confidence"] * (UNCLEAR_PENALTY ** len(unclear))
                                * (CONTRADICTED_PENALTY ** len(contradicted)), 3)
        if contradicted:                # run 5: "flour fills the bowl" was contradicted, yet "someone mixed
            e["status"] = "contested"   # flour" stayed in the story because another fact held
        else:
            e["status"] = "verified" if confirmed and not unclear else ("partly_verified" if confirmed else "unverified")
        kept.append(e)
    return kept, rejected


async def verify(models, image_jpeg: bytes, events: list, labels: dict) -> dict:
    claims = build_claims(events, labels)
    if not claims:
        return {"events": [], "rejected": [], "claims": 0}
    text = "\n".join(f"- {c['claim_id']}: {c['text']}" for c in claims)
    report = await models.spark_json(VERIFY_PROMPT.format(claims=text), VerifyReport, jpeg=image_jpeg, tag="verify")
    verdicts = {c.claim_id: (c.verdict, c.note) for c in report.checks}
    kept, rejected = apply_checks(events, claims, verdicts)
    return {"events": kept, "rejected": rejected, "claims": len(claims)}
