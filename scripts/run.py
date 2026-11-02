"""Run the full Retrodict pipeline on one photo and print the reconstructed timeline.
Usage: python scripts/run.py path\\to\\photo.jpg
Re-running the same photo is free and instant: every model result is cached."""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                  # noqa: E402
from app.clients.models import Models   # noqa: E402
from app.cost import BudgetExceeded     # noqa: E402
from app.errors import explain          # noqa: E402
from app.pipeline import run_photo      # noqa: E402
from app.rewind import make_rewind, when_text  # noqa: E402


# The verifier checks the visible facts each event cites, not the story itself, so say exactly that.
STATUS_TEXT = {"verified": "evidence confirmed", "partly_verified": "evidence partly confirmed",
               "unverified": "evidence not confirmed", "contested": "evidence partly CONTRADICTED"}


def when(e) -> str:
    return when_text(e)


async def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run.py path/to/photo.jpg")
    photo = Path(sys.argv[1])
    if not photo.exists():
        raise SystemExit(f"Photo not found: {photo}")
    out_dir = config.OUT_DIR / photo.stem

    t = time.time()
    models = Models()
    try:
        res = await run_photo(models, photo, out_dir)
    except BudgetExceeded as e:
        raise SystemExit(f"Stopped by spend guard: {e}")
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {str(e)[:400]}\nfix: {explain(e)}")
    finally:
        await models.aclose()

    print("\nSteps:")
    for s in res["trace"]:
        print(f"  {s['step']:<11} {s['seconds']:>6}s  {s['detail']}")
    if res.get("moves"):
        print("\nInvestigation (agent moves):")
        for i, m in enumerate(res["moves"], 1):
            arg = m.get("object_id") or m.get("phrase") or m.get("trace") or ""
            print(f"  {i}. {m['move']} {arg}".rstrip() + f"  -> {m['result']}")
            if m.get("thought"):
                print(f"       why: {m['thought']}")
    print("\nObjects:")
    for oid, s in res["states"].items():
        traces = f"  [traces: {', '.join(s['traces'])}]" if s["traces"] else ""
        print(f"  {oid}: {s['what']}{traces}")
    if res.get("photo_note"):
        print(f"\nNote: {res['photo_note']}")
    print(f"\nSummary: {res['summary']}")
    print("\nTimeline (earliest first):")
    if not res["events"]:
        print("  (no events)")
    for e in res["events"]:
        clock = f", {e['clock'].replace('_', ' ')} ({e['trace']})" if e["clock"] != "no_clock" else ""
        status = STATUS_TEXT.get(e["status"], e["status"])
        print(f"  {e['id']}  {when(e)}  [{status}, confidence {e['confidence']:.2f}{clock}]")
        print(f"      {e['description']}")
        if e.get("clock_rule"):
            print(f"      why this time: {e['clock_rule']}")
        if e.get("bounded_by"):
            print(f"      time also limited because it {e['bounded_by']}")
        tc = e.get("trace_check")
        if tc:
            note = f" ({tc['note']})" if tc.get("note") else ""
            print(f"      trace check: {str(tc['trace']).replace('_', ' ')} -> {tc['verdict']}{note}")
        for ev in e["evidence"]:
            print(f"      - {ev['object_id']}: {ev['observation']}  -> {ev['verdict']}")
        print(f"      rewind: {e.get('undo', 'mark')}")
        for alt in e.get("alternatives", []):
            print(f"      or: {alt}")
    for e in res["rejected"]:
        print(f"  REJECTED by verifier: {e['description']} ({e['reason']})")
    issues = res.get("quality_issues", [])
    print("\nQuality check: " + ("OK" if not issues else f"{len(issues)} PROBLEM(S)"))
    for i in issues:
        print(f"  ! {i}")
    print(f"\nSaved: {out_dir / 'result.json'} and {out_dir / 'sheet.jpg'}")
    if res["events"]:
        try:
            clip = make_rewind(out_dir)          # free: no model calls, only the real photo is edited
            print(f"Rewind: {out_dir / clip['video']} ({clip['seconds']}s, {len(clip['steps'])} events undone)")
        except Exception as e:                   # the clip is a bonus; never lose the result over it
            print(f"Rewind clip skipped: {type(e).__name__}: {str(e)[:200]}")
    print(f"Time: {time.time() - t:.0f}s | paid LLM calls: {models.llm.ledger.calls} | cache hits: {models.cache_hits}")
    print(f"Estimated spend this run: ${models.llm.ledger.run_spent:.4f} | all runs: ${models.llm.ledger.total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
