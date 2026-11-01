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


def when(e) -> str:
    lo, hi = e["minutes_before_min"], e["minutes_before_max"]
    def fmt(m):
        return f"{m:g} min" if m < 120 else f"{m / 60:.1f} h"
    return f"{fmt(lo)} to {fmt(hi)} before"


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
        print(f"  {s['step']:<9} {s['seconds']:>6}s  {s['detail']}")
    print("\nObjects:")
    for oid, s in res["states"].items():
        traces = f"  [traces: {', '.join(s['traces'])}]" if s["traces"] else ""
        print(f"  {oid}: {s['what']}{traces}")
    print(f"\nSummary: {res['summary']}")
    print("\nTimeline (earliest first):")
    if not res["events"]:
        print("  (no events)")
    for e in res["events"]:
        clock = f", {e['clock'].replace('_', ' ')} ({e['trace']})" if e["clock"] != "no_clock" else ""
        print(f"  {e['id']}  {when(e)}  [{e['status']}, confidence {e['confidence']:.2f}{clock}]")
        print(f"      {e['description']}")
        for ev in e["evidence"]:
            print(f"      - {ev['object_id']}: {ev['observation']}  -> {ev['verdict']}")
        for alt in e.get("alternatives", []):
            print(f"      or: {alt}")
    for e in res["rejected"]:
        print(f"  REJECTED by verifier: {e['description']} ({e['reason']})")
    issues = res.get("quality_issues", [])
    print("\nQuality check: " + ("OK" if not issues else f"{len(issues)} PROBLEM(S)"))
    for i in issues:
        print(f"  ! {i}")
    print(f"\nSaved: {out_dir / 'result.json'} and {out_dir / 'sheet.jpg'}")
    print(f"Time: {time.time() - t:.0f}s | paid LLM calls: {models.llm.ledger.calls} | cache hits: {models.cache_hits}")
    print(f"Estimated spend this run: ${models.llm.ledger.run_spent:.4f} | all runs: ${models.llm.ledger.total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
