"""Day 1: one photo -> object list (Muse Spark Open) -> masks (SAM 3.1, local).
Usage (from the project folder):  python scripts/day1_run.py path\\to\\photo.jpg
Re-running the same photo is free and instant: results come from .cache/."""
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
from app.perceive import perceive       # noqa: E402


async def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/day1_run.py path/to/photo.jpg")
    photo = Path(sys.argv[1])
    if not photo.exists():
        raise SystemExit(f"Photo not found: {photo}")
    out_dir = config.OUT_DIR / photo.stem

    t = time.time()
    models = Models()
    try:
        res = await perceive(models, photo, out_dir)
    except BudgetExceeded as e:
        raise SystemExit(f"Stopped by spend guard: {e}")
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {str(e)[:400]}\nfix: {explain(e)}")
    finally:
        await models.aclose()

    print("\nObject kinds from Muse Spark Open: "
          + ", ".join(f"{c['name']} x{c['count']}" for c in res["concepts"]))
    for s in res["search"]:
        if s["phrase"] != s["name"]:
            print(f"  (SAM found nothing for {s['name']!r}, retried as {s['phrase']!r}: {s['found']} found)")
    if res.get("not_found"):
        print(f"  (SAM found no separate match for: {', '.join(res['not_found'])})")
    print(f"Masks from SAM 3.1: {len(res['objects'])}")
    for o in res["objects"]:
        print(f"  {o['id']:>3}  {o['label']:<16} area={o['area']}")
    print(f"\nSaved: {out_dir / 'overlay.png'}")
    print(f"       {out_dir / 'objects.json'}")
    print(f"Time: {time.time() - t:.0f}s | paid LLM calls: {models.llm.ledger.calls} | "
          f"SAM computed: {models.sam.computed} | cache hits: {models.cache_hits}")
    print(f"Estimated spend this run: ${models.llm.ledger.run_spent:.4f} | all runs: ${models.llm.ledger.total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
