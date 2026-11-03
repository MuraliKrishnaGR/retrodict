"""The comparison Retrodict has to beat: ONE Muse Spark Open call on the photo (about 1 US cent, cached).

  python scripts/baseline.py data/milk_photo_original.jpeg
  python scripts/score.py out/baselines/milk_photo_original eval/keys/milk_01.json

Saves out/baselines/<photo name>/result.json in the same shape score.py reads."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                       # noqa: E402
from app.baseline import run_baseline        # noqa: E402
from app.clients.models import Models        # noqa: E402
from app.cost import BudgetExceeded          # noqa: E402
from app.imaging import prepare_image        # noqa: E402


async def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/baseline.py path/to/photo.jpg")
    photo = Path(sys.argv[1])
    if not photo.exists():
        raise SystemExit(f"Photo not found: {photo}")
    img, jpeg = prepare_image(photo)
    out = config.OUT_DIR / "baselines" / photo.stem
    out.mkdir(parents=True, exist_ok=True)
    models = Models()
    try:
        res = await run_baseline(models, jpeg)
    except BudgetExceeded as e:
        raise SystemExit(f"Stopped by spend guard: {e}")
    finally:
        await models.aclose()
    res["photo"] = str(photo)
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    img.save(out / "input.jpg", quality=90)
    print(f"Baseline (one call): {res['summary']}")
    for e in res["events"]:
        print(f"  {e['id']}  {e['description']}")
    print(f"\nSaved {out / 'result.json'}")
    print(f"Spend this run: ${models.llm.ledger.run_spent:.4f} | cache hits: {models.cache_hits}")


if __name__ == "__main__":
    asyncio.run(main())
