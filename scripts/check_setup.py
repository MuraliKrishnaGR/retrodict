"""Step-by-step check of the two models Retrodict uses. Stops at the first failure and says how to fix it.
Usage:
  python scripts/check_setup.py                 key + Muse Spark Open (text)
  python scripts/check_setup.py data\\test1.jpg  + Muse Spark Open sees the photo + SAM 3.1 on your CPU
Finished steps are cached: running it again does not pay or compute again."""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                          # noqa: E402
from app.clients.models import Models           # noqa: E402
from app.errors import explain                  # noqa: E402
from app.imaging import overlay, prepare_image  # noqa: E402
from app.schemas import ObjectList              # noqa: E402


def ok(step, text):
    print(f"[PASS] {step}: {text}", flush=True)


def fail(step, err):
    print(f"[FAIL] {step}")
    print(f"       error: {type(err).__name__}: {str(err)[:400]}")
    print(f"       fix:   {explain(err)}")


async def run(photo: Path | None) -> bool:
    try:
        key = config.api_key()
    except SystemExit as e:
        print(f"[FAIL] 1 key: {e}")
        return False
    if len(key) < 20:
        print("[FAIL] 1 key: the key in .env is too short to be real.\n       fix:   python scripts/set_key.py")
        return False
    ok("1 key", f"found ({len(key)} characters), host {config.LLM_BASE_URL}, model {config.LLM_MODEL}")

    models = Models()
    try:
        step = "2 Muse Spark Open (text)"
        try:
            text = await models.llm.raw("Reply with exactly one word: working", None, "check_text")
            ok(step, f"model replied {text.strip()[:40]!r}")
        except Exception as e:
            fail(step, e)
            return False

        if photo is None:
            print("\nAdd a photo to test vision and SAM:  python scripts/check_setup.py data\\test1.jpg")
            return True

        img, jpeg = prepare_image(photo)
        step = "3 Muse Spark Open (sees the photo)"
        try:
            prompt = ('List up to 3 physical objects you see in this photo as short nouns. '
                      'Return ONLY JSON: {"objects": [{"name": "...", "why": ""}]}')
            res = await models.spark_json(prompt, ObjectList, jpeg=jpeg, tag="check_vision")
            names = [o.name.strip().lower() for o in res.objects if o.name.strip()]
            if not names:
                raise RuntimeError("the model returned no objects for this photo")
            ok(step, ", ".join(names))
        except Exception as e:
            fail(step, e)
            return False

        step = "4 SAM 3.1 on your CPU (segments one object)"
        try:
            t = time.time()
            items = await models.sam_segment(jpeg, img.width, img.height, names[0])
            out = config.OUT_DIR / "check"
            out.mkdir(parents=True, exist_ok=True)
            drawn = [{"id": f"o{i + 1}", "label": names[0], "mask": it["mask"], "bbox": it["bbox"]}
                     for i, it in enumerate(items)]
            overlay(img, drawn).save(out / "sam_check.png")
            took = f"{time.time() - t:.0f}s"
            if items:
                ok(step, f"found {len(items)} x {names[0]!r} in {took}, saved {out / 'sam_check.png'}")
            else:
                print(f"[WARN] {step}: SAM ran ({took}) but found no {names[0]!r}. Try a clearer photo.")
        except Exception as e:
            fail(step, e)
            return False
        return True
    finally:
        print(f"\nPaid LLM calls: {models.llm.ledger.calls} | cache hits: {models.cache_hits} | "
              f"est. this run ${models.llm.ledger.run_spent:.4f} | est. all runs ${models.llm.ledger.total:.4f}")
        await models.aclose()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    photo = Path(args[0]) if args else None
    if photo is not None and not photo.exists():
        raise SystemExit(f"Photo not found: {photo}")
    good = asyncio.run(run(photo))
    print("\nAll checks passed." if good else "\nFix the failed step above, then run this again.")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
