"""Can we use Muse Image with the hackathon credits? Two steps; stops at the first failure.

  python scripts/check_meta_image.py                      1. key works, Muse Image is listed (free)
  python scripts/check_meta_image.py data\\photo.jpeg      2. ONE real edit of your photo (uses one image of credit)

The edit request uses the OpenAI-compatible images API that the Meta Model API follows. If step 2 fails
with a format error, open dev.meta.ai/docs/image-generation and send the example request it shows."""
import base64
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI      # noqa: E402

load_dotenv(ROOT / ".env")
BASE = "https://api.meta.ai/v1"
MODEL = os.getenv("META_IMAGE_MODEL", "muse-image-1.0")


def say_fix(err):
    code, text = getattr(err, "status_code", None), str(err).lower()
    if code == 401:
        return "The key was refused. Create a new key on dev.meta.ai and run scripts/set_meta_key.py again."
    if code == 402 or any(w in text for w in ("billing", "payment", "credit", "insufficient", "card")):
        return "A billing problem: the account may need a working card even with credits. Ask in the hackathon Discord."
    if code == 403:
        return "Access denied: Muse Image may not be enabled for this account or region."
    if code == 404:
        return f"The model id {MODEL!r} was not found. Check the list printed in step 1."
    if code in (400, 422):
        return "The request format was rejected. Send me the example from dev.meta.ai/docs/image-generation."
    return "Unexpected. Send me this exact error text."


def main():
    key = os.getenv("META_API_KEY", "").strip()
    if not key:
        raise SystemExit("META_API_KEY is missing. Run: python scripts/set_meta_key.py")
    client = OpenAI(api_key=key, base_url=BASE, timeout=180)

    try:
        ids = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        raise SystemExit(f"[FAIL] 1 key: {type(e).__name__}: {str(e)[:300]}\n       fix: {say_fix(e)}")
    image_ids = [i for i in ids if "image" in i.lower()]
    print(f"[PASS] 1 key works. {len(ids)} models; image models: {', '.join(image_ids) or 'none listed'}")
    if MODEL not in ids:
        print(f"       note: {MODEL!r} is not in the list; set META_IMAGE_MODEL in .env to one of the image models")

    if len(sys.argv) < 2:
        print("Next (uses one image of credit): python scripts/check_meta_image.py path\\to\\photo.jpeg")
        return
    photo = Path(sys.argv[1])
    try:
        with open(photo, "rb") as f:
            resp = client.images.edit(model=MODEL, image=f, prompt=(
                "Edit this exact photo. Remove the steel glass of milk from the counter and show the bare black "
                "counter and the wall behind it where it stood. Keep everything else exactly the same: camera, "
                "light, every other object and its position."))
    except Exception as e:
        raise SystemExit(f"[FAIL] 2 edit: {type(e).__name__}: {str(e)[:400]}\n       fix: {say_fix(e)}")
    item = resp.data[0]
    out = ROOT / "out" / "meta_image_check.png"
    out.parent.mkdir(exist_ok=True)
    if getattr(item, "b64_json", None):
        out.write_bytes(base64.b64decode(item.b64_json))
    elif getattr(item, "url", None):
        urllib.request.urlretrieve(item.url, out)
    else:
        raise SystemExit(f"[FAIL] 2 edit: unexpected response: {str(resp)[:300]}")
    print(f"[PASS] 2 edit works. Saved {out}")


if __name__ == "__main__":
    main()
