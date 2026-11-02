"""Try a free, open-source instruction image editor (Qwen-Image-Edit, Apache-2.0) on a free Hugging Face GPU.

  python scripts/check_image_edit.py                       1. connect and list what the Space accepts (free)
  python scripts/check_image_edit.py data\\photo.jpeg      2. ONE edit of your photo (uses ~10 s of free GPU)

Needs: pip install gradio_client. A free Hugging Face account token (scripts/set_hf_token.py) gives your
own daily GPU quota (about 5 GPU-minutes a day on a free account); without one, anonymous limits apply.
Saves out/edit_check.png. Nothing here is paid."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
SPACE = os.getenv("IMAGE_EDIT_SPACE", "multimodalart/Qwen-Image-Edit-Fast")
PROMPT = ("Make the steel glass empty: remove only the milk inside it, so the bare inside of the glass shows. "
          "Keep everything else exactly the same: camera, light, the glass itself and every other object.")


def find_endpoint(api: dict):
    """The endpoint that takes an image and a text instruction."""
    for name, ep in (api.get("named_endpoints") or {}).items():
        comps = [p.get("component", "") for p in ep.get("parameters", [])]
        if any("Image" in c or "Gallery" in c for c in comps) and any("Textbox" in c for c in comps):
            return name, ep
    return None, None


def first_image_path(result):
    """Results come back as paths, dicts with 'path'/'image', or lists of those."""
    if isinstance(result, str) and Path(result).exists():
        return result
    if isinstance(result, dict):
        for k in ("path", "image", "value"):
            if k in result:
                p = first_image_path(result[k])
                if p:
                    return p
    if isinstance(result, (list, tuple)):
        for r in result:
            p = first_image_path(r)
            if p:
                return p
    return None


def main():
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise SystemExit("Install it first: pip install gradio_client")
    token = os.getenv("HF_TOKEN") or None
    try:
        client = Client(SPACE, token=token, verbose=False)
        api = client.view_api(print_info=False, return_format="dict")
    except Exception as e:
        raise SystemExit(f"[FAIL] 1 connect to {SPACE}: {type(e).__name__}: {str(e)[:300]}\n"
                         "       fix: the Space may be paused or busy; try again later or set IMAGE_EDIT_SPACE in .env")
    name, ep = find_endpoint(api)
    if not name:
        raise SystemExit(f"[FAIL] 1 {SPACE} has no image + text endpoint. Endpoints: {list((api.get('named_endpoints') or {}))}")
    params = [(p.get("parameter_name"), p.get("component"), p.get("parameter_has_default")) for p in ep["parameters"]]
    print(f"[PASS] 1 connected to {SPACE} ({'with your token' if token else 'anonymous'}); endpoint {name}")
    for p in params:
        print(f"       {p[0]} ({p[1]}){'' if p[2] else '  required'}")
    if len(sys.argv) < 2:
        print("Next (one edit, ~10 s of free GPU): python scripts/check_image_edit.py path\\to\\photo.jpeg")
        return

    photo = Path(sys.argv[1])
    kwargs, text_done = {}, False
    for pname, comp, has_default in params:
        if "Image" in comp or "Gallery" in comp:
            kwargs[pname] = [handle_file(str(photo))] if "Gallery" in comp else handle_file(str(photo))
        elif "Textbox" in comp and not text_done:
            kwargs[pname], text_done = PROMPT, True
        elif not has_default:
            raise SystemExit(f"[FAIL] 2 the Space needs {pname} ({comp}); send me the list printed above")
    try:
        result = client.predict(api_name=name, **kwargs)
    except Exception as e:
        raise SystemExit(f"[FAIL] 2 edit: {type(e).__name__}: {str(e)[:400]}\n"
                         "       fix: a quota message means the free GPU minutes for today are used up")
    path = first_image_path(result)
    if not path:
        raise SystemExit(f"[FAIL] 2 unexpected result: {str(result)[:300]}")
    out = ROOT / "out" / "edit_check.png"
    out.parent.mkdir(exist_ok=True)
    from PIL import Image
    Image.open(path).convert("RGB").save(out)
    print(f"[PASS] 2 edit works. Saved {out}")


if __name__ == "__main__":
    main()
