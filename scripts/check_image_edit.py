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
# Tried in order until one works (Spaces get paused or busy). Override with IMAGE_EDIT_SPACE in .env.
SPACES = [os.getenv("IMAGE_EDIT_SPACE")] if os.getenv("IMAGE_EDIT_SPACE") else \
    ["multimodalart/Qwen-Image-Edit-Fast", "Qwen/Qwen-Image-Edit"]
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


def small_copy(photo: Path) -> Path:
    """At most 1024 px on the long side: what these Spaces are built for, and less GPU time."""
    from PIL import Image
    im = Image.open(photo).convert("RGB")
    im.thumbnail((1024, 1024))
    out = ROOT / "out" / "edit_input.jpg"
    out.parent.mkdir(exist_ok=True)
    im.save(out, quality=92)
    return out


def error_text(e: Exception) -> str:
    """Gradio often hides the real reason; show everything it gives."""
    parts = [type(e).__name__, str(e)] + [str(a) for a in getattr(e, "args", ())]
    return " | ".join(dict.fromkeys(p for p in parts if p))[:600]


def try_space(space, token, photo):
    from gradio_client import Client, handle_file
    try:
        client = Client(space, token=token, verbose=False)
        api = client.view_api(print_info=False, return_format="dict")
    except Exception as e:
        print(f"[FAIL] connect to {space}: {error_text(e)}")
        return None
    name, ep = find_endpoint(api)
    if not name:
        print(f"[FAIL] {space} has no image + text endpoint: {list((api.get('named_endpoints') or {}))}")
        return None
    params = [(p.get("parameter_name"), p.get("component"), p.get("parameter_has_default")) for p in ep["parameters"]]
    print(f"[PASS] connected to {space} ({'with your token' if token else 'anonymous'}); endpoint {name}")
    for p in params:
        print(f"       {p[0]} ({p[1]}){'' if p[2] else '  required'}")
    if photo is None:
        return "listed"
    kwargs, text_done = {}, False
    for pname, comp, has_default in params:
        if "Image" in comp or "Gallery" in comp:
            kwargs[pname] = [handle_file(str(photo))] if "Gallery" in comp else handle_file(str(photo))
        elif "Textbox" in comp and not text_done:
            kwargs[pname], text_done = PROMPT, True
        elif pname == "rewrite_prompt":
            kwargs[pname] = False            # rewriting calls another model inside the Space; keep our exact words
        elif not has_default:
            print(f"[FAIL] {space} needs {pname} ({comp}); send me this output")
            return None
    try:
        result = client.predict(api_name=name, **kwargs)
    except Exception as e:
        print(f"[FAIL] edit on {space}: {error_text(e)}")
        return None
    path = first_image_path(result)
    if not path:
        print(f"[FAIL] {space} returned something unexpected: {str(result)[:300]}")
        return None
    return path


def main():
    try:
        import gradio_client  # noqa: F401
    except ImportError:
        raise SystemExit("Install it first: pip install gradio_client")
    token = os.getenv("HF_TOKEN") or None
    photo = small_copy(Path(sys.argv[1])) if len(sys.argv) > 1 else None
    for space in SPACES:
        path = try_space(space, token, photo)
        if path == "listed":
            print("Next (one edit, ~10 s of free GPU): python scripts/check_image_edit.py path\\to\\photo.jpeg")
            return
        if path:
            from PIL import Image
            out = ROOT / "out" / "edit_check.png"
            Image.open(path).convert("RGB").save(out)
            print(f"[PASS] edit works on {space}. Saved {out}")
            return
    print("No free editor worked." + ("" if token else
          " You are anonymous, which gets the smallest free GPU share (sometimes none): make a free Hugging Face"
          " account, create a Read token, run python scripts/set_hf_token.py, then try again."))


if __name__ == "__main__":
    main()
