"""Save a free Hugging Face access token into .env as HF_TOKEN (gives your account's free GPU quota).
Make one at huggingface.co -> Settings -> Access Tokens (type: Read). Usage: python scripts/set_hf_token.py"""
import getpass
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"


def main():
    key = getpass.getpass("Paste your Hugging Face token (hidden), then press Enter: ").strip().strip('"').strip("'")
    if not key.startswith("hf_") or len(key) < 20:
        raise SystemExit("That does not look like a Hugging Face token (it starts with hf_). Copy it again.")
    old = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    lines = [ln for ln in old.splitlines() if not ln.strip().startswith("HF_TOKEN=")] + [f"HF_TOKEN={key}"]
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved as HF_TOKEN (ends with ...{key[-4:]}). Next: python scripts/check_image_edit.py")


if __name__ == "__main__":
    main()
