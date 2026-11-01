"""Save your LLM host API key (Baseten) into .env without opening an editor.
Usage: python scripts/set_key.py   (paste the key when asked; it is hidden and never printed)"""
import getpass
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"
NAME = "LLM_API_KEY"


def upsert_key(text: str, key: str) -> str:
    drop = (NAME, "BASETEN_API_KEY", "MODEL_API_KEY")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(drop)]
    lines.insert(0, f"{NAME}={key}")
    return "\n".join(lines) + "\n"


def main():
    key = getpass.getpass("Paste your Baseten API key (hidden), then press Enter: ").strip().strip('"').strip("'")
    if len(key) < 20 or " " in key:
        raise SystemExit("That does not look like a full API key. Copy it again from your Baseten account -> API keys.")
    old = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    ENV.write_text(upsert_key(old, key), encoding="utf-8")
    print(f"Saved. Key length: {len(key)} characters, ends with ...{key[-4:]}")
    print("Next: python scripts/check_setup.py data\\test1.jpg")


if __name__ == "__main__":
    main()
