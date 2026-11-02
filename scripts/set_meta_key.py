"""Save your Meta Model API key (the account with the hackathon credits) into .env as META_API_KEY.
Usage: python scripts/set_meta_key.py   (paste the key when asked; it is hidden and never printed)"""
import getpass
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"
NAME = "META_API_KEY"


def upsert(text: str, key: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(NAME + "=")]
    lines.append(f"{NAME}={key}")
    return "\n".join(lines) + "\n"


def main():
    key = getpass.getpass("Paste your Meta Model API key (hidden), then press Enter: ").strip().strip('"').strip("'")
    if len(key) < 20 or " " in key:
        raise SystemExit("That does not look like a full API key. Copy it again from dev.meta.ai -> API keys.")
    old = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    ENV.write_text(upsert(old, key), encoding="utf-8")
    print(f"Saved as {NAME}. Key length: {len(key)} characters, ends with ...{key[-4:]}")
    print("Next: python scripts/check_meta_image.py")


if __name__ == "__main__":
    main()
