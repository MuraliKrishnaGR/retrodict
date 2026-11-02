"""Find which Muse Spark Open model id your LLM key can use, and save the working one to .env.
Usage: python scripts/find_model.py   (each try is a tiny call, well under $0.01)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI      # noqa: E402

from app import config         # noqa: E402

CANDIDATES = [
    "meta-models/muse-spark-open-1026",
    "meta-models/muse-spark-open",
    "meta-models/Muse-Spark-Open",
    "meta-models/Muse-Spark-Open-816B-A42B",
]


def save_model(model_id: str) -> None:
    env = ROOT / ".env"
    lines = [ln for ln in env.read_text(encoding="utf-8").splitlines() if not ln.startswith("LLM_MODEL=")]
    lines.append(f"LLM_MODEL={model_id}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    client = OpenAI(api_key=config.api_key(), base_url=config.LLM_BASE_URL, max_retries=0, timeout=120)

    print("Models this key can see:")
    try:
        ids = sorted(m.id for m in client.models.list())
        muse = [i for i in ids if "muse" in i.lower()]
        for i in (muse or ids[:40]):
            print("  ", i)
        CANDIDATES[:0] = [i for i in muse if i not in CANDIDATES]
    except Exception as e:
        print(f"   (could not list models: {type(e).__name__}: {str(e)[:200]})")

    print("\nTrying ids:")
    for model_id in CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=model_id, max_tokens=2000,
                messages=[{"role": "user", "content": "Reply with exactly one word: working"}])
            text = (r.choices[0].message.content or "").strip()
            print(f"  [OK]   {model_id} -> {text[:40]!r}")
            save_model(model_id)
            print(f"\nSaved LLM_MODEL={model_id} to .env")
            print("Next: python scripts\\check_setup.py data\\test1.jpeg")
            return
        except Exception as e:
            print(f"  [NO]   {model_id} -> {type(e).__name__}: {str(e)[:160]}")
    print("\nNo id worked. Check the model name on your Baseten model page and set LLM_MODEL in .env.")


if __name__ == "__main__":
    main()
