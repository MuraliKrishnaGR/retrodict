"""Point Retrodict at another host of Muse Spark Open (the same Meta open-weight model; open-weight Meta models
count wherever they run). Asks for the host's API key (hidden), finds the exact Muse Spark Open model name the
host uses, and writes the settings into .env (never committed).

  python scripts/use_host.py nebius
  python scripts/use_host.py groq
  python scripts/use_host.py baseten
  python scripts/use_host.py modal       asks for the endpoint URL and a proxy token (ID and secret)
  python scripts/check_setup.py           one tiny call to confirm it works
"""
import getpass
import json
import sys
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"
HOSTS = {
    "nebius": ["https://api.tokenfactory.nebius.com/v1", "https://api.studio.nebius.com/v1"],
    "groq": ["https://api.groq.com/openai/v1"],
    "baseten": ["https://inference.baseten.co/v1"],
    "together": ["https://api.together.xyz/v1"],
    "fireworks": ["https://api.fireworks.ai/inference/v1"],
    "openrouter": ["https://openrouter.ai/api/v1"],
    "modal": [],                        # the endpoint URL is per workspace: asked for below
}
COMMON = {"LLM_REASONING_EFFORT": "high", "LLM_MAX_RETRIES": "8", "LLM_MAX_TOKENS": "20000"}


def list_models(base: str, key: str) -> list:
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "retrodict"}
    if key.startswith("wk-") and "." in key:            # Modal proxy token: also as Modal's own headers
        headers["Modal-Key"], headers["Modal-Secret"] = key.split(".", 1)
    req = urllib.request.Request(base.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    items = data.get("data", data) if isinstance(data, dict) else data
    return [m.get("id", "") if isinstance(m, dict) else str(m) for m in items]


def pick(models: list):
    hits = [m for m in models if "muse" in m.lower() and "spark" in m.lower() and "open" in m.lower()]
    if len(hits) > 1:
        print("Muse Spark Open names on this host: " + ", ".join(hits))
    return sorted(hits, key=lambda m: ("meta" not in m.lower(), any(w in m.lower() for w in ("fast", "test", "turbo")), len(m)))[0] if hits else None


def update(text: str, values: dict) -> str:
    names = set(values) | {"LLM_API_KEY", "BASETEN_API_KEY"}
    lines = [ln for ln in text.splitlines() if ln.split("=", 1)[0].strip() not in names]
    return "\n".join([f"{k}={v}" for k, v in values.items()] + lines) + "\n"


def main():
    host = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if host not in HOSTS:
        raise SystemExit(f"usage: python scripts/use_host.py <{'|'.join(HOSTS)}>")
    if host == "modal":
        url = input("Paste the Muse Spark Open endpoint URL from your Modal dashboard: ").strip().rstrip("/")
        if not url.startswith("https://"):
            raise SystemExit("That does not look like the endpoint URL (it starts with https://).")
        HOSTS["modal"] = [url if url.endswith("/v1") else url + "/v1"]
        tid = getpass.getpass("Proxy token ID (hidden): ").strip()
        tsec = getpass.getpass("Proxy token secret (hidden): ").strip()
        key = f"{tid}.{tsec}"                           # Modal: ID and secret joined by a dot, as the bearer token
    else:
        key = getpass.getpass(f"Paste your {host} API key (hidden), then press Enter: ").strip().strip('"').strip("'")
    if len(key) < 16 or " " in key:
        raise SystemExit("That does not look like a full API key. Copy it again from the host's API keys page.")
    found, base, last = None, None, ""
    for b in HOSTS[host]:
        try:
            found, base = pick(list_models(b, key)), b
            break
        except Exception as e:                          # wrong endpoint or key: try the next one
            last = f"{type(e).__name__}: {str(e)[:150]}"
    if base is None:
        raise SystemExit(f"Could not list {host}'s models with that key ({last}). Check the key and try again.")
    if not found:
        raise SystemExit(f"{host} does not offer Muse Spark Open to this account (no such model in its list).")
    values = {"LLM_API_KEY": key, "LLM_BASE_URL": base, "LLM_MODEL": found, **COMMON}
    ENV.write_text(update(ENV.read_text(encoding="utf-8") if ENV.exists() else "", values), encoding="utf-8")
    print(f"Saved. Host {base}, model {found}, key ends with ...{key[-4:]}")
    print("Next: python scripts/check_setup.py")


if __name__ == "__main__":
    main()
