"""All settings in one place. Values come from .env so nothing secret is in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Muse Spark Open through any OpenAI-compatible host (Baseten by default; see scripts/use_host.py)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://inference.baseten.co/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-models/muse-spark-open")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))       # reasoning tokens count toward this
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")     # e.g. "low"; empty = provider default
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))          # free tiers rate-limit: retry with back-off

# SAM 3.1 open weights, run locally with transformers
SAM_MODEL_ID = os.getenv("SAM_MODEL_ID", "unsloth/sam3.1")
SAM_SCORE_THRESHOLD = float(os.getenv("SAM_SCORE_THRESHOLD", "0.5"))
SAM_MASK_THRESHOLD = float(os.getenv("SAM_MASK_THRESHOLD", "0.5"))
SAM_MAX_INSTANCES = int(os.getenv("SAM_MAX_INSTANCES", "8"))   # keep the most confident per phrase

CACHE_DIR = ROOT / ".cache"
OUT_DIR = ROOT / "out"

MAX_SPEND_USD = float(os.getenv("MAX_SPEND_USD", "50"))
MAX_RUN_USD = float(os.getenv("MAX_RUN_USD", "1.00"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))

# Estimated token prices for the LLM host. Update from your host's pricing page.
PRICE_LLM_IN_PER_M = float(os.getenv("PRICE_LLM_IN_PER_M", "1.25"))
PRICE_LLM_OUT_PER_M = float(os.getenv("PRICE_LLM_OUT_PER_M", "4.25"))


def api_key() -> str:
    key = (os.getenv("LLM_API_KEY") or os.getenv("BASETEN_API_KEY") or "").strip()
    if not key or key == "paste_your_key_here":
        raise SystemExit("LLM_API_KEY is missing. Run: python scripts/use_host.py <host>  (baseten, openrouter, together, fireworks or modal)")
    return key
