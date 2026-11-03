"""Muse Spark Open through an OpenAI-compatible host (Baseten by default).
Every call: disk cache first, spend guard before paying, cost logged after."""
import base64
import json
import re

from pydantic import BaseModel, ValidationError

from app import config
from app.cache import DiskCache, make_key
from app.cost import SpendLedger, llm_cost

CALL_ESTIMATE_USD = 0.04   # rough upper bound per call, checked before paying


def strip_reasoning(text: str) -> str:
    """Remove any <think>...</think> block some hosts put in the reply."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def extract_json(text: str):
    """Pull the JSON object out of a model reply (handles ```json fences and reasoning blocks)."""
    text = re.sub(r"```(?:json)?", "", strip_reasoning(text)).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def jpeg_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class LLMClient:
    def __init__(self, client=None, cache: DiskCache | None = None, ledger: SpendLedger | None = None):
        if client is None:
            from openai import AsyncOpenAI
            key = config.api_key()
            headers = {}
            if "." in key and key.startswith("wk-"):     # a Modal proxy token "ID.secret": also send Modal's headers
                tid, tsec = key.split(".", 1)
                headers = {"Modal-Key": tid, "Modal-Secret": tsec}
            client = AsyncOpenAI(api_key=key, base_url=config.LLM_BASE_URL, default_headers=headers or None,
                                 max_retries=config.LLM_MAX_RETRIES, timeout=300)
        self.client = client
        self.cache = cache or DiskCache(config.CACHE_DIR)
        self.ledger = ledger or SpendLedger(config.CACHE_DIR / "spend.json",
                                            config.MAX_SPEND_USD, config.MAX_RUN_USD)
        self.cache_hits = 0

    async def aclose(self):
        close = getattr(self.client, "close", None)
        if close:
            await close()

    async def raw(self, prompt: str, jpeg: bytes | None = None, tag: str = "llm") -> str:
        key = make_key("llm", config.LLM_BASE_URL, config.LLM_MODEL, prompt, jpeg or b"")
        hit = self.cache.get_json("llm", key)
        if hit is not None:
            self.cache_hits += 1
            return hit["text"]

        content = [{"type": "text", "text": prompt}]
        if jpeg:
            content.append({"type": "image_url", "image_url": {"url": jpeg_data_uri(jpeg)}})
        kwargs = {"model": config.LLM_MODEL, "messages": [{"role": "user", "content": content}],
                  "max_tokens": config.LLM_MAX_TOKENS}
        if config.LLM_REASONING_EFFORT:
            kwargs["extra_body"] = {"reasoning_effort": config.LLM_REASONING_EFFORT}

        self.ledger.check(CALL_ESTIMATE_USD)
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:  # a host may reject an optional field: retry once without it
            if "max_tokens" not in str(e) and "reasoning_effort" not in str(e):
                raise
            kwargs.pop("max_tokens", None)
            kwargs.pop("extra_body", None)
            resp = await self.client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = strip_reasoning(choice.message.content or "")
        u = getattr(resp, "usage", None)
        pin, pout = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
        self.ledger.add(llm_cost(pin, pout), f"llm:{tag} in={pin} out={pout}")
        if not text and getattr(choice, "finish_reason", "") == "length":
            raise RuntimeError("Model used all its tokens on reasoning. Raise LLM_MAX_TOKENS in .env.")
        self.cache.put_json("llm", key, {"text": text, "prompt_tokens": pin, "completion_tokens": pout})
        return text

    async def json(self, prompt: str, schema: type[BaseModel], jpeg: bytes | None = None,
                   tag: str = "llm") -> BaseModel:
        """Ask for JSON and validate it. On bad JSON, retry ONCE with the error shown to the model."""
        text = await self.raw(prompt, jpeg, tag)
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as err:
            fix = (f"{prompt}\n\nYour previous reply was not valid. Error: {str(err)[:500]}\n"
                   "Reply again with ONLY the JSON object, nothing else.")
            text = await self.raw(fix, jpeg, tag + ":retry")
            return schema.model_validate(extract_json(text))
