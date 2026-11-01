"""Turn errors into a plain-language fix. Never prints the key."""


def explain(err: Exception) -> str:
    name = type(err).__name__
    low = str(err).lower()
    code = getattr(err, "status_code", None)
    if name == "BudgetExceeded":
        return "Spend guard stopped the call. Raise MAX_RUN_USD / MAX_SPEND_USD in .env only if you mean to."
    if code == 401 or name == "AuthenticationError":
        return "Key rejected. Run: python scripts/set_key.py and paste the full Baseten API key again."
    if code == 402 or "billing" in low or "payment" in low or "credit" in low or "insufficient" in low:
        return "Billing problem on the LLM host. Check that your Baseten credits are active."
    if code == 403 or name == "PermissionDeniedError":
        return "Access denied. The model may not be enabled for your account yet. Ask in the Baseten Discord channel."
    if code == 404 or name == "NotFoundError":
        return "Model not found. Check LLM_MODEL in .env or run: python scripts/find_model.py."
    if code == 429 or name == "RateLimitError":
        return "Rate limit. Wait a minute and run again (finished steps are cached, not paid twice)."
    if code == 400 or name == "BadRequestError":
        return "Request format rejected. Send Claude this exact error so the request can be fixed."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "TimeoutError"):
        return "Network problem. Check your internet and run again."
    if name in ("ModuleNotFoundError", "ImportError"):
        return "A package is missing. Run: pip install -r requirements.txt"
    if "401" in low or "gated" in low or "huggingface" in low or "repository not found" in low:
        return "Could not download SAM 3.1 weights. Check your internet; then run the same command again."
    if name == "OutOfMemoryError" or "out of memory" in low or "memoryerror" in low:
        return "Not enough RAM for SAM. Close other apps, or set MAX_IMAGE_SIDE=768 in .env."
    return "Unexpected error. Send Claude this exact error text."
