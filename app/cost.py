"""Spend guard. Estimates cost of every paid call and stops before a limit is crossed.
It cannot see the host's real bill, so also check the billing page."""
import json
import time
from pathlib import Path

from app import config


class BudgetExceeded(RuntimeError):
    pass


def llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * config.PRICE_LLM_IN_PER_M
            + completion_tokens * config.PRICE_LLM_OUT_PER_M) / 1_000_000


class SpendLedger:
    """Running total across all runs, saved in .cache/spend.json, plus a per-run cap."""

    def __init__(self, path: Path, max_total: float, max_run: float):
        self.path = Path(path)
        self.max_total = max_total
        self.max_run = max_run
        self.run_spent = 0.0
        self.calls = 0
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.total = float(data.get("total_usd", 0.0))

    def check(self, upcoming: float) -> None:
        """Call BEFORE a paid request with a rough upper estimate."""
        if self.total + upcoming > self.max_total:
            raise BudgetExceeded(f"Total limit ${self.max_total:.2f} reached (spent ~${self.total:.4f}).")
        if self.run_spent + upcoming > self.max_run:
            raise BudgetExceeded(f"Per-run limit ${self.max_run:.2f} reached (this run ~${self.run_spent:.4f}).")

    def add(self, usd: float, what: str) -> None:
        self.total += usd
        self.run_spent += usd
        self.calls += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"total_usd": round(self.total, 6)}, indent=2))
        with open(self.path.with_name("calls.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{what}\t${usd:.5f}\n")
