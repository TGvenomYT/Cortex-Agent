"""
usage.py — what the models have used, what that cost, and the limits on it.

Every model call is recorded. Cloud calls are checked against the limits before they're
made: a day, a month, and one job (Settings → Usage). Dollars need a known price; a model
with no price (Ollama Cloud, unless you add one) is held by the token limits instead, so
nothing cloud-side is ever unlimited. Local models cost nothing and are never stopped.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("cortex.usage")

# The running job (a runtime.Handle), so a call can be charged to it.
current_job = contextvars.ContextVar("cortex_job", default=None)
_db = None
_PRICES = None
LOCAL = "local"


class BudgetExceeded(Exception):
    """A cloud limit has been reached. The message says which, in plain words."""


def bind(db) -> None:
    global _db
    _db = db


def _table() -> dict:
    global _PRICES
    if _PRICES is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm", "prices.json")
        try:
            with open(path, encoding="utf-8") as f:
                _PRICES = json.load(f)["prices"]
        except (OSError, ValueError) as e:
            logger.warning(f"no price table: {e}")
            _PRICES = {}
    return _PRICES


def limits() -> dict:
    import cortex.storage.settings_store as settings_store
    return settings_store.section("limits")


def price_for(model: str) -> list | None:
    """The price entry for a model: yours first, then the longest matching published name."""
    name = (model or "").lower().split("/")[-1]
    for table in (limits().get("prices") or {}, _table()):
        match = max((k for k in table if name.startswith(k.lower())), key=len, default=None)
        if match:
            return table[match]
    return None


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = price_for(model)
    if not p:
        return None
    rate_in, rate_out = p[0], p[1]
    if len(p) >= 5 and input_tokens > p[2]:
        rate_in, rate_out = p[3], p[4]
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def record(provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Note one model call. Never lets bookkeeping break the job."""
    input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
    job = current_job.get()
    cost = None if provider == LOCAL else cost_of(model, input_tokens, output_tokens)
    if job is not None and provider != LOCAL:
        job.spent_tokens += input_tokens + output_tokens
        job.spent_usd += cost or 0.0
    if _db is None:
        return
    try:
        _db.add_usage(provider, model, input_tokens, output_tokens, cost,
                      job.id if job else None, job.session_id if job else None)
    except Exception as e:
        logger.debug(f"usage not recorded: {e}")


def _since(period: str) -> datetime:
    """Start of today or this month in local time, as naive UTC like the stored stamps."""
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        start = start.replace(day=1)
    return start.astimezone(timezone.utc).replace(tzinfo=None)


def totals(period: str = "day") -> dict:
    """{usd, tokens} of CLOUD use so far this day or month."""
    if _db is None:
        return {"usd": 0.0, "tokens": 0}
    return _db.usage_totals(_since(period))


def check(provider: str, model: str) -> None:
    """Raise BudgetExceeded if this call would go over a limit. Local calls always pass."""
    if provider == LOCAL:
        return
    lim = limits()
    job = current_job.get()
    day, month = totals("day"), totals("month")
    rules = [
        (lim.get("daily_usd"), day["usd"], "today's ${:.2f} cloud limit"),
        (lim.get("monthly_usd"), month["usd"], "this month's ${:.2f} cloud limit"),
        (lim.get("daily_tokens"), day["tokens"], "today's {:,} cloud token limit"),
    ]
    if job is not None:
        rules += [(lim.get("job_usd"), job.spent_usd, "this job's ${:.2f} limit"),
                  (lim.get("job_tokens"), job.spent_tokens, "this job's {:,} token limit")]
    for limit, used, words in rules:
        if limit and used >= limit:
            raise BudgetExceeded("I've reached " + words.format(limit))


def near_limit() -> str | None:
    """A heads-up once cloud use passes 80% of a daily or monthly limit."""
    lim = limits()
    day, month = totals("day"), totals("month")
    for limit, used, what in ((lim.get("daily_usd"), day["usd"], "today's dollar"),
                              (lim.get("monthly_usd"), month["usd"], "this month's dollar"),
                              (lim.get("daily_tokens"), day["tokens"], "today's token")):
        if limit and used >= 0.8 * limit:
            return f"Heads up: that's {min(100, round(used / limit * 100))}% of {what} limit for cloud models."
    return None


def report() -> dict:
    """Everything Settings → Usage and `cx usage` show."""
    if _db is None:
        return {"day": {"usd": 0.0, "tokens": 0}, "month": {"usd": 0.0, "tokens": 0}, "models": [], "limits": limits()}
    lim = {k: v for k, v in limits().items() if k != "prices"}
    return {"day": totals("day"), "month": totals("month"), "limits": lim,
            "models": _db.usage_by_model(_since("month"))}
