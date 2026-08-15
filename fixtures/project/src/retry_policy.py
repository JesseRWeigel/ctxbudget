"""Backoff policy. The bug in the fixture task lives here, and the client calls into it."""

from __future__ import annotations

import random

from .settings import (
    BACKOFF_JITTER_RATIO,
    BASE_BACKOFF_MS,
    MAX_BACKOFF_MS,
    RETRYABLE_STATUS,
    RETRY_LIMIT,
)


class RetryPolicy:
    """Decides whether to try again and how long to wait first."""

    def __init__(self, limit: int = RETRY_LIMIT, base_ms: int = BASE_BACKOFF_MS,
                 ceiling_ms: int = MAX_BACKOFF_MS, rng: random.Random | None = None):
        self.limit = limit
        self.base_ms = base_ms
        self.ceiling_ms = ceiling_ms
        self.rng = rng or random.Random()

    def should_retry(self, attempt: int, status: int | None, method: str) -> bool:
        if attempt >= self.limit:
            return False
        if status is None:
            return True
        return status in RETRYABLE_STATUS

    def backoff_ms(self, attempt: int) -> int:
        raw = min(self.ceiling_ms, self.base_ms * (2 ** max(0, attempt)))
        jitter = raw * BACKOFF_JITTER_RATIO
        return int(max(0.0, raw - jitter + self.rng.random() * 2 * jitter))

    def describe(self) -> str:
        steps = [self.base_ms * (2 ** attempt) for attempt in range(self.limit)]
        capped = [min(self.ceiling_ms, step) for step in steps]
        return f"limit={self.limit} backoff_ms={capped}"


def total_wait_ms(policy: RetryPolicy) -> int:
    """Worst case time spent waiting if every attempt fails."""
    return sum(min(policy.ceiling_ms, policy.base_ms * (2 ** attempt))
               for attempt in range(policy.limit))
