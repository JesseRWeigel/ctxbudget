"""The outbound HTTP client. Deliberately the second largest file in this fixture project.

Ranking by size cuts this third. It is the file the task is about, and three other files in the
set refer to it. That is the disagreement the fixture exists to demonstrate.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .retry_policy import RetryPolicy, total_wait_ms
from .settings import (
    CONNECT_TIMEOUT_S,
    IDEMPOTENT_METHODS,
    POOL_SIZE,
    READ_TIMEOUT_S,
    USER_AGENT,
)


class TransportFailure(RuntimeError):
    """The request never produced a response."""


class GaveUp(RuntimeError):
    """The retry policy ran out of attempts."""

    def __init__(self, attempts: int, last_status: int | None):
        super().__init__(f"gave up after {attempts} attempts, last status {last_status}")
        self.attempts = attempts
        self.last_status = last_status


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    attempts: int
    waited_ms: int

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass
class HttpClient:
    """A small retrying client. One connection pool, one retry policy, no global state."""

    base_url: str
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    pool_size: int = POOL_SIZE
    user_agent: str = USER_AGENT
    connect_timeout_s: float = CONNECT_TIMEOUT_S
    read_timeout_s: float = READ_TIMEOUT_S
    _sleep = staticmethod(time.sleep)

    def url_for(self, route: str) -> str:
        if route.startswith("http://") or route.startswith("https://"):
            return route
        return self.base_url.rstrip("/") + "/" + route.lstrip("/")

    def build_request(self, method: str, route: str, body: bytes | None,
                      headers: dict[str, str] | None) -> urllib.request.Request:
        merged = {"User-Agent": self.user_agent, "Accept": "application/json"}
        merged.update(headers or {})
        if body is not None and "Content-Type" not in merged:
            merged["Content-Type"] = "application/json"
        return urllib.request.Request(self.url_for(route), data=body, headers=merged,
                                      method=method.upper())

    def is_safe_to_repeat(self, method: str, idempotency_key: str | None) -> bool:
        if method.upper() in IDEMPOTENT_METHODS:
            return True
        return idempotency_key is not None

    def send(self, method: str, route: str, body: bytes | None = None,
             headers: dict[str, str] | None = None,
             idempotency_key: str | None = None) -> HttpResponse:
        """Send with retries. Raises GaveUp when the policy stops allowing attempts."""
        attempt = 0
        waited_ms = 0
        last_status: int | None = None
        repeatable = self.is_safe_to_repeat(method, idempotency_key)
        while True:
            request = self.build_request(method, route, body, headers)
            try:
                with urllib.request.urlopen(request, timeout=self.read_timeout_s) as handle:
                    payload = handle.read()
                    status = handle.status
                    response_headers = dict(handle.headers.items())
                if 200 <= status < 400:
                    return HttpResponse(status, response_headers, payload, attempt + 1, waited_ms)
                last_status = status
            except urllib.error.HTTPError as error:
                last_status = error.code
            except (urllib.error.URLError, TimeoutError) as error:
                if not repeatable:
                    raise TransportFailure(str(error)) from error
                last_status = None

            if not repeatable or not self.policy.should_retry(attempt, last_status, method):
                raise GaveUp(attempt + 1, last_status)
            delay = self.policy.backoff_ms(attempt)
            waited_ms += delay
            self._sleep(delay / 1000.0)
            attempt += 1

    def get(self, route: str, headers: dict[str, str] | None = None) -> HttpResponse:
        return self.send("GET", route, None, headers)

    def post_json(self, route: str, payload, headers: dict[str, str] | None = None,
                  idempotency_key: str | None = None) -> HttpResponse:
        body = json.dumps(payload).encode("utf-8")
        return self.send("POST", route, body, headers, idempotency_key)

    def put_json(self, route: str, payload, headers: dict[str, str] | None = None) -> HttpResponse:
        body = json.dumps(payload).encode("utf-8")
        return self.send("PUT", route, body, headers)

    def delete(self, route: str, headers: dict[str, str] | None = None) -> HttpResponse:
        return self.send("DELETE", route, None, headers)

    def worst_case_wait_ms(self) -> int:
        return total_wait_ms(self.policy)

    def describe(self) -> str:
        return (f"HttpClient(base_url={self.base_url!r}, pool={self.pool_size}, "
                f"{self.policy.describe()}, worst case wait {self.worst_case_wait_ms()} ms)")


def client_from_env(environ: dict[str, str]) -> HttpClient:
    base = environ.get("INGEST_BASE_URL", "https://localhost:8443")
    limit = int(environ.get("INGEST_RETRY_LIMIT", "0")) or None
    policy = RetryPolicy(limit=limit) if limit else RetryPolicy()
    return HttpClient(base_url=base, policy=policy)
