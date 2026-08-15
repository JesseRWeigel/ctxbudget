"""Current tests for the client. Refers to HttpClient, RetryPolicy and GaveUp by name."""

import random
import unittest

from src.http_client import GaveUp, HttpClient
from src.retry_policy import RetryPolicy


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class RetryPolicyTest(unittest.TestCase):
    def test_backoff_grows_and_is_capped(self):
        policy = RetryPolicy(limit=6, base_ms=100, ceiling_ms=1000,
                             rng=random.Random(7))
        waits = [policy.backoff_ms(attempt) for attempt in range(6)]
        self.assertTrue(all(wait <= 1200 for wait in waits))
        self.assertGreater(waits[3], waits[0])

    def test_stops_at_the_limit(self):
        policy = RetryPolicy(limit=3)
        self.assertTrue(policy.should_retry(0, 503, "GET"))
        self.assertFalse(policy.should_retry(3, 503, "GET"))

    def test_non_retryable_status_is_final(self):
        policy = RetryPolicy(limit=5)
        self.assertFalse(policy.should_retry(0, 404, "GET"))


class HttpClientTest(unittest.TestCase):
    def test_url_for_joins_without_doubling_the_slash(self):
        client = HttpClient(base_url="https://example.invalid/api/")
        self.assertEqual(client.url_for("/v1/rows"), "https://example.invalid/api/v1/rows")

    def test_absolute_route_is_left_alone(self):
        client = HttpClient(base_url="https://example.invalid")
        self.assertEqual(client.url_for("https://other.invalid/x"), "https://other.invalid/x")

    def test_post_is_only_repeatable_with_an_idempotency_key(self):
        client = HttpClient(base_url="https://example.invalid")
        self.assertFalse(client.is_safe_to_repeat("POST", None))
        self.assertTrue(client.is_safe_to_repeat("POST", "key-1"))
        self.assertTrue(client.is_safe_to_repeat("GET", None))

    def test_gave_up_carries_the_attempt_count(self):
        error = GaveUp(4, 503)
        self.assertEqual(error.attempts, 4)
        self.assertEqual(error.last_status, 503)


if __name__ == "__main__":
    unittest.main()
