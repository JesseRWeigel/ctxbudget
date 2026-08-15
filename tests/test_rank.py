"""The cut ranking, against a fixture project whose load-bearing files are known.

`fixtures/project/` is a small, deliberately shaped codebase. Four of its files are dead weight
for the stated task and four are load-bearing, and one of the load-bearing ones is the second
largest file in the set. That is the shape where ranking by size cuts the wrong thing.
"""

import unittest
from pathlib import Path

from ctxbudget import budget as budget_mod
from ctxbudget import rank as rank_mod

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "fixtures" / "project"

QUERY = "the retry backoff in the http client keeps retrying past the limit, fix it"

FILES = [
    "src/http_client.py",
    "src/retry_policy.py",
    "src/settings.py",
    "src/legacy_uploader.py",
    "tests/test_http_client.py",
    "tests/test_http_client_legacy.py",
    "vendor/deps.lock",
    "docs/CHANGELOG.md",
]

# Ground truth for this fixture, decided when the fixture was written and not by running the
# tool. Safe to cut: nothing in the set refers to it, or its content is duplicated elsewhere.
SAFE_TO_CUT = {
    "vendor/deps.lock",
    "docs/CHANGELOG.md",
    "src/legacy_uploader.py",
    "tests/test_http_client_legacy.py",
}
LOAD_BEARING = set(FILES) - SAFE_TO_CUT


def load():
    return [(name, (PROJECT / name).read_text(encoding="utf-8")) for name in FILES]


def token_map(parts):
    report = budget_mod.build("qwen2.5-7b-instruct", parts, None, budget_mod.load_models(),
                              allow_exact=False)
    return {part.label: part.tokens for part in report.parts if part.kind == "file"}


class RankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = load()
        cls.tokens = token_map(cls.parts)
        cls.signals = rank_mod.rank(cls.parts, cls.tokens, QUERY)

    def test_the_fixture_is_shaped_as_claimed(self):
        # A load-bearing file must be among the largest, or the naive baseline cannot lose and
        # the comparison below would be meaningless.
        by_size = sorted(self.tokens, key=lambda name: -self.tokens[name])
        self.assertTrue(set(by_size[:4]) & LOAD_BEARING)

    def test_the_first_four_cuts_are_all_dead_weight(self):
        first_four = [signal.label for signal in self.signals[:4]]
        self.assertEqual(set(first_four), SAFE_TO_CUT)

    def test_it_beats_ranking_by_size(self):
        ours = sum(1 for signal in self.signals[:4] if signal.label in SAFE_TO_CUT)
        naive = sum(1 for signal in rank_mod.largest_first(self.signals)[:4]
                    if signal.label in SAFE_TO_CUT)
        self.assertEqual(ours, 4)
        self.assertLess(naive, ours)

    def test_the_file_the_task_is_about_is_cut_last_of_the_large_ones(self):
        order = [signal.label for signal in self.signals]
        self.assertGreater(order.index("src/http_client.py"),
                           order.index("vendor/deps.lock"))
        self.assertGreater(order.index("src/http_client.py"),
                           order.index("tests/test_http_client_legacy.py"))

    def test_the_duplicate_test_file_outranks_its_twin(self):
        order = [signal.label for signal in self.signals]
        self.assertLess(order.index("tests/test_http_client_legacy.py"),
                        order.index("tests/test_http_client.py"))

    def test_inbound_references_are_found_and_named(self):
        client = next(s for s in self.signals if s.label == "src/http_client.py")
        self.assertGreaterEqual(client.inbound, 2)
        self.assertIn("HttpClient", client.inbound_via)

    def test_the_lockfile_has_no_inbound_reference(self):
        lock = next(s for s in self.signals if s.label == "vendor/deps.lock")
        self.assertEqual(lock.inbound, 0)

    def test_redundancy_is_found_between_the_two_test_files(self):
        legacy = next(s for s in self.signals
                      if s.label == "tests/test_http_client_legacy.py")
        self.assertGreater(legacy.redundancy, 0.5)
        self.assertEqual(legacy.redundant_with, "tests/test_http_client.py")

    def test_a_generic_name_is_not_counted_as_a_reference(self):
        # `length` and similar names appear everywhere and prove nothing.
        parts = [("a.py", "def length(self):\n    return 1\n"),
                 ("b.py", "x = length\n"),
                 ("c.py", "y = length\n"),
                 ("d.py", "z = length\n")]
        signals = rank_mod.analyse(parts)
        first = next(s for s in signals if s.label == "a.py")
        self.assertEqual(first.inbound, 0)

    def test_the_greedy_cut_stops_as_soon_as_it_fits(self):
        chosen = rank_mod.greedy_cut(self.signals, 100)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].label, self.signals[0].label)

    def test_nothing_is_cut_when_nothing_is_over(self):
        self.assertEqual(rank_mod.greedy_cut(self.signals, 0), [])

    def test_the_greedy_cut_frees_at_least_what_was_needed(self):
        over = sum(self.tokens.values()) // 2
        chosen = rank_mod.greedy_cut(self.signals, over)
        self.assertGreaterEqual(sum(signal.tokens for signal in chosen), over)

    def test_the_reason_string_names_the_evidence(self):
        lock = next(s for s in self.signals if s.label == "vendor/deps.lock")
        self.assertIn("nothing else in the context refers to it", lock.reason())

    def test_an_exact_duplicate_does_not_divide_by_zero(self):
        text = "alpha beta gamma\ndelta epsilon\nzeta eta theta\n"
        parts = [("one.txt", text), ("two.txt", text)]
        signals = rank_mod.rank(parts, {"one.txt": 100, "two.txt": 100})
        for signal in signals:
            self.assertTrue(signal.score < float("inf"))
            self.assertGreater(signal.score, 100)


if __name__ == "__main__":
    unittest.main()
