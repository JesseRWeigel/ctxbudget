"""The estimator against real token counts, with no tokenizer installed.

`fixtures/truth/counts.json` holds the counts four real tokenizers produced for the committed
corpus. Those numbers were measured once by `scripts/calibrate.py` and committed, so this suite
tests against reality without depending on `tiktoken`, on a network, or on anybody's files.
"""

import json
import unittest
from pathlib import Path

from ctxbudget.tokens import Counter, naive_count

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "fixtures" / "corpus"
TRUTH = json.loads((ROOT / "fixtures" / "truth" / "counts.json").read_text(
    encoding="utf-8"))["counts"]
FAMILIES = ("cl100k_base", "o200k_base", "qwen2.5", "llama3")

# The estimator's measured worst case is CJK, and this suite states the bound rather than
# hiding it. Anything outside these numbers means the table was refitted and the README claims
# need refreshing.
PER_FILE_LIMIT_PCT = 15.0
CJK_LIMIT_PCT = 40.0
CORPUS_TOTAL_LIMIT_PCT = 3.0
CJK_FILES = {"japanese.txt"}


class EstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # allow_exact=False so the test measures the shipped table and not whatever tokenizer
        # happens to be installed on the machine running it.
        cls.counters = {family: Counter(family, allow_exact=False) for family in FAMILIES}

    def error_pct(self, family, name):
        text = (CORPUS / name).read_text(encoding="utf-8")
        truth = TRUTH[name][family]
        got = self.counters[family].count(text).tokens
        return abs(got - truth) / truth * 100.0

    def test_every_fixture_is_within_its_documented_bound(self):
        for name in sorted(TRUTH):
            limit = CJK_LIMIT_PCT if name in CJK_FILES else PER_FILE_LIMIT_PCT
            for family in FAMILIES:
                with self.subTest(f"{name}/{family}"):
                    self.assertLessEqual(self.error_pct(family, name), limit)

    def test_the_whole_corpus_totals_within_three_percent(self):
        for family in FAMILIES:
            estimated = sum(self.counters[family].count(
                (CORPUS / name).read_text(encoding="utf-8")).tokens for name in TRUTH)
            truth = sum(TRUTH[name][family] for name in TRUTH)
            with self.subTest(family):
                self.assertLessEqual(abs(estimated - truth) / truth * 100.0,
                                     CORPUS_TOTAL_LIMIT_PCT)

    def test_it_beats_characters_over_four_on_the_corpus_total(self):
        for family in FAMILIES:
            estimated = 0
            naive = 0
            truth = 0
            for name in TRUTH:
                text = (CORPUS / name).read_text(encoding="utf-8")
                estimated += self.counters[family].count(text).tokens
                naive += naive_count(text)
                truth += TRUTH[name][family]
            with self.subTest(family):
                self.assertLess(abs(estimated - truth), abs(naive - truth))

    def test_it_beats_characters_over_four_on_most_individual_files(self):
        for family in FAMILIES:
            better = 0
            for name in TRUTH:
                text = (CORPUS / name).read_text(encoding="utf-8")
                truth = TRUTH[name][family]
                ours = abs(self.counters[family].count(text).tokens - truth)
                theirs = abs(naive_count(text) - truth)
                better += ours < theirs
            with self.subTest(family):
                self.assertGreaterEqual(better, len(TRUTH) - 2)

    def test_an_estimate_carries_a_band_and_an_exact_count_does_not(self):
        count = self.counters["cl100k_base"].count("hello world")
        self.assertEqual(count.method, "estimate")
        self.assertIsNotNone(count.band_pct)
        self.assertLess(count.low, count.high)
        self.assertIn("p95 error", count.describe())

    def test_the_shipped_band_matches_the_shipped_accuracy_record(self):
        for family in FAMILIES:
            counter = self.counters[family]
            with self.subTest(family):
                self.assertEqual(counter.count("some text").band_pct,
                                 counter.accuracy["p95_abs_pct"])

    def test_empty_text_counts_zero(self):
        self.assertEqual(self.counters["cl100k_base"].count("").tokens, 0)

    def test_an_unknown_family_fails_loudly(self):
        from ctxbudget.tokens import TableUnavailable
        with self.assertRaises(TableUnavailable):
            Counter("no-such-family")


if __name__ == "__main__":
    unittest.main()
