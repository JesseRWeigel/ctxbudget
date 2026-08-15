import unittest
from pathlib import Path

from ctxbudget import pretok

CORPUS = Path(__file__).resolve().parent.parent / "fixtures" / "corpus"


class PretokTest(unittest.TestCase):
    def test_split_is_lossless_on_every_fixture(self):
        for path in sorted(CORPUS.iterdir()):
            with self.subTest(path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual("".join(pretok.chunks(text)), text)

    def test_chunk_count_never_exceeds_the_real_token_count(self):
        # Merges only happen inside a chunk, so chunks are a lower bound on tokens. If this ever
        # fails the regex has stopped resembling a real pretokenizer.
        import json
        truth = json.loads(
            (CORPUS.parent / "truth" / "counts.json").read_text(encoding="utf-8"))["counts"]
        for name, counts in truth.items():
            text = (CORPUS / name).read_text(encoding="utf-8")
            chunks = len(pretok.chunks(text))
            for family, tokens in counts.items():
                with self.subTest(f"{name}/{family}"):
                    self.assertLessEqual(chunks, tokens)

    def test_classes_are_assigned_as_documented(self):
        self.assertEqual(pretok.classify("hello"), pretok.WORD_ASCII)
        self.assertEqual(pretok.classify("café"), pretok.WORD_WIDE)
        self.assertEqual(pretok.classify("日本語"), pretok.WORD_CJK)
        self.assertEqual(pretok.classify("한국어"), pretok.WORD_HANGUL)
        self.assertEqual(pretok.classify("123"), pretok.NUMBER)
        self.assertEqual(pretok.classify(" ==="), pretok.PUNCT_ASCII)
        self.assertEqual(pretok.classify("\n\n"), pretok.SPACE)

    def test_a_word_is_classified_by_what_it_starts_with(self):
        # " window" is one token and ",window" is two, so they cannot share a bucket.
        self.assertEqual(pretok.classify(" hello"), pretok.WORD_ASCII + pretok.LEAD_SPACE)
        self.assertEqual(pretok.classify(",hello"), pretok.WORD_ASCII + pretok.LEAD_PUNCT)
        self.assertEqual(pretok.classify(" 日本語"), pretok.WORD_CJK + pretok.LEAD_SPACE)
        self.assertEqual(pretok.classify("、日本語"), pretok.WORD_CJK + pretok.LEAD_PUNCT)
        self.assertEqual(pretok.classify(" 한국어"), pretok.WORD_HANGUL + pretok.LEAD_SPACE)
        for cls in (pretok.classify(" hello"), pretok.classify(",hello")):
            self.assertIn(cls, pretok.CLASSES)
            self.assertIn(cls, pretok.CAPS)

    def test_a_random_run_is_split_by_case_and_only_inside_the_run(self):
        blob = "sha512-aBcD1234efGH5678ijKL90mnOP"
        classes = {cls for _, cls, _ in pretok.walk(blob)}
        self.assertIn(pretok.RANDOM_UPPER, classes)
        self.assertIn(pretok.RANDOM_LOWER, classes)
        # The identical letters outside a run are ordinary words, not random-run pieces.
        outside = {cls for _, cls, _ in pretok.walk("aBcD efGH")}
        self.assertNotIn(pretok.RANDOM_UPPER, outside)
        self.assertNotIn(pretok.RANDOM_LOWER, outside)

    def test_a_chunk_over_the_cap_is_charged_an_overflow_offset_as_well_as_a_rate(self):
        short = pretok.features("日本")
        long_run = pretok.features("日本語" * 20)
        self.assertNotIn(pretok.over_key(pretok.WORD_CJK), short)
        self.assertEqual(long_run[pretok.over_key(pretok.WORD_CJK)], 1.0)
        self.assertEqual(long_run[pretok.tail_key(pretok.WORD_CJK)],
                         len(("日本語" * 20).encode()) - pretok.CAPS[pretok.WORD_CJK])

    def test_byte_length_not_character_length_drives_the_bucket(self):
        self.assertEqual(pretok.bucket_key(pretok.WORD_CJK, 9), "word_cjk:9")
        features = pretok.features("日本")
        self.assertIn("word_cjk:6", features)  # two characters, six bytes

    def test_empty_chunk_is_refused_rather_than_silently_classified(self):
        with self.assertRaises(ValueError):
            pretok.classify("")

    def test_features_sum_to_something_for_every_fixture(self):
        for path in sorted(CORPUS.iterdir()):
            with self.subTest(path.name):
                self.assertGreater(sum(pretok.features(
                    path.read_text(encoding="utf-8")).values()), 0)


if __name__ == "__main__":
    unittest.main()
