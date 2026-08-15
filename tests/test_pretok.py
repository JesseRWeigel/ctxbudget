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
        self.assertEqual(pretok.classify(" hello"), pretok.WORD_ASCII)
        self.assertEqual(pretok.classify("café"), pretok.WORD_WIDE)
        self.assertEqual(pretok.classify("日本語"), pretok.WORD_CJK)
        self.assertEqual(pretok.classify("한국어"), pretok.WORD_CJK)
        self.assertEqual(pretok.classify("123"), pretok.NUMBER)
        self.assertEqual(pretok.classify(" ==="), pretok.PUNCT_ASCII)
        self.assertEqual(pretok.classify("\n\n"), pretok.SPACE)

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
