import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ctxbudget import cli

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "fixtures" / "corpus"
PROJECT = ROOT / "fixtures" / "project"


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_a_comfortable_fit_exits_zero_and_says_what_is_left(self):
        code, out, _ = run(["-m", "gpt-4o", "--no-exact", str(CORPUS / "prose_en.md")])
        self.assertEqual(code, cli.EXIT_FITS)
        self.assertIn("FITS", out)
        self.assertIn("reserved for the reply", out)

    def test_over_budget_exits_three_and_names_what_to_cut(self):
        argv = ["-m", "qwen2.5-7b-instruct", "--window", "8192", "--no-exact",
                "-q", "the retry backoff in the http client keeps retrying past the limit"]
        argv += [str(PROJECT / name) for name in
                 ("vendor/deps.lock", "docs/CHANGELOG.md", "src/http_client.py",
                  "src/retry_policy.py")]
        code, out, _ = run(argv)
        self.assertEqual(code, cli.EXIT_OVER)
        self.assertIn("OVER BUDGET", out)
        self.assertIn("CUT FIRST", out)
        self.assertIn("deps.lock", out.split("CUT FIRST")[1].splitlines()[1])

    def test_a_missing_file_exits_two_rather_than_counting_nothing(self):
        code, _, err = run(["-m", "gpt-4o", str(CORPUS / "does-not-exist.txt")])
        self.assertEqual(code, cli.EXIT_UNREADABLE)
        self.assertIn("CANNOT READ", err)

    def test_a_binary_file_exits_two_and_says_why(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
            handle.write(b"abc\x00def")
            name = handle.name
        try:
            code, _, err = run(["-m", "gpt-4o", name])
            self.assertEqual(code, cli.EXIT_UNREADABLE)
            self.assertIn("NUL byte", err)
        finally:
            Path(name).unlink()

    def test_a_directory_counts_the_files_under_it(self):
        code, out, _ = run(["-m", "gpt-4o", "--no-exact", "--json", str(PROJECT)])
        self.assertEqual(code, cli.EXIT_FITS)
        payload = json.loads(out)
        labels = {part["label"] for part in payload["parts"] if part["kind"] == "file"}
        self.assertIn(str(PROJECT / "src" / "http_client.py"), labels)
        self.assertIn(str(PROJECT / "vendor" / "deps.lock"), labels)
        self.assertEqual(len(labels), sum(1 for path in PROJECT.rglob("*") if path.is_file()))
        # The same files named one by one must come to the same number.
        argv = ["-m", "gpt-4o", "--no-exact", "--json"]
        argv += sorted(str(path) for path in PROJECT.rglob("*") if path.is_file())
        _, one_by_one, _ = run(argv)
        self.assertEqual(payload["input_tokens"], json.loads(one_by_one)["input_tokens"])

    def test_a_binary_file_inside_a_directory_is_skipped_out_loud(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "notes.md").write_text("a paragraph of ordinary text\n", encoding="utf-8")
            (root / "icon.png").write_bytes(b"\x89PNG\x00\x00binary")
            (root / "sub").mkdir()
            (root / "sub" / "deep.txt").write_text("more text\n", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "huge.js").write_text("x" * 5000, encoding="utf-8")
            code, out, err = run(["-m", "gpt-4o", "--no-exact", "--json", str(root)])
            self.assertEqual(code, cli.EXIT_FITS)
            payload = json.loads(out)
            labels = {part["label"] for part in payload["parts"] if part["kind"] == "file"}
            self.assertEqual(labels, {str(root / "notes.md"), str(root / "sub" / "deep.txt")})
            # Skipping silently is the failure this is guarding against, so the name and the
            # reason both have to appear.
            self.assertIn("SKIPPED", err)
            self.assertIn("icon.png", err)
            self.assertIn("NUL byte", err)
            self.assertNotIn("huge.js", err)

    def test_an_empty_directory_exits_two_rather_than_counting_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            code, _, err = run(["-m", "gpt-4o", folder])
            self.assertEqual(code, cli.EXIT_UNREADABLE)
            self.assertIn("CANNOT READ", err)

    def test_standard_input_is_counted_and_labelled(self):
        parts, skipped = cli.expand("-", "diff --git a/x b/x\n+one added line\n")
        self.assertEqual(skipped, [])
        self.assertEqual(parts[0][0], cli.STDIN_LABEL)
        self.assertIn("added line", parts[0][1])

    def test_standard_input_with_nothing_on_it_is_an_error(self):
        with self.assertRaises(ValueError) as caught:
            cli.expand("-", None)
        self.assertIn("standard input", str(caught.exception))

    def test_nothing_to_count_is_an_error_and_not_an_empty_pass(self):
        code, _, err = run(["-m", "gpt-4o"])
        self.assertEqual(code, cli.EXIT_UNREADABLE)
        self.assertIn("nothing to count", err)

    def test_an_unknown_model_exits_two_and_lists_the_known_ones(self):
        code, _, err = run(["-m", "gpt-9", str(CORPUS / "prose_en.md")])
        self.assertEqual(code, cli.EXIT_UNREADABLE)
        self.assertIn("unknown model", err)
        self.assertIn("gpt-4o", err)

    def test_json_output_carries_the_numbers_and_the_provenance(self):
        code, out, _ = run(["-m", "gpt-4o", "--json", "--no-exact",
                            str(CORPUS / "prose_en.md"), str(CORPUS / "code_python.py")])
        self.assertEqual(code, cli.EXIT_FITS)
        payload = json.loads(out)
        for key in ("window", "reserved_for_reply", "input_tokens", "usable_input_tokens",
                    "left_for_reply", "counting", "tokenizer", "cut_order",
                    "chars_over_four_would_say"):
            self.assertIn(key, payload)
        self.assertEqual(payload["counting"], "estimate")
        self.assertEqual(payload["usable_input_tokens"],
                         payload["window"] - payload["reserved_for_reply"])
        self.assertEqual(len(payload["cut_order"]), 2)

    def test_the_system_prompt_is_a_part_of_its_own(self):
        code, out, _ = run(["-m", "gpt-4o", "--json", "--no-exact",
                            "-s", str(CORPUS / "prose_en.md"),
                            str(CORPUS / "code_python.py")])
        self.assertEqual(code, cli.EXIT_FITS)
        payload = json.loads(out)
        kinds = [part["kind"] for part in payload["parts"]]
        self.assertIn("system", kinds)

    def test_cjk_heavy_input_gets_the_measured_warning(self):
        code, out, _ = run(["-m", "gpt-4o", "--no-exact", str(CORPUS / "japanese.txt")])
        self.assertEqual(code, cli.EXIT_FITS)
        self.assertIn("Han, Kana or Hangul", out)

    def test_ascii_input_does_not_get_the_cjk_warning(self):
        code, out, _ = run(["-m", "gpt-4o", "--no-exact", str(CORPUS / "prose_en.md")])
        self.assertEqual(code, cli.EXIT_FITS)
        self.assertNotIn("Han, Kana or Hangul", out)

    def test_explain_shows_what_characters_over_four_would_have_said(self):
        code, out, _ = run(["-m", "gpt-4o", "--no-exact", "--explain",
                            str(CORPUS / "numbers.csv")])
        self.assertEqual(code, cli.EXIT_FITS)
        self.assertIn("characters over four", out)


if __name__ == "__main__":
    unittest.main()
