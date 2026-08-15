import unittest

from ctxbudget import budget as budget_mod

MODELS = budget_mod.load_models()


def build(files, **kwargs):
    kwargs.setdefault("system_prompt", None)
    return budget_mod.build(kwargs.pop("model", "gpt-4o"), files,
                            kwargs.pop("system_prompt"), MODELS, allow_exact=False, **kwargs)


class ReserveTest(unittest.TestCase):
    def test_a_separate_output_cap_becomes_the_reserve(self):
        report = build([("a.txt", "hello")], model="gpt-4-turbo")
        self.assertEqual(report.reserve, 4096)
        self.assertIn("model table", report.reserve_source)

    def test_a_shared_window_gets_a_quarter_and_says_it_was_a_choice(self):
        report = build([("a.txt", "hello")], model="qwen2.5-7b-instruct")
        self.assertEqual(report.reserve, 32768 // 4)
        self.assertIn("chosen by this tool", report.reserve_source)

    def test_the_caller_always_wins(self):
        report = build([("a.txt", "hello")], model="gpt-4o", reserve=1000)
        self.assertEqual(report.reserve, 1000)
        self.assertIn("command line", report.reserve_source)


class ArithmeticTest(unittest.TestCase):
    def test_usable_input_is_the_window_minus_the_reserve(self):
        report = build([("a.txt", "hello")], model="gpt-4o", window=1000, reserve=400)
        self.assertEqual(report.usable_input, 600)

    def test_over_budget_is_measured_against_usable_input_not_the_window(self):
        # The classic mistake: input under the window, so it looks fine, but the reply has
        # nowhere to go.
        big = "word " * 400
        report = build([("a.txt", big)], model="gpt-4o", window=1000, reserve=600)
        self.assertGreater(report.input_tokens, 400)
        self.assertLess(report.input_tokens, 1000)
        self.assertEqual(report.status, budget_mod.OVER)
        self.assertEqual(report.over_by, report.input_tokens - 400)

    def test_reply_room_shrinks_before_the_request_is_refused(self):
        big = "word " * 400
        report = build([("a.txt", big)], model="gpt-4o", window=1000, reserve=600)
        self.assertGreater(report.reply_room, 0)
        self.assertLess(report.reply_room, report.reserve)

    def test_reply_room_is_zero_once_the_input_fills_the_window(self):
        report = build([("a.txt", "word " * 4000)], model="gpt-4o", window=1000, reserve=600)
        self.assertEqual(report.reply_room, 0)

    def test_a_fit_inside_the_error_band_is_reported_as_tight(self):
        text = "word " * 200
        rough = build([("a.txt", text)], model="gpt-4o", window=10_000, reserve=100)
        tight = build([("a.txt", text)], model="gpt-4o",
                      window=rough.input_tokens + 105, reserve=100)
        self.assertEqual(tight.status, budget_mod.FITS)
        self.assertTrue(tight.tight)
        self.assertFalse(rough.tight)

    def test_input_total_includes_the_system_prompt_and_the_template_overhead(self):
        without = build([("a.txt", "hello there")], model="gpt-4o")
        with_system = build([("a.txt", "hello there")], model="gpt-4o",
                            system_prompt="you are a helpful assistant")
        self.assertGreater(with_system.input_tokens, without.input_tokens)
        kinds = {part.kind for part in with_system.parts}
        self.assertEqual(kinds, {"system", "file", "overhead"})

    def test_more_messages_cost_more_overhead(self):
        one = build([("a.txt", "hello")], model="gpt-4o", messages=1)
        ten = build([("a.txt", "hello")], model="gpt-4o", messages=10)
        self.assertGreater(ten.input_tokens, one.input_tokens)


class ModelTableTest(unittest.TestCase):
    def test_every_entry_has_the_fields_the_report_prints(self):
        for name, spec in MODELS["models"].items():
            with self.subTest(name):
                self.assertIn(spec["family"], MODELS["families"])
                self.assertGreater(spec["window"], 0)
                self.assertIn("source", spec)
                if spec["output_capped_separately"]:
                    self.assertIsNotNone(spec["max_output"])
                    self.assertLess(spec["max_output"], spec["window"])

    def test_an_unknown_model_fails_unless_it_is_described(self):
        with self.assertRaises(budget_mod.UnknownModel):
            build([("a.txt", "hello")], model="not-a-real-model")
        report = build([("a.txt", "hello")], model="not-a-real-model",
                       window=4096, family="llama3")
        self.assertEqual(report.window, 4096)

    def test_claude_is_reported_as_unmeasured_with_no_error_band(self):
        report = build([("a.txt", "hello")], model="claude-3-5-sonnet")
        self.assertEqual(report.counting, "unmeasured")
        self.assertIsNone(report.band_pct)
        self.assertTrue(any("no tokenizer" in warning for warning in report.warnings))

    def test_a_local_model_carries_the_num_ctx_warning(self):
        report = build([("a.txt", "hello")], model="llama-3.1-8b-instruct")
        self.assertTrue(any("num_ctx" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
