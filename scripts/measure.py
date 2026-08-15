#!/usr/bin/env python3
"""Every headline number this project claims, printed once and hashed.

The fingerprint is what the sabotage harness compares. Two properties matter and both are
deliberate:

  - it depends only on the repository's own content, never on where the repository sits. No
    absolute path, no timestamp, no hostname reaches the hash. A fingerprint that tracks the
    working directory hands gate 2 to every sabotage for free, which is how eleven sabotages in
    this fleet once scored as proven while proving nothing.
  - it covers all three claims: the token counts, the budget arithmetic, and the cut order. A
    fingerprint over only one of them cannot catch a sabotage of the other two.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ctxbudget import budget as budget_mod  # noqa: E402
from ctxbudget import rank as rank_mod  # noqa: E402
from ctxbudget.tokens import Counter, naive_count  # noqa: E402

CORPUS = ROOT / "fixtures" / "corpus"
PROJECT = ROOT / "fixtures" / "project"
TRUTH = json.loads((ROOT / "fixtures" / "truth" / "counts.json").read_text(
    encoding="utf-8"))["counts"]
FAMILIES = ("cl100k_base", "o200k_base", "qwen2.5", "llama3")

PROJECT_FILES = [
    "src/http_client.py", "src/retry_policy.py", "src/settings.py", "src/legacy_uploader.py",
    "tests/test_http_client.py", "tests/test_http_client_legacy.py",
    "vendor/deps.lock", "docs/CHANGELOG.md",
]
QUERY = "the retry backoff in the http client keeps retrying past the limit, fix it"
SAFE_TO_CUT = {"vendor/deps.lock", "docs/CHANGELOG.md", "src/legacy_uploader.py",
               "tests/test_http_client_legacy.py"}


def main() -> int:
    lines: list[str] = []

    # ---------------------------------------------------------------- counting
    counters = {family: Counter(family, allow_exact=False) for family in FAMILIES}
    lines.append("COUNTING  estimator against committed real token counts, per family")
    for family in FAMILIES:
        estimated = naive = truth = 0
        worst = (0.0, "")
        for name in sorted(TRUTH):
            text = (CORPUS / name).read_text(encoding="utf-8")
            got = counters[family].count(text).tokens
            estimated += got
            naive += naive_count(text)
            truth += TRUTH[name][family]
            error = abs(got - TRUTH[name][family]) / TRUTH[name][family] * 100
            if error > worst[0]:
                worst = (error, name)
        lines.append(
            f"  {family:<12} estimated {estimated:>6}  truth {truth:>6}  "
            f"error {abs(estimated - truth) / truth * 100:6.2f}%  "
            f"chars/4 error {abs(naive - truth) / truth * 100:6.2f}%  "
            f"worst file {worst[1]} at {worst[0]:.1f}%")

    lines.append("")
    lines.append("PER FIXTURE  absolute percent error, cl100k_base")
    for name in sorted(TRUTH):
        text = (CORPUS / name).read_text(encoding="utf-8")
        truth = TRUTH[name]["cl100k_base"]
        got = counters["cl100k_base"].count(text).tokens
        lines.append(f"  {name:<20} {got:>6} vs {truth:>6}  "
                     f"{abs(got - truth) / truth * 100:6.2f}%")

    # ---------------------------------------------------------------- budget
    lines.append("")
    lines.append("BUDGET  the fixture project against three models")
    parts = [(name, (PROJECT / name).read_text(encoding="utf-8")) for name in PROJECT_FILES]
    models = budget_mod.load_models()
    for model, window in (("gpt-4o", None), ("gpt-4-turbo", None),
                          ("qwen2.5-7b-instruct", 8192)):
        report = budget_mod.build(model, parts, None, models, window=window, allow_exact=False)
        lines.append(f"  {model:<22} window {report.window:>7}  reserve {report.reserve:>6}  "
                     f"input {report.input_tokens:>6}  usable {report.usable_input:>7}  "
                     f"reply room {report.reply_room:>6}  {report.status}"
                     f"{' TIGHT' if report.tight else ''}")

    # ---------------------------------------------------------------- ranking
    lines.append("")
    lines.append("CUT ORDER  fixture project, ours against ranking by size")
    tokens = {label: counters["qwen2.5"].count(text).tokens for label, text in parts}
    signals = rank_mod.rank(parts, tokens, QUERY)
    for index, signal in enumerate(signals, start=1):
        verdict = "dead-weight" if signal.label in SAFE_TO_CUT else "LOAD-BEARING"
        lines.append(f"  {index}. {signal.label:<38} {signal.tokens:>6} tokens  "
                     f"score {signal.score:>9.1f}  inbound {signal.inbound}  "
                     f"query {signal.query_demand:>4.1f}  dup {signal.redundancy:.2f}  {verdict}")
    ours = sum(1 for signal in signals[:4] if signal.label in SAFE_TO_CUT)
    naive_order = rank_mod.largest_first(signals)
    theirs = sum(1 for signal in naive_order[:4] if signal.label in SAFE_TO_CUT)
    lines.append(f"  first four cuts are dead weight: ours {ours}/4, ranking by size {theirs}/4")
    lines.append("  by size:  " + ", ".join(signal.label for signal in naive_order[:4]))

    body = "\n".join(lines)
    print(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print()
    print(f"FINGERPRINT {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
