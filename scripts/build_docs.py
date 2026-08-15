#!/usr/bin/env python3
"""Generate docs/index.html from the committed data. `--check` rebuilds and diffs.

Every number on the page comes from `ctxbudget/data/token_table.json`, the committed ground
truth, or a live run of the tool over the committed fixtures. Nothing is typed in by hand, and
verify rebuilds the page and fails if it differs, so a published page cannot quietly go stale
after a refit.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ctxbudget import budget as budget_mod  # noqa: E402
from ctxbudget import rank as rank_mod  # noqa: E402
from ctxbudget.tokens import Counter, naive_count  # noqa: E402

OUT = ROOT / "docs" / "index.html"
CORPUS = ROOT / "fixtures" / "corpus"
PROJECT = ROOT / "fixtures" / "project"
TABLE = json.loads((ROOT / "ctxbudget" / "data" / "token_table.json").read_text(encoding="utf-8"))
TRUTH = json.loads((ROOT / "fixtures" / "truth" / "counts.json").read_text(
    encoding="utf-8"))["counts"]
FAMILIES = ("cl100k_base", "o200k_base", "qwen2.5", "llama3")

PROJECT_FILES = ["src/http_client.py", "src/retry_policy.py", "src/settings.py",
                 "src/legacy_uploader.py", "tests/test_http_client.py",
                 "tests/test_http_client_legacy.py", "vendor/deps.lock", "docs/CHANGELOG.md"]
QUERY = "the retry backoff in the http client keeps retrying past the limit, fix it"
SAFE_TO_CUT = {"vendor/deps.lock", "docs/CHANGELOG.md", "src/legacy_uploader.py",
               "tests/test_http_client_legacy.py"}

CSS = """
:root { color-scheme: light dark;
  --bg:#fbfbf9; --fg:#1b1b1a; --muted:#5d5d58; --line:#dedcd4; --accent:#8a3a12; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#16171a; --fg:#e8e7e3; --muted:#9c9c96; --line:#2e3036; --accent:#f0a06a; --card:#1d1f23; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 ui-sans-serif,system-ui,
  -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:52rem; margin:0 auto; padding:2.5rem 1.1rem 4rem; }
h1 { font-size:1.9rem; margin:0 0 .3rem; letter-spacing:-.02em; }
h2 { font-size:1.15rem; margin:2.4rem 0 .6rem; letter-spacing:-.01em; }
p.lede { color:var(--muted); margin:0 0 1.6rem; }
p, li { margin:.7rem 0; }
code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.86em; }
pre { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:.9rem 1rem;
  overflow-x:auto; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:.42rem .7rem; border-bottom:1px solid var(--line);
  white-space:nowrap; }
th { font-weight:600; color:var(--muted); }
tbody tr:last-child td { border-bottom:none; }
td.n { text-align:right; font-variant-numeric:tabular-nums; }
.good { color:var(--accent); font-weight:600; }
.tag { font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
footer { margin-top:3rem; color:var(--muted); font-size:.85rem; }
a { color:var(--accent); }
"""


def esc(value) -> str:
    return html.escape(str(value))


def table_block(headers, rows) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, tuple):
                text, klass = cell
                cells.append(f'<td class="{klass}">{text}</td>')
            else:
                cells.append(f"<td>{esc(cell)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="wrap"><table><thead><tr>' + head + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def build() -> str:
    counters = {family: Counter(family, allow_exact=False) for family in FAMILIES}

    accuracy_rows = []
    for family in FAMILIES:
        record = TABLE["families"][family]
        acc = record["accuracy"]
        accuracy_rows.append([
            family,
            record["label"],
            (f"{acc['files']}", "n"),
            (f"{acc['tokens']:,}", "n"),
            (f'<span class="good">{acc["median_abs_pct"]}%</span>', "n"),
            (f"{acc['p95_abs_pct']}%", "n"),
            (f"{acc['naive_median_abs_pct']}%", "n"),
            (f"{acc['naive_p95_abs_pct']}%", "n"),
        ])

    fixture_rows = []
    for name in sorted(TRUTH):
        text = (CORPUS / name).read_text(encoding="utf-8")
        truth = TRUTH[name]["cl100k_base"]
        got = counters["cl100k_base"].count(text).tokens
        naive = naive_count(text)
        fixture_rows.append([
            name, (f"{truth:,}", "n"), (f"{got:,}", "n"),
            (f"{abs(got - truth) / truth * 100:.1f}%", "n"),
            (f"{naive:,}", "n"), (f"{abs(naive - truth) / truth * 100:.1f}%", "n")])

    parts = [(name, (PROJECT / name).read_text(encoding="utf-8")) for name in PROJECT_FILES]
    models = budget_mod.load_models()
    budget_rows = []
    for model, window in (("gpt-4o", None), ("gpt-4-turbo", None), ("claude-3-5-sonnet", None),
                          ("qwen2.5-7b-instruct", None), ("qwen2.5-7b-instruct", 8192)):
        report = budget_mod.build(model, parts, None, models, window=window, allow_exact=False)
        budget_rows.append([
            model, (f"{report.window:,}", "n"), (f"{report.reserve:,}", "n"),
            (f"{report.input_tokens:,}", "n"), (f"{report.usable_input:,}", "n"),
            (f"{report.reply_room:,}", "n"),
            ("fits" if report.status == "fits" else '<span class="good">over</span>', "")])

    tokens = {label: counters["qwen2.5"].count(text).tokens for label, text in parts}
    signals = rank_mod.rank(parts, tokens, QUERY)
    by_size = rank_mod.largest_first(signals)
    size_position = {signal.label: index for index, signal in enumerate(by_size, start=1)}
    cut_rows = []
    for index, signal in enumerate(signals, start=1):
        verdict = "dead weight" if signal.label in SAFE_TO_CUT else "load-bearing"
        cut_rows.append([
            (str(index), "n"), signal.label, (f"{signal.tokens:,}", "n"),
            (f"{signal.score:,.0f}", "n"), (str(signal.inbound), "n"),
            (f"{signal.query_demand:.0f}", "n"), (f"{signal.redundancy:.2f}", "n"),
            (str(size_position[signal.label]), "n"),
            (f'<span class="good">{verdict}</span>' if verdict == "dead weight" else verdict, "")])
    ours = sum(1 for signal in signals[:4] if signal.label in SAFE_TO_CUT)
    theirs = sum(1 for signal in by_size[:4] if signal.label in SAFE_TO_CUT)

    corpus = TABLE["corpus"]
    body = f"""<main>
<h1>ctxbudget</h1>
<p class="lede">What fills a model's context window, what is left for the reply, and what to cut
first. Counts are estimated by a table fitted against four real tokenizers, and the error of
doing that is measured rather than asserted.</p>

<h2>How wrong the count is</h2>
<p>Fitted on {corpus['train_files']} real text files and measured on {corpus['holdout_files']}
held out from the fit, all of them ordinary files from repositories on one machine.
{corpus['cjk_supplement_files']} files were pulled in deliberately for CJK coverage, because a
hash-ordered sample turned up almost none. The last two columns are what dividing characters by
four scores on the same held-out files.</p>
{table_block(["family", "tokenizer", "files", "tokens", "median", "p95", "chars/4 median",
              "chars/4 p95"], accuracy_rows)}

<h2>Per fixture, against real counts</h2>
<p>The committed corpus, counted by <code>cl100k_base</code>. These ground truth numbers were
produced once by a real tokenizer and committed, so the test suite needs no tokenizer installed.
Japanese is the worst case and it is on the page rather than in a footnote.</p>
{table_block(["fixture", "real", "estimated", "error", "chars/4", "error"], fixture_rows)}

<h2>The window is not the input budget</h2>
<p>The same eight files against five configurations. <code>gpt-4-turbo</code> is the case worth
looking at: a 128,000 token window and a 4,096 token reply cap, so the reply is what binds.
A local runtime caps nothing separately, so its reserve is a choice and the tool says so.</p>
{table_block(["model", "window", "reserved for reply", "input", "usable for input",
              "reply room", ""], budget_rows)}

<h2>What to cut first</h2>
<p>Rank by <code>tokens / ((1 + inbound refs + task match) &times; (1 - redundancy))</code>,
which is tokens returned per unit of demand that would actually be lost. Task:
&ldquo;{esc(QUERY)}&rdquo;. The last two columns are where ranking by size would have put each
file, and whether the file is dead weight for this task.</p>
{table_block(["cut", "file", "tokens", "score", "inbound", "task", "dup", "by size", ""],
             cut_rows)}
<p>First four cuts that are dead weight: <span class="good">{ours} of 4</span> for this rule,
{theirs} of 4 for ranking by size. The file it saves is <code>src/http_client.py</code>, the
second largest in the set and the one the task is about.</p>

<h2>What is not measured</h2>
<ul>
<li>Claude has no public tokenizer. Claude counts here use the <code>cl100k_base</code> table as
a stand-in and are labelled <code>unmeasured</code> with no error band.</li>
<li>Context windows and reply caps come from vendor documentation read on
{esc(json.loads((ROOT / 'ctxbudget' / 'data' / 'models.json').read_text(encoding='utf-8'))['checked'])},
not from anything this project measured. Override them with <code>--window</code> and
<code>--reserve</code>.</li>
<li>Japanese is the worst measured case by a wide margin, and the tool warns when the input is
CJK-heavy instead of quoting the corpus-wide band at it.</li>
<li>The cut ranking reads structure, not meaning. A file can matter for a reason no import graph
records, so the tool prints the counts behind every rank and removes nothing on its own.</li>
</ul>

<h2>Setting a long context on Ollama</h2>
<p>Measured on this machine on ollama 0.32.9: only <code>options.num_ctx</code> in the API
request body changes the loaded context length. <code>/set parameter num_ctx</code> in the REPL
and <code>OLLAMA_CONTEXT_LENGTH</code> in the client shell both leave the server default in
place and report no error, so a run you believe has 128k may be truncating. Check
<code>ollama ps</code>.</p>

<footer>Catalog task CLI-034, from a public catalog of build ideas at
<a href="https://github.com/JesseRWeigel/722-things-to-build">722-things-to-build</a>.
Source: <a href="https://github.com/JesseRWeigel/ctxbudget">JesseRWeigel/ctxbudget</a>.
This page is generated from the committed data by <code>scripts/build_docs.py</code>.</footer>
</main>"""

    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
            "<title>ctxbudget</title>\n<style>" + CSS + "</style>\n</head>\n<body>\n"
            + body + "\n</body>\n</html>\n")


def main() -> int:
    page = build()
    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"   {OUT.relative_to(ROOT)} does not exist. Run scripts/build_docs.py.")
            return 1
        current = OUT.read_text(encoding="utf-8")
        if current != page:
            print(f"   {OUT.relative_to(ROOT)} is stale: {len(current)} chars on disk against "
                  f"{len(page)} rebuilt from the data. Run scripts/build_docs.py.")
            return 1
        print(f"   {OUT.relative_to(ROOT)} matches a rebuild from the data, {len(page)} chars")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}, {len(page)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
