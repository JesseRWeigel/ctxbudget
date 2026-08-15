#!/usr/bin/env python3
"""A second opinion that imports nothing from the package under test.

Two things happen here.

First it proves its own independence by walking its import graph with `ast`, not with grep. A
grep is satisfied by a comment and misses `importlib.import_module("ctxbudget.tokens")`. The
prover is then shown three probe files it must reject and one it must accept, because a prover
that has never rejected anything is a prover nobody has tested.

Then it recomputes the headline numbers by a different route. The tool is run as a subprocess and
only its JSON is read. Token counts are recomputed from the committed table by a hand-rolled
character scanner that shares no code and not even an approach with the regex the package uses.
The chars-over-four comparison is recomputed straight from the fixture bytes. The redundancy
between the two near-identical test files is recomputed with `difflib`, which has nothing to do
with the shingled Jaccard the package uses. And the budget arithmetic is re-derived from the
reported parts.

Two negative controls run at the end. A deliberately broken recount must DISAGREE with the tool,
and a scanner with its tail term removed must fail the agreement bound. If either agrees anyway,
the comparison is vacuous and nothing above it means anything.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "scripts", "probes")
CORPUS = os.path.join(ROOT, "fixtures", "corpus")
PROJECT = os.path.join(ROOT, "fixtures", "project")
TABLE = os.path.join(ROOT, "ctxbudget", "data", "token_table.json")
TRUTH = os.path.join(ROOT, "fixtures", "truth", "counts.json")
FORBIDDEN = "ctxbudget"

FAMILIES = ("cl100k_base", "o200k_base", "qwen2.5", "llama3")
#: How far the independent recount may sit from the real token counts. The estimator's own
#: measured error lives inside this.
AGREEMENT_LIMIT_PCT = 6.0
#: How far the independent scanner may sit from the package on the same file. These are two
#: implementations of one documented split, so they should agree to the token.
IMPLEMENTATION_LIMIT_PCT = 0.5

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"   FAIL {message}")


# ---------------------------------------------------------------- independence, proved with ast

def imported_names(path: str) -> set[str]:
    """Every module name this file imports, including the dynamic ones."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import escapes this file's package. It cannot be shown to be clean.
                names.add("." * node.level + (node.module or ""))
            elif node.module:
                names.add(node.module)
        elif isinstance(node, ast.Call):
            function = node.func
            dotted = ""
            if isinstance(function, ast.Attribute):
                dotted = function.attr
            elif isinstance(function, ast.Name):
                dotted = function.id
            if dotted in ("import_module", "__import__") and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    names.add(argument.value)
    return names


def is_clean(path: str) -> bool:
    for name in imported_names(path):
        if name.startswith("."):
            return False
        root = name.split(".")[0]
        if root == FORBIDDEN:
            return False
    return True


print("== independence")
if is_clean(__file__):
    print(f"   this file imports nothing from {FORBIDDEN}, proved by walking its AST")
else:
    fail(f"this file imports {FORBIDDEN}")

for probe, expected in (("probe_clean.py", True), ("probe_imports.py", False),
                        ("probe_computed.py", False), ("probe_relative.py", False)):
    path = os.path.join(PROBES, probe)
    if not os.path.exists(path):
        fail(f"probe {probe} is missing, so the prover was never tested")
        continue
    got = is_clean(path)
    verdict = "accepted" if got else "rejected"
    if got == expected:
        print(f"   {probe:<20} {verdict} as required")
    else:
        fail(f"{probe} was {verdict} and should not have been")


# ---------------------------------------------------------------- a second, unrelated scanner

def _cjk(ch: str) -> bool:
    point = ord(ch)
    for low, high in ((0x2E80, 0x9FFF), (0xA960, 0xA97F), (0xAC00, 0xD7FF),
                      (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
                      (0x1B000, 0x1B2FF), (0x20000, 0x3FFFF)):
        if low <= point <= high:
            return True
    return False


def split(text: str) -> list[str]:
    """Split text the way the documented pretokenizer does, without a regular expression.

    The package compiles one alternation and lets Python's engine pick a branch. This walks the
    same documented branches by hand, in the same order, with an explicit index. No shared code
    and not even a shared mechanism, which is the point: a validator built out of the thing it
    validates cannot catch that thing's bugs.

        1. one optional character that is not CR, LF, a letter or a digit, then letters
        2. one to three digits
        3. one optional space, then non-space non-letter non-digit characters, then any CR or LF
        4. whitespace ending in one or more CR or LF
        5. a whitespace run minus its last character, when something non-space follows
        6. a whitespace run
    """
    chunks: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        ch = text[index]

        # 1. letters, with at most one leading non-letter that is not a line break
        if ch.isalpha():
            end = index
            while end < size and text[end].isalpha():
                end += 1
            chunks.append(text[index:end])
            index = end
            continue
        if not (ch in "\r\n" or ch.isdigit()):
            end = index + 1
            while end < size and text[end].isalpha():
                end += 1
            if end > index + 1:
                chunks.append(text[index:end])
                index = end
                continue

        # 2. up to three digits
        if ch.isdigit():
            end = index
            while end < size and text[end].isdigit() and end - index < 3:
                end += 1
            chunks.append(text[index:end])
            index = end
            continue

        # 3. punctuation, with at most one leading space, then any trailing line breaks
        body = index + 1 if ch == " " else index
        end = body
        while end < size and not (text[end].isspace() or text[end].isalpha()
                                  or text[end].isdigit()):
            end += 1
        if end > body:
            while end < size and text[end] in "\r\n":
                end += 1
            chunks.append(text[index:end])
            index = end
            continue

        # whitespace runs, branches 4 to 6
        run_end = index
        while run_end < size and text[run_end].isspace():
            run_end += 1
        run = text[index:run_end]
        last_break = max(run.rfind("\n"), run.rfind("\r"))
        if last_break >= 0:
            chunks.append(run[:last_break + 1])
            index += last_break + 1
        elif run_end == size or len(run) < 2:
            chunks.append(run)
            index = run_end
        else:
            chunks.append(run[:-1])
            index = run_end - 1
    return chunks


def random_runs(text: str) -> list[tuple[int, int]]:
    """Spans of 20 or more alphanumeric characters holding both a letter and a digit.

    Found by walking the string, not with a regular expression. These are integrity hashes,
    base64 blobs and content-addressed names, and the chunks inside them cost far more per byte
    than the identical chunks would cost inside a sentence.
    """
    allowed = set("+/=_-")
    spans: list[tuple[int, int]] = []
    index = 0
    size = len(text)
    while index < size:
        ch = text[index]
        if ch.isalnum() or ch in allowed:
            end = index
            while end < size and (text[end].isalnum() or text[end] in allowed):
                end += 1
            body = text[index:end]
            if (len(body) >= 20 and any(c.isdigit() for c in body)
                    and any(c.isalpha() for c in body)):
                spans.append((index, end))
            index = end
        else:
            index += 1
    return spans


def scan(text: str) -> dict[str, float]:
    """Bucket the split chunks the way the committed table is keyed."""
    caps = {"word_ascii": 16, "word_wide": 12, "word_cjk": 12, "number": 4,
            "punct_ascii": 10, "punct_wide": 8, "space": 10, "random": 12}
    spans = random_runs(text)
    buckets: dict[str, float] = {}
    offset = 0
    span_index = 0
    for chunk in split(text):
        start = offset
        offset += len(chunk)
        while span_index < len(spans) and spans[span_index][1] <= start:
            span_index += 1
        if (span_index < len(spans) and spans[span_index][0] <= start
                and offset <= spans[span_index][1]):
            nbytes = len(chunk.encode("utf-8"))
            cap = caps["random"]
            key = f"random:{min(nbytes, cap)}"
            buckets[key] = buckets.get(key, 0.0) + 1.0
            if nbytes > cap:
                buckets["random:tail"] = buckets.get("random:tail", 0.0) + (nbytes - cap)
            continue
        stripped = chunk.strip()
        wide = not chunk.isascii()
        core = stripped[1:] if stripped and not stripped[0].isalpha() else stripped

        if not stripped:
            cls = "space"
        elif any(_cjk(ch) for ch in chunk):
            cls = "word_cjk"
        elif stripped[0].isdigit():
            cls = "number"
        elif core and all(ch.isalpha() for ch in core):
            cls = "word_wide" if wide else "word_ascii"
        else:
            cls = "punct_wide" if wide else "punct_ascii"
        nbytes = len(chunk.encode("utf-8"))
        cap = caps[cls]
        key = f"{cls}:{min(nbytes, cap)}"
        buckets[key] = buckets.get(key, 0.0) + 1.0
        if nbytes > cap:
            buckets[f"{cls}:tail"] = buckets.get(f"{cls}:tail", 0.0) + (nbytes - cap)
    return buckets


def independent_count(table: dict, family: str, text: str, use_tail: bool = True) -> int:
    weights = table["tables"][family]
    total = 0.0
    for key, count in scan(text).items():
        if not use_tail and key.endswith(":tail"):
            continue
        total += weights.get(key, 0.25) * count
    return max(0, round(total))


# ---------------------------------------------------------------- the recomputation

print()
print("== recounting the fixtures by a different route")
with open(TABLE, encoding="utf-8") as handle:
    table = json.load(handle)
with open(TRUTH, encoding="utf-8") as handle:
    truth = json.load(handle)["counts"]

corpus_files = sorted(truth)
texts = {}
for name in corpus_files:
    with open(os.path.join(CORPUS, name), encoding="utf-8") as handle:
        texts[name] = handle.read()

for family in FAMILIES:
    mine = sum(independent_count(table, family, texts[name]) for name in corpus_files)
    real = sum(truth[name][family] for name in corpus_files)
    naive = sum(round(len(texts[name]) / 4) for name in corpus_files)
    my_error = abs(mine - real) / real * 100
    naive_error = abs(naive - real) / real * 100
    print(f"   {family:<12} independent {mine:>6}  truth {real:>6}  error {my_error:5.2f}%  "
          f"chars/4 error {naive_error:5.2f}%")
    if my_error > AGREEMENT_LIMIT_PCT:
        fail(f"the independent recount of {family} is {my_error:.2f}% off the real counts, "
             f"over the {AGREEMENT_LIMIT_PCT}% bound. Either the table is wrong or the "
             f"package's own splitter disagrees with a plainly written one.")
    if my_error >= naive_error:
        fail(f"for {family} an independent implementation of the documented method does not "
             f"beat dividing characters by four, so the method itself is not carrying the "
             f"result")

# ---------------------------------------------------------------- against the tool's own output

print()
print("== against the tool, run as a subprocess and read only through its JSON")
project_files = ["src/http_client.py", "src/retry_policy.py", "src/settings.py",
                 "src/legacy_uploader.py", "tests/test_http_client.py",
                 "tests/test_http_client_legacy.py", "vendor/deps.lock", "docs/CHANGELOG.md"]
argv = [sys.executable, "-m", "ctxbudget", "-m", "qwen2.5-7b-instruct", "--window", "8192",
        "--no-exact", "--json", "-q",
        "the retry backoff in the http client keeps retrying past the limit, fix it"]
argv += [os.path.join("fixtures", "project", name) for name in project_files]
result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ROOT,
                             "PYTHONDONTWRITEBYTECODE": "1"})
if result.returncode not in (0, 3):
    fail(f"the tool exited {result.returncode}: {result.stderr.strip()[:200]}")
    payload = None
else:
    payload = json.loads(result.stdout)

if payload:
    reported = {part["label"]: part["tokens"] for part in payload["parts"]}

    # 1. the parts must add up to the input total
    if sum(reported.values()) != payload["input_tokens"]:
        fail(f"the parts sum to {sum(reported.values())} and the report says "
             f"{payload['input_tokens']}")
    else:
        print(f"   parts sum to the reported input total, {payload['input_tokens']}")

    # 2. usable input and reply room, re-derived
    usable = payload["window"] - payload["reserved_for_reply"]
    reply = max(0, min(payload["reserved_for_reply"],
                       payload["window"] - payload["input_tokens"]))
    over = max(0, payload["input_tokens"] - usable)
    for label, mine_value, theirs in (("usable input", usable, payload["usable_input_tokens"]),
                                      ("left for reply", reply, payload["left_for_reply"]),
                                      ("over by", over, payload["over_by"])):
        if mine_value != theirs:
            fail(f"{label}: independent arithmetic says {mine_value}, the tool says {theirs}")
    print(f"   window {payload['window']} minus reserve {payload['reserved_for_reply']} "
          f"leaves {usable} for input, and the tool agrees")
    if payload["status"] != ("fits" if over == 0 else "over"):
        fail(f"status {payload['status']} disagrees with the re-derived arithmetic")

    # 3. every per-file count, recomputed by the independent scanner
    worst = (0.0, "")
    for name in project_files:
        label = os.path.join("fixtures", "project", name)
        with open(os.path.join(PROJECT, name), encoding="utf-8") as handle:
            body = handle.read()
        mine_value = independent_count(table, "qwen2.5", body)
        theirs = reported[label]
        gap = abs(mine_value - theirs) / max(1, theirs) * 100
        if gap > worst[0]:
            worst = (gap, name)
    if worst[0] > IMPLEMENTATION_LIMIT_PCT:
        fail(f"the independent scanner disagrees with the tool by {worst[0]:.1f}% on "
             f"{worst[1]}, over the {IMPLEMENTATION_LIMIT_PCT}% bound. Two implementations of "
             f"one documented split should not diverge.")
    else:
        print(f"   all {len(project_files)} per-file counts reproduced to within "
              f"{worst[0]:.1f}% by a scanner that shares no code with the package")

    # 4. redundancy, recomputed with difflib rather than shingled Jaccard
    with open(os.path.join(PROJECT, "tests/test_http_client.py"), encoding="utf-8") as handle:
        left = handle.readlines()
    with open(os.path.join(PROJECT, "tests/test_http_client_legacy.py"),
              encoding="utf-8") as handle:
        right = handle.readlines()
    ratio = difflib.SequenceMatcher(None, left, right).ratio()
    claimed = next(entry["redundancy"] for entry in payload["cut_order"]
                   if entry["label"].endswith("test_http_client_legacy.py"))
    print(f"   difflib puts the two test files at {ratio:.2f} similar, the tool claims "
          f"{claimed:.2f} redundancy")
    if ratio < 0.5 or claimed < 0.5:
        fail("the two test fixtures were supposed to be near duplicates and one of the two "
             "measures says they are not")

    # 5. the cut order beats ranking by size, decided from scratch
    safe = {"fixtures/project/vendor/deps.lock", "fixtures/project/docs/CHANGELOG.md",
            "fixtures/project/src/legacy_uploader.py",
            "fixtures/project/tests/test_http_client_legacy.py"}
    order = [entry["label"] for entry in payload["cut_order"]]
    by_size = sorted((entry for entry in payload["cut_order"]),
                     key=lambda entry: -entry["tokens"])
    ours = sum(1 for label in order[:4] if label in safe)
    theirs = sum(1 for entry in by_size[:4] if entry["label"] in safe)
    print(f"   first four cuts that are dead weight: tool {ours}/4, ranking by size {theirs}/4")
    if ours != 4:
        fail(f"the tool put a load-bearing file in its first four cuts: {order[:4]}")
    if theirs >= ours:
        fail("ranking by size did just as well on this fixture, so the fixture does not "
             "demonstrate anything and the comparison is vacuous")

# ---------------------------------------------------------------- negative controls

print()
print("== negative controls, these must DISAGREE")
broken = sum(round(len(texts[name]) / 8) for name in corpus_files)
real = sum(truth[name]["cl100k_base"] for name in corpus_files)
if abs(broken - real) / real * 100 <= AGREEMENT_LIMIT_PCT:
    fail("a deliberately wrong recount agreed with the truth inside the bound, so the bound "
         "is too loose to mean anything")
else:
    print(f"   characters over eight lands {abs(broken - real) / real * 100:.1f}% out, "
          f"outside the {AGREEMENT_LIMIT_PCT}% bound, as it must")

# The tail term carries almost nothing on ASCII code, where chunks are short. It carries the
# whole cost of Japanese, where a spaceless sentence arrives as one very long chunk. So the
# control is run where the term does the work, not where it is idle.
cjk_truth = truth["japanese.txt"]["cl100k_base"]
cjk_full = independent_count(table, "cl100k_base", texts["japanese.txt"])
cjk_no_tail = independent_count(table, "cl100k_base", texts["japanese.txt"], use_tail=False)
full_error = abs(cjk_full - cjk_truth) / cjk_truth * 100
no_tail_error = abs(cjk_no_tail - cjk_truth) / cjk_truth * 100
print(f"   on japanese.txt the scanner is {full_error:.1f}% out with the tail term and "
      f"{no_tail_error:.1f}% out without it")
if no_tail_error <= full_error * 2:
    fail("dropping the tail term barely changed the Japanese count, so the term that is "
         "supposed to carry long chunks is not carrying them")

print()
if failures:
    print(f"INDEPENDENT CHECK FAILED: {len(failures)} disagreement(s)")
    raise SystemExit(1)
print("INDEPENDENT CHECK PASSED")
