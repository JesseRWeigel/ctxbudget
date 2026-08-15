#!/usr/bin/env python3
"""Fit the token table against real tokenizers, then measure the error of doing so.

This script is NOT part of `verify.sh` and is not needed to run the tool. It needs `tiktoken`
and `tokenizers` and a corpus of real files, and it produces two committed artifacts:

    ctxbudget/data/token_table.json      the fitted table the tool ships with
    fixtures/evidence/calibration.md     what was fitted, on what, and how wrong it is

Method. Every chunk of the training corpus is encoded ON ITS OWN by the real tokenizer, and the
mean token count is recorded per (class, byte length) bucket. Chunks longer than their class cap
get a per-class straight line instead, an intercept and a per-byte rate fitted by
occurrence-weighted least squares over the chunks that overflow. The estimate for a whole text is
the sum of those per-chunk numbers. That sum is then compared against the real tokenizer's count
for the WHOLE text, on files the fit never saw. The held-out comparison is the honest one, because
chunk boundaries here are a port of one family's split regex and are not identical to any of
these tokenizers' own.

Usage:
    python3 scripts/calibrate.py --corpus DIR [DIR ...] --tokenizers-dir DIR --out-table PATH

Paths are never written into the output. Only counts, extensions and aggregates are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctxbudget import pretok  # noqa: E402

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".json", ".yml", ".yaml",
    ".sh", ".html", ".css", ".toml", ".cfg", ".ini", ".java", ".c", ".h", ".go", ".rs",
    ".sql", ".csv", ".xml", ".rb", ".php", ".lua", ".vue", ".svelte",
}
MAX_BYTES = 200_000


def collect(roots: list[str], limit: int, exclude: list[str] | None = None) -> list[Path]:
    """Every candidate file under the roots, then a deterministic sample of `limit` of them.

    The sample is ordered by a hash of the path rather than by directory order, so the corpus is
    spread across every repository under the roots instead of being the first N files of the
    alphabetically first one.
    """
    excluded = exclude or []
    found: list[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if any(fragment in dirpath for fragment in excluded):
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".venv", "venv",
                             "dist", "build", ".next", "target", ".cache", "site-packages"}
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() not in TEXT_EXT:
                    continue
                try:
                    if not (200 <= path.stat().st_size <= MAX_BYTES):
                        continue
                except OSError:
                    continue
                found.append(path)
    found.sort(key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
    return found[:limit]


def cjk_supplement(roots: list[str], exclude: list[str], already: set[Path],
                   limit: int) -> list[Path]:
    """Files containing a meaningful amount of CJK text.

    Sampling by hash across the machine turned up almost no CJK, which left the CJK buckets
    unfitted and Japanese underestimated by roughly half. These files are real files on the same
    machine, pulled in deliberately rather than by chance, and the evidence file says so.
    """
    def is_cjk(ch: str) -> bool:
        point = ord(ch)
        return 0x2E80 <= point <= 0x9FFF or 0xAC00 <= point <= 0xD7FF or 0x3040 <= point <= 0x30FF

    found: list[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if any(fragment in dirpath for fragment in exclude):
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".venv", "venv",
                             "dist", "build", ".next", "target", ".cache", "site-packages"}
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() not in TEXT_EXT or path in already:
                    continue
                try:
                    if not (200 <= path.stat().st_size <= MAX_BYTES):
                        continue
                except OSError:
                    continue
                text = read_text(path)
                if text is None:
                    continue
                if sum(1 for ch in text[:20000] if is_cjk(ch)) > 50:
                    found.append(path)
    found.sort(key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
    return found[:limit]


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def split_train_holdout(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Deterministic split on a hash of the file's own content, so it does not depend on paths."""
    train, holdout = [], []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (holdout if int(digest[:2], 16) < 64 else train).append(path)  # 25 percent held out
    return train, holdout


class Family:
    def __init__(self, key: str, label: str, encode):
        self.key = key
        self.label = label
        self.encode = encode


def load_families(tokenizers_dir: Path) -> list[Family]:
    families: list[Family] = []
    import tiktoken

    for name, label in (("cl100k_base", "tiktoken cl100k_base"),
                        ("o200k_base", "tiktoken o200k_base")):
        enc = tiktoken.get_encoding(name)
        families.append(Family(name, f"{label} (tiktoken {tiktoken.__version__})",
                               lambda text, enc=enc: len(enc.encode(text,
                                                                    disallowed_special=()))))
    from tokenizers import Tokenizer
    import tokenizers as tk

    for key, filename, label in (
        ("qwen2.5", "qwen-tokenizer.json", "Qwen2.5 tokenizer.json"),
        ("llama3", "llama3-tokenizer.json", "Llama 3 tokenizer.json"),
    ):
        path = tokenizers_dir / filename
        if not path.exists():
            print(f"missing tokenizer for {key}: {path.name}", file=sys.stderr)
            continue
        tok = Tokenizer.from_file(str(path))
        families.append(Family(
            key, f"{label} (tokenizers {tk.__version__})",
            lambda text, tok=tok: len(tok.encode(text, add_special_tokens=False).ids)))
    return families


def class_support(train: list[Path]) -> dict[str, dict[str, int]]:
    """How many chunks and bytes of the training corpus actually fitted each class.

    A class the corpus barely contains still gets numbers, because the loop below fills every
    empty bucket from the one before it, and those numbers look exactly like fitted ones. The
    Hangul class is the live example: this machine holds 66 Korean chunks. Recording the support
    is what lets a reader tell a fitted class from a filled-in one.
    """
    support: dict[str, dict[str, int]] = {cls: {"chunks": 0, "bytes": 0} for cls in pretok.CLASSES}
    for path in train:
        text = read_text(path)
        if text is None:
            continue
        for _, cls, nbytes in pretok.walk(text):
            support[cls]["chunks"] += 1
            support[cls]["bytes"] += nbytes
    return support


def base_class(cls: str) -> str:
    """The class a lead variant belongs to, e.g. `word_hangul_px` -> `word_hangul`."""
    return cls.removesuffix(pretok.LEAD_SPACE).removesuffix(pretok.LEAD_PUNCT)


def fit_family(family: Family, train: list[Path],
               support: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean isolated-chunk token count per bucket, plus a per-byte tail rate per class."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    # For the tail: (bytes over cap, tokens, occurrences) triples per class.
    over: dict[str, list[tuple[int, int, int]]] = {cls: [] for cls in pretok.CLASSES}

    # Frequency matters. The mean must be weighted by how often a chunk actually occurs, because
    # a bucket contains both " the" and " Rueckwaertskompatibilitaet" and the first one is what
    # real text is mostly made of. An unweighted mean over distinct chunks was measured at 30
    # percent median error, three times worse than dividing characters by four.
    frequency: dict[tuple[str, str], int] = {}
    for path in train:
        text = read_text(path)
        if text is None:
            continue
        for chunk, cls, _ in pretok.walk(text):
            frequency[(chunk, cls)] = frequency.get((chunk, cls), 0) + 1

    for (chunk, cls), occurrences in frequency.items():
        nbytes = len(chunk.encode("utf-8"))
        ntok = family.encode(chunk)
        key = pretok.bucket_key(cls, nbytes)
        sums[key] = sums.get(key, 0.0) + ntok * occurrences
        counts[key] = counts.get(key, 0.0) + occurrences
        if nbytes > pretok.CAPS[cls]:
            over[cls].append((nbytes - pretok.CAPS[cls], ntok, occurrences))

    table: dict[str, float] = {}
    for key, total in sums.items():
        table[key] = total / counts[key]

    for cls in pretok.CLASSES:
        cap = pretok.CAPS[cls]
        base_key = pretok.bucket_key(cls, cap)
        base = table.get(base_key)
        if base is None:
            # No chunk of exactly the cap length. Interpolate from the largest bucket seen.
            below = [table[pretok.bucket_key(cls, n)] for n in range(1, cap)
                     if pretok.bucket_key(cls, n) in table]
            base = below[-1] if below else 1.0
            table[base_key] = base
        pairs = over[cls]
        if pairs:
            # Occurrence-weighted least squares of token count on bytes past the cap, with a free
            # intercept. Forcing the line through `base`, the mean cost of a chunk of exactly the
            # cap length, is what broke CJK: that mean is carried by short full-width punctuation
            # runs and sits well above the real cost of a run of Japanese, so every long chunk
            # contributed a negative residual and dragged the fitted rate to half its true value.
            weight = sum(occurrences for _, _, occurrences in pairs)
            mean_x = sum(extra * occurrences for extra, _, occurrences in pairs) / weight
            mean_y = sum(ntok * occurrences for _, ntok, occurrences in pairs) / weight
            sxx = sum(occurrences * (extra - mean_x) ** 2
                      for extra, _, occurrences in pairs)
            sxy = sum(occurrences * (extra - mean_x) * (ntok - mean_y)
                      for extra, ntok, occurrences in pairs)
            slope = max(0.0, sxy / sxx) if sxx else 0.0
            table[pretok.tail_key(cls)] = slope
            table[pretok.over_key(cls)] = mean_y - slope * mean_x - base
        else:
            table[pretok.tail_key(cls)] = 0.25
            table[pretok.over_key(cls)] = 0.0
    # Every bucket below the cap must exist so the tool never has to guess at runtime.
    for cls in pretok.CLASSES:
        previous = 1.0
        for n in range(1, pretok.CAPS[cls] + 1):
            key = pretok.bucket_key(cls, n)
            if key not in table:
                table[key] = previous
            previous = table[key]

    # A lead variant the corpus never produced is copied wholesale from its base class rather
    # than filled in bucket by bucket from its own empty neighbours, which would invent a shape
    # out of nothing. This machine has Korean strings but no Korean text with punctuation in
    # front of a word, so `word_hangul_px` is copied from `word_hangul`. The copy is recorded in
    # class_support, because a copied class must not read as a fitted one.
    for cls in pretok.CLASSES:
        if support[cls]["chunks"] or base_class(cls) == cls:
            continue
        source = base_class(cls)
        for n in range(1, pretok.CAPS[cls] + 1):
            table[pretok.bucket_key(cls, n)] = table[pretok.bucket_key(source, n)]
        table[pretok.tail_key(cls)] = table[pretok.tail_key(source)]
        table[pretok.over_key(cls)] = table[pretok.over_key(source)]
    return table


def estimate(table: dict[str, float], text: str) -> int:
    total = 0.0
    for key, count in pretok.features(text).items():
        total += table.get(key, 0.25) * count
    return max(0, round(total))


def cjk_share(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if pretok._is_cjk(ch)) / len(text)


def evaluate_cjk(family: Family, table: dict[str, float], fixtures: dict[str, str],
                 corpus: list[Path], share: float) -> dict:
    """Error on CJK-heavy text, measured separately from the corpus-wide band.

    The corpus-wide band is dominated by ASCII code and prose and says nothing useful about a
    file of Japanese, so the tool quotes this number instead once an input crosses the same share
    threshold. It has to be measured on the same kind of file.

    That measurement cannot come from the sampled corpus. Scanning every candidate file on this
    machine for CJK found 80 that contain any at all and none above nine percent: they are source
    files with Japanese UI strings in them, so their error is really the error on TypeScript. The
    measurement therefore runs on the committed CJK fixtures, which sit inside the excluded
    project directory and are never part of the fit. `corpus_files_at_threshold` records how many
    real corpus files would have qualified, so nobody has to take that reasoning on trust.
    """
    errors: list[tuple[float, str]] = []
    for name, text in sorted(fixtures.items()):
        if cjk_share(text) < share:
            continue
        truth = family.encode(text)
        errors.append((abs(estimate(table, text) - truth) / truth * 100.0, name))
    if not errors:
        raise SystemExit(
            f"no committed fixture is at least {share:.0%} Han, Kana or Hangul, so the CJK "
            f"warning would quote an error nothing measured. Add a CJK fixture to "
            f"fixtures/corpus or lower --cjk-warning-share.")
    qualifying = 0
    for path in corpus:
        text = read_text(path)
        if text is not None and cjk_share(text) >= share:
            qualifying += 1
    values = sorted(value for value, _ in errors)
    return {
        "files": len(errors),
        "fixtures": [name for _, name in sorted(errors, key=lambda pair: pair[1])],
        "source": "committed fixtures, which the fit never sees",
        "corpus_files_at_threshold": qualifying,
        "share_threshold": share,
        "median_abs_pct": round(statistics.median(values), 3),
        "max_abs_pct": round(max(values), 3),
    }


def evaluate(family: Family, table: dict[str, float], holdout: list[Path]) -> dict:
    errors: list[float] = []
    signed: list[float] = []
    naive_errors: list[float] = []
    per_ext: dict[str, list[float]] = {}
    total_true = 0
    worst = (0.0, "", 0, 0)
    for path in holdout:
        text = read_text(path)
        if text is None:
            continue
        truth = family.encode(text)
        if truth < 50:
            continue
        got = estimate(table, text)
        naive = round(len(text) / 4)
        err = abs(got - truth) / truth * 100.0
        errors.append(err)
        signed.append((got - truth) / truth * 100.0)
        naive_errors.append(abs(naive - truth) / truth * 100.0)
        per_ext.setdefault(path.suffix.lower(), []).append(err)
        total_true += truth
        if err > worst[0]:
            worst = (err, path.suffix.lower(), truth, got)
    errors.sort()
    naive_errors.sort()

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return values[index]

    return {
        "files": len(errors),
        "tokens": total_true,
        "median_abs_pct": round(pct(errors, 0.5), 3),
        "p95_abs_pct": round(pct(errors, 0.95), 3),
        "max_abs_pct": round(max(errors) if errors else 0.0, 3),
        "mean_signed_pct": round(statistics.fmean(signed) if signed else 0.0, 3),
        "naive_median_abs_pct": round(pct(naive_errors, 0.5), 3),
        "naive_p95_abs_pct": round(pct(naive_errors, 0.95), 3),
        "worst": {"pct": round(worst[0], 2), "ext": worst[1],
                  "true": worst[2], "estimated": worst[3]},
        "by_ext": {ext: {"files": len(vals),
                         "median_abs_pct": round(statistics.median(vals), 3)}
                   for ext, vals in sorted(per_ext.items()) if len(vals) >= 3},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--tokenizers-dir", required=True)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--cjk-limit", type=int, default=60,
                        help="how many CJK-bearing files to add deliberately")
    parser.add_argument("--cjk-warning-share", type=float, default=0.10,
                        help="CJK share at which the tool warns, and so the share the CJK "
                             "error is measured over. Must match tokens.CJK_WARNING_SHARE.")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="path fragments to skip, so the tool does not train on itself")
    parser.add_argument("--out-table", default="ctxbudget/data/token_table.json")
    parser.add_argument("--out-evidence", default="fixtures/evidence/calibration.md")
    parser.add_argument("--out-truth", default="fixtures/truth/counts.json")
    parser.add_argument("--fixture-dir", default="fixtures/corpus")
    args = parser.parse_args()

    paths = collect(args.corpus, args.limit, args.exclude)
    supplement = cjk_supplement(args.corpus, args.exclude, set(paths), args.cjk_limit)
    paths.extend(supplement)
    if len(paths) < 100:
        print(f"corpus too small: {len(paths)} files", file=sys.stderr)
        return 1
    train, holdout = split_train_holdout(paths)
    families = load_families(Path(args.tokenizers_dir))
    if len(families) < 3:
        print("need at least three real tokenizers to calibrate against", file=sys.stderr)
        return 1

    fixture_dir = Path(args.fixture_dir)
    fixtures = {path.name: path.read_text(encoding="utf-8")
                for path in sorted(fixture_dir.iterdir()) if path.is_file()}
    if not fixtures:
        print(f"no fixtures under {fixture_dir}", file=sys.stderr)
        return 1

    print(f"corpus {len(paths)} files ({len(supplement)} added for CJK coverage), "
          f"train {len(train)}, holdout {len(holdout)}, {len(fixtures)} fixtures")

    support = class_support(train)
    empty_bases = sorted(cls for cls, info in support.items()
                         if info["chunks"] == 0 and base_class(cls) == cls)
    if empty_bases:
        print(f"no training chunk landed in {', '.join(empty_bases)}. Every bucket of that class "
              f"would be filled in from its neighbour and would look fitted. Either widen the "
              f"corpus or drop the class.", file=sys.stderr)
        return 1
    for cls in pretok.CLASSES:
        if support[cls]["chunks"] == 0:
            support[cls]["copied_from"] = base_class(cls)
            print(f"  {cls}: no training chunk, copied from {base_class(cls)}")
        elif support[cls]["chunks"] < 500:
            print(f"  thin class {cls}: {support[cls]['chunks']} training chunks")
    tables: dict[str, dict[str, float]] = {}
    reports: dict[str, dict] = {}
    cjk_reports: dict[str, dict] = {}
    for family in families:
        print(f"fitting {family.key} ...", flush=True)
        table = fit_family(family, train, support)
        tables[family.key] = {key: round(value, 5) for key, value in sorted(table.items())}
        reports[family.key] = evaluate(family, tables[family.key], holdout)
        cjk_reports[family.key] = evaluate_cjk(family, tables[family.key], fixtures, paths,
                                               args.cjk_warning_share)
        print(f"  {family.key}: median {reports[family.key]['median_abs_pct']}% "
              f"p95 {reports[family.key]['p95_abs_pct']}% "
              f"(chars/4 median {reports[family.key]['naive_median_abs_pct']}%), "
              f"CJK fixtures median {cjk_reports[family.key]['median_abs_pct']}% "
              f"max {cjk_reports[family.key]['max_abs_pct']}% "
              f"on {cjk_reports[family.key]['files']} files")

    # Chat template overhead. For the two local families the wrapper is a literal string and the
    # real tokenizer is right here, so this is measured. For the OpenAI families the wrapper is
    # applied server side and cannot be reproduced locally, so the documented constants are
    # recorded and labelled as documentation rather than measurement.
    CHATML = "<|im_start|>user\n\n<|im_end|>\n"
    LLAMA3 = "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
    overhead: dict[str, dict] = {}
    for family in families:
        if family.key in ("cl100k_base", "o200k_base"):
            overhead[family.key] = {
                "per_message": 3, "priming": 3, "source": "vendor-doc",
                "what": "OpenAI's documented per-message and priming allowance for the chat "
                        "completions format. Applied server side, so it cannot be measured here."}
        else:
            wrapper = CHATML if family.key == "qwen2.5" else LLAMA3
            role = "ChatML" if family.key == "qwen2.5" else "Llama 3 header"
            overhead[family.key] = {
                "per_message": family.encode(wrapper), "priming": 1, "source": "measured",
                "what": f"{role} wrapper for one message, encoded with the real tokenizer.",
                "wrapper_repr": repr(wrapper)}

    ext_counts: dict[str, int] = {}
    for path in paths:
        ext_counts[path.suffix.lower()] = ext_counts.get(path.suffix.lower(), 0) + 1

    payload = {
        "note": "Fitted by scripts/calibrate.py. Keys are class:bytelength buckets; "
                "class:tail is a per-byte rate charged beyond the class cap.",
        "corpus": {
            "files": len(paths),
            "train_files": len(train),
            "holdout_files": len(holdout),
            "cjk_supplement_files": len(supplement),
            "extensions": dict(sorted(ext_counts.items(), key=lambda kv: -kv[1])),
        },
        "class_support": support,
        "families": {family.key: {"label": family.label,
                                  "accuracy": reports[family.key],
                                  "cjk_accuracy": cjk_reports[family.key],
                                  "chat_overhead": overhead[family.key]}
                     for family in families},
        "tables": tables,
    }
    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out_table}")

    # Ground truth for the committed fixture corpus, so the test suite needs no tokenizer.
    truth: dict[str, dict[str, int]] = {}
    if fixtures:
        for name, text in fixtures.items():
            truth[name] = {family.key: family.encode(text) for family in families}
        out_truth = Path(args.out_truth)
        out_truth.parent.mkdir(parents=True, exist_ok=True)
        out_truth.write_text(json.dumps(
            {"note": "Real token counts for the committed fixture corpus, measured with the "
                     "tokenizers named in ctxbudget/data/token_table.json. The test suite "
                     "compares the estimator against these, so it needs no tokenizer installed.",
             "counts": truth}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out_truth} for {len(truth)} fixtures")

    evidence = Path(args.out_evidence)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Calibration, measured not asserted",
        "",
        f"Fitted and measured on {len(paths)} real text files taken read-only from repositories "
        f"on the machine that built this project. {len(train)} files fitted the table, "
        f"{len(holdout)} were held out and never seen by the fit. {len(supplement)} of the "
        "files were pulled in deliberately because they contain CJK text, since a hash-ordered "
        "sample of this machine turned up almost none and the CJK buckets were going unfitted. "
        "File paths are deliberately not recorded here; only counts and extensions are.",
        "",
        "Held-out error, whole-file token count, estimator against the real tokenizer:",
        "",
        "| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median "
        "| chars/4 p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for family in families:
        report = reports[family.key]
        lines.append(
            f"| `{family.key}` | {family.label} | {report['files']} | {report['tokens']:,} | "
            f"{report['median_abs_pct']}% | {report['p95_abs_pct']}% | "
            f"{report['max_abs_pct']}% | {report['naive_median_abs_pct']}% | "
            f"{report['naive_p95_abs_pct']}% |")
    lines += [
        "",
        "The last two columns are the thing this replaces. Dividing characters by four is the "
        "usual shortcut, and its error on the same held-out files is there beside ours.",
        "",
        "## CJK, measured on its own",
        "",
        f"The band above comes from a corpus that is mostly ASCII code, and it does not describe "
        f"a file of Japanese. So the tool quotes a separate number once an input is at least "
        f"{args.cjk_warning_share:.0%} Han, Kana or Hangul, and that number is measured on text "
        f"of that kind. It cannot come from the sampled corpus: of the "
        f"{len(paths)} files in it, "
        f"{next(iter(cjk_reports.values()))['corpus_files_at_threshold']} reach that share. The "
        "machine has Japanese UI strings inside TypeScript files and no Japanese documents, so "
        "an error measured on those files would be the error on TypeScript. It is measured "
        "instead on the committed CJK fixtures, which live inside the excluded project directory "
        "and are never part of the fit:",
        "",
        "| family | fixtures | median | worst |",
        "|---|---|---|---|",
    ]
    for family in families:
        report = cjk_reports[family.key]
        lines.append(f"| `{family.key}` | {', '.join(report['fixtures'])} | "
                     f"{report['median_abs_pct']}% | {report['max_abs_pct']}% |")
    lines += [
        "",
        "Three short documents is a thin measurement and the tool says so rather than implying a "
        "corpus stands behind it. It is still the honest number: the alternative was a constant "
        "in the source, which is how this file previously came to claim 37.7% while the shipped "
        "table was off by five.",
        "",
        "## Corpus shape",
        "",
        "| extension | files |",
        "|---|---|",
    ]
    for ext, count in sorted(ext_counts.items(), key=lambda kv: -kv[1])[:14]:
        lines.append(f"| `{ext}` | {count} |")
    lines += ["", "## Where each family is worst", ""]
    for family in families:
        report = reports[family.key]
        worst = report["worst"]
        by_ext = ", ".join(f"`{ext}` {info['median_abs_pct']}%"
                           for ext, info in report["by_ext"].items())
        lines.append(f"- **{family.key}**: worst single file {worst['pct']}% off "
                     f"(`{worst['ext']}`, {worst['true']} true, {worst['estimated']} estimated). "
                     f"Median by extension: {by_ext}.")
    lines += [
        "",
        "## What this does not cover",
        "",
        "- Claude has no public tokenizer, so no Claude number here is measured. The tool "
        "labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.",
        "- The chunk regex is a port of the `cl100k_base` split pattern. The other three "
        "families split differently, and the fitted table absorbs that difference rather than "
        "reproducing their regexes. The error above is the price of that shortcut, measured.",
        "- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded "
        "from the corpus and are outside anything measured here.",
        "",
    ]
    evidence.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
