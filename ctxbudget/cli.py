"""The command line. Exit code 0 fits, 3 is over budget, 2 is could not read."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import budget as budget_mod
from . import rank as rank_mod
from .tokens import CJK_WARNING_SHARE, TableUnavailable, cjk_share, naive_count

EXIT_FITS = 0
EXIT_UNREADABLE = 2
EXIT_OVER = 3


def read_file(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"{path} contains a NUL byte, so it is not text and cannot be counted")
    return data.decode("utf-8")


def _thousands(value: int) -> str:
    return f"{value:,}"


def render(report: budget_mod.Budget, signals: list[rank_mod.Signals],
           naive_total: int, explain: bool) -> str:
    out: list[str] = []
    out.append(f"model     {report.model}, {_thousands(report.window)} token window, "
               f"family {report.family}")
    out.append(f"reserve   {_thousands(report.reserve)} tokens held back for the reply "
               f"({report.reserve_source})")
    band = "" if report.band_pct is None else f", p95 error {report.band_pct}%"
    out.append(f"counting  {report.counting}, {report.tokenizer}{band}")
    out.append("")

    width = max(24, max((len(part.label) for part in report.parts), default=24))
    width = min(width, 56)
    out.append(f"{'PART'.ljust(width)}  {'TOKENS':>9}  {'SHARE':>7}")
    for part in sorted(report.parts, key=lambda p: (p.kind != "system", -p.tokens)):
        label = part.label if len(part.label) <= width else "..." + part.label[-(width - 3):]
        share = part.tokens / report.window * 100 if report.window else 0.0
        out.append(f"{label.ljust(width)}  {_thousands(part.tokens):>9}  {share:6.1f}%")
    out.append(f"{'-' * width}  {'-' * 9}  {'-' * 7}")
    out.append(f"{'input total'.ljust(width)}  {_thousands(report.input_tokens):>9}  "
               f"{report.input_tokens / report.window * 100:6.1f}%")
    out.append(f"{'reserved for the reply'.ljust(width)}  {_thousands(report.reserve):>9}  "
               f"{report.reserve / report.window * 100:6.1f}%")
    free = report.window - report.input_tokens - report.reserve
    out.append(f"{'unused'.ljust(width)}  {_thousands(free):>9}  "
               f"{free / report.window * 100:6.1f}%")
    out.append("")

    if report.status == budget_mod.FITS:
        headline = (f"FITS. {_thousands(report.spare)} tokens spare, and the reply can use its "
                    f"full {_thousands(report.reply_room)}.")
        out.append(headline)
        if report.tight:
            out.append(f"  TIGHT. At the top of the measured error band the input is "
                       f"{_thousands(report.worst_case_input)} tokens, which is over the "
                       f"{_thousands(report.usable_input)} available. Treat this as not shown "
                       f"to fit.")
    else:
        out.append(f"OVER BUDGET by {_thousands(report.over_by)} tokens. The input is "
                   f"{_thousands(report.input_tokens)} and only "
                   f"{_thousands(report.usable_input)} is available once "
                   f"{_thousands(report.reserve)} is kept back for the reply.")
        if report.input_tokens < report.window:
            out.append(f"  The request would be accepted. It leaves "
                       f"{_thousands(report.reply_room)} tokens for the reply instead of "
                       f"{_thousands(report.reserve)}, so the answer gets cut off rather than "
                       f"refused.")
        else:
            out.append("  The input alone exceeds the window, so the request is refused outright.")

    if signals:
        out.append("")
        out.append("CUT FIRST  (tokens / ((1 + inbound refs + task match) x "
                   "(1 - redundancy)))")
        needed = rank_mod.greedy_cut(signals, report.over_by)
        needed_labels = {signal.label for signal in needed}
        for index, signal in enumerate(signals[:6], start=1):
            marker = "*" if signal.label in needed_labels else " "
            out.append(f" {marker}{index}. {signal.label}  {_thousands(signal.tokens)} tokens, "
                       f"score {signal.score:,.0f}")
            out.append(f"      {signal.reason()}")
        if needed:
            saved = sum(signal.tokens for signal in needed)
            out.append(f"  Cutting the {len(needed)} marked * frees {_thousands(saved)} tokens "
                       f"and brings the input to "
                       f"{_thousands(report.input_tokens - saved)}, under the "
                       f"{_thousands(report.usable_input)} available.")

    if explain:
        out.append("")
        out.append("HOW THE COUNT WAS MADE")
        out.append(f"  characters over four would have said {_thousands(naive_total)} tokens, "
                   f"a {abs(naive_total - report.input_tokens) / max(1, report.input_tokens) * 100:.1f}% "
                   f"difference from the count above.")
        if report.counting == "estimate":
            out.append(f"  The estimate carries a p95 error of {report.band_pct}% measured on "
                       f"held-out real files. See fixtures/evidence/calibration.md.")
        elif report.counting == "exact":
            out.append("  A real tokenizer for this family is installed, so the count is the "
                       "count and carries no error band.")

    for warning in report.warnings:
        if warning:
            out.append("")
            out.append("NOTE  " + warning)
    return "\n".join(out)


def to_json(report: budget_mod.Budget, signals: list[rank_mod.Signals], naive_total: int) -> str:
    payload = {
        "model": report.model,
        "family": report.family,
        "window": report.window,
        "reserved_for_reply": report.reserve,
        "reserve_source": report.reserve_source,
        "counting": report.counting,
        "tokenizer": report.tokenizer,
        "p95_error_pct": report.band_pct,
        "parts": [{"label": part.label, "kind": part.kind, "tokens": part.tokens,
                   "method": part.count.method} for part in report.parts],
        "input_tokens": report.input_tokens,
        "usable_input_tokens": report.usable_input,
        "left_for_reply": report.reply_room,
        "over_by": report.over_by,
        "status": report.status,
        "tight": report.tight,
        "worst_case_input_tokens": report.worst_case_input,
        "chars_over_four_would_say": naive_total,
        "cut_order": [{"label": signal.label, "tokens": signal.tokens,
                       "score": round(signal.score, 2), "inbound_refs": signal.inbound,
                       "query_demand": signal.query_demand,
                       "redundancy": signal.redundancy,
                       "reason": signal.reason()} for signal in signals],
        "warnings": [warning for warning in report.warnings if warning],
    }
    return json.dumps(payload, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctxbudget",
        description="What fills a model's context window, what is left for the reply, "
                    "and what to cut first.")
    parser.add_argument("files", nargs="*", help="files to put in the context")
    parser.add_argument("--model", "-m", default="gpt-4o")
    parser.add_argument("--system", "-s", help="path to the system prompt")
    parser.add_argument("--query", "-q", default="",
                        help="the task. Used to judge which files answer to it")
    parser.add_argument("--window", type=int, help="override the context window")
    parser.add_argument("--reserve", type=int, help="tokens to hold back for the reply")
    parser.add_argument("--family", help="override the tokenizer family")
    parser.add_argument("--messages", type=int, default=1,
                        help="how many chat messages the request will contain")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--no-exact", action="store_true",
                        help="ignore an installed tokenizer and use the fitted table")
    parser.add_argument("--models", help="path to an alternative model table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        models = budget_mod.load_models(Path(args.models) if args.models else None)
    except OSError as error:
        print(f"CANNOT READ the model table: {error}", file=sys.stderr)
        return EXIT_UNREADABLE

    loaded: list[tuple[str, str]] = []
    for name in args.files:
        path = Path(name)
        try:
            loaded.append((name, read_file(path)))
        except (OSError, ValueError, UnicodeDecodeError) as error:
            print(f"CANNOT READ {name}: {error}", file=sys.stderr)
            return EXIT_UNREADABLE

    system_prompt = None
    if args.system:
        try:
            system_prompt = read_file(Path(args.system))
        except (OSError, ValueError, UnicodeDecodeError) as error:
            print(f"CANNOT READ {args.system}: {error}", file=sys.stderr)
            return EXIT_UNREADABLE

    if not loaded and system_prompt is None:
        print("nothing to count. Give at least one file or --system.", file=sys.stderr)
        return EXIT_UNREADABLE

    try:
        report = budget_mod.build(
            args.model, loaded, system_prompt, models, window=args.window,
            reserve=args.reserve, family=args.family, messages=args.messages,
            allow_exact=not args.no_exact)
    except (budget_mod.UnknownModel, TableUnavailable) as error:
        print(f"CANNOT READ: {error.args[0] if error.args else error}", file=sys.stderr)
        return EXIT_UNREADABLE

    everything = "".join(text for _, text in loaded) + (system_prompt or "")
    share = cjk_share(everything)
    if report.counting != "exact" and share >= CJK_WARNING_SHARE and report.cjk_accuracy is None:
        # The count is not exact, the input is CJK-heavy, and nothing measured the error on that
        # kind of input. Saying so is the point; staying quiet would read as "no problem here".
        report.warnings.append(
            f"{share * 100:.0f}% of this input is Han, Kana or Hangul and nothing here measured "
            f"the error on CJK text, so the band printed above does not describe this input and "
            f"no other number is available. Treat the count as a rough lower bound.")
    elif report.counting != "exact" and share >= CJK_WARNING_SHARE:
        cjk = report.cjk_accuracy
        report.warnings.append(
            f"{share * 100:.0f}% of this input is Han, Kana or Hangul, so the band printed above "
            f"does not describe it: that band comes from a corpus that is mostly ASCII code and "
            f"prose. Measured separately on the {cjk['files']} held-out files that are at least "
            f"{cjk['share_threshold'] * 100:.0f}% CJK, this estimate is off by a median of "
            f"{cjk['median_abs_pct']}% and by up to {cjk['max_abs_pct']}%. Install tiktoken for "
            f"an exact count on an OpenAI family.")

    tokens = {part.label: part.tokens for part in report.parts if part.kind == "file"}
    signals = rank_mod.rank(loaded, tokens, args.query) if loaded else []
    naive_total = sum(naive_count(text) for _, text in loaded)
    if system_prompt is not None:
        naive_total += naive_count(system_prompt)

    if args.json:
        print(to_json(report, signals, naive_total))
    else:
        print(render(report, signals, naive_total, args.explain))
    return EXIT_FITS if report.status == budget_mod.FITS else EXIT_OVER
