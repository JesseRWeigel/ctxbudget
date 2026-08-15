#!/usr/bin/env python3
"""Break the code on purpose and check the suite notices. Three gates, and a null control first.

A sabotage counts only if it

  1. APPLIES.  The patch must actually change the file. `KNOWN = [] or [...]` looks like it
     empties a list and does not, and a sabotage that silently did nothing has been written up
     in this fleet as proof of a gap that was never there.
  2. CHANGES THE MEASURED OUTPUT.  The fingerprint from `measure.py` must move. If it does not,
     either the sabotage is dormant or the fixtures never exercise the code.
  3. IS CAUGHT.  The unit suite must fail.

Gate 2 is the one that lies. A harness whose fingerprint contains the temporary directory it is
running in passes gate 2 for free, every time, for every sabotage. So the NULL CONTROL runs
first: an unmodified copy of the tree, in a different directory, must fingerprint identically to
the original. If it does not, the run is void and this script exits nonzero without scoring
anything.

Guard sabotages invert gate 2. Code that is dormant when the input is well formed, such as the
NUL-byte check that only fires on a binary file, cannot change the fingerprint when you disable
it. For those the requirement is the opposite and stricter: output UNCHANGED and the unit suite
FAILS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTACK = "attack"
GUARD = "guard"


@dataclass
class Sabotage:
    name: str
    kind: str
    path: str
    old: str
    new: str
    what: str


SABOTAGES = [
    Sabotage(
        "cjk-class-collapsed", ATTACK, "ctxbudget/pretok.py",
        "    return any(low <= point <= high for low, high in _CJK_RANGES)",
        "    return False",
        "Treat Han and Kana as ordinary non-ASCII text again, which is the bug this project "
        "found and fixed."),
    Sabotage(
        "counting-back-to-chars-over-four", ATTACK, "ctxbudget/tokens.py",
        "        total = 0.0\n"
        "        for key, occurrences in pretok.features(text).items():\n"
        "            total += self.table[key] * occurrences",
        "        total = len(text) / 4",
        "Throw the fitted table away and divide characters by four."),
    Sabotage(
        "tail-rate-ignored", ATTACK, "ctxbudget/pretok.py",
        "        if nbytes > cap:\n"
        "            tkey = tail_key(cls)\n"
        "            counts[tkey] = counts.get(tkey, 0.0) + (nbytes - cap)",
        "        if False:\n"
        "            pass",
        "Stop charging for bytes past the bucket cap, so long chunks become free."),
    Sabotage(
        "reserve-not-subtracted", ATTACK, "ctxbudget/budget.py",
        "        return self.window - self.reserve",
        "        return self.window",
        "Treat the whole context window as available for input, which is the mistake the tool "
        "exists to catch."),
    Sabotage(
        "reply-room-always-full", ATTACK, "ctxbudget/budget.py",
        "        return max(0, min(self.reserve, self.window - self.input_tokens))",
        "        return self.reserve",
        "Report the reply as having its full allowance even when the input has eaten it."),
    Sabotage(
        "ranking-by-size", ATTACK, "ctxbudget/rank.py",
        "        return self.tokens / self.surviving_demand",
        "        return float(self.tokens)",
        "Fall back to cutting the largest file first."),
    Sabotage(
        "redundancy-ignored", ATTACK, "ctxbudget/rank.py",
        "        return self.demand * max(0.05, 1.0 - self.redundancy)",
        "        return self.demand",
        "Stop discounting content that is already in the window twice."),
    Sabotage(
        "inbound-references-ignored", ATTACK, "ctxbudget/rank.py",
        "            hits = (names[label] - generic) & words[other_label]",
        "            hits = set()",
        "Stop noticing that other files refer to this one."),
    Sabotage(
        "generic-name-filter-removed", ATTACK, "ctxbudget/rank.py",
        "        cutoff = max(2, (len(parts) + 1) // 2)",
        "        cutoff = 10_000",
        "Let a name that appears in every file count as a reference."),
    Sabotage(
        "nul-byte-check-disabled", GUARD, "ctxbudget/cli.py",
        '    if b"\\x00" in data:',
        "    if False:",
        "Accept a binary file as text. Dormant on well-formed input, so the fingerprint must "
        "not move and the suite must still fail."),
    Sabotage(
        "error-band-dropped", GUARD, "ctxbudget/tokens.py",
        "        return Count(max(0, round(total)), ESTIMATE, self.tokenizer_label,\n"
        "                     self.accuracy[\"p95_abs_pct\"])",
        "        return Count(max(0, round(total)), ESTIMATE, self.tokenizer_label, None)",
        "Strip the measured error band off every estimate. The counts do not move, so this is "
        "judged as a guard: the suite must fail with the output unchanged."),
]


def copy_tree(destination: Path) -> None:
    subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    listing = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    for name in listing:
        source = ROOT / name
        if not source.exists():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def fingerprint(tree: Path) -> str | None:
    """Run measure.py inside `tree`, with relative paths only, and return its fingerprint."""
    result = subprocess.run([sys.executable, "scripts/measure.py"], cwd=tree,
                            capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("FINGERPRINT "):
            return line.split()[1]
    return None


def suite_passes(tree: Path) -> bool:
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                            cwd=tree, capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tree),
                                 "PYTHONDONTWRITEBYTECODE": "1"})
    return result.returncode == 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ctxbudget-sabotage-") as workdir:
        work = Path(workdir)
        baseline_tree = work / "baseline"
        baseline_tree.mkdir()
        copy_tree(baseline_tree)

        baseline = fingerprint(baseline_tree)
        if baseline is None:
            print("VOID: the baseline copy does not run at all")
            return 1
        print(f"baseline fingerprint  {baseline}")

        # ------------------------------------------------------------ null control
        control_tree = work / "null-control"
        control_tree.mkdir()
        copy_tree(control_tree)
        control = fingerprint(control_tree)
        if control != baseline:
            print(f"null control fingerprint {control}")
            print("VOID: an unmodified copy in a different directory fingerprinted differently, "
                  "so the measurement tracks where the code lives rather than what it does. "
                  "Every gate 2 below would pass for free. Nothing is scored.")
            return 1
        print(f"null control          {control}  identical, so gate 2 means something")
        if not suite_passes(control_tree):
            print("VOID: the unmodified copy fails its own suite")
            return 1
        print("null control suite    passes")
        print()

        proven = 0
        failures: list[str] = []
        for sabotage in SABOTAGES:
            tree = work / sabotage.name
            tree.mkdir()
            copy_tree(tree)
            target = tree / sabotage.path
            original = target.read_text(encoding="utf-8")
            if sabotage.old not in original:
                failures.append(f"{sabotage.name}: NO-OP, the text to patch is not in "
                                f"{sabotage.path}. The sabotage proves nothing.")
                print(f"  {sabotage.name:<34} gate 1 FAILED, patch text not found")
                continue
            target.write_text(original.replace(sabotage.old, sabotage.new, 1), encoding="utf-8")
            applied = target.read_text(encoding="utf-8") != original

            got = fingerprint(tree)
            moved = got is not None and got != baseline
            crashed = got is None
            caught = not suite_passes(tree)

            if sabotage.kind == ATTACK:
                if crashed:
                    failures.append(f"{sabotage.name}: the tool CRASHED rather than producing "
                                    f"different output. A crash clears gate 2 for free and "
                                    f"proves nothing about the checks.")
                    print(f"  {sabotage.name:<34} gate 2 FAILED, crashed instead of running")
                    continue
                gates = (applied, moved, caught)
                labels = ("applies", "changes output", "caught")
            else:
                if crashed:
                    failures.append(f"{sabotage.name}: guard sabotage crashed the tool, so it "
                                    f"was never dormant and should be rerun as an attack.")
                    print(f"  {sabotage.name:<34} gate 2 FAILED, crashed instead of running")
                    continue
                if moved:
                    failures.append(f"{sabotage.name}: classified as a guard but it CHANGED the "
                                    f"output, so it was never dormant. Rerun it as an attack.")
                    print(f"  {sabotage.name:<34} gate 2 FAILED, a guard must not move the "
                          f"output")
                    continue
                gates = (applied, not moved, caught)
                labels = ("applies", "output unchanged", "caught")

            marks = " ".join(f"{label}={'yes' if value else 'NO'}"
                             for label, value in zip(labels, gates))
            if all(gates):
                proven += 1
                print(f"  {sabotage.name:<34} {sabotage.kind:<6} {marks}")
            else:
                failures.append(f"{sabotage.name}: {marks}")
                print(f"  {sabotage.name:<34} {sabotage.kind:<6} {marks}   FAILED")

        print()
        print(f"{proven} of {len(SABOTAGES)} sabotages proven under the three-gate rule")
        for failure in failures:
            print(f"  {failure}")
        return 0 if proven == len(SABOTAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
