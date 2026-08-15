#!/usr/bin/env python3
"""Nothing private reaches a committed file. Proved with planted controls, not by finding zero.

This project reads arbitrary paths off the command line and prints them back, and its calibration
walked a whole home directory. So the failure mode is specific: a home path, a username or a
credential-shaped string ending up in a committed fixture, a README transcript or an evidence
file.

A scanner that reads nothing reports exactly the same clean result as a scanner that read
everything and found nothing. Three things guard against that here.

  - POSITIVE CONTROLS ARE PLANTED, in a temporary file this script writes and then feeds to the
    same scanning function the real files go through. They are not borrowed from where this
    repository happens to sit, because a control that only exists because of the checkout
    location proves nothing in a clone. Every planted control must be found.
  - A FLOOR ON FILES READ. Before the first commit `git ls-files` returns nothing and a scan of
    nothing passes. Fewer files than the floor is a failure.
  - NUL BYTES ARE HUNTED SEPARATELY, in Python rather than with grep. One NUL makes git and grep
    classify a file as binary and skip it silently, which has hidden a real token in this fleet
    before. Detection is checked against a planted NUL too, because `grep -P '\\x00'` is not
    available in every grep on every machine and returns no matches where it is not.

Patterns are assembled from fragments so this file does not match its own pattern list.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_TRACKED_FILES = 30

_HOME = "/" + "home" + "/"
_USERS = "/" + "Users" + "/"
_GH = "gh" + "p_"
_SK = "sk-" + "ant-" + "api03-"
_OR = "sk-" + "or-" + "v1-"
_AWS = "AK" + "IA"
_PEM = "-----BEGIN " + "RSA PRIVATE KEY" + "-----"

# (name, compiled pattern, case sensitivity is baked into the pattern on purpose). AWS key ids are
# uppercase by definition, and matching them case-insensitively turns every base64 blob in an
# inline image into a false alarm. That happened in this fleet.
PATTERNS = [
    ("home path", re.compile(re.escape(_HOME) + r"[a-z][a-z0-9_.-]{1,31}/")),
    ("mac home path", re.compile(re.escape(_USERS) + r"[a-z][a-z0-9_.-]{1,31}/")),
    ("github token", re.compile(re.escape(_GH) + r"[A-Za-z0-9]{36}")),
    ("anthropic key", re.compile(re.escape(_SK) + r"[A-Za-z0-9_-]{20}")),
    ("openrouter key", re.compile(re.escape(_OR) + r"[a-f0-9]{16}")),
    ("aws key id", re.compile(re.escape(_AWS) + r"[0-9A-Z]{16}")),
    ("private key block", re.compile(re.escape(_PEM))),
    ("bearer header", re.compile(r"[Aa]uthorization:\s*Bearer\s+[A-Za-z0-9._-]{20}")),
]

# Files whose whole job is to name these patterns. This script and nothing else.
SELF = os.path.relpath(os.path.abspath(__file__), ROOT)


def scan_text(text: str) -> list[tuple[str, str]]:
    """Every pattern hit in one blob. The single place both the controls and the tree go through."""
    hits: list[tuple[str, str]] = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            hits.append((name, match.group(0)))
    return hits


def has_nul(data: bytes) -> bool:
    return b"\x00" in data


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.split() if name]


def main() -> int:
    problems: list[str] = []

    # ------------------------------------------------------------ the scanner can find things
    print("== positive controls, planted here and now")
    with tempfile.TemporaryDirectory(prefix="ctxbudget-privacy-") as workdir:
        planted = {
            "home path": _HOME + "someone" + "/Projects/thing.py",
            "mac home path": _USERS + "someone" + "/Documents/notes.md",
            "github token": _GH + "A" * 36,
            "anthropic key": _SK + "B" * 20,
            "openrouter key": _OR + "abcdef0123456789",
            "aws key id": _AWS + "ABCDEFGHIJKLMNOP",
            "private key block": _PEM,
            "bearer header": "Authorization: Bearer " + "C" * 20,
        }
        control_path = os.path.join(workdir, "planted.txt")
        with open(control_path, "w", encoding="utf-8") as handle:
            for label, value in planted.items():
                handle.write(f"{label}: {value}\n")
        with open(control_path, encoding="utf-8") as handle:
            found = {name for name, _ in scan_text(handle.read())}
        for label in planted:
            if label in found:
                print(f"   {label:<20} found")
            else:
                problems.append(f"the scanner missed a planted {label}, so a real one would "
                                f"also pass unnoticed")
                print(f"   {label:<20} MISSED")

        # A planted NUL, because grep-based detection silently fails on some machines.
        nul_path = os.path.join(workdir, "planted.bin")
        with open(nul_path, "wb") as handle:
            handle.write(b"before\x00" + (_HOME + "someone/secret.txt").encode())
        with open(nul_path, "rb") as handle:
            data = handle.read()
        if has_nul(data):
            print("   nul byte            found")
        else:
            problems.append("NUL detection does not work, so any file containing one would be "
                            "skipped silently by this scan")
        if not scan_text(data.decode("utf-8", "replace")):
            problems.append("a home path hiding after a NUL byte was not found")
        else:
            print("   path behind a nul   found")

        # A negative control: clean text must produce nothing, or every pattern is too loose.
        if scan_text("a perfectly ordinary sentence about tokens and windows, 128000 of them"):
            problems.append("ordinary text matched a pattern, so the patterns are too loose to "
                            "mean anything")
        else:
            print("   clean text          correctly produced nothing")

    # ------------------------------------------------------------ the tree itself
    print()
    print("== the committed tree")
    names = tracked_files()
    if len(names) < MIN_TRACKED_FILES:
        problems.append(f"only {len(names)} tracked files, under the floor of "
                        f"{MIN_TRACKED_FILES}. A scan of almost nothing passes for free.")
    read = 0
    binary = 0
    for name in names:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            data = handle.read()
        read += 1
        if has_nul(data):
            binary += 1
            problems.append(f"{name} contains a NUL byte. git and grep will treat it as binary "
                            f"and skip it, which makes this scan blind to that whole file. "
                            f"Write the byte as the two-character escape instead.")
        text = data.decode("utf-8", "replace")
        for label, value in scan_text(text):
            if name == SELF:
                continue
            problems.append(f"{name}: {label} -> {value[:48]}")
    print(f"   {read} tracked files read, {binary} containing a NUL byte")
    print(f"   {len(PATTERNS)} patterns, every one of them proven to fire above")

    print()
    if problems:
        print(f"PRIVACY SCAN FAILED: {len(problems)} problem(s)")
        for problem in problems:
            print(f"   {problem}")
        return 1
    print("PRIVACY SCAN PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
