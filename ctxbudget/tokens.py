"""Counting tokens, and being honest about how the number was arrived at.

Two ways of getting a count, and the difference between them is the whole point of this module.

`exact`     a real tokenizer for that model family is installed here, so the number is the
            number. Only `tiktoken` qualifies today, and only for the two OpenAI encodings.
`estimate`  the fitted table in `data/token_table.json` was used. It carries a measured error
            band from held-out files, and the band is printed with every number.

Nothing returns a bare integer. A count that has lost the record of where it came from is the
thing that lets a pipeline quietly overflow, so `Count` carries the method with it and the
report prints it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import pretok

DATA = Path(__file__).parent / "data"

EXACT = "exact"
ESTIMATE = "estimate"
UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class Count:
    """A token count and its provenance."""

    tokens: int
    method: str
    tokenizer: str
    #: Held-out p95 absolute error, in percent, or None when nothing measured it.
    band_pct: float | None

    @property
    def low(self) -> int:
        if self.band_pct is None:
            return self.tokens
        return int(self.tokens * (1 - self.band_pct / 100))

    @property
    def high(self) -> int:
        if self.band_pct is None:
            return self.tokens
        return int(round(self.tokens * (1 + self.band_pct / 100)))

    def describe(self) -> str:
        if self.method == EXACT:
            return f"exact, {self.tokenizer}"
        if self.method == UNMEASURED:
            return f"estimate, {self.tokenizer}, error never measured"
        return f"estimate, {self.tokenizer}, p95 error {self.band_pct}%"


class TableUnavailable(RuntimeError):
    """The fitted table is missing or does not cover the requested family."""


_TABLE_CACHE: dict | None = None


def load_table(path: Path | None = None) -> dict:
    global _TABLE_CACHE
    if path is None:
        if _TABLE_CACHE is not None:
            return _TABLE_CACHE
        path = DATA / "token_table.json"
    if not path.exists():
        raise TableUnavailable(
            f"{path} is missing. It is produced by scripts/calibrate.py and is meant to be "
            f"committed. Without it no count can be produced at all.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path == DATA / "token_table.json":
        _TABLE_CACHE = payload
    return payload


def tiktoken_encoding_for(family: str):
    """Return a real tiktoken encoding for this family, or None.

    Returns None for two different reasons, and the caller does not need to tell them apart:
    tiktoken is not installed, or this family has no tiktoken encoding. Either way the estimate
    path is what runs, and the report says so rather than pretending the count is exact.
    """
    if family not in ("cl100k_base", "o200k_base"):
        return None
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding(family)
    except Exception:
        return None


class Counter:
    """Counts text for one model family, exactly if it can and by the fitted table if it cannot."""

    def __init__(self, family: str, table_path: Path | None = None, allow_exact: bool = True):
        payload = load_table(table_path)
        self.family = family
        self.families = payload["families"]
        if family not in payload["tables"]:
            known = ", ".join(sorted(payload["tables"]))
            raise TableUnavailable(
                f"no fitted table for family {family!r}. Fitted families: {known}. "
                f"Refit with scripts/calibrate.py to add one.")
        self.table = payload["tables"][family]
        self.accuracy = self.families[family]["accuracy"]
        self.encoding = tiktoken_encoding_for(family) if allow_exact else None
        if self.encoding is not None:
            import tiktoken
            self.tokenizer_label = f"tiktoken {tiktoken.__version__} {family}"
        else:
            self.tokenizer_label = f"fitted table for {family}"

    @property
    def exact(self) -> bool:
        return self.encoding is not None

    def count(self, text: str) -> Count:
        if self.encoding is not None:
            return Count(len(self.encoding.encode(text, disallowed_special=())),
                         EXACT, self.tokenizer_label, None)
        total = 0.0
        for key, occurrences in pretok.features(text).items():
            total += self.table[key] * occurrences
        return Count(max(0, round(total)), ESTIMATE, self.tokenizer_label,
                     self.accuracy["p95_abs_pct"])


def naive_count(text: str) -> int:
    """Characters over four. Here only so the report can show what it saved you from."""
    return round(len(text) / 4)


#: Measured on the committed fixture `japanese.txt` against all four real tokenizers. This is the
#: estimator's worst known case and the number is real, so the tool says it out loud rather than
#: quoting the corpus-wide band on text the corpus barely contained.
CJK_WORST_ERROR_PCT = 37.7
CJK_WARNING_SHARE = 0.10


def cjk_share(text: str) -> float:
    """Fraction of characters that are Han, Kana or Hangul."""
    if not text:
        return 0.0
    hits = sum(1 for ch in text if pretok._is_cjk(ch))
    return hits / len(text)
