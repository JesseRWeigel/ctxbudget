"""Pretokenization, the step every BPE tokenizer does before it merges anything.

A BPE tokenizer does not run over the raw string. It first splits the text with a fixed regex
into chunks, and merges only ever happen inside a chunk. So the chunk sequence is a hard
structural skeleton of the token sequence, and the number of chunks is a lower bound on the
number of tokens. Counting characters and dividing by four throws that structure away, which is
why it is wrong by different amounts on prose, on minified JavaScript and on Japanese.

The regex below is a port of the `cl100k_base` pattern to Python's `re`, which has no `\\p{L}`.
The substitutions:

    \\p{L}   ->  [^\\W\\d_]     (a word character that is neither a digit nor an underscore)
    \\p{N}   ->  \\d

Both are exact for the Basic Multilingual Plane under `re.UNICODE`, which is Python 3's default
for `str` patterns.

The other families this project measures (`o200k_base`, Qwen 2.5, Llama 3) use different split
regexes. This module does not try to reproduce each of them. It uses one skeleton and lets the
per-family fitted table absorb the difference, and then the error of doing that is measured per
family rather than assumed. See `fixtures/evidence/calibration.md`.
"""

from __future__ import annotations

import re

# A single character that is not CR, not LF, not a letter and not a digit. Python cannot negate a
# positive class inline, so it is spelled as a dot guarded by three negative lookaheads.
_NOT_LETTER_DIGIT_CRLF = r"(?:(?![\r\n])(?![^\W\d_])(?!\d).)"
# The same thing but also excluding whitespace, for the punctuation-run branch.
_NOT_LETTER_DIGIT_SPACE = r"(?:(?!\s)(?![^\W\d_])(?!\d).)"

PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|" + _NOT_LETTER_DIGIT_CRLF + r"?[^\W\d_]+"
    r"|\d{1,3}"
    r"|[ ]?" + _NOT_LETTER_DIGIT_SPACE + r"+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Class names are part of the committed table's key space. Do not rename without refitting.
WORD_ASCII = "word_ascii"
WORD_WIDE = "word_wide"
WORD_CJK = "word_cjk"
NUMBER = "number"
PUNCT_ASCII = "punct_ascii"
PUNCT_WIDE = "punct_wide"
SPACE = "space"

RANDOM = "random"

CLASSES = (WORD_ASCII, WORD_WIDE, WORD_CJK, NUMBER, PUNCT_ASCII, PUNCT_WIDE, SPACE, RANDOM)

# Byte length at which each class stops getting its own bucket and switches to a per-byte tail
# rate. Chosen so that every bucket below the cap is populated by the calibration corpus.
CAPS = {
    WORD_ASCII: 16,
    WORD_WIDE: 12,
    WORD_CJK: 12,
    NUMBER: 4,
    PUNCT_ASCII: 10,
    PUNCT_WIDE: 8,
    SPACE: 10,
    RANDOM: 12,
}

# A long alphanumeric run with both letters and digits in it: an integrity hash, a base64 blob,
# a content-addressed filename, a minified identifier table. BPE has never seen these sequences,
# so it falls back to very short pieces and they cost two to three bytes per token instead of
# the four to six an English word costs.
#
# The reason this needs its own class rather than a per-byte surcharge: the split regex shreds a
# hash into many SHORT chunks, alternating letter runs and digit runs of at most three, none of
# them long enough to reach a tail rate. Priced from the ordinary buckets they look like little
# words and cost about one token each, and measured against a real package-lock.json that
# under-counted by 21 percent. Classifying by the run a chunk sits inside, rather than by the
# chunk alone, is what fixes it.
_RANDOM_RUN = re.compile(r"[A-Za-z0-9+/=_-]{20,}")


def random_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in _RANDOM_RUN.finditer(text):
        body = match.group(0)
        if any(ch.isdigit() for ch in body) and any(ch.isalpha() for ch in body):
            spans.append(match.span())
    return spans

# Han, Hiragana, Katakana, Hangul and the CJK punctuation around them. Splitting these out of
# the general non-ASCII class was worth doing: a run of Japanese has no spaces in it, so the
# whole sentence arrives as one chunk and its cost is carried entirely by the per-byte tail
# rate. Fitted together with accented Latin, that rate came out roughly three times too low and
# Japanese was underestimated by about half. Separated, the same fixture lands far closer.
_CJK_RANGES = (
    (0x2E80, 0x9FFF), (0xA960, 0xA97F), (0xAC00, 0xD7FF),
    (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
    (0x1B000, 0x1B2FF), (0x20000, 0x3FFFF),
)


def _is_cjk(ch: str) -> bool:
    point = ord(ch)
    return any(low <= point <= high for low, high in _CJK_RANGES)


def classify(chunk: str) -> str:
    """Bucket one pretoken chunk into a class.

    Byte length rather than character length is what matters downstream, because BPE merges run
    over UTF-8 bytes. A three-byte CJK character is three merge candidates, not one.
    """
    if not chunk:
        raise ValueError("empty chunk, the pretokenizer never emits one")
    wide = not chunk.isascii()
    if chunk.strip() == "":
        return SPACE
    stripped = chunk.lstrip()
    if stripped and stripped[0].isdigit():
        return NUMBER
    # A word chunk is letters, optionally with one leading non-letter character.
    core = stripped[1:] if (stripped and not (stripped[0].isalpha())) else stripped
    cjk = wide and any(_is_cjk(ch) for ch in chunk)
    if core and all(ch.isalpha() for ch in core):
        if cjk:
            return WORD_CJK
        return WORD_WIDE if wide else WORD_ASCII
    if cjk:
        return WORD_CJK
    return PUNCT_WIDE if wide else PUNCT_ASCII


def chunks(text: str) -> list[str]:
    """Split text into pretoken chunks. Concatenating the result reproduces the input exactly."""
    return PATTERN.findall(text)


def bucket_key(cls: str, nbytes: int) -> str:
    """The table key for a chunk of this class and byte length, saturating at the class cap."""
    cap = CAPS[cls]
    return f"{cls}:{min(nbytes, cap)}"


def tail_key(cls: str) -> str:
    """The table key for the per-byte rate charged on bytes beyond the class cap."""
    return f"{cls}:tail"


def walk(text: str):
    """Yield (chunk, class, byte length) for the whole text.

    The one place classification happens, so the calibration and the runtime estimate cannot
    drift apart. Position matters here: a chunk inside a hash is priced differently from the
    identical chunk in a sentence.
    """
    spans = random_spans(text)
    index = 0
    for match in PATTERN.finditer(text):
        chunk = match.group(0)
        start, end = match.span()
        while index < len(spans) and spans[index][1] <= start:
            index += 1
        inside = index < len(spans) and spans[index][0] <= start and end <= spans[index][1]
        cls = RANDOM if inside else classify(chunk)
        yield chunk, cls, len(chunk.encode("utf-8"))


def features(text: str) -> dict[str, float]:
    """Count the buckets in a piece of text.

    Returns a mapping from table key to a count. Bucket keys carry an integer count of chunks,
    tail keys carry a count of overflow bytes. The estimate is the dot product of this mapping
    with a fitted table, which is what makes the table refittable per model family without
    touching this function.
    """
    counts: dict[str, float] = {}
    for _, cls, nbytes in walk(text):
        key = bucket_key(cls, nbytes)
        counts[key] = counts.get(key, 0.0) + 1.0
        cap = CAPS[cls]
        if nbytes > cap:
            tkey = tail_key(cls)
            counts[tkey] = counts.get(tkey, 0.0) + (nbytes - cap)
    return counts
