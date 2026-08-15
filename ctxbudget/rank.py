"""What to cut first.

Largest file first is the obvious rule and it is wrong in a specific, common way: the biggest
file in a bundle is often the schema, the router or the type definitions that everything else
refers to, and the small file nobody has touched in two years is the one you can lose. Ranking
by size cuts the load-bearing wall and keeps the dead weight.

The rule here is tokens returned per unit of demand that would actually be lost:

    cut_score = tokens / ((1 + inbound + query_demand) * (1 - redundancy))

Every term is a count of something observable in the input, not a tuned weight.

    tokens        what you get back by removing it. Cutting is worth doing in proportion to this.
    inbound       how many OTHER parts of the context refer to this one, by its filename, its
                  module name, or a name it defines. This is the demand signal. A file that
                  three other files import is holding them up. A file nothing mentions is
                  carrying itself only.
    query_demand  how much of the task description this file answers to. A term matching the
                  path or a defined name counts double, since that is a much stronger signal
                  than the same word appearing somewhere in the body.
    redundancy    the largest line overlap with another included file, between 0 and 1. It
                  divides rather than adds, because it says how much of this file's demand
                  survives the cut. Remove one of two near-identical files and the model can
                  still read the content in the other, so almost none of the demand is lost and
                  the file is nearly free to drop. At redundancy 0 the term is 1 and does
                  nothing.

Two free parameters, both structural. The 1 in the first factor stops a file with no demand from
dividing by zero and sets how much one reference protects a file. The second factor is floored at
0.05 so an exact duplicate scores high rather than infinite.

What this rule cannot do. It reads structure and not meaning. A file can be essential for a
reason no import graph records, and this will rank it for the chop. So the output prints the
counts behind every rank and never removes anything on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Names too common to count as evidence of a reference. Two kinds are in here: English function
# words, and identifiers so generic that finding one in another file says nothing about whether
# that file depends on this one.
STOPNAMES = {
    "self", "this", "true", "false", "null", "none", "return", "import", "export", "from",
    "const", "class", "function", "async", "await", "type", "interface", "value", "data",
    "result", "error", "config", "index", "main", "test", "name", "item", "list", "dict",
    "string", "number", "object", "array", "default", "module", "params", "props", "state",
    "with", "that", "have", "which", "when", "then", "than", "into", "your", "will", "would",
    "there", "their", "about", "other", "some", "what", "does", "make", "such", "only", "them",
    "they", "been", "were", "also", "more", "most", "over", "each", "these",
    "length", "size", "count", "start", "stop", "offset", "label", "total", "values", "keys",
    "items", "text", "path", "file", "line", "lines", "read", "write", "open", "close", "load",
    "save", "print", "input", "output", "table", "field", "fields", "entry", "record", "records",
    "message", "messages", "request", "response", "status", "version", "options", "args",
}

DEFINITION_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M),
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M),
    re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*[:=]", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:function|class)\s+([A-Za-z_$]\w*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$]\w*)", re.M),
    re.compile(r"^\s*CREATE\s+(?:TABLE|VIEW|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_]\w*)",
               re.M | re.I),
)


@dataclass
class Signals:
    label: str
    tokens: int
    inbound: int
    inbound_via: list[str]
    query_demand: float
    query_terms: list[str]
    redundancy: float
    redundant_with: str | None
    repeat_ratio: float

    @property
    def demand(self) -> float:
        return 1.0 + self.inbound + self.query_demand

    @property
    def surviving_demand(self) -> float:
        """Demand that would actually be lost by cutting this, after duplication elsewhere."""
        return self.demand * max(0.05, 1.0 - self.redundancy)

    @property
    def score(self) -> float:
        return self.tokens / self.surviving_demand

    def reason(self) -> str:
        bits: list[str] = []
        if self.inbound == 0:
            bits.append("nothing else in the context refers to it")
        else:
            via = ", ".join(self.inbound_via[:3])
            bits.append(f"{self.inbound} other part(s) refer to it via {via}")
        if self.query_terms:
            bits.append("the task mentions " + ", ".join(self.query_terms[:3]))
        else:
            bits.append("no term from the task appears in it")
        if self.redundancy >= 0.30 and self.redundant_with:
            bits.append(f"{round(self.redundancy * 100)}% line overlap with {self.redundant_with}")
        if self.repeat_ratio >= 0.30:
            bits.append(f"{round(self.repeat_ratio * 100)}% of its lines repeat inside it")
        return "; ".join(bits)


def defined_names(text: str) -> set[str]:
    names: set[str] = set()
    for pattern in DEFINITION_PATTERNS:
        for match in pattern.findall(text):
            if len(match) >= 4 and match.lower() not in STOPNAMES:
                names.add(match)
    return names


def path_names(label: str) -> set[str]:
    """The filename and its stem, plus a snake or camel split, all as reference candidates."""
    base = label.replace("\\", "/").rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    names = {base, stem}
    return {name for name in names if len(name) >= 4}


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[A-Za-z_$][\w$]{3,}", text)}


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _shingles(text: str) -> set[str]:
    lines = _lines(text)
    return set(lines) if len(lines) < 3 else {
        "\n".join(lines[index:index + 2]) for index in range(len(lines) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def query_terms(query: str) -> list[str]:
    seen: list[str] = []
    for raw in re.findall(r"[A-Za-z_][\w_]{3,}", query or ""):
        term = raw.lower()
        if term in STOPNAMES or term in seen:
            continue
        seen.append(term)
    return seen


def analyse(parts: list[tuple[str, str]], query: str = "",
            protected: set[str] | None = None) -> list[Signals]:
    """Score every part. `parts` is a list of (label, text); returns them in cut order."""
    protected = protected or set()
    texts = {label: text for label, text in parts}
    names = {label: defined_names(text) | path_names(label) for label, text in parts}
    words = {label: _words(text) for label, text in parts}
    lowered = {label: text.lower() for label, text in parts}
    shingles = {label: _shingles(text) for label, text in parts}
    terms = query_terms(query)

    # A name that turns up in most of the context is not evidence that anything depends on the
    # file that happens to define it. Drop names carried by more than half the other parts, which
    # is the same reason a search engine discounts a word appearing in every document.
    generic: set[str] = set()
    if len(parts) > 2:
        appearances: dict[str, int] = {}
        for name_set in names.values():
            for name in name_set:
                appearances[name] = 0
        for name in appearances:
            appearances[name] = sum(1 for label in texts if name in words[label])
        cutoff = max(2, (len(parts) + 1) // 2)
        generic = {name for name, seen in appearances.items() if seen > cutoff}

    signals: list[Signals] = []
    for label, text in parts:
        inbound = 0
        via: list[str] = []
        for other_label in texts:
            if other_label == label:
                continue
            hits = (names[label] - generic) & words[other_label]
            if hits:
                inbound += 1
                via.extend(sorted(hits)[:2])

        demand = 0.0
        matched: list[str] = []
        haystack_strong = (label + " " + " ".join(sorted(names[label]))).lower()
        body_hits = 0
        for term in terms:
            if term in haystack_strong:
                demand += 2.0
                matched.append(term)
            elif term in lowered[label] and body_hits < 3:
                demand += 1.0
                body_hits += 1
                matched.append(term)

        redundancy = 0.0
        twin = None
        for other_label in texts:
            if other_label == label:
                continue
            overlap = _jaccard(shingles[label], shingles[other_label])
            if overlap > redundancy:
                redundancy = overlap
                twin = other_label

        lines = _lines(text)
        repeat = 0.0 if not lines else 1.0 - (len(set(lines)) / len(lines))

        signals.append(Signals(
            label=label, tokens=0, inbound=inbound, inbound_via=sorted(set(via)),
            query_demand=demand, query_terms=matched, redundancy=round(redundancy, 4),
            redundant_with=twin if redundancy >= 0.30 else None,
            repeat_ratio=round(repeat, 4)))
    return signals


def rank(parts: list[tuple[str, str]], tokens: dict[str, int], query: str = "") -> list[Signals]:
    signals = analyse(parts, query)
    for signal in signals:
        signal.tokens = tokens.get(signal.label, 0)
    signals.sort(key=lambda s: (-s.score, s.label))
    return signals


def greedy_cut(signals: list[Signals], over_by: int) -> list[Signals]:
    """The shortest prefix of the cut order that gets the input back under budget."""
    if over_by <= 0:
        return []
    chosen: list[Signals] = []
    saved = 0
    for signal in signals:
        if saved >= over_by:
            break
        chosen.append(signal)
        saved += signal.tokens
    return chosen


def largest_first(signals: list[Signals]) -> list[Signals]:
    """The naive baseline, kept so the ranking can be compared against it rather than praised."""
    return sorted(signals, key=lambda s: (-s.tokens, s.label))
