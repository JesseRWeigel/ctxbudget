"""The arithmetic. What goes in, what the reply needs, what is actually left.

The mistake this module exists to prevent: treating the context window as the input budget. It
is not. The reply comes out of the same window, and on several hosted models the reply is
separately capped far below the window, so the binding constraint is often the output cap and
not the window at all. A request that fills the window to the brim is accepted and then
truncated mid-sentence, which looks nothing like an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import Count, Counter

DATA = Path(__file__).parent / "data"

FITS = "fits"
OVER = "over"


class UnknownModel(KeyError):
    pass


def load_models(path: Path | None = None) -> dict:
    path = path or (DATA / "models.json")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Part:
    """One thing occupying the window."""

    label: str
    kind: str  # system | file | overhead
    count: Count
    text: str = ""
    path: str | None = None

    @property
    def tokens(self) -> int:
        return self.count.tokens


@dataclass
class Budget:
    model: str
    family: str
    window: int
    reserve: int
    reserve_source: str
    counting: str
    tokenizer: str
    band_pct: float | None
    parts: list[Part] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(part.tokens for part in self.parts)

    @property
    def usable_input(self) -> int:
        """The window minus what the reply needs. This is the number that decides fit."""
        return self.window - self.reserve

    @property
    def over_by(self) -> int:
        return max(0, self.input_tokens - self.usable_input)

    @property
    def spare(self) -> int:
        """Tokens left over after the input and the full reply allowance are both accounted for."""
        return self.usable_input - self.input_tokens

    @property
    def reply_room(self) -> int:
        """How long a reply actually fits, which is not the same as the reserve you asked for."""
        return max(0, min(self.reserve, self.window - self.input_tokens))

    @property
    def status(self) -> str:
        return FITS if self.over_by == 0 else OVER

    @property
    def worst_case_input(self) -> int:
        """Input tokens at the top of the measured error band.

        An estimate that fits by less than its own error bar has not been shown to fit.
        """
        if self.band_pct is None:
            return self.input_tokens
        return int(round(self.input_tokens * (1 + self.band_pct / 100)))

    @property
    def tight(self) -> bool:
        return self.status == FITS and self.worst_case_input > self.usable_input


def resolve_model(name: str, models: dict, window: int | None, reserve: int | None,
                  family: str | None) -> tuple[str, dict]:
    """Look a model up, allowing every field to be overridden.

    An unknown name with an explicit --window and --family is not an error. A table of context
    windows goes stale, and refusing to answer for anything not in it would make the tool useless
    the week after it shipped.
    """
    table = models["models"]
    if name in table:
        spec = dict(table[name])
    elif window is not None and family is not None:
        spec = {"family": family, "window": window, "max_output": reserve,
                "output_capped_separately": reserve is not None,
                "deployment": "unknown", "source": "given on the command line"}
    else:
        known = ", ".join(sorted(table))
        raise UnknownModel(
            f"unknown model {name!r}. Either pick one of: {known}. Or describe it: "
            f"--window N --family FAMILY [--reserve N].")
    if window is not None:
        spec["window"] = window
        spec["source"] = "window given on the command line"
    if family is not None:
        spec["family"] = family
    return name, spec


def default_reserve(spec: dict, requested: int | None) -> tuple[int, str]:
    """Decide how much of the window to keep back for the reply, and say why.

    Three cases, and they are genuinely different:
      - the caller said so, so use that;
      - the model caps the reply separately, so the cap is the right reserve;
      - the runtime shares one window between prompt and reply, in which case nothing external
        reserves anything and the number is a choice. A quarter of the window is the default and
        the report labels it a choice, so nobody reads it as a limit.
    """
    if requested is not None:
        return requested, "given on the command line"
    if spec.get("output_capped_separately") and spec.get("max_output"):
        return int(spec["max_output"]), "the model's own maximum output, from the model table"
    quarter = max(256, spec["window"] // 4)
    return quarter, ("a quarter of the window, chosen by this tool because the runtime caps "
                     "nothing separately. Set --reserve to the longest reply you expect")


def build(model_name: str, files: list[tuple[str, str]], system_prompt: str | None,
          models: dict, window: int | None = None, reserve: int | None = None,
          family: str | None = None, messages: int = 1, table_path: Path | None = None,
          allow_exact: bool = True) -> Budget:
    """Count every part and assemble the budget. `files` is a list of (label, text)."""
    name, spec = resolve_model(model_name, models, window, reserve, family)
    fam = spec["family"]
    family_meta = models["families"].get(fam, {})
    counting_family = family_meta.get("stand_in", fam)
    counter = Counter(counting_family, table_path=table_path, allow_exact=allow_exact)

    reserve_tokens, reserve_source = default_reserve(spec, reserve)
    band = None if counter.exact else counter.accuracy["p95_abs_pct"]
    tokenizer = counter.tokenizer_label
    counting = "exact" if counter.exact else "estimate"
    warnings: list[str] = []

    if family_meta.get("counting") == "unmeasured":
        counting = "unmeasured"
        band = None
        tokenizer = (f"{counter.tokenizer_label}, standing in for "
                     f"{family_meta.get('label', fam)}")
        warnings.append(family_meta.get("why", ""))

    budget = Budget(model=name, family=fam, window=int(spec["window"]),
                    reserve=int(reserve_tokens), reserve_source=reserve_source,
                    counting=counting, tokenizer=tokenizer, band_pct=band, warnings=warnings)

    if system_prompt is not None:
        budget.parts.append(Part("system prompt", "system", counter.count(system_prompt),
                                 text=system_prompt))
    for label, text in files:
        budget.parts.append(Part(label, "file", counter.count(text), text=text, path=label))

    overhead_spec = counter.families[counting_family]["chat_overhead"]
    total_overhead = overhead_spec["per_message"] * max(0, messages) + overhead_spec["priming"]
    if total_overhead:
        budget.parts.append(Part(
            f"chat template overhead, {messages} message(s), {overhead_spec['source']}",
            "overhead",
            Count(total_overhead, "exact" if overhead_spec["source"] == "measured" else "estimate",
                  overhead_spec["source"], None)))

    if spec.get("deployment") == "local":
        budget.warnings.append(models["local_runtime_warning"])
    if spec.get("note"):
        budget.warnings.append(spec["note"])
    return budget
