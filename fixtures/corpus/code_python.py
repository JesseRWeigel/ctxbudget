"""A small module with the shapes a tokenizer finds ordinary in Python source."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    label: str
    start_offset: int
    end_offset: int

    @property
    def length(self) -> int:
        return self.end_offset - self.start_offset


def normalise_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in raw.values())
    if total <= 0.0:
        return {key: 0.0 for key in raw}
    return {key: max(0.0, value) / total for key, value in raw.items()}


def harmonic_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1.0 / value for value in positive)


def bucket_by_length(segments: list[Segment], width: int = 64) -> dict[int, list[Segment]]:
    buckets: dict[int, list[Segment]] = {}
    for segment in segments:
        index = segment.length // width
        buckets.setdefault(index, []).append(segment)
    return buckets


def summarise(segments: list[Segment]) -> str:
    if not segments:
        return json.dumps({"count": 0})
    lengths = [segment.length for segment in segments]
    payload = {
        "count": len(segments),
        "total": sum(lengths),
        "mean": round(sum(lengths) / len(lengths), 3),
        "stdev": round(math.sqrt(sum((x - sum(lengths) / len(lengths)) ** 2
                                     for x in lengths) / len(lengths)), 3),
        "labels": sorted({segment.label for segment in segments}),
    }
    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    demo = [Segment("header", 0, 128), Segment("body", 128, 4096), Segment("footer", 4096, 4200)]
    print(summarise(demo))
