"""A probe the prover must ACCEPT: standard library only, arithmetic written out longhand."""

import json


def estimate_from_table(table_path, buckets):
    with open(table_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    table = payload["tables"]["cl100k_base"]
    return sum(table[key] * count for key, count in buckets.items())
