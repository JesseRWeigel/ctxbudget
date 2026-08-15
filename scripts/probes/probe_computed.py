"""A probe the prover must REJECT: the import is hidden behind importlib."""

import importlib


def count(text):
    module = importlib.import_module("ctxbudget.tokens")
    return module.Counter("cl100k_base").count(text).tokens
