"""A probe the prover must REJECT: it reaches into the package under test."""

from ctxbudget.tokens import Counter


def count(text):
    return Counter("cl100k_base").count(text).tokens
