"""A probe the prover must REJECT: a relative import reaches somewhere it cannot see."""

from ..ctxbudget import pretok


def chunks(text):
    return pretok.chunks(text)
