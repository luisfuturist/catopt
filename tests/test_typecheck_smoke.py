"""Smoke test: the package and its public API import cleanly.

Pairs with the pyright ratchet in pyproject.toml — if module-level code
breaks at import time, this fails before any typecheck output matters.
"""


def test_modules_import():
    import catopt

    assert catopt.EGraph
    assert catopt.IR
    assert catopt.IRModule
    assert catopt.__version__


def test_submodules_import():
    import catopt.egraph
    import catopt.ir
    import catopt.laws
    import catopt.models
    import catopt.optimize
    import catopt.torch_bridge

    assert catopt.egraph.ENode
    assert catopt.optimize.optimize_model
