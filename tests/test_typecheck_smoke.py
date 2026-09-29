"""Smoke test: the domain packages and their public API import cleanly.

Pairs with the ty ratchet in pyproject.toml — if module-level code
breaks at import time, this fails before any typecheck output matters.
Plan 0008 removed the ``catopt`` façade: ``import catopt`` must fail
and every public name lives at its real package path.
"""

import importlib
import sys

import pytest


def test_modules_import():
    import catopt_core
    import catopt_orchestrator
    import catopt_torch
    from catopt_core.egraph import EGraph
    from catopt_core.ir import IR
    from catopt_orchestrator import Optimizer



    from catopt_torch.torch_bridge import IRModule

    assert catopt_core and catopt_orchestrator and catopt_torch
    assert EGraph and IR and IRModule
    assert Optimizer


def test_submodules_import():
    import catopt_core.egraph
    import catopt_core.ir
    import catopt_core.laws
    import catopt_orchestrator.optimize
    import catopt_torch.models
    import catopt_torch.torch_bridge

    assert catopt_core.egraph.ENode
    assert catopt_orchestrator.optimize.Optimizer


def test_catopt_facade_is_gone():
    """Plan 0008 deleted the ``catopt`` façade — the name fails."""
    sys.modules.pop("catopt", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("catopt")
