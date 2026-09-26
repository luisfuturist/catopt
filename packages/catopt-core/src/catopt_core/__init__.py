"""catopt-core — the dependency-free semantic engine.

The categorical optimizer's abstract core: typed term IR
(:mod:`~catopt_core.ir`), canonical attribute contract
(:mod:`~catopt_core.attrs`), shape/type inference
(:mod:`~catopt_core.typing`), the e-graph engine
(:mod:`~catopt_core.egraph`), equational + non-local laws
(:mod:`~catopt_core.laws`), cost algebra
(:mod:`~catopt_core.cost`), rule synthesis
(:mod:`~catopt_core.meta`), the ``OpTable`` adapter registry
(:mod:`~catopt_core.ops`), and the ports layer
(:mod:`~catopt_core.ports`).  No torch, no numpy — pure Python.
"""

from catopt_core.egraph import EGraph, ENode, Rewrite
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.laws import CATEGORICAL_RULES

__all__ = [
    "CATEGORICAL_RULES",
    "IR",
    "Const",
    "EGraph",
    "ENode",
    "Op",
    "Param",
    "Rewrite",
    "TensorType",
    "Var",
]
