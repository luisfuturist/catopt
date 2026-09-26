"""Concrete numeric evaluation for catopt-core's rule synthesis.

``catopt_core.meta.synthesize_rules`` validates a candidate rewrite by
instantiating it on random fp64 tensors and comparing both sides with
``allclose``.  The tensor machinery — ``torch.randn`` / ``torch.tensor``
/ ``torch.allclose`` plus the ``_IR_TO_TORCH`` lowering bindings — is
torch-specific, so it lives here in the adapter.  Importing this module
registers the backend with core
(:func:`catopt_core.meta.register_concrete_eval`); core imports neither
``torch`` nor ``catopt_torch``.
"""

from __future__ import annotations

from typing import Any

import torch
from catopt_core.ir import Const, Op, Param, Var
from catopt_core.meta import register_concrete_eval

__all__ = ["TorchConcreteEval"]


def _eval_term(term: Any, env: dict) -> Any:
    """Evaluate *term* against *env* (leaf -> tensor) through the
    ``_IR_TO_TORCH`` bindings.  Returns a tensor or a nested tuple
    (aff/om carriers)."""
    from catopt_torch.torch_bridge import _IR_TO_TORCH

    if isinstance(term, Const):
        return torch.tensor(term.value)
    if isinstance(term, (Var, Param)):
        return env[term]
    if isinstance(term, Op):
        fn = _IR_TO_TORCH.get(term.op)
        if fn is None:
            raise KeyError(term.op)
        args = [_eval_term(a, env) for a in term.args]
        return fn(*args, **dict(term.attrs))
    raise TypeError(term)


def _eval_allclose(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _eval_allclose(x, y, tol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.allclose(a, b, atol=tol, rtol=tol))
    return False


class TorchConcreteEval:
    """The ``catopt_core.meta.ConcreteEval`` implementation over torch.

    Registered with core at import (see the module docstring).
    """

    def make_env(self, leaf_shapes: dict) -> dict:
        return {
            leaf: torch.randn(*shape, dtype=torch.float64)
            for leaf, shape in leaf_shapes.items()
        }

    def eval_term(self, term: Any, env: dict) -> Any:
        return _eval_term(term, env)

    def allclose(self, a: Any, b: Any, tol: float) -> bool:
        return _eval_allclose(a, b, tol)


register_concrete_eval(TorchConcreteEval())
