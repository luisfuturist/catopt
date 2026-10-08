"""The reverse handler — gradient programs as derived terms.

The load-bearing claims: ``backward`` derives a correct gradient
program from the REVERSE table (checked against torch.autograd),
the result is an ordinary term the board optimizes unchanged, and
an op with no VJP row declines honestly.
"""

import torch
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_torch.meta_eval import _eval_allclose, _eval_term
import pytest

from catopt_discovery import meta_arena as ma
from catopt_discovery import training


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _env(term, dtype=torch.float64):
    torch.manual_seed(0)
    return {
        lf: torch.randn(lf.typ.shape, dtype=dtype)
        for lf in training.leaves_of(term).values()
    }


def _autograd(term, env, leaf, cotangent=None):
    """torch.autograd's answer for ``d(term)/d(leaf)``."""
    tenv = {k: v.clone().requires_grad_(True) for k, v in env.items()}
    out = _eval_term(term, tenv)
    g = torch.ones_like(out) if cotangent is None else cotangent
    return torch.autograd.grad(out, tenv[leaf], grad_outputs=g)[0]


class TestBackward:
    def test_elementwise_chain_matches_autograd(self):
        # f = x * sigmoid(x):  df/dx = sig + x*sig*(1-sig)
        x = _v("x", 4, 4)
        term = _p("mul", x, _p("sigmoid", x))
        grads = training.backward(term)
        assert set(grads) == {"x"}
        env = _env(term)
        leaf = training.leaves_of(term)["x"]
        mine = _eval_term(grads["x"], env)
        true = _autograd(term, env, leaf)
        assert _eval_allclose(mine, true, 1e-9)

    def test_shared_leaf_accumulates_contributions(self):
        # f = x + x*x: three cotangent paths must sum
        x = _v("x", 4, 4)
        term = _p("add", x, _p("mul", x, x))
        grads = training.backward(term)
        env = _env(term)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf),
            1e-9,
        )

    def test_matmul_matches_autograd_with_shaped_cotangent(self):
        # f = x @ W: dL/dx = g @ W^T, dL/dW = x^T @ g
        x, w = _v("x", 4, 8), _v("w", 8, 4)
        term = _p("matmul", x, w)
        ones = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(term, cotangent=ones)
        assert set(grads) == {"x", "w"}
        # Const(1.0) evals float32 — keep the env in dtype so the
        # broadcast cotangent matmul matches
        env = _env(term, dtype=torch.float32)
        g0 = torch.ones(4, 4, dtype=torch.float32)
        for name in ("x", "w"):
            leaf = training.leaves_of(term)[name]
            assert _eval_allclose(
                _eval_term(grads[name], env),
                _autograd(term, env, leaf, cotangent=g0),
                1e-4,
            )

    def test_mlp_chain_matches_autograd(self):
        # f = (xW1 + b) relu-ish:  tanh(xW+b) as a deeper chain
        x, w, b = _v("x", 4, 8), _v("w", 8, 4), _v("b", 4, 4)
        term = _p("tanh", _p("add", _p("matmul", x, w), b))
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(term, cotangent=g0)
        assert set(grads) == {"x", "w", "b"}
        env = _env(term, dtype=torch.float32)
        tg0 = torch.ones(4, 4, dtype=torch.float32)
        for name in ("x", "w", "b"):
            leaf = training.leaves_of(term)[name]
            assert _eval_allclose(
                _eval_term(grads[name], env),
                _autograd(term, env, leaf, cotangent=tg0),
                1e-4,
            )

    def test_unknown_op_declines(self):
        term = _p("freeze_dry", _v("x", 4, 4))
        with pytest.raises(ValueError, match="no VJP row"):
            training.backward(term)

    def test_param_leaves_get_grads(self):
        from catopt_core.ir import Param

        x = _v("x", 4, 4)
        p = Param("scale", TensorType((4, 4)))
        term = _p("mul", x, p)
        grads = training.backward(term)
        assert "scale" in grads

    def test_backward_term_is_a_board_citizen(self):
        # the derived gradient program lives on the board: laws
        # fire, extraction certifies — no new semantics stack
        from catopt_core.cost.basic import count_cost

        x = _v("x", 4, 4)
        fwd = _p("mul", x, _p("sigmoid", x))
        g = training.backward(fwd)["x"]
        arena = ma.MetaArena(
            g,
            supported=None,
            cost_fn=count_cost,
            max_specs=8,
        )
        arena.step(ma.Action.saturate(budget=512))
        _, rep = arena.step(ma.Action.extract())
        assert rep.applied and rep.certificate_ok
        # the certified extraction stays numerically the gradient
        env = _env(fwd)
        best = arena.eg.extract_best(arena.root, arena.feasible_cost)
        assert _eval_allclose(
            _eval_term(best, env),
            _autograd(fwd, env, training.leaves_of(fwd)["x"]),
            1e-9,
        )
