"""The reverse handler — gradient programs as derived terms.

The load-bearing claims: ``backward`` derives a correct gradient
program from the REVERSE table (checked against torch.autograd),
the result is an ordinary term the board optimizes unchanged, and
an op with no VJP row declines honestly.
"""

import pytest
import torch
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_discovery import meta_arena as ma
from catopt_discovery import training
from catopt_torch.meta_eval import _eval_allclose, _eval_term


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
        with pytest.raises(ValueError, match="no applicable VJP row"):
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


class TestShapedVJPs:
    """Reduction/view VJPs — the shape-binding extension."""

    def test_sum_full_reduce_matches_autograd(self):
        # f = sum(x): every input element's grad is the cotangent
        x = _v("x", 4, 4)
        term = _p("sum", x)
        grads = training.backward(term)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf),
            1e-4,
        )

    def test_sum_dim_keepdim_matches_autograd(self):
        x = _v("x", 4, 4)
        term = _p("sum", x, dim=1, keepdim=True)
        grads = training.backward(term)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf),
            1e-4,
        )

    def test_sum_dim_no_keepdim_declines(self):
        # keepdim=False needs an un-reduce the spec cannot spell
        x = _v("x", 4, 4)
        term = _p("sum", x, dim=1, keepdim=False)
        with pytest.raises(ValueError, match="declines"):
            training.backward(term)

    def test_mean_full_reduce_matches_autograd(self):
        x = _v("x", 4, 4)
        term = _p("mean", x)
        grads = training.backward(term)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf),
            1e-4,
        )

    def test_reshape_matches_autograd(self):
        x = _v("x", 4, 4)
        term = _p("reshape", x, shape=(2, 8))
        g0 = _p("broadcast_to", Const(1.0), shape=(2, 8))
        grads = training.backward(term, cotangent=g0)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(2, 8)),
            1e-4,
        )

    def test_permute_matches_autograd(self):
        x = _v("x", 2, 4, 8)
        term = _p("permute", x, dims=(1, 2, 0))
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 8, 2))
        grads = training.backward(term, cotangent=g0)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(4, 8, 2)),
            1e-4,
        )

    def test_expand_matches_autograd(self):
        # x (4,1) expanded to (4,8): the grad sums the copies back
        x = _v("x", 4, 1)
        term = _p("expand", x, shape=(4, 8))
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 8))
        grads = training.backward(term, cotangent=g0)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(4, 8)),
            1e-4,
        )

    def test_broadcast_to_matches_autograd(self):
        x = _v("x", 1, 8)
        term = _p("broadcast_to", x, shape=(4, 8))
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 8))
        grads = training.backward(term, cotangent=g0)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(4, 8)),
            1e-4,
        )


class TestTrainingProbe:
    def test_backward_programs_certify_on_the_board(self):
        # real-shaped forwards → derived gradients → the board's
        # saturate/extract/certify pipeline, unchanged
        rows = training.training_probe()
        names = {r["name"] for r in rows}
        assert {"silu_fwd", "softmax_fwd", "linear_fwd"} <= names
        certified = [r for r in rows if r.get("cert")]
        assert len(certified) == 3
        # the e-graph grew: laws fired on the gradient programs
        assert all(r["n_enodes"] > 5 for r in certified)
        text = training.training_table(rows)
        assert "softmax_fwd" in text

    def test_relu_matches_autograd(self):
        x = _v("x", 4, 4)
        term = _p("relu", x)
        grads = training.backward(term)
        env = _env(term)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env).double(),
            _autograd(term, env, leaf),
            1e-9,
        )

    def test_unsqueeze_squeeze_match_autograd(self):
        x = _v("x", 4, 4)
        term = _p("squeeze", _p("unsqueeze", x, dim=1), dim=1)
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(term, cotangent=g0)
        env = _env(term, dtype=torch.float32)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(4, 4)),
            1e-4,
        )

    def test_softmax_matches_autograd(self):
        # d softmax(x)_i = s_i (g_i - Σ_j g_j s_j) — the VJP re-spells
        # the forward output twice; the joint e-graph is where those
        # dedup.  dim plumbs through ``$dim``.
        x = _v("x", 4, 4)
        term = _p("softmax", x, dim=-1)
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(term, cotangent=g0)
        env = _env(term)
        leaf = training.leaves_of(term)["x"]
        assert _eval_allclose(
            _eval_term(grads["x"], env),
            _autograd(term, env, leaf, torch.ones(4, 4).double()),
            1e-9,
        )

    def test_log_softmax_matches_autograd(self):
        # d lsm_i = g_i - s_i Σ_j g_j with s = exp(lsm(x))
        x = _v("x", 4, 4)
        term = _p("log_softmax", x, dim=-1)
        g0 = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(term, cotangent=g0)
        env = _env(term)
        leaf = training.leaves_of(term)["x"]
        tenv = {k: v.clone().requires_grad_(True) for k, v in env.items()}
        out = torch.log_softmax(tenv[leaf], dim=-1)
        out.backward(torch.ones_like(out))
        assert _eval_allclose(
            _eval_term(grads["x"], env), tenv[leaf].grad, 1e-9
        )


class TestRealModels:
    """The domain claim end-to-end: a torch module's backward."""

    def test_linear_relu_backward_matches_autograd(self):
        import torch.nn as nn
        from catopt_torch.adapters import TorchSource

        torch.manual_seed(0)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.fc(x))

        mod = MLP().eval()
        xin = torch.randn(4, 8)
        ir, leaves = TorchSource().to_ir(mod, xin)
        grads = training.backward(
            ir.root,
            cotangent=Op.make(
                "broadcast_to", Const(1.0), shape=tuple(xin.shape)
            ),
        )
        # the real claim: params got gradient terms
        names = set(grads)
        assert ir.params and names & set(ir.params)

        # numeric check vs autograd on the same weights
        mod2 = MLP().eval()
        mod2.load_state_dict(mod.state_dict())
        x2 = xin.clone().requires_grad_(True)
        torch.manual_seed(0)
        out = mod2(x2)
        out.sum().backward()
        leaves_ = training.leaves_of(ir.root)
        env = {lf: v for lf, v in zip(leaves_.values(), [xin])}
        env.update(
            {
                leaves_[n]: t
                for n, t in mod.state_dict().items()
                if n in leaves_
            }
        )
        for name, tensor in mod.state_dict().items():
            if name not in grads:
                continue
            assert _eval_allclose(
                _eval_term(grads[name], env).double(),
                dict(
                    (n, p.grad)
                    for n, p in mod2.named_parameters()
                    if p.grad is not None
                ).get(name),
                1e-5,
            )


class TestJointProbe:
    """Forward+backward on one e-graph — the cross-boundary claim."""

    def test_joint_rows(self):
        rows = training.joint_probe()
        names = {r["name"] for r in rows}
        assert {"silu_fwd", "softmax_fwd", "linear_fwd"} <= names
        for r in rows:
            # sharing never costs more than two pipelines
            assert r["joint"] <= r["separate"]
            assert r["enodes_joint"] <= r["enodes_sep"]
        # measured sharing: at least one case bills shared work once
        assert any(r["joint"] < r["separate"] for r in rows)
        assert all(r["shared"] > 0 for r in rows)
        text = training.joint_table(rows)
        assert "linear_fwd" in text

    def test_joint_declines(self):
        rows = training.joint_probe(
            forwards=[
                ("const", Const(1.0), "x"),
                ("no_vjp", _p("nonexistent_op", _v("x", 2, 2)), "x"),
            ]
        )
        assert [r["name"] for r in rows] == ["const", "no_vjp"]
        assert all("declined" in r for r in rows)
        text = training.joint_table(rows)
        assert "declined" in text

    def test_joint_extraction_stays_correct(self):
        # the winning case end-to-end: extract grads THROUGH the
        # shared e-graph, still numerically equal to autograd
        from catopt_core.cost.basic import count_cost
        from catopt_core.egraph import EGraph
        from catopt_core.laws import DEFAULT

        x, w, b = _v("x", 4, 4), _v("w", 4, 4), _v("b", 4, 4)
        fwd = _p("tanh", _p("add", _p("matmul", x, w), b))
        cot = _p("broadcast_to", Const(1.0), shape=(4, 4))
        grads = training.backward(fwd, cotangent=cot)
        eg = EGraph()
        rf = eg.add_term(fwd)
        rg = {n: eg.add_term(g) for n, g in grads.items()}
        eg.run(DEFAULT, rf, max_nodes=20_000)
        env = _env(fwd, dtype=torch.float32)
        leaves = training.leaves_of(fwd)
        for name, rid in rg.items():
            got = _eval_term(eg.extract_best(rid, count_cost), env)
            true = _autograd(fwd, env, leaves[name], torch.ones(4, 4))
            assert _eval_allclose(got, true, 1e-4)

    def test_main_joint(self, capsys):
        assert training.main(["--joint"]) == 0
        assert "separate" in capsys.readouterr().out

    def test_main_default(self, capsys, monkeypatch):
        monkeypatch.setattr(
            training,
            "training_probe",
            lambda **kw: [
                {
                    "name": "stub",
                    "baseline": 1.0,
                    "cost": 1.0,
                    "cert": True,
                    "n_enodes": 1,
                }
            ],
        )
        assert training.main([]) == 0
        assert "baseline" in capsys.readouterr().out
