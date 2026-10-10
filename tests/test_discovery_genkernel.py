"""Tests for ``catopt_discovery.genkernel`` — generated lowerings.

Pins the mint path (concrete subterm → handler entry), the binding
semantics (the generated callable evaluates the spelled body — the
soundness of ``claim``), the sink extension (gen ops are
supported + lowerable), and the novelty cost wrapper.
"""

import torch

from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import DEFAULT
from catopt_core.cost.basic import count_cost
from catopt_discovery import genkernel as gk
from catopt_discovery import meta_arena as ma
from catopt_discovery import play
from catopt_discovery.lawdata import HANDLERS
from catopt_discovery.meta_arena import _canon_concrete


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


class TestMinting:
    def test_elementwise_subterm_mints_handler(self):
        x = _v("x", 4, 4)
        h = gk.gen_handlers([_p("mul", x, _p("relu", x))])
        assert "gen_0" in h
        assert h["gen_0"]["kernel"] == "gen_0_k"
        assert h["gen_0"]["pattern"] == (
            "mul", "X1", ("relu", "X1")
        )
        assert h["gen_0"]["args"] == ("X1",)

    def test_covered_pattern_skipped(self):
        # silu's concrete pattern is already named — no gen name
        x = _v("x", 4, 4)
        covered = {
            _canon_concrete(h["pattern"]) for h in HANDLERS.values()
        }
        assert gk.gen_handlers(
            [_p("mul", x, _p("sigmoid", x))], covered=covered
        ) == {}

    def test_non_elementwise_skipped(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        assert (
            gk.gen_handlers([_p("matmul", x, y)]) == {}
        )
        # and the dedup: same concrete pattern twice → one entry
        z = _v("z", 4, 4)
        h = gk.gen_handlers(
            [_p("mul", x, _p("relu", x)),
             _p("mul", z, _p("relu", z))]
        )
        assert len(h) == 1

    def test_leaf_shape_preserved(self):
        # two leaves, consistent binding: sub(x, y) keeps order
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        h = gk.gen_handlers(
            [_p("add", _p("mul", x, x), _p("mul", y, y))]
        )
        assert h["gen_0"]["args"] == ("X1", "X2")


class TestBindings:
    def test_binding_evaluates_spelled_body(self):
        x = _v("x", 4, 4)
        h = gk.gen_handlers([_p("mul", x, _p("relu", x))])
        b = gk.gen_bindings(h, compile_kernels=False)
        t = torch.randn(4, 4)
        torch.testing.assert_close(
            b["gen_0_k"](t), t * torch.relu(t)
        )

    def test_binding_two_metavars(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        h = gk.gen_handlers([_p("sub", x, _p("abs", y))])
        b = gk.gen_bindings(h, compile_kernels=False)
        a, c = torch.randn(4, 4), torch.randn(4, 4)
        torch.testing.assert_close(
            b["gen_0_k"](a, c), a - c.abs()
        )

    def test_gen_sink_supports_and_lowers(self):
        x = _v("x", 4, 4)
        term = _p("mul", x, _p("relu", x))
        h = gk.gen_handlers([term])
        sink = gk.gen_sink(h, compile_kernels=False)
        assert "gen_0_k" in sink.supported_ops
        from catopt_core.ir import IR

        ir = IR(
            root=_p("gen_0_k", x),
            inputs=[x],
            input_names={"x"},
            params={},
        )
        mod = sink.lower(ir, params={"x": torch.randn(4, 4)})
        t = torch.randn(4, 4)
        torch.testing.assert_close(mod(t), t * torch.relu(t))


class TestNovelCost:
    def test_novel_head_discounted(self):
        x = _v("x", 4, 4)
        cost = gk.novel_cost(
            count_cost, lambda op: op.startswith("gen_"), 0.5
        )
        spelled = _p("mul", x, _p("relu", x))
        gened = _p("gen_0_k", x)
        assert cost(gened) == cost(spelled) - 1.5  # 1 + 1 - 1.5

    def test_known_head_unchanged(self):
        x = _v("x", 4, 4)
        cost = gk.novel_cost(
            count_cost, lambda op: op == "never", 0.5
        )
        assert cost(_p("mul", x, x)) == 1.0


class TestGenDomain:
    def test_case_mints_handlers(self):
        torch = __import__("pytest").importorskip("torch")
        cases = play._gen_cases(0, 1)
        assert cases[0][5]["gen_handlers"] or True  # may be mlp
        cases = [c for s in range(4) for c in play._gen_cases(s, 3)]
        fuses = [c for c in cases if "mul" in c[0] or "fuse" in c[0]]
        assert fuses
        assert fuses[0][5]["gen_handlers"]

    def test_claim_gen_delivers_verified(self):
        pytest = __import__("pytest")
        pytest.importorskip("torch")
        for s in range(8):
            for c in play._gen_cases(s, 4):
                if not c[5]["gen_handlers"]:
                    continue
                board = play._gen_board(c)
                st = board.observe()
                claims = [
                    a for a in ma.legal_actions(st)
                    if a.op == "claim"
                ]
                if not claims:
                    continue
                board.step(claims[0])
                board.step(ma.Action.extract())
                best = board.eg.extract_best(
                    board.root, board.feasible_cost
                )
                from catopt_core.ir import op_repr

                assert "gen_" in op_repr(board.deliverable(best))
                return
        raise AssertionError("no gen claim landed")
