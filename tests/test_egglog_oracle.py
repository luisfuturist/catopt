"""Differential oracle: catopt's e-graph vs the ``egglog`` engine.

OPT-IN and TEST-ONLY (Phase 4).  This test cross-checks catopt's
hand-rolled equality-saturation search against the Rust ``egglog``
library on a small op/law subset — see ``tests/egglog_oracle.py`` for
the ported prototype and its documented limitations.

``egglog`` is not a default dependency: the test self-skips when the
package is absent, so the rest of the suite stays green.  Run the
oracle explicitly with::

    uv sync --group oracle
    .venv/bin/python -m pytest tests/test_egglog_oracle.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

# Keep the suite green without the opt-in `oracle` dependency group.
pytest.importorskip("egglog")

from catopt_core.ir import (
    Op,
    Param,
    TensorType,
    Var,
    op_repr,
)
from catopt_torch.torch_bridge import export_to_ir

from tests.egglog_oracle import (
    EgglogEngine,
    canon,
    catopt_equivalent,
    catopt_extract,
    evaluate,
    folded_flops,
)

# ---------------------------------------------------------------------------
#  Fixtures / builders
# ---------------------------------------------------------------------------


def _sequential_ir(with_relu: bool):
    """A 2-Linear chain (optionally ReLU-separated) + torch reference."""
    torch.manual_seed(0)
    layers: list[nn.Module] = [nn.Linear(4, 4)]
    if with_relu:
        layers.append(nn.ReLU())
    layers.append(nn.Linear(4, 4))
    m = nn.Sequential(*layers)
    x = torch.randn(2, 4)
    ir, src = export_to_ir(m, x)
    inputs = {v.name: tuple(v.typ.shape) for v in ir.inputs}
    params = {p.name: tuple(p.typ.shape) for p in ir.params.values()}
    vals_in = {ir.inputs[0].name: x.detach().numpy()}
    vals_p = {n: t.detach().numpy() for n, t in src.items()}
    ref = m(x).detach().numpy()
    return ir.root, inputs, params, vals_in, vals_p, ref


def _matmul_chain():
    """``A @ (B @ x)`` — reassociation changes the FLOP count.

    x (2,4); B (64,2); A (4,64).  Right-assoc A@(B@x) = 1024 + 2048 =
    3072; left-assoc (A@B)@x = 1024 + 64 = 1088 (cheaper).
    """
    a = Param("A", TensorType((4, 64)))
    b = Param("B", TensorType((64, 2)))
    x = Var("x", TensorType((2, 4)))
    term = Op.make("matmul", a, Op.make("matmul", b, x))
    inputs = {"x": (2, 4)}
    params = {"A": (4, 64), "B": (64, 2)}
    rng = np.random.default_rng(0)
    vals_in = {"x": rng.standard_normal((2, 4))}
    vals_p = {
        "A": rng.standard_normal((4, 64)),
        "B": rng.standard_normal((64, 2)),
    }
    return term, inputs, params, vals_in, vals_p


def _elementwise_term():
    """Exercise the remaining ops: ``relu(neg(mul(transpose(x), P)))``."""
    x = Var("x", TensorType((2, 4)))
    p = Param("P", TensorType((4, 2)))
    term = Op.make(
        "relu",
        Op.make(
            "neg",
            Op.make("mul", Op.make("transpose", x, dim0=-2, dim1=-1), p),
        ),
    )
    inputs = {"x": (2, 4)}
    params = {"P": (4, 2)}
    rng = np.random.default_rng(1)
    vals_in = {"x": rng.standard_normal((2, 4))}
    vals_p = {"P": rng.standard_normal((4, 2))}
    return term, inputs, params, vals_in, vals_p


def _comm_add_term():
    """Two parameter leaves under ``add`` — comm_add is representative-only."""
    p = Param("P", TensorType((4,)))
    q = Param("Q", TensorType((4,)))
    term = Op.make("add", p, q)
    params = {"P": (4,), "Q": (4,)}
    rng = np.random.default_rng(2)
    vals_p = {"P": rng.standard_normal((4,)), "Q": rng.standard_normal((4,))}
    return term, {}, params, {}, vals_p


# ---------------------------------------------------------------------------
#  The differential comparison itself
# ---------------------------------------------------------------------------


def _differential(term, inputs, params, vals_in, vals_p, ref=None):
    """Run both engines and assert they agree.

    Agreement means (a) numeric equality of the original and both
    extracted terms — against a torch ground truth when supplied, else
    against the original term's NumPy evaluation; (b) the extracted
    terms are structurally identical up to commutative-operand
    representative choice; (c) the folded (param-discounted) costs
    coincide.
    """
    cat_best, cat_fires = catopt_extract(term)

    eng = EgglogEngine.from_term(term, inputs, params)
    egg_fires = eng.run(10)
    egg_best, egg_cost = eng.extract_best()

    ground = (
        ref if ref is not None else evaluate(term, vals_in, vals_p)
    )
    for label, t in (
        ("original", term),
        ("catopt", cat_best),
        ("egglog", egg_best),
    ):
        got = evaluate(t, vals_in, vals_p)
        assert np.allclose(got, ground, atol=1e-5), (
            f"{label} term is not numerically equal to the ground truth"
        )

    assert op_repr(canon(cat_best)) == op_repr(canon(egg_best)), (
        "engines extracted different forms (beyond comm representative)"
    )
    assert folded_flops(cat_best) == pytest.approx(egg_cost)

    return cat_best, egg_best, cat_fires, egg_fires


# ---------------------------------------------------------------------------
#  Cases mirroring the spike (op/law subset coverage)
# ---------------------------------------------------------------------------


def test_relu_blocks_linear_composition():
    """ReLU between the linears: neither engine rewrites; both agree."""
    term, inputs, params, vals_in, vals_p, ref = _sequential_ir(
        with_relu=True
    )
    cat_best, egg_best, cat_fires, egg_fires = _differential(
        term, inputs, params, vals_in, vals_p, ref
    )

    # No rule fires in either engine: the ReLU blocks assoc_linear_bias.
    assert cat_fires == {}
    assert egg_fires == {}
    assert op_repr(cat_best) == op_repr(term)
    assert op_repr(egg_best) == op_repr(term)


def test_matmul_reassociation_both_find_cheaper_form():
    """A @ (B @ x) reassociates to the cheaper (A @ B) @ x in both."""
    term, inputs, params, vals_in, vals_p = _matmul_chain()
    cat_best, egg_best, cat_fires, _egg_fires = _differential(
        term, inputs, params, vals_in, vals_p
    )

    a = Param("A", TensorType((4, 64)))
    b = Param("B", TensorType((64, 2)))
    x = Var("x", TensorType((2, 4)))
    expected = op_repr(Op.make("matmul", Op.make("matmul", a, b), x))
    assert op_repr(cat_best) == expected
    assert op_repr(egg_best) == expected
    assert "assoc_matmul" in cat_fires


def test_linear_composition_both_find_fused_form():
    """Two stacked Linears fuse to the biased affine composition in both."""
    term, inputs, params, vals_in, vals_p, ref = _sequential_ir(
        with_relu=False
    )
    cat_best, egg_best, cat_fires, _egg_fires = _differential(
        term, inputs, params, vals_in, vals_p, ref
    )

    # Both fuse: the W1@W0 product appears in the extracted term.
    assert "matmul" in op_repr(cat_best)
    assert "matmul" in op_repr(egg_best)
    assert "assoc_linear_bias" in cat_fires
    # Folded cost: fused 72 < the chain's 128.
    assert folded_flops(cat_best) == pytest.approx(72.0)
    assert folded_flops(egg_best) == pytest.approx(72.0)


def test_elementwise_ops_agree():
    """neg / transpose / mul / relu: comm_mul fires in both; forms agree."""
    term, inputs, params, vals_in, vals_p = _elementwise_term()
    _cat_best, _egg_best, cat_fires, egg_fires = _differential(
        term, inputs, params, vals_in, vals_p
    )
    assert "comm_mul" in cat_fires
    assert "comm_mul" in egg_fires


def test_comm_add_representative_only():
    """add(P, Q): comm_add fires; the two forms differ only by operand order."""
    term, inputs, params, vals_in, vals_p = _comm_add_term()
    cat_best, egg_best, cat_fires, egg_fires = _differential(
        term, inputs, params, vals_in, vals_p
    )
    assert "comm_add" in cat_fires
    assert "comm_add" in egg_fires
    # The raw forms may differ by operand order; canonicalised they match.
    assert op_repr(canon(cat_best)) == op_repr(canon(egg_best))


# ---------------------------------------------------------------------------
#  Discovered equivalences (not just the extracted form)
# ---------------------------------------------------------------------------


def test_equivalence_discovery_agrees():
    """Both engines *prove* A@(B@x) == (A@B)@x, not merely extract it."""
    term, inputs, params, _vals_in, _vals_p = _matmul_chain()
    a = Param("A", TensorType((4, 64)))
    b = Param("B", TensorType((64, 2)))
    x = Var("x", TensorType((2, 4)))
    other = Op.make("matmul", Op.make("matmul", a, b), x)

    # catopt: the two terms land in the same e-class.
    assert catopt_equivalent(term, other)

    # egglog: prove the same equation after saturation.
    eng = EgglogEngine.from_term(term, inputs, params)
    eng.run(10)
    other_term = eng.register(other)
    assert eng.check_equal(eng.root, other_term)


def test_oracle_covers_supported_ops():
    """The oracle op subset is exactly the spike's seven ops."""
    from tests.egglog_oracle import SUPPORTED_OPS

    assert frozenset(
        {"add", "mul", "neg", "matmul", "linear", "relu", "transpose"}
    ) == SUPPORTED_OPS
