"""Structurally-special weight offers — the exact-elision pass.

Pins :func:`catopt_core.laws.specials.offer_weight_specials` — the
sibling of ``offer_low_rank_factors`` covering *structural* weight
specials rather than numerical rank:

* **identity** — ``W = I`` collapses ``linear(x, I)``/``matmul(x, I)``
  to ``x`` (bias: ``add(x, b)``);
* **diagonal** — ``W = diag(d)`` collapses to ``mul(x, d)``;
* **zero** — ``W = 0`` collapses to an exact zeros member;
* **elide** — bitwise-dead input slices gather away on the data side
  and bitwise-duplicate output slices (dead rows are the all-zero
  group) re-expand by ``index_select`` around a shrunk weight;
* **block_diag** — contiguous diagonal-ordered blocks split into
  per-block projections on ``split`` inputs, concatenated.

All exact — every offer carries ``error_bound=0.0``; detection is
bitwise (a dead slice is exactly 0.0, duplicates dedupe on raw
bytes), and each member is verified numerically against the dense
reference after lowering.
"""

import torch
import torch.nn as nn
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt_core.laws.specials import (
    _analyse,
    _blocks,
    _dedup,
    _offer_one_site,
    offer_weight_specials,
)
from catopt_orchestrator.optimize import Optimizer
from catopt_torch.backend import TorchBackend
from catopt_torch.torch_bridge import IRModule


def _lower(term, x, tensors, xv):
    """Lower *term* through the torch sink and evaluate on *xv*."""
    ir = IR(
        root=term, inputs=[x], input_names={x.name}, params={}
    )
    mod = IRModule(ir, param_values=dict(tensors))
    with torch.no_grad():
        return mod(xv)


def _site(term):
    """Intern *term* in a fresh graph; return (eg, root_eid)."""
    eg = EGraph()
    return eg, eg.add_term(term)


# ---------------------------------------------------------------------------
#  Unit cover — analysis helpers
# ---------------------------------------------------------------------------


def test_dedup_index_map():
    """First-occurrence grouping on raw-byte signatures."""
    sigs = [b"aa", b"bb", b"aa", b"cc", b"bb"]
    imap, firsts = _dedup(sigs)
    assert imap == (0, 1, 0, 2, 1)
    assert firsts == (0, 1, 3)


def test_analyse_dense_weight_has_no_structure():
    """A generic dense weight: no dead slices, no dups, one component."""
    torch.manual_seed(0)
    w = torch.randn(8, 6, dtype=torch.float64)
    a = _analyse(w, 8, 6)
    assert a["imap"] == tuple(range(8)) and a["firsts"] == tuple(
        range(8)
    )
    assert a["keep"] == tuple(range(6))
    assert not a["diag"] and not a["ident"] and not a["zero"]
    assert a["blocks"] is None  # all spans overlap -> one component


def test_blocks_decline_cases():
    """``_blocks`` declines: <2 components, non-contiguous rows, dead
    rows splitting the tiling."""
    # one component — every row spans the full input
    assert _blocks([(0, 4), (0, 4)], 2, 4) is None
    # two components but row sets interleave ({0,2} vs {1})
    assert _blocks([(0, 1), (2, 3), (0, 1)], 3, 4) is None
    # clean 2-block tiling plus a trailing dead row -> not a pure
    # block-diagonal (the elide member covers dead outputs)
    assert _blocks([(0, 2), (2, 4), None], 3, 4) is None
    # clean diagonal-ordered 2x2 tiling
    assert _blocks([(0, 2), (0, 2), (2, 4), (2, 4)], 4, 4) == [
        (0, 2, 0, 2),
        (2, 4, 2, 4),
    ]


# ---------------------------------------------------------------------------
#  identity / diagonal / zero
# ---------------------------------------------------------------------------


def test_identity_linear_is_input():
    """``linear(x, I)`` unions with ``x``'s class — the GEMM vanishes."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_i", TensorType((d, d)))
    eg, eid = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(
        eg, {"p_i": torch.eye(d, dtype=torch.float64)}
    )
    assert [r["kind"] for r in offers] == ["identity"]
    assert eg.extract_best(eid, flops_cost) == x


def test_identity_linear_with_bias_is_add():
    """``linear(x, I, b)`` -> ``add(x, b)`` — verified numerically."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_i", TensorType((d, d)))
    b = Param("p_b", TensorType((d,)))
    eg, eid = _site(Op.make("linear", x, w, b))
    bv = torch.randn(d, dtype=torch.float64)
    tensors = {"p_i": torch.eye(d, dtype=torch.float64), "p_b": bv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["identity"]
    best = eg.extract_best(eid, flops_cost)
    assert best == Op.make("add", x, b)
    xv = torch.randn(8, d, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.equal(out, xv + bv)


def test_identity_matmul():
    """``matmul(x, I)`` — same collapse in the matmul orientation."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_m", TensorType((d, d)))
    eg, eid = _site(Op.make("matmul", x, w))
    offers = offer_weight_specials(
        eg, {"p_m": torch.eye(d, dtype=torch.float64)}
    )
    assert [r["kind"] for r in offers] == ["identity"]
    assert offers[0]["orient"] == "matmul_r"
    assert eg.extract_best(eid, flops_cost) == x


def test_diagonal_is_pointwise_mul():
    """``linear(x, diag(d))`` -> ``mul(x, d_param)`` — flop-free."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_d", TensorType((d, d)))
    eg, eid = _site(Op.make("linear", x, w))
    dv = torch.randn(d, dtype=torch.float64)
    tensors = {"p_d": torch.diag(dv)}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["diagonal"]
    d_name = offers[0]["d_param"]
    assert torch.equal(tensors[d_name], dv)
    best = eg.extract_best(eid, flops_cost)
    assert best == Op.make(
        "mul", x, Param(d_name, TensorType((d,)))
    )
    xv = torch.randn(8, d, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv * dv)
    assert dag_cost(best, flops_cost) < dag_cost(
        Op.make("linear", x, w), flops_cost
    )


def test_diagonal_matmul():
    """``matmul(x, diag(d))`` -> ``mul(x, d)`` — same collapse."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_dm", TensorType((d, d)))
    eg, eid = _site(Op.make("matmul", x, w))
    dv = torch.randn(d, dtype=torch.float64)
    tensors = {"p_dm": torch.diag(dv)}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["diagonal"]
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(8, d, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv * dv)


def test_zero_weight_is_zeros():
    """``linear(x, 0, b)`` -> ``add(zeros, b)`` — exact zeros."""
    o, i = 16, 8
    x = Var("x", TensorType((4, i)))
    w = Param("p_z", TensorType((o, i)))
    b = Param("p_b", TensorType((o,)))
    eg, eid = _site(Op.make("linear", x, w, b))
    bv = torch.randn(o, dtype=torch.float64)
    tensors = {"p_z": torch.zeros(o, i, dtype=torch.float64), "p_b": bv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["zero"]
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(4, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.equal(out, bv.expand(4, o).contiguous())


def test_near_zero_is_not_special():
    """Exactness is bitwise — a *small* weight is not dead/zero."""
    o, i = 16, 8
    x = Var("x", TensorType((4, i)))
    w = Param("p_eps", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    wv = torch.full((o, i), 1e-30, dtype=torch.float64)
    wv[0] = 0.0  # one truly dead row, rest merely tiny
    offers = offer_weight_specials(eg, {"p_eps": wv})
    # the dead row collapses into one zero slice -> elide, NOT zero
    assert [r["kind"] for r in offers] == ["elide"]
    assert offers[0]["out_dim"] == o


# ---------------------------------------------------------------------------
#  elide — dead rows/cols and duplicate output slices
# ---------------------------------------------------------------------------


def _elided_weight(o=64, i=32):
    """(o, i) weight with dead rows 10:20 and row 31 == row 30."""
    torch.manual_seed(0)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[10:20] = 0.0
    wv[31] = wv[30]
    return wv


def test_dead_and_duplicate_rows_gather():
    """Dead rows collapse to one zero slice; dup rows compute once."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["elide"]
    rec = offers[0]
    # 64 rows -> 54 unique (10 dead -> 1 zero slice, 1 dup removed)
    assert tensors[rec["w_param"]].shape == (54, i)
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "index_select"
    assert best.args[0] == Op.make(
        "linear", x, Param(rec["w_param"], TensorType((54, i)))
    )
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.equal(out, xv @ wv.T)
    dense = Op.make("linear", x, w)
    assert dag_cost(best, flops_cost) < dag_cost(dense, flops_cost)


def test_dead_cols_gather_input():
    """Dead input slices gather away on the DATA side — output stays
    full-width, no output gather needed."""
    o, i = 64, 32
    torch.manual_seed(1)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[:, 5:9] = 0.0
    x = Var("x", TensorType((8, i)))
    w = Param("p_wc", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_wc": wv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["elide"]
    rec = offers[0]
    assert tensors[rec["w_param"]].shape == (o, i - 4)
    best = eg.extract_best(eid, flops_cost)
    # linear(index_select(x, -1, keep), W')
    assert best.op == "linear" and best.args[0].op == "index_select"
    xv = torch.randn(8, i, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv @ wv.T)


def test_elide_matmul_orientation():
    """``matmul(x, W)`` — dead cols of x (dead ROWS of W) gather on the
    input; duplicated output cols re-expand on the output."""
    i, o = 32, 64
    torch.manual_seed(2)
    wv = torch.randn(i, o, dtype=torch.float64)
    wv[5:9] = 0.0  # dead input dims
    wv[:, 20] = wv[:, 10]  # duplicate output col
    x = Var("x", TensorType((8, i)))
    w = Param("p_wm", TensorType((i, o)))
    eg, eid = _site(Op.make("matmul", x, w))
    tensors = {"p_wm": wv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["elide"]
    rec = offers[0]
    assert tensors[rec["w_param"]].shape == (i - 4, o - 1)
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "index_select"
    inner = best.args[0]
    assert inner.op == "matmul" and inner.args[0].op == "index_select"
    xv = torch.randn(8, i, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv @ wv)


def test_elide_with_bias_readds_after_gather():
    """``linear(x, W, b)`` offers gather-then-add — the bias pairs with
    the *expanded* output positions."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    bv = torch.randn(o, dtype=torch.float64)
    x = Var("x", TensorType((8, i)))
    w = Param("p_wb", TensorType((o, i)))
    b = Param("p_bb", TensorType((o,)))
    eg, eid = _site(Op.make("linear", x, w, b))
    tensors = {"p_wb": wv, "p_bb": bv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["elide"]
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "add"
    xv = torch.randn(8, i, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv @ wv.T + bv)


# ---------------------------------------------------------------------------
#  block_diag
# ---------------------------------------------------------------------------


def test_block_diag_linear_splits():
    """2-block diagonal weight -> concat of per-block linears."""
    o, i = 8, 16
    x = Var("x", TensorType((8, i)))
    w = Param("p_bd", TensorType((o, i)))
    torch.manual_seed(3)
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[:4, :8] = torch.randn(4, 8, dtype=torch.float64)
    wv[4:, 8:] = torch.randn(4, 8, dtype=torch.float64)
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_bd": wv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["block_diag"]
    rec = offers[0]
    assert rec["sizes"] == (8, 8)
    assert len(rec["w_params"]) == 2
    assert tensors[rec["w_params"][0]].shape == (4, 8)
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "concat"
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.allclose(out, xv @ wv.T, atol=1e-12)
    # flops: 2·M·(4·8 + 4·8) vs 2·M·8·16 — halved.
    dense = Op.make("linear", x, w)
    assert dag_cost(best, flops_cost) == pytest_half(
        dag_cost(dense, flops_cost)
    )


def pytest_half(v):
    """Half of a flop count — the 2-block equal-size split."""
    return v / 2


def test_block_diag_with_bias():
    """Biased block-diagonal linear: bias re-added once after concat."""
    o, i = 8, 16
    x = Var("x", TensorType((8, i)))
    w = Param("p_bdb", TensorType((o, i)))
    b = Param("p_bbb", TensorType((o,)))
    torch.manual_seed(4)
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[:4, :8] = torch.randn(4, 8, dtype=torch.float64)
    wv[4:, 8:] = torch.randn(4, 8, dtype=torch.float64)
    bv = torch.randn(o, dtype=torch.float64)
    eg, eid = _site(Op.make("linear", x, w, b))
    tensors = {"p_bdb": wv, "p_bbb": bv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["block_diag"]
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "add" and best.args[0].op == "concat"
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.allclose(out, xv @ wv.T + bv, atol=1e-12)


def test_block_diag_matmul_orientation():
    """``matmul(x, W)`` block-diag — rows of W are the input axis."""
    i, o = 16, 8
    x = Var("x", TensorType((8, i)))
    w = Param("p_bdm", TensorType((i, o)))
    torch.manual_seed(5)
    wv = torch.zeros(i, o, dtype=torch.float64)
    wv[:8, :4] = torch.randn(8, 4, dtype=torch.float64)
    wv[8:, 4:] = torch.randn(8, 4, dtype=torch.float64)
    eg, eid = _site(Op.make("matmul", x, w))
    tensors = {"p_bdm": wv}
    offers = offer_weight_specials(eg, tensors)
    assert [r["kind"] for r in offers] == ["block_diag"]
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "concat"
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.allclose(out, xv @ wv, atol=1e-12)


def test_block_diag_with_dead_col_inside_slice():
    """An interior dead column folds into a block's slice — still
    exact, just a zero column in the block weight."""
    o, i = 6, 9
    x = Var("x", TensorType((4, i)))
    w = Param("p_bdz", TensorType((o, i)))
    torch.manual_seed(6)
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[:3, :4] = torch.randn(3, 4, dtype=torch.float64)
    wv[3:, 5:] = torch.randn(3, 4, dtype=torch.float64)
    # col 4 fully dead — folds into the first block's slice [0, 5)
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_bdz": wv}
    offers = offer_weight_specials(eg, tensors)
    kinds = {r["kind"] for r in offers}
    assert "block_diag" in kinds
    xv = torch.randn(4, i, dtype=torch.float64)
    best = eg.extract_best(eid, flops_cost)
    assert torch.allclose(
        _lower(best, x, tensors, xv), xv @ wv.T, atol=1e-12
    )


def test_elide_and_block_both_offered():
    """A block-diagonal weight with a dead input column gets BOTH
    members — the dead dim also folds into a block slice, so the two
    offers compete in the same e-class."""
    o, i = 6, 9
    x = Var("x", TensorType((4, i)))
    w = Param("p_be", TensorType((o, i)))
    torch.manual_seed(7)
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[:3, :4] = torch.randn(3, 4, dtype=torch.float64)
    wv[3:, 5:] = torch.randn(3, 4, dtype=torch.float64)
    # col 4 dead: elide gathers it away; blocks fold it into slice 0.
    eg, eid = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_be": wv})
    assert {r["kind"] for r in offers} == {"elide", "block_diag"}


def test_noncontiguous_rows_decline_blocks_but_elide():
    """Interleaved component rows aren't diagonal-ordered — no block
    offer; the dup/dead structure still elides."""
    o, i = 3, 4
    x = Var("x", TensorType((2, i)))
    w = Param("p_nc", TensorType((o, i)))
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[0, 0] = 1.0
    wv[1, 2] = 2.0
    wv[2, 0] = 1.0  # duplicates row 0 — components interleave
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_nc": wv}
    offers = offer_weight_specials(eg, tensors)
    assert {r["kind"] for r in offers} == {"elide"}
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(2, i, dtype=torch.float64)
    assert torch.equal(_lower(best, x, tensors, xv), xv @ wv.T)


# ---------------------------------------------------------------------------
#  Declines, guards, idempotence
# ---------------------------------------------------------------------------


def test_dense_weight_no_offers():
    """A generic dense weight has no structure to exploit."""
    torch.manual_seed(8)
    x = Var("x", TensorType((8, 32)))
    w = Param("p_w", TensorType((64, 32)))
    eg, _ = _site(Op.make("linear", x, w))
    assert (
        offer_weight_specials(
            eg, {"p_w": torch.randn(64, 32, dtype=torch.float64)}
        )
        == []
    )


def test_offer_guards():
    """Honest declines: missing value, non-tensor, non-2-D, empty axis,
    param-only consumer, left-weight matmul, var weight."""
    o, i = 8, 4
    x = Var("x", TensorType((2, i)))
    w = Param("p_w", TensorType((o, i)))
    wv = torch.zeros(o, i, dtype=torch.float64)
    wv[0, 0] = 1.0
    wv[1, 1] = 2.0  # block-diag-able structure, ready to fire

    eg, _ = _site(Op.make("linear", x, w))
    assert offer_weight_specials(eg, {}) == []  # no value
    eg, _ = _site(Op.make("linear", x, w))
    assert offer_weight_specials(eg, {"p_w": 3.0}) == []
    eg, _ = _site(Op.make("linear", x, w))
    assert (
        offer_weight_specials(
            eg, {"p_w": torch.randn(o, dtype=torch.float64)}
        )
        == []
    )
    # degenerate axis — nothing safe to spell
    w0 = Param("p_w0", TensorType((o, 0)))
    x0 = Var("x0", TensorType((2, 0)))
    eg, _ = _site(Op.make("linear", x0, w0))
    assert (
        offer_weight_specials(
            eg, {"p_w0": torch.zeros(o, 0, dtype=torch.float64)}
        )
        == []
    )
    # param-only consumer folds either way
    wm = Param("p_m", TensorType((i, o)))
    eg, _ = _site(Op.make("matmul", w, wm))
    assert (
        offer_weight_specials(
            eg, {"p_w": wv, "p_m": torch.randn(i, o)}
        )
        == []
    )
    # left-weight matmul is out of scope (data_c carries no Var)
    eg, _ = _site(Op.make("matmul", wm, x))
    assert offer_weight_specials(eg, {"p_m": torch.randn(i, o)}) == []
    # var-vs-var site — the weight class holds no Param leaf
    y = Var("y", TensorType((i, o)))
    eg, _ = _site(Op.make("matmul", x, y))
    assert offer_weight_specials(eg, {}) == []


def test_idempotent_and_witness_flag():
    """A second run re-offers nothing; ``witness=False`` merges free."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    tensors = {"p_w": wv}
    eg, _ = _site(Op.make("linear", x, w))
    assert len(offer_weight_specials(eg, tensors)) == 1
    assert offer_weight_specials(eg, tensors) == []  # union already done

    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_w": wv}, witness=False)
    assert len(offers) == 1


def test_shared_weight_two_consumers_reuse():
    """Two consumers of one dead-row weight share the derived param;
    the second site's analysis is a cache hit."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    x = Var("x", TensorType((8, i)))
    y = Var("y", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    term = Op.make(
        "add", Op.make("linear", x, w), Op.make("linear", y, w)
    )
    eg, _ = _site(term)
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors)
    assert len(offers) == 2
    assert offers[0]["w_param"] == offers[1]["w_param"]
    assert sum(k.startswith("p_w__el") for k in tensors) == 1


def test_two_param_leaves_first_declines():
    """A weight class holding two Param leaves: the first declines
    (no value), the second offers — the leaf loop continues."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    x = Var("x", TensorType((8, i)))
    w1 = Param("p_w1", TensorType((o, i)))
    w2 = Param("p_w2", TensorType((o, i)))
    eg = EGraph()
    eid = eg.add_term(Op.make("linear", x, w1))
    # union the two param leaves into one weight class
    eg.union(eg.add_term(w1), eg.add_term(w2))
    offers = offer_weight_specials(eg, {"p_w2": wv})
    assert len(offers) == 1 and offers[0]["param"] == "p_w2"
    _ = eid


def test_poisoned_name_takes_suffix():
    """A ``__diag`` name already taken by different values gets _1."""
    d = 16
    x = Var("x", TensorType((8, d)))
    w = Param("p_d", TensorType((d, d)))
    dv = torch.randn(d, dtype=torch.float64)
    tensors = {
        "p_d": torch.diag(dv),
        "p_d__diag": torch.zeros(d, dtype=torch.float64),
    }
    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, tensors)
    assert offers[0]["d_param"] == "p_d__diag_1"


def test_witness_carries_zero_bound():
    """The certified bound is exactly 0 — a proven equality."""
    o, i = 64, 32
    wv = _elided_weight(o, i)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_w": wv}, witness=True)
    assert len(offers) == 1
    # the synthetic rewrite registered for the merge has bound 0
    wits = [
        r
        for n, r in eg._rule_objs.items()
        if n.startswith("weight_special#")
    ]
    assert wits
    assert all(r.error_bound == 0.0 for r in wits)


def test_offer_one_site_direct_decline():
    """Direct unit cover for the guard clause."""
    x = Var("x", TensorType((8, 32)))
    w = Param("p_w", TensorType((64, 32)))
    eg = EGraph()
    eid = eg.add_term(Op.make("linear", x, w))
    cid = eg.find(eid)
    node = next(
        n for n in eg._classes[cid].nodes if n.op == "linear"
    )
    assert (
        _offer_one_site(
            eg, cid, node, "linear", node.children[0], w, {}, {},
            True,
        )
        == []
    )


# ---------------------------------------------------------------------------
#  Pipeline wiring — detect_specials=
# ---------------------------------------------------------------------------


def test_detect_specials_pipeline_flag():
    """``detect_specials=True`` flows the pass through Optimizer.search;
    flop pricing selects the elided member; lowering verifies."""
    torch.manual_seed(9)
    i, o = 32, 64

    class Dead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(i, o, bias=False).double()
            with torch.no_grad():
                self.lin.weight[16:48] = 0.0

        def forward(self, t):
            return self.lin(t)

    model = Dead().eval().double()
    xv = torch.randn(8, i, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, xv, detect_specials=True, cost_fn=flops_cost)
    assert res.stats["weight_specials"]
    rec = res.stats["weight_specials"][0]
    assert rec["kind"] == "elide" and "eid" not in rec
    assert res.term.op == "index_select"
    low = opt.lower(res, xv)
    assert low.verified.passed
    with torch.no_grad():
        assert torch.allclose(model(xv), low.module(xv), atol=1e-12)


def test_detect_specials_identity_pipeline():
    """An identity-weight Linear end to end: search, lower, verify."""
    d = 32

    class Eye(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d, d, bias=False).double()
            with torch.no_grad():
                self.lin.weight.copy_(torch.eye(d, dtype=torch.float64))

        def forward(self, t):
            return self.lin(t)

    model = Eye().eval().double()
    xv = torch.randn(8, d, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, xv, detect_specials=True)
    kinds = {r["kind"] for r in res.stats["weight_specials"]}
    assert "identity" in kinds
    low = opt.lower(res, xv)
    assert low.verified.passed


def test_detect_specials_no_candidates():
    """Flag on + dense weights: the pass runs, offers nothing, records
    no stats key."""
    torch.manual_seed(10)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(32, 64, bias=False).double()

        def forward(self, t):
            return self.lin(t)

    res = Optimizer(backend=TorchBackend()).search(
        Full().eval().double(),
        torch.randn(8, 32, dtype=torch.float64),
        detect_specials=True,
    )
    assert "weight_specials" not in res.stats


def test_detect_specials_off_by_default():
    """The flag is opt-in: default search leaves the dense member."""
    torch.manual_seed(11)
    i, o = 32, 64

    class Dead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(i, o, bias=False).double()
            with torch.no_grad():
                self.lin.weight[16:48] = 0.0

        def forward(self, t):
            return self.lin(t)

    res = Optimizer(backend=TorchBackend()).search(
        Dead().eval().double(), torch.randn(8, 32, dtype=torch.float64)
    )
    assert "weight_specials" not in res.stats


def test_detect_specials_and_factors_compose():
    """Both opt-ins coexist — a low-rank weight AND a dead-row weight
    in one model each get their pass's member."""
    torch.manual_seed(12)
    i, o, r = 32, 64, 4

    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(i, o, bias=False).double()
            self.b = nn.Linear(i, o, bias=False).double()
            with torch.no_grad():
                self.a.weight.copy_(
                    torch.randn(o, r, dtype=torch.float64)
                    @ torch.randn(r, i, dtype=torch.float64)
                )
                self.b.weight[32:] = 0.0

        def forward(self, t):
            return self.a(t) + self.b(t)

    res = Optimizer(backend=TorchBackend()).search(
        Both().eval().double(),
        torch.randn(8, i, dtype=torch.float64),
        detect_factors=True,
        detect_specials=True,
        cost_fn=flops_cost,
    )
    assert res.stats["low_rank_factors"]
    assert res.stats["weight_specials"]


def test_op_repr_smoke():
    """op_repr keeps the offered members readable."""
    x = Var("x", TensorType((8, 32)))
    w = Param("p_w", TensorType((64, 32)))
    wv = _elided_weight(64, 32)
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offer_weight_specials(eg, tensors)
    rep = op_repr(eg.extract_best(eid, flops_cost))
    assert "index_select" in rep and "linear" in rep
