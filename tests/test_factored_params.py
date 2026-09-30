"""Factored-parameter execution path — the low-rank first-class path.

Pins what ``x @ (A·B)`` does end to end:

* **term level** — ``assoc_matmul`` / ``assoc_matmul_rev`` /
  ``assoc_linear`` / ``assoc_linear_rev`` put both associations in
  the e-class and cost-driven extraction picks: under flop pricing
  the factored ``(x@A)@B`` wins iff ``r·(i+o) < i·o``; under the
  roofline/executor default the same member wins only once the saved
  work also beats the extra kernel launch (~8.7 µs on the bundled
  profile) — tiny workloads honestly prefer the fused single GEMM.
* **lowering** — a selected ``matmul(A, B)`` param-only subtree folds
  to a materialised weight at construction
  (``IRModule._fold_weight_chains``); a selected factored chain stays
  two runtime GEMMs.
* **detection** — :func:`offer_low_rank_factors` covers the case the
  laws cannot see: a dense ``Param`` whose stored *value* is
  numerically low-rank.
* **KVLatentShare interaction** — the morphism's shared latent
  ``C = x@U`` with recoveries ``k_i = C@D_i`` must survive extraction
  and lowering: no ``matmul(param, param)`` subtree may re-materialise
  ``U·D_i`` into a dense weight.
* **LoRA** — ``x@W + x@A@B`` delivery is numerically identical
  whether extraction keeps the adapter factored or merges it into
  the base weight (the merge is flop-optimal and may legitimately
  win; the guard is the verify, not the spelling).
"""

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import torch
import torch.nn as nn
from catopt_core.cost import (
    dag_cost,
    executor_cost_for,
    flops_cost,
    launch_aware_cost,
)
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import (
    ALL_RULES,
    ASSOC_LINEAR,
    ASSOC_LINEAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
)
from catopt_core.laws.factored import (
    _cls_has_var,
    _factor_cols,
    _factor_rows,
    _fresh_name,
    _leaf_params,
    offer_low_rank_factors,
)
from catopt_core.typing import has_var_leaf
from catopt_orchestrator.optimize import Optimizer
from catopt_torch.adapters import TorchSink
from catopt_torch.backend import TorchBackend

from tests.test_morphism_kv import (
    _lift,
    _SharedKVStack,
    _SingleKV,
    _SingleMatmulKV,
    _x,
)


def _mm(x, a, b):
    """``(x @ A) @ B`` — the factored spelling."""
    return Op.make("matmul", Op.make("matmul", x, a), b)


def _mm_mat(x, a, b):
    """``x @ (A @ B)`` — the materialised spelling."""
    return Op.make("matmul", x, Op.make("matmul", a, b))


def _assoc_run(term, iters=8):
    """Saturate with just the assoc laws and return (eg, eid)."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        [
            ASSOC_MATMUL,
            ASSOC_MATMUL_REV,
            ASSOC_LINEAR,
            ASSOC_LINEAR_REV,
        ],
        eid,
        max_iterations=iters,
    )
    return eg, eid


def _param_only_subterms(term):
    """Op subtrees with no Var leaf — the lowerer's fold sites."""
    out = []

    def go(t):
        if isinstance(t, Op):
            if not has_var_leaf(t):
                out.append(t)
            for a in t.args:
                go(a)

    go(term)
    return out


# ---------------------------------------------------------------------------
#  Task 1 — the association laws already carry the factored order
# ---------------------------------------------------------------------------


def test_matmul_assoc_picks_factored_low_rank():
    """``x@(A@B)`` and ``(x@A)@B`` unify; flops pick the chain at r≪d."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("A", TensorType((i, r)))
    b = Param("B", TensorType((r, o)))
    for seed_term in (_mm(x, a, b), _mm_mat(x, a, b)):
        eg, eid = _assoc_run(seed_term)
        best = eg.extract_best(eid, flops_cost)
        assert best == _mm(x, a, b)
        # The flop break-even is r·(i+o) < i·o — here 4·192 < 8192.
        assert dag_cost(best, flops_cost) < dag_cost(
            _mm_mat(x, a, b), flops_cost
        )


def test_matmul_assoc_picks_materialized_high_rank():
    """At r ≥ i·o/(i+o) the dense product is the honest winner."""
    m, i, r, o = 8, 64, 48, 128  # r·(i+o)=9216 ≥ 8192
    x = Var("x", TensorType((m, i)))
    a = Param("A", TensorType((i, r)))
    b = Param("B", TensorType((r, o)))
    eg, eid = _assoc_run(_mm(x, a, b))
    best = eg.extract_best(eid, flops_cost)
    assert best == _mm_mat(x, a, b)


def test_assoc_linear_rev_splits_fused_weight():
    """``linear(x, B@A)`` reaches the factored member — the gap the
    new ``assoc_linear_rev`` law closes (``assoc_linear`` alone is
    one-directional)."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("lora_A", TensorType((r, i)))  # HF convention names
    b = Param("lora_B", TensorType((o, r)))
    fused = Op.make("linear", x, Op.make("matmul", b, a))
    eg = EGraph()
    eid = eg.add_term(fused)
    # Forward alone cannot reach the chain.
    eg.run([ASSOC_LINEAR], eid, max_iterations=4)
    assert eg.extract_best(eid, flops_cost) == fused
    # With the reverse law it can — and flops select it.
    eg, eid = _assoc_run(fused)
    assert eg.extract_best(eid, flops_cost) == Op.make(
        "linear", Op.make("linear", x, a), b
    )


def test_linear_chain_fuses_and_splits_both_directions():
    """``linear(linear(x,A),B)`` keeps both members; flops pick split."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("lora_A", TensorType((r, i)))
    b = Param("lora_B", TensorType((o, r)))
    chain = Op.make("linear", Op.make("linear", x, a), b)
    eg, eid = _assoc_run(chain)
    best = eg.extract_best(eid, flops_cost)
    assert best == chain
    # The fused member was generated — it just loses on flops.
    assert eg.rule_fires.get("assoc_linear")
    fused = Op.make("linear", x, Op.make("matmul", b, a))
    assert dag_cost(fused, flops_cost) > dag_cost(chain, flops_cost)


def test_executor_cost_launch_overhead_crossover():
    """The default pipeline model is honest, not flops-blind: at tiny
    dims the extra kernel launch beats the flop saving and the fused
    single GEMM wins; once the work is real, factored wins.  This is
    the documented regime boundary — selection is cost-driven, so the
    'factored path' is *where the cost model lands*."""
    cf = executor_cost_for(lowering="generic")
    x = Var("x", TensorType((8, 64)))
    a = Param("A", TensorType((64, 4)))
    b = Param("B", TensorType((4, 128)))
    assert dag_cost(_mm_mat(x, a, b), cf) < dag_cost(_mm(x, a, b), cf)
    # Compute-bound regime: the same rank at d=1024/2048 flips the
    # verdict — two launches no longer dominate ~10x fewer flops.
    x2 = Var("x", TensorType((64, 1024)))
    a2 = Param("A2", TensorType((1024, 64)))
    b2 = Param("B2", TensorType((64, 2048)))
    assert dag_cost(_mm(x2, a2, b2), cf) < dag_cost(
        _mm_mat(x2, a2, b2), cf
    )
    # ...and extraction agrees.
    eg, eid = _assoc_run(_mm_mat(x2, a2, b2))
    assert eg.extract_best(eid, cf) == _mm(x2, a2, b2)


def test_fold_weight_chains_materialises_selected_form():
    """Lowering folds a *selected* param-only matmul to one stored
    weight — and leaves a selected factored chain as two GEMMs."""
    from catopt_torch.torch_bridge import IRModule

    x = Var("x", TensorType((8, 64)))
    a = Param("A", TensorType((64, 4)))
    b = Param("B", TensorType((4, 128)))
    av = torch.randn(64, 4, dtype=torch.float64)
    bv = torch.randn(4, 128, dtype=torch.float64)
    xv = torch.randn(8, 64, dtype=torch.float64)
    leaves = {"A": av, "B": bv}

    ref = (xv @ av) @ bv
    for term, expect_fused in (
        (_mm_mat(x, a, b), True),
        (_mm(x, a, b), False),
    ):
        ir = IR(
            root=term,
            inputs=[x],
            input_names={"x"},
            params={"A": a, "B": b},
        )
        mod = IRModule(ir, param_values=dict(leaves))
        with torch.no_grad():
            assert torch.allclose(mod(xv), ref, atol=1e-9)
        fused_params = [k for k in mod._param_map if "fused" in k]
        assert bool(fused_params) == expect_fused


# ---------------------------------------------------------------------------
#  Task 4 — measured flop deltas on the Linear(d, 2d) + lora(r) fixture
# ---------------------------------------------------------------------------


def test_lora_flop_arithmetic():
    """Document the break-even on the headline fixture.

    ``Linear(d, 2d)`` adapter, rank r, batch M: materialised adapter
    costs ``2·M·d·2d`` flops; factored ``x@A@B`` costs
    ``2·M·r·d + 2·M·r·2d = 2·M·r·3d``.  Factored wins iff
    ``r < 2d/3`` — the ``r·(i+o) < i·o`` break-even, far more generous
    than the 'r < ~d/4' rule of thumb.  (Launch overhead tightens the
    real crossover — see ``test_executor_cost_launch_overhead_crossover``.)
    """
    d, r, m = 256, 4, 32
    x = Var("x", TensorType((m, d)))
    a = Param("lora_A", TensorType((d, r)))
    b = Param("lora_B", TensorType((r, 2 * d)))
    fac, mat = _mm(x, a, b), _mm_mat(x, a, b)
    f_fac, f_mat = dag_cost(fac, flops_cost), dag_cost(mat, flops_cost)
    assert f_fac == 2 * m * r * (d + 2 * d)
    assert f_mat == 2 * m * d * 2 * d  # inner A@B folds: compile-time
    assert f_fac / f_mat == r * 3 * d / (d * 2 * d)
    assert f_fac < f_mat  # r=4 ≪ 2d/3 ≈ 170


def test_lora_base_merge_numerics():
    """``x@W + x@A@B`` vs ``x@(W + A@B)`` — numerically identical
    delivery whichever member extraction prefers (the merge is
    flop-optimal, so flops pricing *should* fold the adapter into
    the base weight; the verify is what makes that safe)."""
    torch.manual_seed(0)
    d, r, o, m = 32, 4, 64, 8
    sink = TorchSink()
    x = Var("x", TensorType((m, d)))
    w = Param("p_w", TensorType((o, d)))
    a = Param("lora_A", TensorType((r, d)))
    b = Param("lora_B", TensorType((o, r)))
    wv = torch.randn(o, d, dtype=torch.float64)
    av = torch.randn(r, d, dtype=torch.float64)
    bv = torch.randn(o, r, dtype=torch.float64)
    xv = torch.randn(m, d, dtype=torch.float64)
    leaves = {"p_w": wv, "lora_A": av, "lora_B": bv}
    params = {"p_w": w, "lora_A": a, "lora_B": b}

    unfused = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make("linear", Op.make("linear", x, a), b),
    )
    merged = Op.make(
        "linear", x, Op.make("add", w, Op.make("matmul", b, a))
    )
    ir = IR(root=unfused, inputs=[x], input_names={"x"}, params=params)
    m_unfused = sink.lower(ir, dict(leaves))
    m_merged = sink.lower(
        IR(root=merged, inputs=[x], input_names={"x"}, params=params),
        dict(leaves),
    )
    with torch.no_grad():
        d_unfused = (
            (m_unfused(xv) - (xv @ wv.T + (xv @ av.T) @ bv.T))
            .abs()
            .max()
        )
        d_cross = (m_unfused(xv) - m_merged(xv)).abs().max()
    assert d_unfused < 1e-9 and d_cross < 1e-9


def test_lora_unscaled_adapter_merges_into_base():
    """``x@W + x@A@B`` (no scaling): the flop-optimal extraction IS
    the merge — ``linear(x, W + B@A)`` materialises one weight at
    compile time and drops the adapter's flops entirely.  Folding
    ``A·B`` into ``W_base`` is what the cost model prefers here, so
    it folds — verified numerically identical on delivery."""
    d, r, o, m = 32, 4, 64, 8
    x = Var("x", TensorType((m, d)))
    w = Param("p_w", TensorType((o, d)))
    a = Param("lora_A", TensorType((r, d)))
    b = Param("lora_B", TensorType((o, r)))
    term = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make("linear", Op.make("linear", x, a), b),
    )
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(ALL_RULES, eid, max_iterations=8)
    best = eg.extract_best(eid, launch_aware_cost)
    assert best == Op.make(
        "linear", x, Op.make("add", w, Op.make("matmul", b, a))
    )
    assert dag_cost(best, flops_cost) < dag_cost(term, flops_cost)


def test_lora_scaled_adapter_stays_factored():
    """HF spelling ``x@W + s·(x@A@B)``: the multiplicative scale sits
    between the merge pattern's operands, so the adapter branch keeps
    a factored chain — the scale gets pushed inside instead."""
    d, r, o, m = 32, 4, 64, 8
    x = Var("x", TensorType((m, d)))
    w = Param("p_w", TensorType((o, d)))
    a = Param("lora_A", TensorType((r, d)))
    b = Param("lora_B", TensorType((o, r)))
    s = Param("lora_s", TensorType((1,)))
    term = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make(
            "mul",
            Op.make("linear", Op.make("linear", x, a), b),
            s,
        ),
    )
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(ALL_RULES, eid, max_iterations=8)
    best = eg.extract_best(eid, launch_aware_cost)
    # Still a two-stage chain under the scale — never collapsed to
    # one dense (o, i) GEMM.
    assert "matmul" not in op_repr(best)
    assert dag_cost(best, flops_cost) <= dag_cost(term, flops_cost)


# ---------------------------------------------------------------------------
#  Task 3 — the detection pass: dense params whose VALUES are low-rank
# ---------------------------------------------------------------------------


def _lowrank_pair(i, o, r, seed=0):
    """An exact rank-r (o, i) factor pair for a ``linear`` weight."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(r, i, generator=g, dtype=torch.float64)
    b = torch.randn(o, r, generator=g, dtype=torch.float64)
    return b @ a, a, b


def test_offer_low_rank_linear():
    """Dense ``linear(x, W)`` with a rank-4 stored weight gets the
    factored member, and flops extraction picks it."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg = EGraph()
    eid = eg.add_term(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_low_rank_factors(eg, tensors)
    assert len(offers) == 1
    rec = offers[0]
    assert rec["orient"] == "linear" and rec["rank"] == r
    assert rec["in_dim"] == i and rec["out_dim"] == o
    assert rec["error"] < 1e-9
    # Derived params registered for the lowerer.
    assert tensors[rec["a_param"]].shape == (r, i)
    assert tensors[rec["b_param"]].shape == (o, r)
    best = eg.extract_best(eid, flops_cost)
    assert best == Op.make(
        "linear",
        Op.make("linear", x, Param(rec["a_param"], TensorType((r, i)))),
        Param(rec["b_param"], TensorType((o, r))),
    )


def test_offer_low_rank_matmul_right():
    """``matmul(x, W)`` (i, o) right-weights factor through the column
    space — ``(x@A)@B``."""
    i, o, r = 32, 64, 4
    g = torch.Generator().manual_seed(1)
    wv = torch.randn(
        i, r, generator=g, dtype=torch.float64
    ) @ torch.randn(r, o, generator=g, dtype=torch.float64)
    x = Var("x", TensorType((8, i)))
    w = Param("p_wm", TensorType((i, o)))
    eg = EGraph()
    eid = eg.add_term(Op.make("matmul", x, w))
    tensors = {"p_wm": wv}
    offers = offer_low_rank_factors(eg, tensors)
    assert len(offers) == 1 and offers[0]["orient"] == "matmul_r"
    best = eg.extract_best(eid, flops_cost)
    assert best == _mm(
        x,
        Param(offers[0]["a_param"], TensorType((i, r))),
        Param(offers[0]["b_param"], TensorType((r, o))),
    )
    assert _param_only_subterms(best) == []


def test_offer_low_rank_biased_linear():
    """``linear(x, W, b)`` offers ``linear(linear(x,A),B) + b``."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    bv = torch.randn(o, dtype=torch.float64)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    bias = Param("p_b", TensorType((o,)))
    eg = EGraph()
    eid = eg.add_term(Op.make("linear", x, w, bias))
    tensors = {"p_w": wv, "p_b": bv}
    offers = offer_low_rank_factors(eg, tensors)
    assert len(offers) == 1
    best = eg.extract_best(eid, flops_cost)
    # Deliverable and correct — lower it and check the value.
    sink = TorchSink()
    xv = torch.randn(8, i, dtype=torch.float64)
    ref = xv @ wv.T + bv
    ir = IR(
        root=best,
        inputs=[x],
        input_names={"x"},
        params={"p_w": w, "p_b": bias},
    )
    with torch.no_grad():
        out = sink.lower(ir, tensors)(xv)
    assert torch.allclose(out, ref, atol=1e-9)


def test_offer_declines():
    """Honest declines: full-rank, missing value, non-2-D, no data
    operand, left-weight matmul, inexact beyond tolerance."""
    o, i, r = 64, 32, 4
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))

    # full-rank value — rank never clears the break-even
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert (
        offer_low_rank_factors(
            eg, {"p_w": torch.randn(o, i, dtype=torch.float64)}
        )
        == []
    )

    # param not in source_tensors
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert offer_low_rank_factors(eg, {}) == []

    # non-tensor / 1-D values skipped
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert offer_low_rank_factors(eg, {"p_w": 3.0}) == []
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert (
        offer_low_rank_factors(
            eg, {"p_w": torch.randn(i, dtype=torch.float64)}
        )
        == []
    )

    # param-only consumer (matmul of two weights) folds anyway
    wm = Param("p_wm", TensorType((i, o)))
    eg = EGraph()
    eg.add_term(Op.make("matmul", w, wm))
    wv, _a, _b = _lowrank_pair(i, o, r)
    mmv = torch.randn(i, o, dtype=torch.float64)
    assert offer_low_rank_factors(eg, {"p_w": wv, "p_wm": mmv}) == []

    # left-weight matmul is out of scope
    eg = EGraph()
    eg.add_term(Op.make("matmul", wm, x))
    assert offer_low_rank_factors(eg, {"p_wm": mmv}) == []


def test_offer_inexact_beyond_tol_declines():
    """A weight whose residual exceeds ``rel_tol`` relative Frobenius
    is not offered."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    wv = wv + torch.randn(o, i, dtype=torch.float64) * 1.0
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert offer_low_rank_factors(eg, {"p_w": wv}, rel_tol=1e-12) == []


def test_offer_rank_gate_and_residual_gate():
    """Two distinct declines inside the factorisation:

    * rank gate — a low-rank weight whose rank still fails
      ``r·(i+o) < i·o`` offers nothing (rank-3 of a (4,8) weight:
      3·12 ≥ 32);
    * residual gate — a dominant row makes the tolerance-certified
      basis cover everything *individually* while the aggregate
      Frobenius residual still exceeds ``rel_tol·‖W‖``.
    """
    # rank gate: (o, i) = (8, 4) weight of exact rank 3
    g = torch.Generator().manual_seed(3)
    wv = torch.randn(8, 3, generator=g, dtype=torch.float64) @ (
        torch.randn(3, 4, generator=g, dtype=torch.float64)
    )
    x = Var("x", TensorType((2, 4)))
    w = Param("p_w", TensorType((8, 4)))
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert offer_low_rank_factors(eg, {"p_w": wv}) == []

    # residual gate: one 1e6 row swamps the per-row tolerance while
    # the rest contribute a collective residual above rel_tol·‖W‖.
    w2 = torch.zeros(64, 32, dtype=torch.float64)
    w2[0, 0] = 1e6
    w2[1:, :32] = torch.eye(63, 32, dtype=torch.float64) * 50.0
    x2 = Var("x", TensorType((8, 32)))
    w_ = Param("p_w2", TensorType((64, 32)))
    eg = EGraph()
    eg.add_term(Op.make("linear", x2, w_))
    assert offer_low_rank_factors(eg, {"p_w2": w2}, rel_tol=1e-4) == []
    # ...and the same shape through the matmul orientation's
    # column-space factorisation.
    wm = Param("p_wm", TensorType((32, 64)))
    eg = EGraph()
    eg.add_term(Op.make("matmul", x2, wm))
    assert (
        offer_low_rank_factors(
            eg, {"p_wm": w2.T.contiguous()}, rel_tol=1e-4
        )
        == []
    )


def test_offer_twice_and_name_reuse():
    """A second pass on the same graph re-offers nothing (the union
    is already done); a fresh graph over already-registered factor
    tensors reuses the identical ``__lr`` names; a poisoned name
    takes the ``_1`` suffix."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    tensors = {"p_w": wv}

    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    assert len(offer_low_rank_factors(eg, tensors)) == 1
    # Same graph, second pass — the member is already there.
    assert offer_low_rank_factors(eg, tensors) == []

    # Fresh graph, same tensors — names with identical values reused.
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    offers = offer_low_rank_factors(eg, tensors)
    assert offers[0]["a_param"] == "p_w__lr4_a"
    assert offers[0]["b_param"] == "p_w__lr4_b"

    # Same names taken by different values -> suffixed fresh names.
    tensors2 = {
        "p_w": wv,
        "p_w__lr4_a": torch.zeros(r, i, dtype=torch.float64),
        "p_w__lr4_b": torch.zeros(o, r, dtype=torch.float64),
    }
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    offers = offer_low_rank_factors(eg, tensors2)
    assert offers[0]["a_param"] == "p_w__lr4_a_1"
    assert offers[0]["b_param"] == "p_w__lr4_b_1"


def test_offer_shared_weight_two_consumers():
    """One low-rank weight read by two consumers: a single derived
    factor pair serves both offers."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    term = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make(
            "linear", Op.make("mul", x, Param("c", TensorType((1,)))), w
        ),
    )
    eg = EGraph()
    eg.add_term(term)
    tensors = {"p_w": wv, "c": torch.ones(1, dtype=torch.float64)}
    offers = offer_low_rank_factors(eg, tensors)
    assert len(offers) == 2
    assert {r_["a_param"] for r_ in offers} == {offers[0]["a_param"]}
    # Only one factor pair was materialised into source_tensors.
    assert sum(k.startswith("p_w__lr") for k in tensors) == 2


def test_detect_factors_pipeline_flag():
    """``detect_factors=True`` on Optimizer.search flows the pass into
    the pipeline; flop pricing selects the factored chain."""
    torch.manual_seed(0)
    i, o, r = 32, 64, 4

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(i, o, bias=False).double()
            with torch.no_grad():
                self.lin.weight.copy_(
                    torch.randn(o, r, dtype=torch.float64)
                    @ torch.randn(r, i, dtype=torch.float64)
                )

        def forward(self, t):
            return self.lin(t)

    model = Dense().eval().double()
    xv = torch.randn(8, i, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, xv, detect_factors=True, cost_fn=flops_cost)
    assert res.stats["low_rank_factors"]
    term = res.term
    assert term.op == "linear" and term.args[0].op == "linear"
    # The extracted program keeps the chain as runtime GEMMs — no
    # param-only subtree waits to fold back into a dense weight.
    assert _param_only_subterms(term) == []
    low = opt.lower(res, xv)
    assert low.verified.passed
    with torch.no_grad():
        assert torch.allclose(model(xv), low.module(xv), atol=1e-9)


def test_detect_factors_no_candidates():
    """Flag on + full-rank weights: the pass runs, offers nothing,
    and records no stats key."""
    torch.manual_seed(0)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(32, 64, bias=False).double()

        def forward(self, t):
            return self.lin(t)

    res = Optimizer(backend=TorchBackend()).search(
        Full().eval().double(),
        torch.randn(8, 32, dtype=torch.float64),
        detect_factors=True,
    )
    assert "low_rank_factors" not in res.stats


def test_detect_factors_off_by_default():
    """The flag is opt-in: default search leaves the dense member."""
    torch.manual_seed(0)
    i, o, r = 32, 64, 4

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(i, o, bias=False).double()
            with torch.no_grad():
                self.lin.weight.copy_(
                    torch.randn(o, r, dtype=torch.float64)
                    @ torch.randn(r, i, dtype=torch.float64)
                )

        def forward(self, t):
            return self.lin(t)

    res = Optimizer(backend=TorchBackend()).search(
        Dense().eval().double(), torch.randn(8, i, dtype=torch.float64)
    )
    assert "low_rank_factors" not in res.stats


# ---------------------------------------------------------------------------
#  Task 2 — KVLatentShare: the factored chain must survive extraction
#  and lowering (no matmul(param, param) re-materialisation)
# ---------------------------------------------------------------------------


def _reify(match, graph, cost_fn):
    """The standard family reify with a chosen cost model."""
    return K._reify_family(
        match,
        graph,
        sink=TorchSink(),
        cost_fn=cost_fn,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )


def test_kv_latent_stays_factored_flops():
    """The reified term keeps ``matmul(x, Ut)`` shared and never holds
    a ``matmul(param, param)`` subtree — nothing folds ``U·D_i``."""
    torch.manual_seed(0)
    g = _lift(_SingleMatmulKV().eval().double(), _x())
    m = K.KVLatentShare().match(g)[0]
    out = _reify(m, g, flops_cost)
    assert out["status"] == "grafted"
    reified = out["reified"]
    assert "p_kv_latent_ut" in reified
    # Factored spelling: matmul(C, D_i) — NOT matmul(x, matmul(U, D_i)).
    assert (
        "(matmul (matmul x, p_kv_latent_ut), p_kv_latent_d" in reified
    )
    assert "(matmul p_kv_latent_ut, p_kv_latent_d" not in reified


def test_kv_latent_linear_sites_stay_factored():
    """Same guard for the ``linear`` orientation recoveries."""
    torch.manual_seed(0)
    g = _lift(_SingleKV().eval().double(), _x())
    m = K.KVLatentShare().match(g)[0]
    out = _reify(m, g, flops_cost)
    assert out["status"] == "grafted"
    reified = out["reified"]
    assert (
        "(linear (matmul x, p_kv_latent_ut), p_kv_latent_d" in reified
    )
    # No param-only composition member selected anywhere.
    assert "matmul p_kv_latent" not in reified


def test_kv_latent_lowered_module_keeps_latent():
    """The lowered graft really stores U and the D_i — not a folded
    dense reconstruction of U·D_i."""
    torch.manual_seed(0)
    model = _SharedKVStack().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=M.MorphismSearch(
            laws=[K.KVLatentShare()], optimize_rest=False
        ),
    )
    key = "kv_latent_share:blocks.0+blocks.1"
    assert stats["matches"][key]["status"] == "grafted"
    rep = opt.blocks[0] if hasattr(opt, "blocks") else None
    # The grafted first slot is the fused family module; its params
    # must include the latent and recoveries, not dense U·D_i folds.
    inner = rep
    pmap = getattr(inner, "_param_map", None)
    if pmap is None and hasattr(inner, "module"):
        pmap = getattr(inner.module, "_param_map", None)
    assert pmap is not None
    names = list(pmap)
    assert any("p_kv_latent_ut" in n for n in names)
    assert any("p_kv_latent_d" in n for n in names)
    # And no fused_ param recreated a (d_in, d_kv) dense weight:
    # the only fused_ params allowed are unrelated (none here).
    assert not any("fused" in n for n in names)


def test_kv_latent_reify_under_exec_cost_is_honest():
    """Under the launch-dominated executor model the latent share is
    *not* a free win at these tiny dims: the reify either declines or
    delivers the re-materialised member — and the reported
    ``kv_flops_after`` is then a factored-form estimate, not the
    delivered program's count.  Pinned so the caveat is visible."""
    torch.manual_seed(0)
    g = _lift(_SingleMatmulKV().eval().double(), _x())
    m = K.KVLatentShare().match(g)[0]
    out = _reify(m, g, executor_cost_for(lowering="generic"))
    if out["status"] == "grafted":
        # When it grafts under exec cost, extraction was free to pick
        # the materialised member — documented behaviour, still
        # verify-correct.
        assert (
            "(matmul x, (matmul p_kv_latent_ut," in out["reified"]
            or "(matmul (matmul x, p_kv_latent_ut)" in out["reified"]
        )


# ---------------------------------------------------------------------------
#  Pass units
# ---------------------------------------------------------------------------


def test_factor_helpers_and_names():
    """Unit cover for the duck-typed numerics and plumbing."""
    wv, _a, _b = _lowrank_pair(32, 64, 4)
    fac = _factor_rows(wv, 1e-8)
    assert fac is not None
    a, b, err = fac
    assert a.shape == (4, 32) and b.shape == (64, 4)
    assert err < 1e-9
    assert torch.allclose(b @ a, wv, atol=1e-9)

    facc = _factor_cols(wv.T, 1e-8)
    assert facc is not None
    ac, bc, _errc = facc
    assert ac.shape == (32, 4) and bc.shape == (4, 64)
    assert torch.allclose(ac @ bc, wv.T, atol=1e-9)

    # zero weight / full-rank decline, both orientations
    z = torch.zeros(4, 8, dtype=torch.float64)
    assert _factor_rows(z, 1e-8) is None
    assert _factor_cols(z, 1e-8) is None
    assert (
        _factor_rows(torch.randn(8, 8, dtype=torch.float64), 1e-8)
        is None
    )

    # fp32 noise needs a looser tolerance
    w32 = torch.randn(64, 4) @ torch.randn(4, 32)
    assert _factor_rows(w32, 1e-8) is None
    assert _factor_rows(w32, 1e-3) is not None

    assert _fresh_name({"a": 1}, "a") == "a_1"
    assert _fresh_name({"a": 1, "a_1": 2}, "a") == "a_2"
    assert _fresh_name({}, "b") == "b"


def test_leaf_params_and_cls_has_var():
    """Class-level leaf/var reachability helpers."""
    x = Var("x", TensorType((8, 32)))
    p = Param("p", TensorType((64, 32)))
    eg = EGraph()
    xid = eg.add_term(x)
    pid = eg.add_term(p)
    tid = eg.add_term(Op.make("linear", x, p))
    leaves = _leaf_params(eg, pid)
    assert leaves == [p]
    # A Var-leaf class yields no Param members; a class with only an
    # op member is walked past it.
    assert _leaf_params(eg, xid) == []
    assert _leaf_params(eg, eg.find(tid)) == []
    memo: dict[int, bool] = {}
    has_var = _cls_has_var(eg, memo)
    assert has_var(tid, frozenset())
    assert not has_var(pid, frozenset())
    # Memoised hit and the cycle guard (on a class never queried —
    # the memo lookup precedes the stack check).
    p2 = Param("p2", TensorType((4,)))
    p2id = eg.add_term(p2)
    assert has_var(tid, frozenset())  # memo hit
    assert not has_var(p2id, frozenset({p2id}))


def test_non_projection_enodes_skipped():
    """Degenerate / non-projection arities are not sites."""
    x = Var("x", TensorType((8, 32)))
    p = Param("p", TensorType((64, 32)))
    eg = EGraph()
    xid = eg.add_term(x)
    pid = eg.add_term(p)
    eg.add_enode("matmul", (xid, pid, pid), {})
    eg.add_enode("linear", (xid,), {})
    eg.add_enode("add", (xid, xid), {})
    assert (
        offer_low_rank_factors(eg, {"p": _lowrank_pair(32, 64, 4)[0]})
        == []
    )


def test_witness_flag_off_still_offers():
    """``witness=False`` records the same members, witness-free."""
    o, i, r = 64, 32, 4
    wv, _a, _b = _lowrank_pair(i, o, r)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    offers = offer_low_rank_factors(eg, {"p_w": wv}, witness=False)
    assert len(offers) == 1


def test_op_repr_smoke():
    """op_repr keeps the factored/member distinction readable."""
    x = Var("x", TensorType((8, 32)))
    a = Param("A", TensorType((32, 4)))
    b = Param("B", TensorType((4, 64)))
    assert "matmul" in op_repr(_mm(x, a, b))
