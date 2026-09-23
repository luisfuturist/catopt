"""Integration tests: torch.export → IR → eqsat → IRModule round-trip.

These cover the real-model path added for the polyhedral-RFC targets
(SwiGLU, RMSNorm) plus matrix-chain weight fusion.
"""

import pytest
import torch

from catopt.models import MatrixChain, RMSNorm, SwiGLU
from catopt.torch_bridge import (
    export_to_ir,
    ir_to_torch_module,
    IRModule,
    _canon_aten_name,
    _SCALAR_OPERAND_OPS,
)
from catopt.ir import IR, Op, Const, Var, Param, TensorType, op_repr
from catopt.egraph import EGraph
from catopt.rules import CATEGORICAL_RULES, SIMPLIFICATION_RULES
from catopt.cost import flops_cost


# ---------------------------------------------------------------------------
#  ATen name canonicalization
# ---------------------------------------------------------------------------

def test_canon_overload_names():
    """Overload-qualified ATen names map to catopt generators."""
    assert _canon_aten_name("mul.Tensor") == "mul"
    assert _canon_aten_name("add.Tensor") == "add"
    assert _canon_aten_name("linear.default") == "linear"
    assert _canon_aten_name("mean.dim") == "mean"
    assert _canon_aten_name("pow.Tensor_Scalar") == "pow"
    assert _canon_aten_name("matmul.default") == "matmul"


def test_canon_plain_names():
    assert _canon_aten_name("matmul") == "matmul"
    assert _canon_aten_name("silu") == "silu"


def test_scalar_operand_ops_contains_pow():
    assert "pow" in _SCALAR_OPERAND_OPS
    assert "mul" in _SCALAR_OPERAND_OPS
    # reductions must NOT be treated as scalar-operand ops
    assert "mean" not in _SCALAR_OPERAND_OPS
    assert "sum" not in _SCALAR_OPERAND_OPS


# ---------------------------------------------------------------------------
#  MatrixChain: associativity + compile-time weight fusion
# ---------------------------------------------------------------------------

def test_matrix_chain_export_and_lower_roundtrip():
    """Exported MatrixChain lowers back with bit-exact equivalence."""
    torch.manual_seed(0)
    model = MatrixChain(16, 8, 4, 2)
    x = torch.randn(3, 16)
    ir, source = export_to_ir(model, x)
    assert ir.root is not None
    assert len(source) == 3
    lowered = ir_to_torch_module(ir, param_values=source)
    model.eval()
    lowered.eval()
    with torch.no_grad():
        d = (model(x.clone()) - lowered(x.clone())).abs().max().item()
    assert d < 1e-6


def test_matrix_chain_associativity_reduces_flops():
    """Eqsat finds the right-associative form with fewer FLOPs."""
    torch.manual_seed(0)
    d0, d1, d2, d3, batch = 128, 64, 32, 8, 256
    model = MatrixChain(d0, d1, d2, d3)
    x = torch.randn(batch, d0)
    ir, _ = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=5000)
    best = eg.extract_best(eid, flops_cost)
    assert flops_cost(best) < flops_cost(ir.root)


def test_ir_module_fuses_weight_chain():
    """IRModule collapses weight-only matmul subtrees into fused params.

    Whatever association the extractor picks, any matmul whose operands
    are both parameters (no data dependence) must be materialised at
    construction time rather than recomputed on every forward pass.
    """
    torch.manual_seed(0)
    model = MatrixChain(16, 8, 4, 2)
    x = torch.randn(2, 16)
    ir, source = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=5000)
    best = eg.extract_best(eid, flops_cost)
    lowered = ir_to_torch_module(
        IR(root=best, inputs=ir.inputs,
           input_names=ir.input_names, params=ir.params),
        param_values=source,
    )
    names = [n for n, _ in lowered.named_parameters()]
    assert any(n.startswith("fused_") for n in names)
    # w2/w3 were consumed by the fused weight -> no longer live params
    assert "p_w2" not in names
    assert "p_w3" not in names
    # and the fused module still evaluates correctly
    model.eval()
    lowered.eval()
    with torch.no_grad():
        d = (model(x.clone()) - lowered(x.clone())).abs().max().item()
    assert d < 1e-5


def test_ir_module_fused_params_match_original():
    """The fused parameter equals the explicit product of the originals."""
    torch.manual_seed(0)
    model = MatrixChain(8, 4, 3, 2)
    x = torch.randn(2, 8)
    ir, source = export_to_ir(model, x)
    fused_term = Op.make(
        "matmul", ir.params["p_w1"],
        Op.make("matmul", ir.params["p_w2"], ir.params["p_w3"]))
    lowered = ir_to_torch_module(
        IR(root=fused_term, inputs=ir.inputs,
           input_names=ir.input_names, params=ir.params),
        param_values=source,
    )
    expected = torch.matmul(model.W1, torch.matmul(model.W2, model.W3))
    got = next(p for _, p in lowered.named_parameters())
    assert torch.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
#  SwiGLU / RMSNorm: the polyhedral-RFC targets
# ---------------------------------------------------------------------------

def test_swiglu_export_shape():
    """SwiGLU exports to the expected linear/mul/silu/linear structure."""
    torch.manual_seed(0)
    model = SwiGLU(16, hidden_mult=2)
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(model, x)
    assert len(source) == 3  # gate, up, down weights
    s = op_repr(ir.root)
    assert "linear" in s
    assert "silu" in s
    assert "mul" in s


def test_swiglu_roundtrip_and_optimize():
    """SwiGLU survives eqsat + lowering with exact equivalence."""
    torch.manual_seed(0)
    model = SwiGLU(16, hidden_mult=2)
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=20, max_nodes=20000)
    best = eg.extract_best(eid, flops_cost)
    lowered = ir_to_torch_module(
        IR(root=best, inputs=ir.inputs,
           input_names=ir.input_names, params=ir.params),
        param_values=source,
    )
    model.eval()
    lowered.eval()
    with torch.no_grad():
        d = (model(x.clone()) - lowered(x.clone())).abs().max().item()
    assert d < 1e-5


def test_rmsnorm_export_captures_reduction_attrs():
    """RMSNorm's mean(-1, keepdim=True) is captured as attrs."""
    torch.manual_seed(0)
    model = RMSNorm(16)
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(model, x)
    s = op_repr(ir.root)
    assert "mean" in s
    assert "rsqrt" in s
    assert "pow" in s


def test_rmsnorm_roundtrip_and_optimize():
    """RMSNorm lowers back with numerical equivalence."""
    torch.manual_seed(0)
    model = RMSNorm(16)
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=20, max_nodes=20000)
    best = eg.extract_best(eid, flops_cost)
    lowered = ir_to_torch_module(
        IR(root=best, inputs=ir.inputs,
           input_names=ir.input_names, params=ir.params),
        param_values=source,
    )
    model.eval()
    lowered.eval()
    with torch.no_grad():
        d = (model(x.clone()) - lowered(x.clone())).abs().max().item()
    assert d < 1e-5


def test_pow_to_square_bridge():
    """pow(x, 2) and square(x) land in the same e-class."""
    from catopt.rules import POW_TO_SQUARE, SQUARE_TO_POW
    from catopt.ir import Var, TensorType
    x = Var("x", TensorType((4, 4)))
    term = Op.make("pow", x, Const(2))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([POW_TO_SQUARE, SQUARE_TO_POW], eid,
           max_iterations=5, max_nodes=1000)
    ops = {n.op for n in eg.get_class(eid).nodes}
    assert "pow" in ops
    assert "square" in ops


def test_silu_mul_form_rule():
    """silu(g)*u expands to (g*sigmoid(g))*u in the e-graph."""
    from catopt.rules import SILU_MUL_FORM
    from catopt.ir import Var, TensorType
    g = Var("g", TensorType((4, 4)))
    u = Var("u", TensorType((4, 4)))
    term = Op.make("mul", Op.make("silu", g), u)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SILU_MUL_FORM], eid, max_iterations=5, max_nodes=1000)
    # The rewritten form lives in the same e-class; silu's inner sigmoid
    # is nested one level down, so check the whole reachable term set.
    def ops_in(eclass_id: int, seen: set[int] | None = None) -> set[str]:
        seen = seen or set()
        eid = eg.find(eclass_id)
        if eid in seen:
            return set()
        seen.add(eid)
        found = set()
        for node in eg.get_class(eid).nodes:
            found.add(node.op)
            for ch in node.children:
                found |= ops_in(ch, seen)
        return found

    ops = ops_in(eid)
    assert "mul" in ops
    assert "sigmoid" in ops


# ---------------------------------------------------------------------------
#  Parallel projections: weight merging (bilinearity) — the composed win
# ---------------------------------------------------------------------------

from catopt.models import ParallelLinear, DeepParallel  # noqa: E402


def _eqsat_best(model, x):
    """Export, saturate with all rules, return (orig_ir, best_term, source)."""
    ir, source = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=50000)
    best = eg.extract_best(eid, flops_cost)
    return ir, best, source


def _lower(ir, best, source):
    return ir_to_torch_module(
        IR(root=best, inputs=ir.inputs,
           input_names=ir.input_names, params=ir.params),
        param_values=source,
    )


def test_weight_factor_linear_halves_flops():
    """x@W1 + x@W2 merges to one matmul: exactly 2x fewer FLOPs."""
    torch.manual_seed(0)
    m = ParallelLinear(64, n_experts=2)
    # batch must be large enough that one matmul beats two even after
    # counting the (one-time) weight-add in the cost model.
    x = torch.randn(4096, 64)
    ir, best, source = _eqsat_best(m, x)
    assert flops_cost(best) == pytest.approx(flops_cost(ir.root) / 2, rel=0.01)
    low = _lower(ir, best, source)
    # merged to a single runtime parameter
    assert len(list(low.named_parameters())) == 1
    m.eval(); low.eval()
    with torch.no_grad():
        assert (m(x.clone()) - low(x.clone())).abs().max() < 1e-4


def test_deepparallel_composed_win():
    """Two laws (weight merge + reassociation) compose into one runtime op."""
    torch.manual_seed(0)
    d = DeepParallel(64, 64, 64)
    # Large batch amortises the one-time W3@(W1+W2) precompute; below the
    # break-even the cost model correctly keeps the outer linear separate.
    x = torch.randn(4096, 64)
    ir, best, source = _eqsat_best(d, x)
    # 3 linears -> 1: expect a large FLOP reduction (>2x)
    assert flops_cost(best) < flops_cost(ir.root) / 2
    low = _lower(ir, best, source)
    assert len(list(low.named_parameters())) == 1
    d.eval(); low.eval()
    with torch.no_grad():
        assert (d(x.clone()) - low(x.clone())).abs().max() < 1e-4


def test_assoc_linear_transpose_order():
    """fused weight must be B @ A (transposes flip the order), not A @ B.

    This is the subtlety that made a hand-derived reference wrong twice;
    the rule must get it right or the result is silently incorrect.
    """
    torch.manual_seed(0)
    d = DeepParallel(8, 12, 16)
    # closed form for stacked F.linear(linear(linear(x,A),B)) with
    # weights A=(12,8), B=(16,12): fused = B @ A  (shapes: 16x8)
    closed = d.W3.weight @ (d.W1.weight + d.W2.weight)
    assert closed.shape == (16, 8)

    x = torch.randn(4096, 8)
    ir, best, source = _eqsat_best(d, x)
    low = _lower(ir, best, source)
    fused = next(p for _, p in low.named_parameters())
    assert torch.allclose(fused, closed, atol=1e-5)

    # The naive order (A @ B) does not merely give wrong VALUES here —
    # with distinct dims it is not even shape-valid (12x8 @ 16x12), so a
    # hand-written merge that gets the order wrong fails loudly or silently
    # broadcasts.  catopt's assoc_linear rule (fused = B @ A) got it right.
    naive_shape_possible = (d.W1.weight.shape[1] == d.W3.weight.shape[0])
    if not naive_shape_possible:
        with pytest.raises(RuntimeError):
            _ = (d.W1.weight + d.W2.weight) @ d.W3.weight


def test_weight_merge_does_not_fire_on_distinct_inputs():
    """Soundness: x@W1 + y@W2 (x != y) must stay unmerged."""
    torch.manual_seed(0)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    w1 = Param("W1", TensorType((4, 4)))
    w2 = Param("W2", TensorType((4, 4)))
    from catopt.rules import WEIGHT_FACTOR

    different = Op.make("add", Op.make("matmul", x, w1),
                        Op.make("matmul", y, w2))
    eg = EGraph()
    eid = eg.add_term(different)
    eg.run([WEIGHT_FACTOR], eid, max_iterations=5, max_nodes=500)

    # The root must still be an `add` of two matmuls — the rule must NOT
    # have collapsed it into a single merged projection.
    root_ops = {n.op for n in eg.get_class(eid).nodes}
    assert "add" in root_ops
    # No merged single-matmul representative may have been added
    # (a merged form would appear as a lone `matmul` leaf-pair).
    matmul_nodes = [
        n for n in eg.get_class(eid).nodes if n.op == "matmul"
    ]
    assert len(matmul_nodes) == 0
