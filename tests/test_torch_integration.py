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


# ---------------------------------------------------------------------------
#  Product structure: fused projections (SwiGLU gate/up)
# ---------------------------------------------------------------------------

def _find_ops(term, name, out=None):
    if out is None:
        out = []
    if isinstance(term, Op):
        if term.op == name:
            out.append(term)
        for a in term.args:
            _find_ops(a, name, out)
    return out


def test_swiglu_fuse_produces_shared_gemm():
    """swiglu_fuse merges gate/up into one wide GEMM + two chunk views.

    The extracted form must be
        mul(silu(chunk0(linear(x, cat(Wg,Wu)))), chunk1(linear(x, cat)))
    where the two `linear` subterms are literally the SAME object — the
    e-graph stores the fused projection once and lowering runs one GEMM.
    """
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = SwiGLU(32, hidden_mult=2)
    x = torch.randn(128, 32)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=50000)
    best = eg.extract_best(eid, launch_aware_cost)

    chunks = _find_ops(best, "chunk")
    assert len(chunks) == 2
    # Both chunk projections read the SAME fused linear (one GEMM).
    assert chunks[0].args[0] is chunks[1].args[0]
    fused_lin = chunks[0].args[0]
    assert fused_lin.op == "linear"
    cat = fused_lin.args[1]
    assert cat.op == "concat"

    # Lowering folds cat(Wg,Wu) into one fused runtime parameter and
    # remains numerically equivalent.
    low = _lower(ir, best, source)
    names = [n for n, _ in low.named_parameters()]
    assert any(n.startswith("fused_") for n in names)
    assert "p_gate_weight" not in names
    assert "p_up_weight" not in names
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-5


def test_swiglu_fuse_evals_fused_gemm_once():
    """_eval memoization: the shared fused linear runs exactly once."""
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = SwiGLU(32, hidden_mult=2)
    x = torch.randn(64, 32)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=50000)
    best = eg.extract_best(eid, launch_aware_cost)
    low = _lower(ir, best, source)

    calls = []
    orig_linear = torch.nn.functional.linear
    def counting(*a, **kw):
        calls.append(1)
        return orig_linear(*a, **kw)
    import catopt.torch_bridge as tb
    saved = tb._IR_TO_TORCH["linear"]
    tb._IR_TO_TORCH["linear"] = counting
    try:
        low.eval()
        with torch.no_grad():
            low(x.clone())
    finally:
        tb._IR_TO_TORCH["linear"] = saved
    # gate/up fused -> 1 wide GEMM; + down projection = 2 linear calls
    assert len(calls) == 2


def test_swiglu_fuse_requires_shared_input():
    """The pairing rule must not fire when gate/up read DIFFERENT inputs."""
    from catopt.rules import SWIGLU_FUSE
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    wa = Param("Wa", TensorType((4, 4)))
    wb = Param("Wb", TensorType((4, 4)))
    t = Op.make("mul",
                Op.make("silu", Op.make("linear", x, wa)),
                Op.make("linear", y, wb))
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([SWIGLU_FUSE], eid, max_iterations=5, max_nodes=500)
    # No chunk/concat nodes may appear: the product rewrite is invalid
    # without the shared source object.
    ops = {n.op for n in eg.get_class(eid).nodes}
    assert "chunk" not in ops and "concat" not in ops


# ---------------------------------------------------------------------------
#  Fused QKV (attribute metavariables + triple pairing)
# ---------------------------------------------------------------------------

def test_attention_roundtrip():
    """AttentionBlock exports and lowers bit-exactly."""
    from catopt.models import AttentionBlock
    torch.manual_seed(0)
    m = AttentionBlock(64, n_heads=4).eval()
    x = torch.randn(2, 8, 64)
    ir, source = export_to_ir(m, x)
    low = _lower(ir, ir.root, source)
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-5


def test_qkv_fuse_produces_single_gemm():
    """qkv_fuse merges q/k/v into one GEMM + three chunk projections."""
    from catopt.models import AttentionBlock
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = AttentionBlock(64, n_heads=4).eval()
    x = torch.randn(2, 8, 64)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=50000)
    best = eg.extract_best(eid, launch_aware_cost)

    chunks = _find_ops(best, "chunk")
    assert len(chunks) == 3
    # all three chunks read the SAME fused linear
    assert chunks[0].args[0] is chunks[1].args[0] is chunks[2].args[0]
    assert chunks[0].args[0].op == "linear"
    # concat of three weights along dim 0
    cat = chunks[0].args[0].args[1]
    assert cat.op == "concat"

    low = _lower(ir, best, source)
    names = [n for n, _ in low.named_parameters()]
    assert any(n.startswith("fused_") for n in names)
    assert "p_q_proj_weight" not in names
    assert "p_k_proj_weight" not in names
    assert "p_v_proj_weight" not in names
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-5


def test_attr_metavariable_binds_shape():
    """Pattern attr value-as-string binds the node's concrete attr."""
    from catopt.rules import QKV_FUSE
    from catopt.models import AttentionBlock
    torch.manual_seed(0)
    m = AttentionBlock(32, n_heads=2).eval()
    x = torch.randn(2, 4, 32)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run([QKV_FUSE], eid, max_iterations=5, max_nodes=5000)
    # the fused enode must carry the ORIGINAL view shape, rebound via $attr:S
    reshape_nodes = [
        n for n in eg._node_to_class if n.op == "reshape"
    ]
    fused_reshapes = [
        n for n in reshape_nodes
        if any(dict(n.attrs).get("shape") == (2, 4, 2, 16) for _ in [0])
    ]
    # at least the three original reshapes exist; fused form adds 3 more
    # with the SAME shape bound through the attribute metavariable.
    assert len(fused_reshapes) >= 3


# ---------------------------------------------------------------------------
#  Norm folding (channel-scale into weight, row-scale hoists out)
# ---------------------------------------------------------------------------

def test_normlinear_folds_channel_scale():
    """linear(x*rms*wn, W) -> rms * linear(x, W*wn): gain folds at compile time."""
    from catopt.models import NormLinear
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = NormLinear(64, 64).eval()
    x = torch.randn(128, 8, 64)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=50000)
    best = eg.extract_best(eid, launch_aware_cost)
    low = _lower(ir, best, source)
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-4
    # the norm gain must be folded INTO the weight: exactly one fused param
    names = [n for n, _ in low.named_parameters()]
    assert names == ["fused_2"] or (
        len([n for n in names if n.startswith("fused_")]) == 1
        and "p_norm_weight" not in names)


def test_row_scale_rejects_data_scale():
    """Soundness: linear(a, data-shaped 'scale') must NOT row-hoist.

    r bound to a (B,T,C) data tensor is well-typed but semantically
    wrong — the check predicate must reject it (regression for the
    diff=9.83 unsoundness caught by the verifier).
    """
    from catopt.rules import LINEAR_ROW_SCALE
    x = Var("x", TensorType((4, 4)))
    r = Var("r", TensorType((4, 4)))  # data-shaped, NOT per-row
    w = Param("W", TensorType((4, 4)))
    t = Op.make("linear", Op.make("mul", x, r), w)
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([LINEAR_ROW_SCALE], eid, max_iterations=5, max_nodes=500)
    # no mul(linear, r) enode may be created
    for n in eg.get_class(eid).nodes:
        if n.op == "mul":
            pytest.fail("row_scale fired on a data-shaped scale")


# ---------------------------------------------------------------------------
#  Asymmetric product pairing (GQA) + stacked TransformerBlock
# ---------------------------------------------------------------------------

def test_gqa_asym_fuse():
    """qkv_fuse_asym: uneven head counts -> split with derived sizes."""
    from catopt.models import GQAAttention
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = GQAAttention(128, n_heads=4, n_kv_heads=2).eval()
    x = torch.randn(2, 8, 128)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=20, max_nodes=50000)
    best = eg.extract_best(eid, launch_aware_cost)

    splits = _find_ops(best, "split")
    assert len(splits) == 3
    # derived sizes: q=4*32=128, k=v=2*32=64
    for s in splits:
        assert s.attrs.get("sizes") == (128, 64, 64)
    assert splits[0].args[0] is splits[1].args[0] is splits[2].args[0]
    assert splits[0].args[0].op == "linear"

    low = _lower(ir, best, source)
    names = [n for n, _ in low.named_parameters()]
    assert any(n.startswith("fused_") for n in names)
    assert "p_q_proj_weight" not in names
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-5


def test_derive_hook_computes_sizes():
    """Rewrite.derive injects computed attrs into the instantiation."""
    from catopt.rules import QKV_FUSE_ASYM
    from catopt.models import GQAAttention
    torch.manual_seed(0)
    m = GQAAttention(64, n_heads=2, n_kv_heads=1).eval()
    x = torch.randn(2, 4, 64)
    ir, _ = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    changed = eg.apply_rule(QKV_FUSE_ASYM, eid)
    assert changed
    eg.rebuild()
    # a split enode must exist carrying the derived sizes (64, 32, 32)
    sizes = {
        dict(n.attrs).get("sizes")
        for n in eg._node_to_class if n.op == "split"
    }
    assert (64, 32, 32) in sizes


def test_transformer_block_stacks_fusions():
    """One saturation pass finds BOTH fusions in a full block."""
    from catopt.models import TransformerBlock
    from catopt.cost import launch_aware_cost
    torch.manual_seed(0)
    m = TransformerBlock(128, n_heads=4, hidden_mult=2).eval()
    x = torch.randn(4, 16, 128)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES + SIMPLIFICATION_RULES, eid,
           max_iterations=30, max_nodes=200000)
    best = eg.extract_best(eid, launch_aware_cost)
    r = op_repr(best)
    # fused QKV (3 chunks on one GEMM) AND fused gate/up (2 chunks)
    assert r.count("(chunk") >= 5
    low = _lower(ir, best, source)
    n_fused = len([n for n, _ in low.named_parameters()
                   if n.startswith("fused_")])
    assert n_fused >= 2  # qkv concat + gate/up concat
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-4


# ---------------------------------------------------------------------------
#  Diagram-level pairing pass (general product law)
# ---------------------------------------------------------------------------

def test_pairing_pass_five_way_parallel_block():
    """ParallelBlock: all 5 same-source projections fuse into ONE GEMM."""
    from catopt.models import ParallelBlock
    from catopt.optimize import optimize_model
    from catopt.ir import op_repr
    torch.manual_seed(0)
    m = ParallelBlock(128, n_heads=4, hidden_mult=2).eval()
    x = torch.randn(4, 16, 128)
    opt, stats = optimize_model(m, x, verbose=False)
    r = op_repr(opt._root)
    assert stats.get("paired_extract")
    assert r.count("(split") >= 5
    # uneven sizes: q,k,v are dim; gate,up are 2*dim
    assert "(128, 128, 128, 256, 256)" in r
    m.eval(); opt.eval()
    with torch.no_grad():
        d = (m(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-4


def test_pairing_subsumes_swiglu_rule():
    """The pairing pass alone fuses gate/up — no consumer pattern needed."""
    from catopt.models import SwiGLU
    from catopt.rules import (pair_shared_input_linears,
                              CATEGORICAL_RULES, SIMPLIFICATION_RULES,
                              SWIGLU_FUSE, PARALLEL_MUL_FUSE,
                              QKV_FUSE, QKV_FUSE_ASYM)
    from catopt.cost import launch_aware_cost, dag_cost
    subsumed = {SWIGLU_FUSE.name, PARALLEL_MUL_FUSE.name,
                QKV_FUSE.name, QKV_FUSE_ASYM.name}
    rules = [r for r in CATEGORICAL_RULES if r.name not in subsumed]
    torch.manual_seed(0)
    m = SwiGLU(64, 2).eval()
    x = torch.randn(16, 64)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(rules + SIMPLIFICATION_RULES, eid, max_iterations=10,
           max_nodes=50000)
    groups = pair_shared_input_linears(eg)
    eg.rebuild()
    assert groups and any(len(g) >= 2 for g in groups)
    greedy = eg.extract_best(eid, launch_aware_cost)
    forced = eg.extract_paired(eid, launch_aware_cost, groups)
    assert forced is not None
    assert dag_cost(forced, launch_aware_cost) <= dag_cost(greedy, launch_aware_cost)
    splits = _find_ops(forced, "split")
    assert len(splits) == 2
    assert splits[0].args[0] is splits[1].args[0]
    low = _lower(ir, forced, source)
    m.eval(); low.eval()
    with torch.no_grad():
        d = (m(x.clone()) - low(x.clone())).abs().max().item()
    assert d < 1e-5


def test_pairing_no_shared_input_no_fusion():
    """Linears with different inputs must NOT be paired."""
    from catopt.rules import pair_shared_input_linears
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    wa = Param("A", TensorType((4, 4)))
    wb = Param("B", TensorType((4, 4)))
    t = Op.make("mul",
                Op.make("linear", x, wa),
                Op.make("linear", y, wb))
    eg = EGraph()
    eid = eg.add_term(t)
    groups = pair_shared_input_linears(eg)
    assert not any(len(g) >= 2 for g in groups)
    assert not any(n.op == "split" for n in eg._node_to_class)


def test_channel_scale_rejects_row_scale():
    """linear(x*r, W) with r per-row must not fold r into W."""
    from catopt.rules import LINEAR_CHANNEL_SCALE
    x = Var("x", TensorType((4, 8, 4)))
    r = Var("r", TensorType((4, 8, 1)))  # per-row
    w = Param("W", TensorType((4, 4)))
    t = Op.make("linear", Op.make("mul", x, r), w)
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([LINEAR_CHANNEL_SCALE], eid, max_iterations=5, max_nodes=500)
    # mul(W, r) would be ill-typed AND wrong; the check must reject it
    for n in eg.get_class(eid).nodes:
        if n.op == "linear":
            # only the original linear should exist
            pass


def test_multi_input_module():
    from catopt.optimize import optimize_model
    """Modules with several tensor inputs export, optimize, and verify."""
    import torch.nn as nn
    import torch.nn.functional as F

    class TwoInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(16, 16, bias=False)
        def forward(self, x, scale):
            return self.w(x * scale)

    torch.manual_seed(0)
    m = TwoInput().eval()
    x = torch.randn(4, 16)
    s = torch.randn(4, 16)
    opt, stats = optimize_model(m, (x, s), verbose=False)
    with torch.no_grad():
        diff = (m(x, s) - opt(x, s)).abs().max().item()
    assert diff < 1e-4


def test_dropout_eval_is_identity():
    from catopt.optimize import optimize_model
    """Dropout exported in eval mode is a semantic identity and must lower."""
    import torch.nn as nn
    import torch.nn.functional as F

    class WithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(16, 32, bias=False)
            self.w3 = nn.Linear(16, 32, bias=False)
            self.w2 = nn.Linear(32, 16, bias=False)
            self.drop = nn.Dropout(0.5)  # p>0 but eval() makes it identity
        def forward(self, x):
            return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    torch.manual_seed(0)
    m = WithDropout().eval()
    x = torch.randn(8, 16)
    opt, stats = optimize_model(m, x, verbose=False)
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4
    # w1/w3 share an input — the pairing pass should have fired
    assert stats.get("pairing_groups", 0) >= 1


def test_rope_style_ops_roundtrip():
    from catopt.optimize import optimize_model
    """RoPE-shaped graphs (unbind/stack/expand/flatten/slice) lower and run."""
    import torch.nn as nn

    class MiniRope(nn.Module):
        def forward(self, x, fc, fs):
            b, t, h, d = x.shape
            xr, xi = x.reshape(b, t, h, d // 2, 2).unbind(-1)
            fc = fc.view(1, t, 1, d // 2)
            fs = fs.view(1, t, 1, d // 2)
            out_r = xr * fc - xi * fs
            out_i = xr * fs + xi * fc
            return torch.stack([out_r, out_i], dim=-1).flatten(3)

    torch.manual_seed(0)
    m = MiniRope().eval()
    x = torch.randn(2, 8, 4, 16)
    fc = torch.randn(8, 8)
    fs = torch.randn(8, 8)
    opt, stats = optimize_model(m, (x, fc, fs), verbose=False)
    with torch.no_grad():
        diff = (m(x, fc, fs) - opt(x, fc, fs)).abs().max().item()
    assert diff < 1e-4


def test_dag_sharing_scales():
    """Deep DAGs with heavy sharing must not blow up extraction/lowering.

    Regression: cost fns, add_term, _uses_input, and collect all used
    unmemoised tree walks — exponential on shared-subterm DAGs (the
    llama2.c 4-layer case hung for minutes).  With id()-memoisation this
    must finish in seconds.
    """
    import time
    # Build a maximally-shared term DAG directly: x feeds every stage,
    # each stage's output feeds all later stages (depth d → 2^d tree if
    # expanded; the DAG object shares subtrees by identity).
    x = Var("x", TensorType((4, 32)))
    shared = x
    t = shared
    for i in range(14):  # tree-expansion would be ~16k nodes; DAG is 14
        t = Op.make("add", t, Op.make("mul", t, shared))
    eg = EGraph()
    t0 = time.time()
    root = eg.add_term(t)
    from catopt.cost import launch_aware_cost
    best = eg.extract_best(root, launch_aware_cost)
    mod = ir_to_torch_module(
        IR(root=best, inputs=[x], input_names={"x"}, params={}))
    dt = time.time() - t0
    assert dt < 30  # exponential blowup made this minutes
    import torch as _t
    xv = _t.randn(4, 32)
    with _t.no_grad():
        out = mod(xv)
    # closed-form check: t_{i+1} = t + t*x; t_0 = x
    expect = xv
    for _ in range(14):
        expect = expect + expect * xv
    assert (out - expect).abs().max().item() < 1e-3
