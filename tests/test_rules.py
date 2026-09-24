"""Tests for rewrite rules."""

import pytest
from catopt.ir import Op, Var, Const, Param, TensorType, op_repr
from catopt.egraph import EGraph, Rewrite
from catopt.rules import (
    all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES,
    COMM_ADD, COMM_MUL, ASSOC_ADD, ASSOC_MUL,
    ID_ADD, ID_MUL, DOUBLE_NEG, SUB_TO_ADD,
    SILU_EXPAND, SQUARE_EXPAND,
    DISTRIBUTE_MUL, FACTOR_MUL,
    NATURALITY_SCALAR, NATURALITY_SCALAR_REV,
    ASSOC_MATMUL, ASSOC_MATMUL_REV,
)


def test_rules_have_correct_names():
    assert COMM_ADD.name == "comm_add"
    assert ASSOC_MATMUL.name == "assoc_matmul"
    assert NATURALITY_SCALAR.name == "naturality_scalar"


def test_all_rules_contains_all():
    """ALL_RULES should be the union of simplification and categorical rules."""
    combined = SIMPLIFICATION_RULES + CATEGORICAL_RULES
    assert len(all_rules()) == len(combined)


def test_id_add_simplifies():
    """add(x, 0) should simplify to x."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("add", x, Const(0))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([ID_ADD], eid, max_iterations=5, max_nodes=100)

    from catopt.cost import count_cost
    best = eg.extract_best(eid, count_cost)
    # The best should be just "x" (leaf, 0 cost)
    # Note: after simplification, the e-class should contain both add(x,0) and x.
    # The lowest-cost term is x (cost 0) vs add(x,0) (cost 1).
    assert isinstance(best, Var) and best.name == "x"


def test_id_mul_simplifies():
    """mul(x, 1) should simplify to x."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("mul", x, Const(1))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([ID_MUL], eid, max_iterations=5, max_nodes=100)

    from catopt.cost import count_cost
    best = eg.extract_best(eid, count_cost)
    assert isinstance(best, Var) and best.name == "x"


def test_silu_expands():
    """silu(x) should expand to mul(x, sigmoid(x))."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("silu", x)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SILU_EXPAND], eid, max_iterations=5, max_nodes=100)

    # The e-class should now contain both silu(x) and mul(x, sigmoid(x))
    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "silu" in op_names  # original
    assert "mul" in op_names   # expanded


def test_sub_converts_to_add():
    """sub(a, b) should convert to add(a, neg(b))."""
    a = Var("a", TensorType((1, 4)))
    b = Var("b", TensorType((1, 4)))
    term = Op.make("sub", a, b)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SUB_TO_ADD], eid, max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "sub" in op_names  # original
    assert "add" in op_names   # converted


def test_square_expands():
    """square(x) should expand to mul(x, x)."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("square", x)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SQUARE_EXPAND], eid, max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "square" in op_names
    assert "mul" in op_names


def test_naturality_round_trip():
    """W @ (x * c) and (W @ x) * c should be equivalent."""
    x = Var("x", TensorType((1, 4)))
    W = Param("W", TensorType((4, 4)))
    c = Const(2.0)

    # Start with: W @ (x * c)  — matmul(W, mul(x, c))
    term = Op.make("matmul", W, Op.make("mul", x, c))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([NATURALITY_SCALAR, NATURALITY_SCALAR_REV], eid,
           max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    # Should contain both forms
    assert len(root_class.nodes) >= 2


def test_matmul_associativity():
    """Both association orders of matmul should be reachable."""
    x = Var("x", TensorType((1, 4)))
    A = Param("A", TensorType((4, 4)))
    B = Param("B", TensorType((4, 4)))
    C = Param("C", TensorType((4, 4)))

    left = Op.make("matmul", Op.make("matmul", Op.make("matmul", x, A), B), C)
    eg = EGraph()
    eid = eg.add_term(left)
    eg.run([ASSOC_MATMUL, ASSOC_MATMUL_REV], eid,
           max_iterations=10, max_nodes=10000)

    root_class = eg.get_class(eid)
    # Both association orders should be present
    assert len(root_class.nodes) >= 2


# ---------------------------------------------------------------------------
#  share_duplicate_param_slices — slice-level (per-head) weight sharing
# ---------------------------------------------------------------------------


def _forced_member_term(eg, eid, op_name):
    """Extract the offered (non-leaf) member of a class by forcing the
    override to the enode with the given op — the same coordinated-
    extraction mechanism the pipeline uses for pairing offers."""
    from catopt.cost import count_cost
    cls = eg.get_class(eid)
    node = next(n for n in cls.nodes if n.op == op_name)
    return eg.extract_best(eid, count_cost,
                           overrides={eg.find(eid): node})


def test_share_duplicate_param_slices_offers_dedup_member():
    """Heads 0,2 bitwise-equal / 1,3 distinct -> the W class gains a
    member that stores only the unique head-blocks and evaluates
    fp64-exact through ir_to_torch_module."""
    import torch
    from catopt.egraph import EGraph
    from catopt.rules import share_duplicate_param_slices
    from catopt.torch_bridge import ir_to_torch_module
    from catopt.ir import IR

    torch.manual_seed(0)
    h, d, i = 4, 3, 5
    o = h * d
    a = torch.randn(d, i, dtype=torch.float64)
    b = torch.randn(d, i, dtype=torch.float64)
    c = torch.randn(d, i, dtype=torch.float64)
    W = torch.cat([a, b, a, c], dim=0)          # heads 0 and 2 tied
    source = {"W": W}

    eg = EGraph()
    w_eid = eg.add_term(Param("W", TensorType((o, i))))
    offers = share_duplicate_param_slices(eg, source)

    assert len(offers) == 1
    off = offers[0]
    assert off["param"] == "W"
    assert off["heads"] == h
    assert off["head_dim"] == d
    assert off["unique"] == 3                    # {a, b, c}
    assert off["index_map"] == (0, 1, 0, 2)
    assert off["stored_after"] == 3 * d * i
    assert off["stored_after"] < off["stored_before"] == o * i

    # The deduplicated tensor was registered for lowering.
    dedup = source[off["dedup_param"]]
    assert tuple(dedup.shape) == (3, d, i)
    assert torch.equal(dedup[0], a) and torch.equal(dedup[1], b)
    assert torch.equal(dedup[2], c)

    # W's e-class now holds the reshape(index_select(D)) member.
    member = _forced_member_term(eg, w_eid, "reshape")
    assert member.op == "reshape"
    assert member.args[0].op == "index_select"
    assert member.args[0].args[0].name == off["dedup_param"]

    # The offered member stores fewer values than the original param.
    ir = IR(root=member, inputs=[], input_names=set(), params={})
    mod = ir_to_torch_module(ir, param_values=source)
    stored = sum(p.numel() for p in mod.parameters())
    assert stored == 3 * d * i < o * i

    # ... and evaluates bitwise-equal to W (float64, exact gather).
    out = mod(torch.zeros(1))                    # no Var leaves: arg unused
    assert out.dtype == torch.float64
    assert torch.equal(out, W)

    # The merge carried a replayable exact witness.
    edge = eg.merge_log[-1]
    assert edge.rule.startswith("share_slices#")
    wit = eg._rule_objs[edge.rule]
    assert wit.error_bound is None
    assert wit.lhs == Param("W", TensorType((o, i)))
    assert wit.rhs == member

    # Under the storage cost model the member wins extraction on its
    # own — 45 stored values < 60 — no coordinated override needed.
    from catopt.cost import param_bytes_cost_for
    best = eg.extract_best(w_eid, param_bytes_cost_for(source))
    assert best == member


def test_share_duplicate_param_slices_no_offer_when_all_distinct():
    """All-distinct head slices -> no member offered, no dedup tensor
    registered, the class keeps only its leaf."""
    import torch
    from catopt.egraph import EGraph
    from catopt.rules import share_duplicate_param_slices

    torch.manual_seed(1)
    W = torch.randn(12, 5, dtype=torch.float64)  # 4 heads x 3 rows, all distinct
    source = {"W": W}

    eg = EGraph()
    w_eid = eg.add_term(Param("W", TensorType((12, 5))))
    offers = share_duplicate_param_slices(eg, source)

    assert offers == []
    assert {n.op for n in eg.get_class(w_eid).nodes} == {"leaf"}
    assert list(source) == ["W"]                 # nothing registered


def test_share_duplicate_param_slices_guards():
    """Only 2-D params that actually appear in the e-graph are
    considered; a dedup form must strictly shrink storage."""
    import torch
    from catopt.egraph import EGraph
    from catopt.rules import share_duplicate_param_slices

    torch.manual_seed(2)
    # A 1-D "weight" (bias-like) with a duplicated half — not 2-D, skip.
    v = torch.randn(4, dtype=torch.float64)
    v1 = torch.cat([v, v])
    # A 2-D param whose duplicated halves don't fit head_counts: o=2,
    # the only divisor in range is h=2 with d=1 — two equal rows DO
    # qualify, so use o=1*2... simplest non-shrinking case: W whose
    # only equal-block factorisation is h=o (all rows unique under it).
    W2 = torch.randn(6, 4, dtype=torch.float64)
    W2[3:] = W2[:3]                              # halves equal: h=2 works
    # W3 is 2-D with duplicated rows but never appears in the e-graph.
    W3 = torch.cat([v.reshape(2, 2), v.reshape(2, 2)], dim=0)
    source = {"bias": v1, "W": W2, "absent": W3}

    eg = EGraph()
    eg.add_term(Param("bias", TensorType((8,))))
    w_eid = eg.add_term(Param("W", TensorType((6, 4))))
    offers = share_duplicate_param_slices(eg, source)

    # Only W qualifies: h=2 splits it into two equal (3,4) halves ->
    # one unique block, index_map (0, 0).
    assert len(offers) == 1
    off = offers[0]
    assert off["param"] == "W" and off["heads"] == 2
    assert off["unique"] == 1 and off["index_map"] == (0, 0)
    assert off["stored_after"] == 3 * 4 < 6 * 4
    # 'bias' (1-D) and 'absent' (not in graph) produced no dedup params.
    assert [n for n in source if n != "W" and n != off["dedup_param"]
            and n != "bias"] == ["absent"]
    assert off["dedup_param"] in source
