"""Tests for catopt.meta — coherence stratification + rule synthesis.

Part A: coherent laws (comm/assoc/id/neg) are *computed* by
``canonicalize`` rather than stored in the e-graph, so ``stratified_run``
saturates with contentful rules only.  Measured on LinearRecurrence
(T=6, d=8): all-laws saturation reaches 100,353 e-nodes while the
stratified run needs 178 — a ~560x reduction — and the extracted term
is fp64-exact.

Part B: ``synthesize_rules`` composes ordered rule pairs (critical-pair
completion).  Fed ``SCAN_LAWS`` minus ``aff_lift_step`` plus a small
unrolled-recurrence seed, it re-derives the unfolded equivalent of
``aff_lift_step`` (two raw steps -> nested ``apply``); with
``aff_lift_step`` present it emits the composed form
``apply(aff_compose(aff(A2,x), aff(A1,u)), h)`` directly.
"""

import torch

from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, Param, Const, TensorType, op_repr
from catopt import rules as R
from catopt import meta
from catopt.models import LinearRecurrence, SwiGLU
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _T(n=4):
    return TensorType((n, n))


def _opdepth(t, memo):
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max((_opdepth(a, memo) for a in t.args), default=0)
    return memo[k]


def _recurrence_seed():
    """Two raw steps with DISTINCT leaves (maximally general seed)."""
    A1, A2 = Param("A1", _T()), Param("A2", _T())
    h0, u, x = Var("h0", _T()), Var("u", _T()), Var("x", _T())
    seed = Op.make(
        "add",
        Op.make("matmul", A2,
                Op.make("add", Op.make("matmul", A1, h0), u)),
        x)
    return seed, A1, A2, h0, u, x


# ---------------------------------------------------------------------------
#  Part A.1 — classification
# ---------------------------------------------------------------------------

def test_classification_partitions_all_rules():
    rules = meta.module_rules(R)
    coherent, contentful = meta.classify_rules(rules)
    assert len(coherent) + len(contentful) == len(rules)
    assert {r.name for r in coherent} <= meta.COHERENT_RULE_NAMES
    # the spec'd coherent members are actually present in rules.py
    for name in ("comm_add", "comm_mul", "assoc_add", "assoc_mul",
                 "id_add", "id_mul", "double_neg",
                 "assoc_matmul", "assoc_matmul_rev",
                 "aff_assoc", "aff_assoc_rev"):
        assert name in {r.name for r in coherent}
    # contentful spot checks
    cnames = {r.name for r in contentful}
    for name in ("distribute_matmul_over_add", "factor_matmul",
                 "weight_factor_matmul", "aff_lift", "aff_lift_step",
                 "aff_unlift", "aff_compose_unfold", "sub_to_add",
                 "silu_expand"):
        assert name in cnames


# ---------------------------------------------------------------------------
#  Part A.2 — canonicalize
# ---------------------------------------------------------------------------

def test_canonicalize_collapses_coherent_variants():
    a, b, c = Var("a", _T()), Var("b", _T()), Var("c", _T())
    t1 = Op.make("add", Op.make("add", a, b), c)
    t2 = Op.make("add", c, Op.make("add", b, a))
    t3 = Op.make("add", Op.make("add", a, Const(0.0)),
                 Op.make("add", b, c))
    canon = {op_repr(meta.canonicalize(t)) for t in (t1, t2, t3)}
    assert len(canon) == 1  # comm+assoc+id all collapse to one form


def test_canonicalize_identity_and_involution():
    x = Var("x", _T())
    assert meta.canonicalize(Op.make("add", x, Const(0.0))) == x
    assert meta.canonicalize(Op.make("add", Const(0.0), x)) == x
    assert meta.canonicalize(Op.make("mul", x, Const(1.0))) == x
    nn = Op.make("neg", Op.make("neg", x))
    assert meta.canonicalize(nn) == x
    # add of only-zeros collapses to the identity
    assert meta.canonicalize(
        Op.make("add", Const(0.0), Const(0.0))) == Const(0.0)


def test_canonicalize_assoc_only_preserves_order():
    """matmul is associative but NOT commutative: order must survive."""
    a, b, c = Var("a", _T()), Var("b", _T()), Var("c", _T())
    left = Op.make("matmul", Op.make("matmul", a, b), c)
    right = Op.make("matmul", a, Op.make("matmul", b, c))
    swapped = Op.make("matmul", Op.make("matmul", c, b), a)
    assert op_repr(meta.canonicalize(left)) == op_repr(
        meta.canonicalize(right))
    assert op_repr(meta.canonicalize(swapped)) != op_repr(
        meta.canonicalize(left))


def test_canonicalize_idempotent_on_exported_ir():
    torch.manual_seed(0)
    for model, x in [
        (LinearRecurrence(8, 6).eval().double(),
         torch.randn(6, 8, dtype=torch.float64)),
        (SwiGLU(16, hidden_mult=2).eval().double(),
         torch.randn(2, 3, 16, dtype=torch.float64)),
    ]:
        ir, _ = export_to_ir(model, x)
        once = meta.canonicalize(ir.root)
        twice = meta.canonicalize(once)
        assert op_repr(once) == op_repr(twice)


def test_canonicalize_semantics_preserved():
    """Hand-scrambled add/mul soup canonicalizes to a term that lowers
    and evaluates identically (fp64)."""
    torch.manual_seed(0)
    x, y = Var("x", _T()), Var("y", _T())
    orig = Op.make("add",
                   Op.make("mul", x, Const(1.0)),
                   Op.make("add", y, Const(0.0)))
    canon = meta.canonicalize(orig)
    assert op_repr(canon) == op_repr(Op.make("add", x, y))

    ir = IR(root=canon, inputs=[x, y],
            input_names={"x", "y"}, params={})
    mod = ir_to_torch_module(ir, param_values={})
    a, b = torch.randn(4, 4, dtype=torch.float64), \
        torch.randn(4, 4, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(mod(a, b), a + b)


def test_canonicalize_rebalances_compose_chain():
    """A right-leaning aff_compose chain canonicalizes to the balanced
    (Blelloch) bracketing — coherence as a computed canonical form."""
    A = Param("A", _T())
    leaves = [Op.make("aff", A, Var(f"x{i}", _T())) for i in range(4)]
    right = leaves[0]
    for lf in leaves[1:]:
        right = Op.make("aff_compose", lf, right)
    canon = meta.canonicalize(right)
    # balanced tree over 4 leaves has depth 3 (aff_compose levels = 2)
    assert _opdepth(canon, {}) == _opdepth(leaves[0], {}) + 2
    # and it is genuinely balanced: root children are both composes
    assert canon.op == "aff_compose"
    assert all(a.op == "aff_compose" for a in canon.args)


# ---------------------------------------------------------------------------
#  Part A.3 — stratified_run
# ---------------------------------------------------------------------------

def test_stratified_run_reduces_enodes_and_stays_exact():
    """The payoff: all-laws saturation vs. canonicalize + contentful-only.

    Measured on this machine (d=8):
      T=4  full 1,671   stratified 73
      T=5  full 11,688  stratified 118
      T=6  full 100,353 stratified 178
    """
    torch.manual_seed(0)
    T, d = 5, 8
    m = LinearRecurrence(d, T).eval().double()
    x = torch.randn(T, d, dtype=torch.float64)
    ir, st = export_to_ir(m, x)
    all_rules = R.all_rules() + R.SCAN_LAWS

    eg_full = EGraph()
    root_full = eg_full.add_term(ir.root)
    stats_full = eg_full.run(all_rules, root_full, max_iterations=14,
                             max_nodes=400_000)

    eg_strat = EGraph()
    out = meta.stratified_run(
        eg_strat, all_rules, ir.root, max_iterations=14,
        max_nodes=400_000,
        cost_fn=meta.canonical_cost(lambda t, **k: _opdepth(t, {})))

    n_full, n_strat = stats_full["n_enodes"], out["stats"]["n_enodes"]
    assert n_strat < n_full / 10
    assert "comm_add" in out["coherent_dropped"]
    assert "aff_lift" in out["contentful_used"]
    # the affine monoid was still reached by contentful rules alone
    assert "aff" in op_repr(out["canonical_best"])

    opt_ir = IR(root=out["canonical_best"], inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        assert torch.allclose(m(x), mod(x), atol=1e-10)


# ---------------------------------------------------------------------------
#  Part B — synthesize_rules
# ---------------------------------------------------------------------------

def test_synthesize_emits_valid_derived_rules():
    seed, *_ = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    assert len(derived) >= 1

    for d in derived:
        # well-formed: every rhs metavar is bound by the lhs
        assert meta.pattern_metavars(d.rhs) <= meta.pattern_metavars(d.lhs)
        # the rule actually rewrites correctly: instantiate the lhs,
        # apply the derived rule, and compare semantically on fp64
        # random tensors.
        mvars = sorted(meta.pattern_metavars(d.lhs))
        leaves = [Var(f"_t{i}", _T()) for i in range(len(mvars))]
        subst = dict(zip(mvars, leaves))
        t0 = meta.instantiate_pattern(d.lhs, subst)
        env = {v: torch.randn(4, 4, dtype=torch.float64)
               for v in leaves}
        a = meta._eval_term(t0, env)
        applied = meta.apply_rewrite_at(d, t0, ())
        assert applied is not None
        b = meta._eval_term(applied, env)
        assert meta._eval_allclose(a, b, tol=1e-6)


def test_synthesis_rediscovers_lift_step_composite():
    """SCAN_LAWS minus aff_lift_step + a 2-step seed re-derives the
    unfolded equivalent: two raw steps -> nested affine applies."""
    seed, A1, A2, h0, u, x = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)

    # some derived rule fires on the raw 2-step term at the root...
    hits = [d for d in derived
            if meta.match_pattern(d.lhs, seed, {}) is not None]
    assert hits, "no derived rule matches the two-step seed"

    # ...and one of them produces the nested-apply fused form —
    # exactly what aff_lift_step + aff_compose_unfold would yield.
    expected = Op.make(
        "apply", Op.make("aff", A2, x),
        Op.make("apply", Op.make("aff", A1, u), h0))
    found = False
    for d in hits:
        subst = meta.match_pattern(d.lhs, seed, {})
        out = meta.instantiate_pattern(d.rhs, subst)
        if out == expected:
            found = True
    assert found, f"no derived rule produced the fused form: " \
                  f"{[op_repr(d.rhs) for d in hits]}"


def test_synthesis_with_lift_step_emits_composed_form():
    """With aff_lift_step available, the same pair composition emits the
    COMPOSED fusion: apply(aff_compose(aff(A2,x), aff(A1,u)), h)."""
    seed, A1, A2, h0, u, x = _recurrence_seed()
    derived = meta.synthesize_rules(R.SCAN_LAWS, [seed], fuel=3000)

    expected = Op.make(
        "apply",
        Op.make("aff_compose",
                Op.make("aff", A2, x),
                Op.make("aff", A1, u)),
        h0)
    found = False
    for d in derived:
        subst = meta.match_pattern(d.lhs, seed, {})
        if subst is not None and meta.instantiate_pattern(
                d.rhs, subst) == expected:
            found = True
    assert found


def test_derived_rules_fire_in_egraph():
    """A synthesized rule plugs back into the e-graph and rewrites."""
    seed, A1, A2, h0, u, x = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    assert derived

    eg = EGraph()
    root = eg.add_term(seed)
    eg.run(derived, root, max_iterations=5, max_nodes=10_000)
    assert any(eg.rule_fires.get(d.name, 0) > 0 for d in derived)
