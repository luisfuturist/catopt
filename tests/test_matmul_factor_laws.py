"""Tests for the matmul/linear factor-distribute rank guard.

``project/retros/law-gap-targeted-gen.md`` §3 exposed a shipped
unsoundness: ``matmul`` contracts a rank>=2 right operand's axis -2 but
a rank-1 operand's ONLY axis, while ``add`` broadcast-aligns TRAILING
axes.  On a mixed-rank binding the distributive law is genuinely false:

    x(16,) @ a(16,) + x(16,) @ b(16,16)   !=   x @ (a + b)

— the scalar ``x·a`` broadcasts across the matvec's output axis
(``allclose=False``, max diff ~13 on randn).  The shipped
``weight_factor_matmul``, ``factor_matmul`` and
``distribute_matmul_over_add`` (plus ``weight_distribute_matmul`` — the
coherence catalogue's 4-member equivalence class) were unconditional;
``_check_mm_rhs_addends`` / ``_check_mm_rhs_weights`` now carry the
side condition.  The same matcher-cannot-see-shapes class affected the
``swiglu_fuse`` / ``parallel_mul_fuse`` pair folds — ``concat`` +
``chunk(·, 2)`` recovers each projection only when the paired weights
share every dim — guarded by ``_check_fuse_pair``.

Covered surface:

* the check hooks unit-tested branch by branch (equal rank, the legal
  rank>=2 batch-broadcast mixture, the rank-1 x rank>=2 veto, scalar
  and unshaped declines);
* the counterexample binding evaluated under the torch oracle —
  ``allclose=False``, and ``True`` on the sound region;
* firing declines on the counterexample term for every guarded rule,
  and firing + RHS-is-member on the equal-rank binding;
* the 4-member equivalence class still holds under the guard: on a
  well-typed binding the pair reach each other's members;
* the ``swiglu_fuse``/``parallel_mul_fuse`` pair guards — decline on a
  broadcastable-but-unequal weight pair, fire on equal weights;
* the unguarded left-operand pair (``right_distribute`` /
  ``right_factor``) stays unguarded and sound — contraction there is
  the last axis, which IS the broadcast axis.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import (
    ALL_RULES,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    PARALLEL_MUL_FUSE,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    SWIGLU_FUSE,
    WEIGHT_DISTRIBUTE,
    WEIGHT_DISTRIBUTE_LINEAR,
    WEIGHT_FACTOR,
    WEIGHT_FACTOR_LINEAR,
    tags,
)
from catopt_core.laws.tensor import (
    _check_fuse_pair,
    _check_mm_rhs_addends,
    _check_mm_rhs_weights,
)
from catopt_torch.adapters import TorchSink
from catopt_torch.meta_eval import _eval_allclose, _eval_term

_SINK = TorchSink()


def _cost_fn():
    """The pipeline's selection model (roofline + per-dispatch term)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), _SINK.supported_ops
    )


def _saturate(term, rules, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _env_of(*terms):
    """Random fp64 tensors keyed by the leaves of *terms*."""
    leaves: dict = {}

    def collect(t):
        if isinstance(t, (Var, Param)):
            leaves[t.name] = t
        elif isinstance(t, Op):
            for a in t.args:
                collect(a)

    for t in terms:
        collect(t)
    return {
        lf: torch.randn(lf.typ.shape, dtype=torch.float64)
        for lf in leaves.values()
    }


# ---------------------------------------------------------------------------
#  The counterexample under the torch oracle
# ---------------------------------------------------------------------------


def test_counterexample_is_numerically_false():
    """``x@a + x@B`` with a vector, B a matrix: the shipped rewrite was
    wrong — the dot product broadcasts over the matvec's output axis."""
    torch.manual_seed(0)
    x, a, b = _v("x", 16), _p("a", 16), _p("b", 16, 16)
    lhs = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    rhs = Op.make("matmul", x, Op.make("add", a, b))
    env = _env_of(lhs, rhs)
    lv, rv = _eval_term(lhs, env), _eval_term(rhs, env)
    assert lv.shape == rv.shape == (16,)  # same SHAPE, wrong VALUES
    assert not _eval_allclose(lv, rv, 1e-6)


def test_sound_region_is_numerically_true():
    """Same-rank (and both-rank>=2) addends still satisfy bilinearity."""
    torch.manual_seed(0)
    # equal rank
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    lhs = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    rhs = Op.make("matmul", x, Op.make("add", a, b))
    env = _env_of(lhs, rhs)
    assert _eval_allclose(_eval_term(lhs, env), _eval_term(rhs, env))
    # mixed rank, both >= 2: broadcast only replicates the batch axis
    w, a2, b2 = _p("w", 5, 4), _p("a2", 4, 3), _p("b2", 2, 4, 3)
    lhs2 = Op.make(
        "add", Op.make("matmul", w, a2), Op.make("matmul", w, b2)
    )
    rhs2 = Op.make("matmul", w, Op.make("add", a2, b2))
    env2 = _env_of(lhs2, rhs2)
    assert _eval_allclose(
        _eval_term(lhs2, env2), _eval_term(rhs2, env2)
    )
    # both rank-1 (dot + dot)
    w3, a3, b3 = _p("w3", 5, 4), _p("a3", 4), _p("b3", 4)
    lhs3 = Op.make(
        "add", Op.make("matmul", w3, a3), Op.make("matmul", w3, b3)
    )
    rhs3 = Op.make("matmul", w3, Op.make("add", a3, b3))
    env3 = _env_of(lhs3, rhs3)
    assert _eval_allclose(
        _eval_term(lhs3, env3), _eval_term(rhs3, env3)
    )


# ---------------------------------------------------------------------------
#  check hooks — branch by branch
# ---------------------------------------------------------------------------


def test_check_mm_rhs_addends_accepts_legal_shapes():
    assert _check_mm_rhs_addends(
        {"a": _p("a", 4, 3), "b": _p("b", 4, 3)}
    )
    # both >= 2: rank mismatch is legal (batch broadcast)
    assert _check_mm_rhs_addends(
        {"a": _p("a", 4, 3), "b": _p("b", 2, 4, 3)}
    )
    # both vectors
    assert _check_mm_rhs_addends({"a": _p("a", 4), "b": _p("b", 4)})


def test_check_mm_rhs_addends_declines_illegal_shapes():
    # the counterexample shape: vector x matrix
    assert not _check_mm_rhs_addends(
        {"a": _p("a", 16), "b": _p("b", 16, 16)}
    )
    assert not _check_mm_rhs_addends(
        {"a": _p("a", 4, 1), "b": _p("b", 4)}
    )
    # scalar addend can never feed matmul
    assert not _check_mm_rhs_addends({"a": _p("a"), "b": _p("b", 4, 3)})
    # unknown shape (a ()-shaped transpose operand reports None):
    # cannot prove alignment — declines
    assert not _check_mm_rhs_addends(
        {"a": Op.make("transpose", Const(1.0)), "b": _p("b", 4)}
    )
    # missing bindings decline rather than crash
    assert not _check_mm_rhs_addends({})


def test_check_mm_rhs_weights_uses_weight_keys():
    assert _check_mm_rhs_weights(
        {"W": _p("w", 3, 4), "W2": _p("w2", 3, 4)}
    )
    assert not _check_mm_rhs_weights(
        {"W": _p("w", 4), "W2": _p("w2", 4, 3)}
    )
    assert not _check_mm_rhs_weights(
        {"W": _p("w"), "W2": _p("w2", 4, 3)}
    )
    assert not _check_mm_rhs_weights({})


# ---------------------------------------------------------------------------
#  Firing: declines on the counterexample, fires on equal-rank bindings
# ---------------------------------------------------------------------------


def _counterexample_term():
    """The documented witness: x(16,)@a(16,) + x(16,)@b(16,16)."""
    x, a, b = _v("x", 16), _p("a", 16), _p("b", 16, 16)
    return Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )


def _legal_term():
    """x(5,4)@a(4,3) + x(5,4)@b(4,3) — the LoRA-merge shape."""
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    return Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )


def test_weight_factor_declines_on_mixed_rank():
    """The counterexample binding must NOT fire."""
    eg, _root = _saturate(_counterexample_term(), [WEIGHT_FACTOR])
    assert eg.rule_fires.get("weight_factor_matmul", 0) == 0


def test_factor_and_distribute_decline_on_mixed_rank():
    """The same binding vetoes the distribute direction too."""
    w, a, b = _p("w", 5, 4), _p("a", 4), _p("b", 4, 1)
    term = Op.make("matmul", w, Op.make("add", a, b))
    eg, _root = _saturate(term, [DISTRIBUTE_MUL, FACTOR_MUL])
    assert eg.rule_fires.get("distribute_matmul_over_add", 0) == 0
    assert eg.rule_fires.get("factor_matmul", 0) == 0


def test_weight_distribute_declines_on_mixed_rank():
    x, a, b = _v("x", 5, 4), _p("a", 4), _p("b", 4, 3)
    term = Op.make("matmul", x, Op.make("add", a, b))
    eg, _root = _saturate(term, [WEIGHT_DISTRIBUTE])
    assert eg.rule_fires.get("weight_distribute_matmul", 0) == 0


def test_equivalence_class_still_holds_under_guard():
    """On a legal binding the 4-member class still reaches both forms.

    ``factor_matmul`` and ``weight_factor_matmul`` mint the IDENTICAL
    merged member, so when both rules run together the second match is
    a no-op merge — honestly not counted as a fire.  Each rule is
    therefore checked alone on its own LHS instance, and the class is
    checked at the member level: saturate one form under all four
    rules, the other form must land in the root e-class.
    """
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    factored = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    merged = Op.make("matmul", x, Op.make("add", a, b))
    rules = [
        WEIGHT_FACTOR,
        WEIGHT_DISTRIBUTE,
        FACTOR_MUL,
        DISTRIBUTE_MUL,
    ]
    # each rule still fires alone on its own LHS instance
    for rule, inst in (
        (WEIGHT_FACTOR, factored),
        (FACTOR_MUL, factored),
        (WEIGHT_DISTRIBUTE, merged),
        (DISTRIBUTE_MUL, merged),
    ):
        eg_i, _ = _saturate(inst, [rule])
        assert eg_i.rule_fires.get(rule.name, 0) > 0, rule.name

    eg, root = _saturate(factored, rules)
    assert eg.find(root) == eg.find(eg.add_term(merged))

    eg2, root2 = _saturate(merged, rules)
    assert eg2.find(root2) == eg2.find(eg2.add_term(factored))


def test_left_operand_pair_unguarded_and_sound():
    """``(a+b)@W = a@W + b@W`` is sound on every evaluable binding —
    the contraction axis IS the last (broadcast-aligned) axis, so a
    mixed-rank left binding stays exact.  No guard by design."""
    assert RIGHT_DISTRIBUTE.check is None
    assert RIGHT_FACTOR.check is None
    torch.manual_seed(0)
    a, b, w = _p("a", 4), _p("b", 5, 4), _p("w", 4, 3)
    lhs = Op.make(
        "add", Op.make("matmul", a, w), Op.make("matmul", b, w)
    )
    rhs = Op.make("matmul", Op.make("add", a, b), w)
    env = _env_of(lhs, rhs)
    assert _eval_allclose(_eval_term(lhs, env), _eval_term(rhs, env))
    # …and it still fires
    eg, root = _saturate(lhs, [RIGHT_FACTOR])
    assert eg.rule_fires.get("right_factor_matmul", 0) > 0
    assert eg.find(root) == eg.find(eg.add_term(rhs))


# ---------------------------------------------------------------------------
#  The same bug class in the product-structure fuses
# ---------------------------------------------------------------------------


def _swiglu_lhs(sa, sb):
    x = _v("x", 5, sa[1])
    a, b = _p("A", *sa), _p("B", *sb)
    return Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )


def test_fuse_pair_declines_on_unequal_weight_dims():
    """A (4,i) + B (1,i): the LHS broadcasts and evals, but the fused
    ``chunk(·,2)`` cannot recover the unequal halves — vetoed."""
    eg, _root = _saturate(
        _swiglu_lhs((4, 4), (1, 4)), [SWIGLU_FUSE, PARALLEL_MUL_FUSE]
    )
    assert eg.rule_fires.get("swiglu_fuse", 0) == 0
    eg2, _r2 = _saturate(
        Op.make(
            "mul",
            Op.make("linear", _v("x", 5, 4), _p("A", 4, 4)),
            Op.make("linear", _v("x", 5, 4), _p("B", 1, 4)),
        ),
        [PARALLEL_MUL_FUSE],
    )
    assert eg2.rule_fires.get("parallel_mul_fuse", 0) == 0


def test_fuse_pair_check_accepts_equal_and_wildcard():
    assert _check_fuse_pair({"A": _p("A", 4, 4), "B": _p("B", 4, 4)})
    # None dims are wildcards — only provable mismatches veto
    assert _check_fuse_pair(
        {"A": _p("A", 4, None), "B": _p("B", 4, 4)}
    )
    assert not _check_fuse_pair({"A": _p("A", 4, 4), "B": _p("B", 1, 4)})
    assert not _check_fuse_pair(
        {"A": _p("A", 4, 4), "B": _p("B", 2, 4, 4)}
    )
    assert not _check_fuse_pair({})


def test_swiglu_fuse_still_fires_on_equal_weights():
    x = _v("x", 5, 4)
    a, b = _p("A", 4, 4), _p("B", 4, 4)
    term = Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )
    eg, root = _saturate(term, [SWIGLU_FUSE])
    assert eg.rule_fires.get("swiglu_fuse", 0) > 0
    fused = Op.make("linear", x, Op.make("concat", a, b, dim=0))
    rhs = Op.make(
        "mul",
        Op.make(
            "silu", Op.make("chunk", fused, chunks=2, dim=-1, index=0)
        ),
        Op.make("chunk", fused, chunks=2, dim=-1, index=1),
    )
    assert eg.find(root) == eg.find(eg.add_term(rhs))


# ---------------------------------------------------------------------------
#  Guards do not regress the legal firing + certificate path
# ---------------------------------------------------------------------------


def test_weight_factor_fires_and_certifies_on_equal_shapes():
    x, w1, w2 = _v("x", 4, 4), _p("W", 4, 4), _p("W2", 4, 4)
    src = Op.make(
        "add", Op.make("matmul", x, w1), Op.make("matmul", x, w2)
    )
    eg, root = _saturate(src, [WEIGHT_FACTOR])
    assert eg.rule_fires.get("weight_factor_matmul", 0) > 0
    merged = Op.make("matmul", x, Op.make("add", w1, w2))
    assert eg.find(root) == eg.find(eg.add_term(merged))
    cert = eg.certificate(src, merged)
    verify_certificate(src, cert)
    assert cert.rules_used == ["weight_factor_matmul"]


def test_all_four_rules_carry_the_guard():
    """Registration check: the whole right-operand family is guarded."""
    for r in (
        DISTRIBUTE_MUL,
        FACTOR_MUL,
        WEIGHT_FACTOR,
        WEIGHT_DISTRIBUTE,
        WEIGHT_FACTOR_LINEAR,
        WEIGHT_DISTRIBUTE_LINEAR,
        SWIGLU_FUSE,
        PARALLEL_MUL_FUSE,
    ):
        assert r.check is not None, r.name
    # and the tags the pipeline budgets on are unchanged
    assert WEIGHT_FACTOR.tags == {tags.CATEGORICAL, tags.EXPANSIVE}
    assert SWIGLU_FUSE.tags == {tags.FUSION, tags.SUBSUMED}
    names = {r.name for r in ALL_RULES}
    assert {
        "distribute_matmul_over_add",
        "factor_matmul",
        "weight_factor_matmul",
        "weight_distribute_matmul",
    } <= names
