"""Memo-safety regression tests for the hash-consing cleanup.

Since phase 2a terms are interned, hashable content objects: the memo
tables that used to key on ``id(t)`` (act_eps's ``_has_var`` /
calibrate-eval, meta's ``canonicalize``, the om/omd shape checks and
plan builders) now key on the term itself.  These tests mint pairs of
structurally identical but OBJECT-distinct terms — direct
``Op(...)``/``Var(...)`` construction bypasses ``Op.make``'s intern
table — and drive every previously id-keyed path.  An id-keyed memo
would count each twin separately (and, once GC freed the first
object, could alias a recycled address to a stale entry); a
content-keyed memo treats them as one entry and returns identical
results either way.
"""

import torch

from catopt.act_eps import _has_var, calibrate
from catopt.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from catopt.meta import canonicalize, match_pattern
from catopt.om_lower import _is_om_tree, build_om_plan
from catopt.omd_lower import _is_omd_tree, build_omd_plan


def _T(*shape):
    return TensorType(tuple(shape))


def _fresh(term):
    """Re-mint *term* as an equal but object-distinct tree.

    Direct dataclass construction — never ``Op.make`` — so nothing is
    interned and no node object is shared with the input.
    """
    if isinstance(term, Op):
        return Op(
            term.op,
            tuple(_fresh(a) for a in term.args),
            dict(term.attrs),
        )
    if isinstance(term, Var):
        return Var(term.name, term.typ)
    if isinstance(term, Param):
        return Param(term.name, term.typ)
    if isinstance(term, Const):
        return Const(term.value)
    return term


def _mm():
    """matmul(x:(2,3), w:(3,4)) — separately minted on every call."""
    return Op(
        "matmul",
        (Var("x", _T(2, 3)), Param("w", _T(3, 4))),
        {},
    )


def test_fresh_mint_is_equal_but_distinct():
    t = _mm()
    t2 = _fresh(t)
    assert t2 is not t
    assert t2 == t and hash(t2) == hash(t)


def test_has_var_shared_memo_content_keyed():
    t1 = Op("add", (_mm(), Const(0.0)), {})
    t2 = _fresh(t1)
    memo: dict = {}
    assert _has_var(t1, memo) is True
    n = len(memo)
    # Second call over the twin: content-keyed hits, id-keyed would
    # append a whole second set of entries for the twin objects.
    assert _has_var(t2, memo) is True
    assert len(memo) == n


def test_canonicalize_shared_memo_consistent():
    x = Var("x", _T(2))
    y = Var("y", _T(2))
    inner = Op("add", (x, Const(0.0)), {})  # identity element drops
    t1 = Op("add", (inner, y), {})
    t2 = _fresh(t1)
    memo: dict = {}
    c1 = canonicalize(t1, memo)
    n = len(memo)
    c2 = canonicalize(t2, memo)
    assert c1 == c2
    assert op_repr(c1) == op_repr(c2)
    assert len(memo) == n


def test_match_pattern_metavar_binds_equal_distinct_terms():
    # A repeated metavar must bind STRUCTURALLY equal subterms — not
    # the same object.  an ``is`` check would refuse the twin.
    pat = Op("add", ("$a", "$a"), {})
    a1 = _mm()
    term = Op("add", (a1, _fresh(a1)), {})
    subst = match_pattern(pat, term)
    assert subst is not None
    assert subst["$a"] == a1


def test_is_om_tree_shared_memo_content_keyed():
    leaf = Op("om_elem", (Var("s", _T(4)), Var("v", _T(8))), {})
    t1 = Op("om_compose", (leaf, _fresh(leaf)), {})
    t2 = _fresh(t1)
    memo: dict = {}
    assert _is_om_tree(t1, memo) is True
    n = len(memo)
    assert _is_om_tree(t2, memo) is True
    assert len(memo) == n


def test_is_omd_tree_shared_memo_content_keyed():
    a = Var("a", _T(4))
    leaf = Op("omd", (a, a, a, a), {})
    t1 = Op("omd_compose", (leaf, _fresh(leaf)), {})
    t2 = _fresh(t1)
    memo: dict = {}
    assert _is_omd_tree(t1, memo) is True
    n = len(memo)
    assert _is_omd_tree(t2, memo) is True
    assert len(memo) == n


def test_build_om_plan_dedups_equal_distinct_leaves():
    # compose(L, L') with L' a separately-minted twin of L is the DAG
    # case: content-keyed bookkeeping sees ONE leaf contributing its
    # carrier twice (multiplicity 2); id-keyed bookkeeping would have
    # scheduled the twin as a second leaf.
    l1 = Op("om_elem", (Var("s", _T(4)), Var("v", _T(8))), {})
    l2 = _fresh(l1)
    assert l2 is not l1 and l2 == l1
    root = Op("om_apply", (Op("om_compose", (l1, l2), {}),), {})
    plan = build_om_plan(root)
    assert plan is not None
    assert plan["leaves"] == [l1]
    grp = plan["leaf_groups"][0]
    assert grp["mults"] == [2]


def test_build_omd_plan_dedups_equal_distinct_leaves():
    a = Var("a", _T(4))
    h = Var("h", _T(4))
    l1 = Op("omd", (a, a, a, a), {})
    l2 = _fresh(l1)
    root = Op("omd_apply", (Op("omd_compose", (l1, l2), {}), h), {})
    plan = build_omd_plan(root)
    assert plan is not None
    assert plan["omd_leaves"] == [l1]
    assert plan["omd_root"] == 1  # slot 0 = leaf, slot 1 = compose


def test_calibrate_ir_eval_consistent_across_mints():
    # The calibrate eval memo is shared with the term-DAG walk: twin
    # subtrees must produce the same site table as a truly shared
    # (interned) subtree.
    x = Var("x", _T(2, 3))
    w = Param("w", _T(3, 4))
    mm = Op.make("matmul", x, w)
    root_twins = Op.make("add", mm, _fresh(mm))
    root_shared = Op.make("add", mm, mm)

    xv = torch.randn(2, 3)
    wv = torch.randn(3, 4)
    r_twins = calibrate(
        IR(root=root_twins, inputs=[x]), (xv,), {"w": wv}
    )
    r_shared = calibrate(
        IR(root=root_shared, inputs=[x]), (xv,), {"w": wv}
    )
    assert r_twins["per_site"] == r_shared["per_site"]
    assert r_twins["global"] == r_shared["global"]
