"""DECODE_GEOM_LAWS — decode-memory geometry rewrites.

The head-packing / KV-layout slice (see the audit in
``catopt_carriers.decode_geom``):

* ``cat_slice_merge`` / ``cat_narrow_merge`` — adjacent slices of
  one base re-join into the wider view; ``slice_full`` /
  ``narrow_full`` close the loop back to the packed buffer.
* ``cat_head_*`` / ``stack_{select,unbind,getitem}_*`` — a view
  covering exactly one packed operand recovers it;
  ``stack_from_cat_unsqueeze`` packs heads materialised apart.
* ``gather_<view>_{out,in}`` — a shared ``index_select`` slides
  across slice/narrow/select/unbind/stack/cat on a different axis;
  ``gather_cat_batch`` folds per-piece gathers on the packed axis.
* ``unary_cat_*`` / ``binary_cat_*`` — the shared-table lift
  (RoPE cos/sin materialised once over the packed buffer),
  broadcast-checked.

Every family is verified fp64-exact through the extract →
``ir_to_torch_module`` path, with check-level and e-graph-level
decline cases for each guard.
"""

import pytest
import torch
from catopt_carriers.decode_geom import (
    DECODE_GEOM_LAWS,
    DECODE_GEOM_RULES,
    _check_binary_cat,
    _check_cat_head_left,
    _check_cat_head_right,
    _check_cat_narrow_left,
    _check_cat_narrow_right,
    _check_cat_pair,
    _check_cat_unsqueeze,
    _check_gather_axes,
    _check_gather_cat,
    _check_gather_same_axis,
    _check_gather_select,
    _check_gather_select_rev,
    _check_gather_stack,
    _check_gather_stack_rev,
    _check_narrow_full,
    _check_narrow_merge,
    _check_slice_bare,
    _check_slice_full,
    _check_slice_full_step,
    _check_slice_merge,
    _check_slice_merge_step,
    _check_stack_axis,
    _check_stack_getitem,
    _check_stack_view,
    _derive_gather_same_axis,
    _derive_gather_select,
    _derive_gather_select_rev,
    _derive_gather_stack,
    _derive_gather_stack_rev,
    _derive_narrow_merge,
)
from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_torch.torch_bridge import ir_to_torch_module

# ---------------------------------------------------------------------------
#  helpers (same protocol as test_decode_laws)
# ---------------------------------------------------------------------------


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _ill_typed():
    """A term whose shape is provably invalid (not a tuple)."""
    return Op.make("add", _v("ia", 2), _v("ib", 3))


def _run(term, rules=None):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        DECODE_GEOM_LAWS if rules is None else rules,
        root,
        max_iterations=10,
        max_nodes=200_000,
    )
    return eg, root


def _class_ops(eg, eid):
    return {n.op for n in eg.get_class(eg.find(eid)).nodes}


def _class_has_op(eg, eid, opname, seen=None):
    """Does the e-class subtree rooted at eid contain op `opname`?"""
    seen = set() if seen is None else seen
    eid = eg.find(eid)
    if eid in seen:
        return False
    seen.add(eid)
    for n in eg.get_class(eid).nodes:
        if n.op == opname:
            return True
        if any(_class_has_op(eg, c, opname, seen) for c in n.children):
            return True
    return False


def _extract_op(eg, root, pred):
    """Force-extract the root member whose enode satisfies ``pred``."""
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if pred(n):
            t = eg.extract_best(canon, flops_cost, overrides={canon: n})
            if t is not None:
                return t
    return None


def _module(term, inputs):
    ir = IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params={},
    )
    return ir_to_torch_module(ir)


# ---------------------------------------------------------------------------
#  1. cat_slice_merge / cat_narrow_merge — fp64 + guards
# ---------------------------------------------------------------------------


def test_cat_slice_merge_fp64():
    """cat(t[:,1:3], t[:,3:7], 1) ≡ t[:,1:7] — exact data motion."""
    t = _v("t", 2, 9, 4)
    term = Op.make(
        "concat",
        Op.make("slice", t, dim=1, start=1, end=3),
        Op.make("slice", t, dim=1, start=3, end=7),
        dim=1,
    )
    eg, root = _run(term)
    merged = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "slice"
            and not _class_has_op(eg, n_.children[0], "concat")
        ),
    )
    assert merged is not None
    mod = _module(merged, [t])
    tt = torch.randn(2, 9, 4, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, tt[:, 1:7])


def test_cat_slice_merge_to_full_buffer():
    """Head slices covering the whole axis merge, then collapse to
    the packed buffer itself (cat_slice_merge ∘ slice_full)."""
    t = _v("t", 2, 8, 4)
    term = Op.make(
        "concat",
        Op.make("slice", t, dim=1, start=0, end=3),
        Op.make("slice", t, dim=1, start=3, end=8),
        dim=1,
    )
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == t


def test_cat_slice_merge_step_spelling():
    """Unit-step slices merge; a strided piece declines."""
    t = _v("t", 2, 9, 4)
    for step, fires in ((1, True), (2, False)):
        term = Op.make(
            "concat",
            Op.make("slice", t, dim=1, start=1, end=3, step=step),
            Op.make("slice", t, dim=1, start=3, end=7, step=step),
            dim=1,
        )
        eg, root = _run(term)
        merged = [
            n
            for n in eg.get_class(eg.find(root)).nodes
            if n.op == "slice"
            and not _class_has_op(eg, n.children[0], "concat")
        ]
        assert bool(merged) is fires, step


def test_cat_narrow_merge_fp64():
    t = _v("t", 2, 9, 4)
    term = Op.make(
        "concat",
        Op.make("narrow", t, dim=1, start=1, length=2),
        Op.make("narrow", t, dim=1, start=3, length=4),
        dim=1,
    )
    eg, root = _run(term)
    merged = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "narrow"
            and not _class_has_op(eg, n_.children[0], "concat")
        ),
    )
    assert merged is not None
    mod = _module(merged, [t])
    tt = torch.randn(2, 9, 4, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, tt.narrow(1, 1, 6))


def _merge_bound(t_shape, *, cd=1, d1=1, d2=1, s1=1, e1=3, s2=3, e2=7):
    return {
        "t": _v("t", *t_shape),
        "$attr:CD": cd,
        "$attr:D1": d1,
        "$attr:D2": d2,
        "$attr:S1": s1,
        "$attr:E1": e1,
        "$attr:S2": s2,
        "$attr:E2": e2,
    }


def test_check_slice_merge_guards():
    ok = _merge_bound((2, 9, 4))
    assert _check_slice_merge(ok)
    # None start reads as 0
    assert _check_slice_merge({**ok, "$attr:S1": None, "$attr:E1": 3})
    # unresolved shape → veto
    assert not _check_slice_merge({**ok, "t": _ill_typed()})
    assert not _check_slice_merge(
        _merge_bound(())
    )  # scalar has no cat axis
    # non-int / mismatched dims → veto
    assert not _check_slice_merge({**ok, "$attr:CD": "1"})
    assert not _check_slice_merge({**ok, "$attr:D1": 0})
    assert not _check_slice_merge({**ok, "$attr:D2": -1})
    # non-int bounds → veto
    assert not _check_slice_merge({**ok, "$attr:E1": "3"})
    # non-contiguous (e1 != s2) / out-of-order bounds → veto
    assert not _check_slice_merge({**ok, "$attr:S2": 4})
    assert not _check_slice_merge({**ok, "$attr:E2": 2})
    assert not _check_slice_merge({**ok, "$attr:S1": 5})


def test_check_slice_merge_step_guards():
    ok = {**_merge_bound((2, 9, 4)), "$attr:P1": 1, "$attr:P2": 1}
    assert _check_slice_merge_step(ok)
    assert _check_slice_merge_step({**ok, "$attr:P1": None})
    assert not _check_slice_merge_step({**ok, "$attr:P1": 2})
    assert not _check_slice_merge_step({**ok, "$attr:P2": -1})
    assert not _check_slice_merge_step({**ok, "$attr:S2": 4})


def _nmerge_bound(t_shape, *, cd=1, d1=1, d2=1, s1=1, l1=2, s2=3, l2=4):
    return {
        "t": _v("t", *t_shape),
        "$attr:CD": cd,
        "$attr:D1": d1,
        "$attr:D2": d2,
        "$attr:S1": s1,
        "$attr:L1": l1,
        "$attr:S2": s2,
        "$attr:L2": l2,
    }


def test_check_narrow_merge_guards():
    ok = _nmerge_bound((2, 9, 4))
    assert _check_narrow_merge(ok)
    assert _check_narrow_merge({**ok, "$attr:L1": 0, "$attr:S2": 1})
    assert not _check_narrow_merge({**ok, "t": _ill_typed()})
    assert not _check_narrow_merge({**ok, "$attr:CD": "1"})
    assert not _check_narrow_merge({**ok, "$attr:D2": 2})
    assert not _check_narrow_merge({**ok, "$attr:S1": -1})
    assert not _check_narrow_merge({**ok, "$attr:L1": "2"})
    assert not _check_narrow_merge({**ok, "$attr:S2": 4})
    assert not _check_narrow_merge({**ok, "$attr:L2": -1})


def test_derive_narrow_merge():
    ok = _nmerge_bound((2, 9, 4))
    assert _derive_narrow_merge(ok) == {"$attr:LT": 6}
    assert _derive_narrow_merge({**ok, "$attr:L1": "2"}) is None
    assert _derive_narrow_merge({**ok, "$attr:L2": "x"}) is None


# ---------------------------------------------------------------------------
#  2. cat_head_* / stack_* — recovery views
# ---------------------------------------------------------------------------


def test_cat_head_left_fp64():
    """slice(cat(x,y),1,0,ex) ≡ x — the left head read back."""
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    term = Op.make(
        "slice",
        Op.make("concat", x, y, dim=1),
        dim=1,
        start=0,
        end=3,
    )
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == x


def test_cat_head_right_fp64():
    """slice(cat(x,y),1,ex,e≥ex+ey) ≡ y — the tail head."""
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    for end in (8, 10**18):  # exact and to-end sentinel spellings
        term = Op.make(
            "slice",
            Op.make("concat", x, y, dim=1),
            dim=1,
            start=3,
            end=end,
        )
        eg, root = _run(term)
        assert eg.extract_best(eg.find(root), flops_cost) == y, end


def test_cat_head_narrow_fp64():
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    for start, length, want in ((0, 3, x), (3, 5, y)):
        term = Op.make(
            "narrow",
            Op.make("concat", x, y, dim=1),
            dim=1,
            start=start,
            length=length,
        )
        eg, root = _run(term)
        got = eg.extract_best(eg.find(root), flops_cost)
        assert got == want, (start, length)


def _ch_bound(x_shape, y_shape, *, cd=1, vd=1, s=0, e=3, le=None):
    b = {
        "x": _v("x", *x_shape),
        "y": _v("y", *y_shape),
        "$attr:CD": cd,
        "$attr:VD": vd,
        "$attr:S": s,
        "$attr:E": e,
        "$attr:L": le,
    }
    return b


def test_check_cat_head_left_guards():
    ok = _ch_bound((2, 3, 4), (2, 5, 4))
    assert _check_cat_head_left(ok)
    assert _check_cat_head_left({**ok, "$attr:S": None})
    # s != 0 / end != left extent / symbolic extent → veto
    assert not _check_cat_head_left({**ok, "$attr:S": 1})
    assert not _check_cat_head_left({**ok, "$attr:E": 4})
    assert not _check_cat_head_left(_ch_bound((2, None, 4), (2, 5, 4)))
    # broken cat pair → veto
    assert not _check_cat_head_left({**ok, "x": _ill_typed()})
    assert not _check_cat_head_left({**ok, "$attr:CD": "1"})
    assert not _check_cat_head_left({**ok, "$attr:VD": 0})
    assert not _check_cat_head_left(_ch_bound((2, 3, 4), (3, 5, 4)))
    assert not _check_cat_head_left(_ch_bound((2, 3), (2, 5, 4)))


def test_check_cat_head_right_guards():
    ok = _ch_bound((2, 3, 4), (2, 5, 4), s=3, e=8)
    assert _check_cat_head_right(ok)
    # to-end read (end absent/None) covers the tail exactly
    assert _check_cat_head_right({**ok, "$attr:E": None})
    # start != left extent / partial tail / unknown extents → veto
    assert not _check_cat_head_right({**ok, "$attr:S": 2})
    assert not _check_cat_head_right({**ok, "$attr:E": 7})
    assert not _check_cat_head_right(
        _ch_bound((2, None, 4), (2, 5, 4), s=3, e=8)
    )
    assert not _check_cat_head_right(
        _ch_bound((2, 3, 4), (2, None, 4), s=3, e=8)
    )
    assert not _check_cat_head_right({**ok, "$attr:E": "x"})
    # broken cat pair → veto
    assert not _check_cat_head_right({**ok, "x": _ill_typed()})
    assert not _check_cat_head_right(
        _ch_bound((2, 3, 4), (3, 5, 4), s=3, e=8)
    )
    assert not _check_cat_head_right(
        _ch_bound((2, 3), (2, 5, 4), s=3, e=8)
    )


def test_check_cat_narrow_guards():
    okl = _ch_bound((2, 3, 4), (2, 5, 4), s=0, le=3)
    assert _check_cat_narrow_left(okl)
    assert not _check_cat_narrow_left({**okl, "$attr:S": 1})
    assert not _check_cat_narrow_left({**okl, "$attr:L": 4})
    assert not _check_cat_narrow_left(
        _ch_bound((2, None, 4), (2, 5, 4), s=0, le=3)
    )
    assert not _check_cat_narrow_left({**okl, "x": _ill_typed()})
    assert not _check_cat_narrow_left(
        _ch_bound((2, 3), (2, 5, 4), s=0, le=3)
    )
    okr = _ch_bound((2, 3, 4), (2, 5, 4), s=3, le=5)
    assert _check_cat_narrow_right(okr)
    assert not _check_cat_narrow_right({**okr, "$attr:S": 2})
    assert not _check_cat_narrow_right({**okr, "$attr:L": 4})
    assert not _check_cat_narrow_right(
        _ch_bound((2, 3, 4), (2, None, 4), s=3, le=5)
    )
    assert not _check_cat_narrow_right({**okr, "x": _ill_typed()})


def test_stack_view_fp64():
    """select/unbind/getitem on a stack recover the stacked head."""
    a, b = _v("a", 2, 4), _v("b", 2, 4)
    packed = Op.make("stack", a, b, dim=0)
    for op, kwargs in (
        ("select", {"dim": 0, "index": 1}),
        ("unbind", {"dim": 0, "index": 1}),
        ("getitem", {"index": 1}),
    ):
        term = Op.make(op, packed, **kwargs)
        eg, root = _run(term)
        got = eg.extract_best(eg.find(root), flops_cost)
        assert got == b, op
    # head 0 as well
    term = Op.make("select", packed, dim=0, index=0)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == a


def test_stack_view_real_eval():
    a, b = _v("a", 2, 4), _v("b", 2, 4)
    term = Op.make(
        "unbind", Op.make("stack", a, b, dim=1), dim=1, index=0
    )
    eg, root = _run(term)
    got = _extract_op(eg, root, lambda n_: n_.op == "leaf")
    assert got == a
    mod = _module(a, [a])
    ta = torch.randn(2, 4, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(mod(ta), ta)


def _stack_bound(a_shape, b_shape, *, sd=1, dd=1):
    return {
        "a": _v("a", *a_shape),
        "b": _v("b", *b_shape),
        "$attr:SD": sd,
        "$attr:DD": dd,
    }


def test_check_stack_view_guards():
    ok = _stack_bound((2, 4), (2, 4))
    assert _check_stack_view(ok)
    assert _check_stack_axis(ok) == (3, 1)
    # view axis != stack axis → veto
    assert not _check_stack_view({**ok, "$attr:DD": 0})
    # non-int dims → veto
    assert not _check_stack_view({**ok, "$attr:SD": "1"})
    assert not _check_stack_view({**ok, "$attr:DD": "d"})
    # rank/shape mismatch or unresolvable → veto
    assert not _check_stack_view(_stack_bound((2, 4), (2, 4, 1)))
    assert not _check_stack_view(_stack_bound((2, 4), (3, 4)))
    assert not _check_stack_view({**ok, "a": _ill_typed()})


def test_check_stack_getitem_guards():
    ok0 = _stack_bound((2, 4), (2, 4), sd=0)
    assert _check_stack_getitem(ok0)
    assert not _check_stack_getitem(_stack_bound((2, 4), (2, 4)))
    assert not _check_stack_getitem(
        _stack_bound((2, 4), (2, 4), sd="x")
    )
    # scalar operands decline — the pair-shape guard needs a rank
    assert not _check_stack_getitem(_stack_bound((), (), sd=0))


def test_stack_from_cat_unsqueeze_fp64():
    """cat(a[d], b[d], d) ≡ stack(a,b,d) — heads pack into a buffer."""
    a, b = _v("a", 2, 4), _v("b", 2, 4)
    term = Op.make(
        "concat",
        Op.make("unsqueeze", a, dim=1),
        Op.make("unsqueeze", b, dim=1),
        dim=1,
    )
    eg, root = _run(term)
    got = eg.extract_best(eg.find(root), flops_cost)
    mod = _module(got, [a, b])
    ta = torch.randn(2, 4, dtype=torch.float64)
    tb = torch.randn(2, 4, dtype=torch.float64)
    with torch.no_grad():
        out = mod(ta, tb)
    assert torch.equal(out, torch.stack([ta, tb], dim=1))


def test_check_cat_unsqueeze_guards():
    ok = {
        "a": _v("a", 2, 4),
        "b": _v("b", 2, 4),
        "$attr:D": 1,
        "$attr:D2": -2,
        "$attr:CD": 1,
    }
    assert _check_cat_unsqueeze(ok)
    assert not _check_cat_unsqueeze({**ok, "$attr:D2": 0})
    assert not _check_cat_unsqueeze({**ok, "$attr:CD": 2})
    assert not _check_cat_unsqueeze({**ok, "$attr:D": "1"})
    assert not _check_cat_unsqueeze({**ok, "a": _ill_typed()})
    assert not _check_cat_unsqueeze(
        {**ok, "b": _v("b", 2, 4, 1)}  # rank mismatch
    )
    assert not _check_cat_unsqueeze({**ok, "b": _v("b", 3, 4)})


# ---------------------------------------------------------------------------
#  3. Full-read identities
# ---------------------------------------------------------------------------


def test_slice_full_elimination():
    """slice(t,1,0,e≥ext) ≡ t — the whole-axis read collapses."""
    t = _p("t", 3, 4)
    for end in (3, 10**18):
        term = Op.make("slice", t, dim=0, start=0, end=end)
        eg, root = _run(term)
        assert eg.extract_best(eg.find(root), flops_cost) == t, end
    # partial read stays
    term = Op.make("slice", t, dim=0, start=0, end=2)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) != t


def test_slice_full_step_and_bare():
    t = _p("t", 3, 4)
    term = Op.make("slice", t, dim=0, start=0, end=9, step=1)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == t
    # strided "full" read is not identity
    term = Op.make("slice", t, dim=0, start=0, end=9, step=2)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) != t
    # bare slice(t,d) is a one-axis [:]
    term = Op.make("slice", t, dim=1)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == t


def test_narrow_full_elimination():
    t = _p("t", 3, 4)
    term = Op.make("narrow", t, dim=0, start=0, length=3)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) == t
    term = Op.make("narrow", t, dim=0, start=0, length=2)
    eg, root = _run(term)
    assert eg.extract_best(eg.find(root), flops_cost) != t


def test_check_slice_full_guards():
    ok = {"t": _p("t", 3, 4), "$attr:D": 0, "$attr:S": 0, "$attr:E": 3}
    assert _check_slice_full(ok)
    assert _check_slice_full({**ok, "$attr:E": 99})
    assert _check_slice_full({**ok, "$attr:S": None})
    assert _check_slice_full({**ok, "$attr:E": None})
    # partial / non-zero start / unknown extent / non-int → veto
    assert not _check_slice_full({**ok, "$attr:E": 2})
    assert not _check_slice_full({**ok, "$attr:S": 1})
    assert not _check_slice_full({**ok, "$attr:E": "x"})
    assert not _check_slice_full(
        {**ok, "t": _p("t", None, 4), "$attr:E": 9}
    )
    assert not _check_slice_full({**ok, "t": _ill_typed()})
    assert not _check_slice_full({**ok, "t": _p("t")})
    assert not _check_slice_full({**ok, "$attr:D": "0"})


def test_check_slice_full_step_guards():
    ok = {
        "t": _p("t", 3, 4),
        "$attr:D": 0,
        "$attr:S": 0,
        "$attr:E": 3,
        "$attr:P": 1,
    }
    assert _check_slice_full_step(ok)
    assert _check_slice_full_step({**ok, "$attr:P": None})
    assert not _check_slice_full_step({**ok, "$attr:P": 2})
    assert not _check_slice_full_step({**ok, "$attr:E": 2})


def test_check_slice_bare_guards():
    ok = {"t": _p("t", 3, 4), "$attr:D": 1}
    assert _check_slice_bare(ok)
    assert not _check_slice_bare({**ok, "t": _ill_typed()})
    assert not _check_slice_bare({**ok, "t": _p("t")})
    assert not _check_slice_bare({**ok, "$attr:D": "1"})


def test_check_narrow_full_guards():
    ok = {"t": _p("t", 3, 4), "$attr:D": 0, "$attr:S": 0, "$attr:L": 3}
    assert _check_narrow_full(ok)
    assert not _check_narrow_full({**ok, "$attr:S": 1})
    assert not _check_narrow_full({**ok, "$attr:L": 2})
    assert not _check_narrow_full(
        {**ok, "t": _p("t", None, 4), "$attr:L": 4}
    )
    assert not _check_narrow_full({**ok, "t": _ill_typed()})
    assert not _check_narrow_full({**ok, "$attr:D": "d"})


# ---------------------------------------------------------------------------
#  4. Gather mobility — one shared index slides across head views
# ---------------------------------------------------------------------------


def test_gather_slice_out_fp64():
    """index_select(t[:,1:3], 2, I) ≡ index_select(t,2,I)[:,1:3]."""
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "index_select",
        Op.make("slice", t, dim=1, start=1, end=3),
        dim=2,
        index=(0, 2, 0),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "slice"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt.index_select(2, torch.tensor([0, 2, 0]))[:, 1:3]
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)


def test_gather_slice_in_fp64():
    """Reverse direction: gather-over-packed-then-slice pushes in."""
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "slice",
        Op.make("index_select", t, dim=2, index=(1, 0)),
        dim=1,
        start=1,
        end=3,
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "slice")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt[:, 1:3].index_select(2, torch.tensor([1, 0]))
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)


def test_gather_slice_tensor_index_and_step():
    """The index-tensor spelling and strided slices commute too."""
    idx = _v("idx", 3)
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "index_select",
        Op.make("slice", t, dim=1, start=0, end=4, step=2),
        idx,
        dim=2,
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "slice"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t, idx])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ti = torch.tensor([2, 0, 2])
    ref = tt[:, 0:4:2].index_select(2, ti)
    with torch.no_grad():
        out = mod(tt, ti)
    assert torch.equal(out, ref)


def test_gather_narrow_out_fp64():
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "index_select",
        Op.make("narrow", t, dim=1, start=1, length=2),
        dim=2,
        index=(3, 1),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "narrow"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt.index_select(2, torch.tensor([3, 1]))[:, 1:3]
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)


def test_gather_select_out_fp64():
    """index_select(t[:,h], s, I) ≡ index_select(t,s,I)[:,h]."""
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "index_select",
        Op.make("select", t, dim=1, index=2),
        dim=0,
        index=(1, 0),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "select"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt.index_select(0, torch.tensor([1, 0]))[:, 2]
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)


def test_gather_unbind_out_and_in():
    """unbind spelling: gather past the head index both ways."""
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "index_select",
        Op.make("unbind", t, dim=1, index=0),
        dim=0,
        index=(1,),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "unbind"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt.index_select(0, torch.tensor([1]))[:, 0]
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)
    # reverse direction materialises too
    inner = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "unbind")
        ),
    )
    assert inner is not None


def test_gather_select_in_axis_map():
    """select(isel(t,2,I),0,i) → isel(t[0,i],1,I) — axis re-mapped."""
    t = _v("t", 2, 4, 6)
    term = Op.make(
        "select",
        Op.make("index_select", t, dim=2, index=(0, 3)),
        dim=0,
        index=1,
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [t])
    tt = torch.randn(2, 4, 6, dtype=torch.float64)
    ref = tt[1].index_select(1, torch.tensor([0, 3]))
    with torch.no_grad():
        out = mod(tt)
    assert torch.equal(out, ref)


def test_gather_stack_out_and_in():
    """index_select(stack(x,y,1), 2, I) ≡ stack(x[2,I], y[2,I], 1)."""
    x, y = _v("x", 2, 4), _v("y", 2, 4)
    term = Op.make(
        "index_select",
        Op.make("stack", x, y, dim=1),
        dim=2,
        index=(2, 0),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "stack"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [x, y])
    tx = torch.randn(2, 4, dtype=torch.float64)
    ty = torch.randn(2, 4, dtype=torch.float64)
    ref = torch.stack([tx, ty], 1).index_select(2, torch.tensor([2, 0]))
    with torch.no_grad():
        out = mod(tx, ty)
    assert torch.equal(out, ref)
    # reverse: same-index gathers under a stack batch to one gather
    term2 = Op.make(
        "stack",
        Op.make("index_select", x, dim=1, index=(2, 0)),
        Op.make("index_select", y, dim=1, index=(2, 0)),
        dim=1,
    )
    eg2, root2 = _run(term2)
    gath2 = _extract_op(
        eg2,
        root2,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg2, n_.children[0], "stack")
        ),
    )
    assert gath2 is not None
    mod2 = _module(gath2, [x, y])
    with torch.no_grad():
        out2 = mod2(tx, ty)
    assert torch.equal(out2, ref)


def test_gather_cat_out_and_in():
    """index_select(cat(x,y,1), 2, I) ≡ cat(x[2,I], y[2,I], 1)."""
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    term = Op.make(
        "index_select",
        Op.make("concat", x, y, dim=1),
        dim=2,
        index=(0, 3),
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "concat"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert gath is not None
    mod = _module(gath, [x, y])
    tx = torch.randn(2, 3, 4, dtype=torch.float64)
    ty = torch.randn(2, 5, 4, dtype=torch.float64)
    ref = torch.cat([tx, ty], 1).index_select(2, torch.tensor([0, 3]))
    with torch.no_grad():
        out = mod(tx, ty)
    assert torch.equal(out, ref)
    # reverse batches the per-piece gathers
    term2 = Op.make(
        "concat",
        Op.make("index_select", x, dim=2, index=(0, 3)),
        Op.make("index_select", y, dim=2, index=(0, 3)),
        dim=1,
    )
    eg2, root2 = _run(term2)
    gath2 = _extract_op(
        eg2,
        root2,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg2, n_.children[0], "concat")
        ),
    )
    assert gath2 is not None
    mod2 = _module(gath2, [x, y])
    with torch.no_grad():
        assert torch.equal(mod2(tx, ty), ref)


def test_gather_cat_batch_fp64():
    """cat(x[Ix], y[Iy], d) ≡ cat(x,y,d)[Ix + (ex+Iy)] — one gather."""
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    term = Op.make(
        "concat",
        Op.make("index_select", x, dim=1, index=(0, 2)),
        Op.make("index_select", y, dim=1, index=(1, 0)),
        dim=1,
    )
    eg, root = _run(term)
    gath = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "concat")
        ),
    )
    assert gath is not None
    mod = _module(gath, [x, y])
    tx = torch.randn(2, 3, 4, dtype=torch.float64)
    ty = torch.randn(2, 5, 4, dtype=torch.float64)
    ref = torch.cat(
        [
            tx.index_select(1, torch.tensor([0, 2])),
            ty.index_select(1, torch.tensor([1, 0])),
        ],
        1,
    )
    with torch.no_grad():
        out = mod(tx, ty)
    assert torch.equal(out, ref)


def test_headline_per_head_gathers_collapse():
    """The decode read: per-head gathers on packed KV → ONE gather.

    cat(isel(K[:,0:2][s,I]), isel(K[:,2:4][s,I]), h) composes
    gather_slice_out ∘ cat_slice_merge ∘ slice_full into
    ``index_select(K, s, I)`` — the extracted form reads the packed
    table once.
    """
    K = _v("K", 2, 4, 6, 3)

    def hg(a, b):
        return Op.make(
            "index_select",
            Op.make("slice", K, dim=1, start=a, end=b),
            dim=2,
            index=(0, 2, 0),
        )

    term = Op.make("concat", hg(0, 2), hg(2, 4), dim=1)
    eg, root = _run(term)
    got = eg.extract_best(eg.find(root), flops_cost)
    assert got == Op.make("index_select", K, dim=2, index=(0, 2, 0))
    mod = _module(got, [K])
    tk = torch.randn(2, 4, 6, 3, dtype=torch.float64)
    ref = torch.cat(
        [
            tk[:, 0:2].index_select(2, torch.tensor([0, 2, 0])),
            tk[:, 2:4].index_select(2, torch.tensor([0, 2, 0])),
        ],
        1,
    )
    with torch.no_grad():
        out = mod(tk)
    assert torch.equal(out, ref)


def test_check_gather_axes_guards():
    ok = {
        "t": _v("t", 2, 4, 6),
        "$attr:GD": 2,
        "$attr:VD": 1,
    }
    assert _check_gather_axes(ok)
    assert _check_gather_axes({**ok, "$attr:GD": -1})
    # same axis / non-int / unresolvable → veto
    assert not _check_gather_axes({**ok, "$attr:GD": 1})
    assert not _check_gather_axes({**ok, "$attr:GD": -2})
    assert not _check_gather_axes({**ok, "$attr:VD": "1"})
    assert not _check_gather_axes({**ok, "t": _ill_typed()})
    assert not _check_gather_axes({**ok, "t": _v("t")})


def test_check_gather_select_guards():
    ok = {
        "t": _v("t", 2, 4, 6),
        "$attr:SD": 1,
        "$attr:GD": 0,
    }
    assert _check_gather_select(ok)
    assert _derive_gather_select(ok) == {"$attr:GT": 0}
    # gather axis past the removed one shifts by +1
    assert _derive_gather_select({**ok, "$attr:GD": 1}) == {
        "$attr:GT": 2
    }
    assert _derive_gather_select({**ok, "$attr:GD": -1}) == {
        "$attr:GT": 2
    }
    # rank < 2 / non-int dims → veto
    assert not _check_gather_select({**ok, "t": _v("t", 4)})
    assert not _check_gather_select({**ok, "$attr:SD": "1"})
    assert not _check_gather_select({**ok, "$attr:GD": "0"})
    assert not _check_gather_select({**ok, "t": _ill_typed()})
    assert _derive_gather_select({**ok, "t": _ill_typed()}) is None
    assert _derive_gather_select({**ok, "$attr:SD": "1"}) is None
    assert _derive_gather_select({**ok, "t": _v("t", 4)}) is None


def test_check_gather_select_rev_guards():
    ok = {
        "t": _v("t", 2, 4, 6),
        "$attr:SD": 1,
        "$attr:GT": 2,
    }
    assert _check_gather_select_rev(ok)
    assert _derive_gather_select_rev(ok) == {"$attr:GD2": 1}
    assert _derive_gather_select_rev({**ok, "$attr:GT": 0}) == {
        "$attr:GD2": 0
    }
    # gather on the removed axis is a different transform → veto
    assert not _check_gather_select_rev({**ok, "$attr:GT": 1})
    assert _derive_gather_select_rev({**ok, "$attr:GT": 1}) is None
    assert not _check_gather_select_rev({**ok, "t": _v("t", 4)})
    assert not _check_gather_select_rev({**ok, "$attr:GT": "2"})
    assert _derive_gather_select_rev({**ok, "t": _ill_typed()}) is None
    assert _derive_gather_select_rev({**ok, "t": _v("t", 4)}) is None


def test_check_gather_stack_guards():
    ok = {
        "x": _v("x", 2, 4),
        "y": _v("y", 2, 4),
        "$attr:SD": 1,
        "$attr:GD": 2,
    }
    assert _check_gather_stack(ok)
    assert _derive_gather_stack(ok) == {"$attr:GA": 1}
    assert _derive_gather_stack({**ok, "$attr:GD": 0}) == {
        "$attr:GA": 0
    }
    # gather on the stacked axis → veto
    assert not _check_gather_stack({**ok, "$attr:GD": 1})
    assert _derive_gather_stack({**ok, "$attr:GD": 1}) is None
    # shape-incompatible pair / unresolvable → veto
    assert not _check_gather_stack({**ok, "y": _v("y", 3, 4)})
    assert not _check_gather_stack({**ok, "x": _ill_typed()})
    assert not _check_gather_stack({**ok, "x": _v("x")})
    assert _derive_gather_stack({**ok, "x": _ill_typed()}) is None
    assert _derive_gather_stack({**ok, "$attr:SD": "1"}) is None


def test_check_gather_stack_rev_guards():
    ok = {
        "x": _v("x", 2, 4),
        "y": _v("y", 2, 4),
        "$attr:SD": 1,
        "$attr:GT": 1,
    }
    assert _check_gather_stack_rev(ok)
    assert _derive_gather_stack_rev(ok) == {"$attr:GD2": 2}
    assert _derive_gather_stack_rev({**ok, "$attr:GT": 0}) == {
        "$attr:GD2": 0
    }
    assert not _check_gather_stack_rev({**ok, "y": _v("y", 2, 4, 1)})
    assert not _check_gather_stack_rev({**ok, "y": _v("y", 3, 4)})
    assert not _check_gather_stack_rev({**ok, "x": _ill_typed()})
    assert not _check_gather_stack_rev({**ok, "$attr:GT": "1"})
    assert _derive_gather_stack_rev({**ok, "x": _ill_typed()}) is None
    assert _derive_gather_stack_rev({**ok, "$attr:SD": "1"}) is None


def test_check_gather_cat_guards():
    ok = {
        "x": _v("x", 2, 3, 4),
        "y": _v("y", 2, 5, 4),
        "$attr:CD": 1,
        "$attr:GD": 2,
    }
    assert _check_gather_cat(ok)
    assert not _check_gather_cat({**ok, "$attr:GD": 1})
    assert not _check_gather_cat({**ok, "$attr:GD": "2"})
    assert not _check_gather_cat({**ok, "$attr:CD": "1"})
    assert not _check_gather_cat({**ok, "y": _v("y", 3, 5, 4)})
    assert not _check_gather_cat({**ok, "x": _ill_typed()})
    assert not _check_gather_cat({**ok, "x": _v("x", 3)})


def test_check_gather_same_axis_guards():
    ok = {
        "x": _v("x", 2, 3, 4),
        "y": _v("y", 2, 5, 4),
        "$attr:CD": 1,
        "$attr:IX": (0, 2),
        "$attr:IY": (1, 0),
    }
    assert _check_gather_same_axis(ok)
    assert _derive_gather_same_axis(ok) == {"$attr:J": (0, 2, 4, 3)}
    # non-tuple / non-int / bool / empty index → veto
    assert not _check_gather_same_axis({**ok, "$attr:IX": 5})
    assert not _check_gather_same_axis({**ok, "$attr:IX": (0, "x")})
    assert not _check_gather_same_axis({**ok, "$attr:IY": (True,)})
    assert not _check_gather_same_axis({**ok, "$attr:IY": ()})
    # symbolic left extent can't mint the offset → veto
    assert not _check_gather_same_axis({**ok, "x": _v("x", 2, None, 4)})
    assert not _check_gather_same_axis({**ok, "$attr:CD": "1"})
    assert not _check_gather_same_axis({**ok, "y": _v("y", 3, 5, 4)})
    assert not _check_gather_same_axis({**ok, "x": _ill_typed()})
    # derive vetoes mirror the check
    assert _derive_gather_same_axis({**ok, "$attr:IX": 5}) is None
    assert _derive_gather_same_axis({**ok, "x": _ill_typed()}) is None
    assert (
        _derive_gather_same_axis({**ok, "x": _v("x", 2, None, 4)})
        is None
    )


# ---------------------------------------------------------------------------
#  5. Shared-table lift — pointwise maps over the pack
# ---------------------------------------------------------------------------


def test_unary_cat_fp64():
    """cat(silu(x), silu(y), 1) ≡ silu(cat(x,y,1))."""
    x, y = _v("x", 2, 3, 4), _v("y", 2, 5, 4)
    term = Op.make(
        "concat",
        Op.make("silu", x),
        Op.make("silu", y),
        dim=1,
    )
    eg, root = _run(term)
    got = _extract_op(
        eg,
        root,
        lambda n_: n_.op == "silu",
    )
    assert got is not None
    mod = _module(got, [x, y])
    tx = torch.randn(2, 3, 4, dtype=torch.float64)
    ty = torch.randn(2, 5, 4, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tx, ty)
    assert torch.equal(
        out, torch.nn.functional.silu(torch.cat([tx, ty], 1))
    )


def test_binary_cat_rope_table_fp64():
    """cat(x·C, y·C, h) ≡ cat(x,y,h)·C — the rope table materialises
    once over the packed heads (per-position table, head-broadcast)."""
    x, y, C = _v("x", 2, 2, 6), _v("y", 2, 2, 6), _v("C", 2, 1, 6)
    term = Op.make(
        "concat",
        Op.make("mul", x, C),
        Op.make("mul", y, C),
        dim=1,
    )
    eg, root = _run(term)
    got = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "mul"
            and _class_has_op(eg, n_.children[0], "concat")
        ),
    )
    assert got is not None
    mod = _module(got, [x, y, C])
    tx = torch.randn(2, 2, 6, dtype=torch.float64)
    ty = torch.randn(2, 2, 6, dtype=torch.float64)
    tc = torch.randn(2, 1, 6, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tx, ty, tc)
    assert torch.equal(out, torch.cat([tx, ty], 1) * tc)


def test_binary_cat_per_piece_table_declines():
    """A table covering only ONE piece's extent on the cat axis
    cannot multiply the packed buffer → no fold."""
    x, y, C = _v("x", 2, 2, 6), _v("y", 2, 2, 6), _v("C", 2, 2, 6)
    term = Op.make(
        "concat",
        Op.make("mul", x, C),
        Op.make("mul", y, C),
        dim=1,
    )
    eg, root = _run(term)
    lifted = [
        n
        for n in eg.get_class(eg.find(root)).nodes
        if n.op == "mul" and _class_has_op(eg, n.children[0], "concat")
    ]
    assert not lifted


def test_binary_cat_other_ops():
    """add/sub/div lift under the same broadcast contract."""
    x, y, C = _v("x", 4), _v("y", 6), _v("C", 1)
    for op in ("add", "sub", "div"):
        term = Op.make(
            "concat",
            Op.make(op, x, C),
            Op.make(op, y, C),
            dim=0,
        )
        eg, root = _run(term)
        got = _extract_op(
            eg,
            root,
            lambda n_, _op=op, _eg=eg: (
                n_.op == _op
                and _class_has_op(_eg, n_.children[0], "concat")
            ),
        )
        assert got is not None, op


def test_check_cat_pair_guards():
    ok = {
        "x": _v("x", 2, 3, 4),
        "y": _v("y", 2, 5, 4),
        "$attr:CD": 1,
    }
    assert _check_cat_pair(ok)
    assert not _check_cat_pair({**ok, "x": _ill_typed()})
    assert not _check_cat_pair({**ok, "y": _v("y", 5, 4)})
    assert not _check_cat_pair({**ok, "y": _v("y", 3, 5, 4)})
    assert not _check_cat_pair({**ok, "$attr:CD": "1"})


def test_check_binary_cat_guards():
    ok = {
        "x": _v("x", 2, 2, 6),
        "y": _v("y", 2, 2, 6),
        "c": _v("c", 2, 1, 6),
        "$attr:CD": 1,
    }
    assert _check_binary_cat(ok)
    # scalar / per-position tables broadcast fine too
    assert _check_binary_cat({**ok, "c": _v("c")})
    # per-piece extent on the cat axis → veto
    assert not _check_binary_cat({**ok, "c": _v("c", 2, 2, 6)})
    # cat-axis extent covering the whole pack DOES broadcast —
    # but then it never fit the pieces → veto on the piece checks
    assert not _check_binary_cat({**ok, "c": _v("c", 2, 4, 6)})
    # unresolvable shapes → veto
    assert not _check_binary_cat({**ok, "c": _ill_typed()})
    assert not _check_binary_cat({**ok, "x": _ill_typed()})
    assert not _check_binary_cat({**ok, "x": _v("x", 2)})
    assert not _check_binary_cat({**ok, "$attr:CD": "1"})


# ---------------------------------------------------------------------------
#  6. Wiring — the geometry set lands in the carrier presets
# ---------------------------------------------------------------------------


def test_decode_geom_ruleset_composition():
    """The RuleSet carries the whole list, tagged CARRIER+DECODE."""
    from catopt_core.laws import tags

    assert len(DECODE_GEOM_RULES) == len(DECODE_GEOM_LAWS)
    for r in DECODE_GEOM_RULES:
        assert tags.CARRIER in r.tags and tags.DECODE in r.tags
    assert "cat_slice_merge" in DECODE_GEOM_RULES
    # standalone subtraction leaves a strict subset
    sub = DECODE_GEOM_RULES - DECODE_GEOM_RULES.named("cat_slice_merge")
    assert "cat_slice_merge" not in sub


def test_wired_into_carriers_and_default():
    """DECODE_GEOM_RULES composes into CARRIERS → pipeline default."""
    import catopt_carriers
    from catopt_orchestrator.optimize import default_rules
    from catopt_orchestrator.regime import default_rules as reg_rules

    carriers = catopt_carriers.CARRIERS
    assert "cat_slice_merge" in carriers
    assert "narrow_full" in catopt_carriers.DECODE_GEOM_RULES
    with pytest.raises(AttributeError):
        _ = catopt_carriers.NO_SUCH_RULES
    assert "gather_cat_batch" in carriers
    assert "binary_cat_mul" in default_rules()
    assert "gather_slice_out" in reg_rules()
