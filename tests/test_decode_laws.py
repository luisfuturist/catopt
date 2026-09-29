"""DECODE_LAWS — decode-time memory-traffic rewrites.

The bandwidth-bound slice of the rewrite surface (see the audit in
``catopt_carriers.decode_laws``):

* ``kv_append_scatter_*`` — ``cat(prefix_view(buf), new)`` ≡
  ``slice_scatter(buf, new, n, n+m)``: the KV-cache append as a buffer
  write, both spellings (``narrow`` / ``slice`` ± ``step``), verified
  bitwise-exact in fp64 via the extract path.
* ``scatter_to_cat_*`` — the reverse bridge.
* ``index_select_dedup`` — the static gather-of-gather dedup
  ``t[I] ≡ t[U][inv]``.
* ``repeat_kv_as_gather`` — the unsqueeze→expand→reshape copy chain
  (repeat_interleave) as an ``index_select`` reindex.
* ``index_select_id`` — identity-gather elimination.
* decline cases for every guard, at both check and e-graph level.
"""

import torch
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_torch.torch_bridge import ir_to_torch_module
from catopt_carriers.decode_laws import (
    DECODE_LAWS,
    _check_cat_prefix,
    _check_dedup_index,
    _check_identity_index,
    _check_scatter_cat,
    _derive_dedup_index,
    _derive_repeat_gather,
    _derive_scatter_end,
)
from catopt_core.cost import flops_cost

# ---------------------------------------------------------------------------
#  helpers
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
        DECODE_LAWS if rules is None else rules,
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
#  1a. KV append — cat(narrow(buf), new) -> slice_scatter, fp64
# ---------------------------------------------------------------------------


def test_kv_append_scatter_narrow_fp64():
    """cat(narrow(buf,1,0,n), new, 1) ≡ slice_scatter — exact data motion."""
    B, n, m, d = 2, 4, 3, 5
    buf, new = _v("buf", B, n + m, d), _v("new", B, m, d)
    term = Op.make(
        "concat",
        Op.make("narrow", buf, dim=1, start=0, length=n),
        new,
        dim=1,
    )
    eg, root = _run(term)
    assert "slice_scatter" in _class_ops(eg, root)
    t = _extract_op(eg, root, lambda n_: n_.op == "slice_scatter")
    assert t is not None
    mod = _module(t, [buf, new])
    tb = torch.randn(B, n + m, d, dtype=torch.float64)
    tn = torch.randn(B, m, d, dtype=torch.float64)
    ref = torch.cat([tb[:, :n], tn], dim=1)
    with torch.no_grad():
        out = mod(tb, tn)
    assert torch.equal(out, ref)


def test_kv_append_scatter_slice_spellings():
    """Both slice spellings (±step attr) fire; the write is exact."""
    B, n, m, d = 2, 5, 2, 4
    for step_attrs in ({"step": 1}, {}):
        buf, new = _v("buf", B, n + m, d), _v("new", B, m, d)
        view = Op.make(
            "slice", buf, dim=1, start=0, end=n, **step_attrs
        )
        term = Op.make("concat", view, new, dim=1)
        eg, root = _run(term)
        assert "slice_scatter" in _class_ops(eg, root), step_attrs
        t = _extract_op(eg, root, lambda n_: n_.op == "slice_scatter")
        mod = _module(t, [buf, new])
        tb = torch.randn(B, n + m, d, dtype=torch.float64)
        tn = torch.randn(B, m, d, dtype=torch.float64)
        ref = torch.cat([tb[:, :n], tn], dim=1)
        with torch.no_grad():
            out = mod(tb, tn)
        assert torch.equal(out, ref)


def test_kv_append_negative_dim():
    """cat dim spelled -2 equals the view's +1 — normalised mod rank."""
    B, n, m, d = 2, 3, 2, 4
    buf, new = _v("buf", B, n + m, d), _v("new", B, m, d)
    term = Op.make(
        "concat",
        Op.make("narrow", buf, dim=-2, start=0, length=n),
        new,
        dim=-2,
    )
    eg, root = _run(term)
    assert "slice_scatter" in _class_ops(eg, root)


def test_kv_append_declines_at_egraph_level():
    """A buffer with a live tail declines: no slice_scatter appears."""
    B, n, m, d = 2, 4, 3, 5
    buf, new = _v("buf", B, n + m + 2, d), _v("new", B, m, d)
    term = Op.make(
        "concat",
        Op.make("narrow", buf, dim=1, start=0, length=n),
        new,
        dim=1,
    )
    eg, root = _run(term)
    assert "slice_scatter" not in _class_ops(eg, root)


def test_kv_append_from_exported_graph():
    """torch.export emits cat(slice(buf,0,n), new) — the law fires on
    the real boundary spelling, and the extracted write is exact."""
    from catopt_torch.torch_bridge import export_to_ir

    class KVAppend(torch.nn.Module):
        def forward(self, buf, new):
            return torch.cat([buf[:, :4], new], dim=1)

    ir, _ = export_to_ir(
        KVAppend(), (torch.randn(2, 7, 5), torch.randn(2, 3, 5))
    )
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(DECODE_LAWS, root, max_iterations=10)
    canon = eg.find(root)
    enode = next(
        n for n in eg.get_class(canon).nodes if n.op == "slice_scatter"
    )
    t = eg.extract_best(canon, flops_cost, overrides={canon: enode})
    mod = ir_to_torch_module(
        IR(
            root=t,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        )
    )
    tb = torch.randn(2, 7, 5, dtype=torch.float64)
    tn = torch.randn(2, 3, 5, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tb, tn)
    assert torch.equal(out, torch.cat([tb[:, :4], tn], dim=1))


# ---------------------------------------------------------------------------
#  1b. check-level decline cases — _check_cat_prefix
# ---------------------------------------------------------------------------


def _cat_bound(buf_shape, new_shape, *, cd=1, vd=1, vs=0, vl=4, vst=1):
    return {
        "buf": _v("buf", *buf_shape),
        "new": _v("new", *new_shape),
        "$attr:CD": cd,
        "$attr:VD": vd,
        "$attr:VS": vs,
        "$attr:VL": vl,
        "$attr:VST": vst,
    }


def test_check_cat_prefix_guards():
    ok = _cat_bound((2, 7, 5), (2, 3, 5))
    assert _check_cat_prefix(ok)
    # no-step spelling: VST simply absent
    no_step = {k: v for k, v in ok.items() if k != "$attr:VST"}
    assert _check_cat_prefix(no_step)
    # strided view / non-prefix view → veto
    assert not _check_cat_prefix({**ok, "$attr:VST": 2})
    assert not _check_cat_prefix({**ok, "$attr:VS": 3})
    # non-int attrs → veto
    assert not _check_cat_prefix({**ok, "$attr:CD": "1"})
    assert not _check_cat_prefix({**ok, "$attr:VL": "4"})
    # unresolvable / rank-mismatched shapes → veto
    assert not _check_cat_prefix({**ok, "buf": _ill_typed()})
    assert not _check_cat_prefix(
        _cat_bound((), (2, 3, 5))
    )  # scalar buffer
    assert not _check_cat_prefix(
        _cat_bound((2, 7, 5), (2, 3))
    )  # rank mismatch
    # view axis != cat axis → veto
    assert not _check_cat_prefix({**ok, "$attr:VD": 0})
    # unknown buffer/new extent on the cat axis → veto
    assert not _check_cat_prefix(_cat_bound((2, None, 5), (2, 3, 5)))
    assert not _check_cat_prefix(_cat_bound((2, 7, 5), (2, None, 5)))
    # buffer tail != write region → veto
    assert not _check_cat_prefix(_cat_bound((2, 9, 5), (2, 3, 5)))
    # off-axis mismatch → veto
    assert not _check_cat_prefix(_cat_bound((2, 7, 5), (3, 3, 5)))
    # unknown off-axis dims pass (the checks veto only provable
    # mismatches — same convention as om.py's _dim_eq).
    assert _check_cat_prefix(_cat_bound((None, 7, 5), (None, 3, 5)))


def test_derive_scatter_end_guards():
    ok = _cat_bound((2, 7, 5), (2, 3, 5))
    assert _derive_scatter_end(ok) == {"$attr:SE": 7}
    assert _derive_scatter_end({**ok, "buf": _ill_typed()}) is None
    assert _derive_scatter_end({**ok, "$attr:CD": "1"}) is None
    assert (
        _derive_scatter_end(_cat_bound((2, None, 5), (2, 3, 5))) is None
    )
    # scalar buffer shape → None
    assert _derive_scatter_end(_cat_bound((), (2, 3, 5))) is None


# ---------------------------------------------------------------------------
#  1c. scatter_to_cat — the reverse bridge
# ---------------------------------------------------------------------------


def test_scatter_to_cat_fp64():
    """slice_scatter(buf, src, 1, s, e) ≡ cat(buf[:s], src, 1)."""
    B, s, m, d = 2, 4, 3, 5
    buf, src = _v("buf", B, s + m, d), _v("src", B, m, d)
    for attrs in (
        {"dim": 1, "start": s, "end": s + m, "step": 1},
        {"dim": 1, "start": s, "end": s + m},
    ):
        term = Op.make("slice_scatter", buf, src, **attrs)
        eg, root = _run(term)
        assert "concat" in _class_ops(eg, root), attrs
        t = _extract_op(eg, root, lambda n_: n_.op == "concat")
        assert t is not None
        mod = _module(t, [buf, src])
        tb = torch.randn(B, s + m, d, dtype=torch.float64)
        ts = torch.randn(B, m, d, dtype=torch.float64)
        ref = tb.slice_scatter(ts, dim=1, start=s, end=s + m)
        with torch.no_grad():
            out = mod(tb, ts)
        assert torch.equal(out, ref)


def _scat_bound(buf_shape, src_shape, *, cd=1, s0=4, se=7, st=1):
    b = {
        "buf": _v("buf", *buf_shape),
        "src": _v("src", *src_shape),
        "$attr:CD": cd,
        "$attr:S0": s0,
        "$attr:SE": se,
    }
    if st != "absent":
        b["$attr:ST"] = st
    return b


def test_check_scatter_cat_guards():
    ok = _scat_bound((2, 7, 5), (2, 3, 5))
    assert _check_scatter_cat(ok)
    # no-step spelling absent
    assert _check_scatter_cat(
        _scat_bound((2, 7, 5), (2, 3, 5), st="absent")
    )
    # strided write → veto
    assert not _check_scatter_cat({**ok, "$attr:ST": 2})
    # non-int / out-of-range bounds → veto
    assert not _check_scatter_cat({**ok, "$attr:CD": "1"})
    assert not _check_scatter_cat({**ok, "$attr:S0": -1})
    assert not _check_scatter_cat({**ok, "$attr:S0": 6, "$attr:SE": 4})
    # unresolvable / rank-mismatched shapes → veto
    assert not _check_scatter_cat({**ok, "buf": _ill_typed()})
    assert not _check_scatter_cat(_scat_bound((), (2, 3, 5)))
    assert not _check_scatter_cat(_scat_bound((2, 7, 5), (2, 3)))
    # unknown extents → veto
    assert not _check_scatter_cat(_scat_bound((2, None, 5), (2, 3, 5)))
    assert not _check_scatter_cat(_scat_bound((2, 7, 5), (2, None, 5)))
    # interior write (e < buf extent) → veto
    assert not _check_scatter_cat(
        _scat_bound((2, 9, 5), (2, 3, 5), s0=4, se=7)
    )
    # src doesn't fill the slice → veto
    assert not _check_scatter_cat(
        _scat_bound((2, 7, 5), (2, 2, 5), s0=4, se=7)
    )
    # off-axis mismatch → veto
    assert not _check_scatter_cat(_scat_bound((2, 7, 5), (3, 3, 5)))
    # unknown off-axis dims pass
    assert _check_scatter_cat(_scat_bound((None, 7, 5), (None, 3, 5)))


# ---------------------------------------------------------------------------
#  2. index_select dedup — static gather-of-gather
# ---------------------------------------------------------------------------


def test_index_select_dedup_fp64():
    """index_select(t, 0, (0,2,0,2,1)) ≡ index_select(t[U], 0, inv)."""
    v, d = 4, 3
    w = _v("w", v, d)
    term = Op.make("index_select", w, dim=0, index=(0, 2, 0, 2, 1))
    eg, root = _run(term)
    # the nested gather member materialised in the root class
    nested = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert nested is not None
    mod = _module(nested, [w])
    tw = torch.randn(v, d, dtype=torch.float64)
    ref = tw[torch.tensor([0, 2, 0, 2, 1])]
    with torch.no_grad():
        out = mod(tw)
    assert torch.equal(out, ref)


def test_index_select_dedup_param_values():
    """The deduped member evals from the param registry, fp64-exact."""
    v, d = 5, 4
    w = _p("w", v, d)
    idx = (4, 0, 4, 1, 1, 0)
    term = Op.make("index_select", w, dim=0, index=idx)
    eg, root = _run(term)
    t = _extract_op(
        eg,
        root,
        lambda n_: (
            n_.op == "index_select"
            and _class_has_op(eg, n_.children[0], "index_select")
        ),
    )
    assert t is not None
    ir = IR(root=t, inputs=[], input_names=set(), params={"w": w})
    tw = torch.randn(v, d, dtype=torch.float64)
    mod = ir_to_torch_module(ir, param_values={"w": tw})
    with torch.no_grad():
        out = mod()
    assert torch.equal(out, tw[torch.tensor(list(idx))])


def test_check_dedup_index_guards():
    assert _check_dedup_index({"$attr:I": (0, 2, 0)})
    # repeats are required, >= 2 elements, all ints
    assert not _check_dedup_index({"$attr:I": 5})
    assert not _check_dedup_index({"$attr:I": (0,)})
    assert not _check_dedup_index({"$attr:I": (0, "x", 0)})
    assert not _check_dedup_index({"$attr:I": (True, True)})
    assert not _check_dedup_index({"$attr:I": (0, 1, 2)})
    assert not _check_dedup_index({})


def test_derive_dedup_index_map():
    got = _derive_dedup_index({"$attr:I": (7, 3, 7, 0, 3)})
    assert got == {"$attr:U": (7, 3, 0), "$attr:VI": (0, 1, 0, 2, 1)}


def test_index_select_dedup_declines_without_repeats():
    """A repeat-free index produces no nested member."""
    w = _p("w", 4, 3)
    term = Op.make("index_select", w, dim=0, index=(0, 2, 1))
    eg, root = _run(term)
    members = [
        n
        for n in eg.get_class(eg.find(root)).nodes
        if n.op == "index_select"
        and _class_has_op(eg, n.children[0], "index_select")
    ]
    assert not members


# ---------------------------------------------------------------------------
#  3. repeat_kv as gather — the copy diagonal as an index map
# ---------------------------------------------------------------------------


def test_repeat_kv_as_gather_fp64():
    """reshape(expand(unsqueeze(k,2),…),…) ≡ index_select(k, 1, i//r)."""
    B, h, r, t_, d = 2, 3, 4, 5, 6
    k = _v("k", B, h, t_, d)
    term = Op.make(
        "reshape",
        Op.make(
            "expand",
            Op.make("unsqueeze", k, dim=2),
            shape=(B, h, r, t_, d),
        ),
        shape=(B, h * r, t_, d),
    )
    eg, root = _run(term)
    assert "index_select" in _class_ops(eg, root)
    gath = _extract_op(eg, root, lambda n_: n_.op == "index_select")
    assert gath is not None
    mod = _module(gath, [k])
    tk = torch.randn(B, h, t_, d, dtype=torch.float64)
    ref = tk.repeat_interleave(r, dim=1)
    with torch.no_grad():
        out = mod(tk)
    assert torch.equal(out, ref)


def test_repeat_kv_as_gather_wrong_chain_declines():
    """An expand that grows a non-inserted dim is not a copy map."""
    B, h, r, t_, d = 2, 3, 4, 5, 6
    k = _v("k", B, h, t_, d)
    term = Op.make(
        "reshape",
        Op.make(
            "expand",
            Op.make("unsqueeze", k, dim=2),
            shape=(B, h * 2, r, t_, d),  # grows the head dim too
        ),
        shape=(B, h * r * 2, t_, d),
    )
    eg, root = _run(term)
    assert "index_select" not in _class_ops(eg, root)


def test_derive_repeat_gather_guards():
    k = _p("k", 1, 4, 2, 3)
    ok = {
        "k": k,
        "$attr:UDk": 3,
        "$attr:ESk": (1, 4, 2, 4, 3),
        "$attr:RSk": (1, 4, 8, 3),
    }
    assert _derive_repeat_gather(ok) == {
        "$attr:GD": 2,
        "$attr:GI": (0, 0, 0, 0, 1, 1, 1, 1),
    }
    # unresolvable base / non-int dim / non-tuple expand shape → None
    assert _derive_repeat_gather({**ok, "k": _ill_typed()}) is None
    assert _derive_repeat_gather({**ok, "$attr:UDk": "3"}) is None
    assert _derive_repeat_gather({**ok, "$attr:ESk": 5}) is None
    # unsqueeze at dim 0 has no merged axis → None
    assert _derive_repeat_gather({**ok, "$attr:UDk": 0}) is None
    # symbolic head extent / repeat factor → None
    bad_h = {
        "k": _p("k", 1, 4, None, 3),
        "$attr:UDk": 3,
        "$attr:ESk": (1, 4, None, 4, 3),
        "$attr:RSk": (1, 4, None, 3),
    }
    assert _derive_repeat_gather(bad_h) is None
    bad_r = {**ok, "$attr:ESk": (1, 4, 2, "r", 3)}
    assert _derive_repeat_gather(bad_r) is None


# ---------------------------------------------------------------------------
#  4. Identity gather
# ---------------------------------------------------------------------------


def test_index_select_id_elimination():
    """index_select(w, 0, (0,1,2)) on (3,x) w merges into w's class."""
    w = _p("w", 3, 4)
    term = Op.make("index_select", w, dim=0, index=(0, 1, 2))
    eg, root = _run(term)
    # the leaf member itself is now a member of the root class
    t = eg.extract_best(eg.find(root), flops_cost)
    assert t == w


def test_index_select_id_fp64_partial_index_stays():
    """(0,1) on a (3,x) table is not identity — nothing merges."""
    w = _p("w", 3, 4)
    term = Op.make("index_select", w, dim=0, index=(0, 1))
    eg, root = _run(term)
    t = eg.extract_best(eg.find(root), flops_cost)
    assert t != w
    mod = ir_to_torch_module(
        IR(
            root=t,
            inputs=[],
            input_names=set(),
            params={"w": w},
        ),
        param_values={
            "w": (tw := torch.randn(3, 4, dtype=torch.float64))
        },
    )
    with torch.no_grad():
        assert torch.equal(mod(), tw[:2])


def test_check_identity_index_guards():
    ok = {"t": _p("t", 3, 4), "$attr:D": 0, "$attr:I": (0, 1, 2)}
    assert _check_identity_index(ok)
    # permuted / partial / non-seq / missing attrs → veto
    assert not _check_identity_index({**ok, "$attr:I": (0, 2, 1)})
    assert not _check_identity_index({**ok, "$attr:I": (0, 1)})
    assert not _check_identity_index({**ok, "$attr:I": 3})
    assert not _check_identity_index({**ok, "$attr:D": "0"})
    # unresolvable / scalar t → veto
    assert not _check_identity_index({**ok, "t": _ill_typed()})
    assert not _check_identity_index({**ok, "t": _p("t")})
    # unknown extent on the gathered axis → veto
    assert not _check_identity_index({**ok, "t": _p("t", None, 4)})
    # wrong axis: identity on dim1 needs (0,1,2,3)
    assert not _check_identity_index({**ok, "$attr:D": 1})
    assert _check_identity_index(
        {**ok, "$attr:D": 1, "$attr:I": (0, 1, 2, 3)}
    )


# ---------------------------------------------------------------------------
#  5. Composition + documented scope limits
# ---------------------------------------------------------------------------


def test_repeat_chain_dedup_compose():
    """Δ_r → gather → dedup → inner identity: the composition lands.

    repeat_kv(k, r=2) becomes index_select(k, 1, (0,0,1,1)); dedup
    refactors it into index_select(index_select(k,1,(0,1)),1,(0,0,1,1))
    whose inner gather is the identity — so the class also holds
    index_select(k,1,(0,0,1,1)) merged through the identity law.
    """
    B, h, r, t_, d = 1, 2, 2, 3, 4
    k = _v("k", B, h, t_, d)
    term = Op.make(
        "reshape",
        Op.make(
            "expand",
            Op.make("unsqueeze", k, dim=2),
            shape=(B, h, r, t_, d),
        ),
        shape=(B, h * r, t_, d),
    )
    eg, root = _run(term)
    ops = _class_ops(eg, root)
    assert "index_select" in ops
    # dedup fired on the derived gather: a nested index_select exists
    nested = [
        n
        for n in eg.get_class(eg.find(root)).nodes
        if n.op == "index_select"
        and _class_has_op(eg, n.children[0], "index_select")
    ]
    assert nested
    # and the whole class is still fp64-exact under either member
    gath = _extract_op(eg, root, lambda n_: n_.op == "index_select")
    mod = _module(gath, [k])
    tk = torch.randn(B, h, t_, d, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tk)
    assert torch.equal(out, tk.repeat_interleave(r, dim=1))


def test_op_table_scope_limits():
    """The documented limits: the op table has no unique/index_copy,
    and the scatter family + index_select ARE the write/gather seam."""
    import catopt_torch.torch_bridge  # noqa: F401 — registers tables
    from catopt_core.ops import OpTable

    names = set(OpTable.full().torch_bindings)
    for present in (
        "slice_scatter",
        "select_scatter",
        "index_put",
        "index_select",
        "gather",
        "embedding",
        "concat",
        "narrow",
        "slice",
    ):
        assert present in names
    for absent in ("unique", "repeat_interleave", "index_copy"):
        assert absent not in names
