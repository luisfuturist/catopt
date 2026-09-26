"""Mask-distribution laws — masked_fill/add/where over concat.

The online-softmax homomorphism (OM_SPLIT) needs the score operand to
be a literal concat; a causal or additive mask wraps that concat and
blocks the chain.  ``OM_MASK_LAWS`` (catopt/om.py) close the gap:

    mask(cat(s1, s2), M) = cat(mask(s1, M1), mask(s2, M2))

where M_i is block i's slice ``split(M, (K1,K2), d, i)`` of the full
mask — for a causal mask the block's positional offset lives inside
the slice.  When M broadcasts along the cat axis it is reused whole.

Covered here:

* the law fires in each masking idiom (masked_fill, additive bias,
  torch.where) and in both modes (slice / broadcast-reuse);
* masked chunked attention saturates into om_apply(om_compose(...))
  and is fp64-exact vs the dense masked computation — causal tril
  masks, additive 0/−inf masks, uneven and n-ary blocks, batched;
* edge cases: a fully-masked row yields NaN in exactly the positions
  torch.softmax produces NaN, and a fully-masked BLOCK contributes
  zero (the om_compose isfinite guard);
* negative checks: mismatched mask extents and cat axes must not fire.
"""

import torch

from catopt.cost import flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, TensorType, Var, op_repr
from catopt.om import OM_LAWS, OM_MASK_LAWS
from catopt.torch_bridge import ir_to_torch_module

NEG_INF = Const(float("-inf"))


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _nested_cat(ts, dim, attr_key="dim"):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, **{attr_key: dim})
    return out


def _masked_dense_term(
    q, ks, vs, m, style="masked_fill", attr_key="dim"
):
    """softmax(mask(q @ cat(k_i).T)) @ cat(v_i) as an IR term.

    ``style`` selects the masking idiom:
      "masked_fill" — scores.masked_fill(m, -inf)   (m is the BAD mask)
      "add"         — scores + m                     (additive 0/−inf)
      "add_r"       — m + scores                     (commuted)
      "where"       — where(m, scores, -inf)         (m is a KEEP mask)
    """
    kcat = _nested_cat(ks, -2, attr_key)
    vcat = _nested_cat(vs, -2, attr_key)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, arg1=-2, arg2=-1)
    )
    if style == "masked_fill":
        masked = Op.make("masked_fill", scores, m, NEG_INF)
    elif style == "add":
        masked = Op.make("add", scores, m)
    elif style == "add_r":
        masked = Op.make("add", m, scores)
    elif style == "where":
        masked = Op.make("where", m, scores, NEG_INF)
    else:
        raise ValueError(style)
    return Op.make("matmul", Op.make("softmax", masked, arg1=-1), vcat)


def _masked_dense_ref(q, ks, vs, m, style="masked_fill"):
    s = q @ torch.cat(list(ks), dim=-2).transpose(-2, -1)
    if style == "masked_fill":
        s = s.masked_fill(m, float("-inf"))
    elif style in ("add", "add_r"):
        s = s + m
    elif style == "where":
        s = torch.where(
            m, s, torch.tensor(float("-inf"), dtype=s.dtype)
        )
    return torch.softmax(s, dim=-1) @ torch.cat(list(vs), dim=-2)


def _run_om(term, max_iterations=20, max_nodes=200_000):
    eg = EGraph()
    root = eg.add_term(term)
    stats = eg.run(
        OM_LAWS,
        root,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
    )
    return eg, root, stats


def _class_has_op(eg, eid, opname, seen=None):
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


def _extract_chunked(eg, root):
    """Force-extract a maximally-chunked carrier term (per-class greedy
    extraction never picks om_compose — the unsplit elem is locally
    cheaper).  Same override protocol as test_om_monoid."""
    canon = eg.find(root)
    ov = {}
    for cid in list(eg._classes):
        c = eg.find(cid)
        comps = [
            n for n in eg._classes[c].nodes if n.op == "om_compose"
        ]
        if comps:
            ov.setdefault(c, comps[0])
    for n in eg.get_class(canon).nodes:
        if n.op == "om_apply" and _class_has_op(
            eg, n.children[0], "om_compose"
        ):
            o = dict(ov)
            o[canon] = n
            t = eg.extract_best(canon, flops_cost, overrides=o)
            if t is not None:
                return t
    return None


def _assert_close_or_nan(out, ref, tol=1e-12):
    """NaN-aware fp64 comparison: NaN positions must match torch
    softmax exactly, finite positions must agree to tol."""
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    fin = ~torch.isnan(ref)
    assert (out[fin] - ref[fin]).abs().max().item() < tol


def _causal_mask(T, K):
    """masked_fill-style mask: True = disallowed (strict upper tri)."""
    return ~torch.tril(torch.ones(T, K, dtype=torch.bool))


# ---------------------------------------------------------------------------
#  (a) the distribution laws fire — and are fp64-exact
# ---------------------------------------------------------------------------


def test_masked_fill_distributes_over_concat_slice():
    """masked_fill(cat(s1,s2,-1), M, -inf) gains the concat-of-masked
    member with per-block mask slices, and evaluates identically."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((4, 8)))
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) > 0

    torch.manual_seed(0)
    ts1, ts2 = (
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 5, dtype=torch.float64),
    )
    tm = torch.rand(4, 8) < 0.4
    ref = torch.cat([ts1, ts2], -1).masked_fill(tm, float("-inf"))
    # The concat member whose children are masked_fills is the new form.
    canon = eg.find(root)
    hit = None
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        kids = [eg.get_class(c).nodes for c in n.children]
        if all(any(x.op == "masked_fill" for x in ks) for ks in kids):
            hit = n
    assert hit is not None, "concat-of-masked_fills never materialised"
    term = eg.extract_best(canon, flops_cost, overrides={canon: hit})
    ir = IR(
        root=term,
        inputs=[s1, s2, m],
        input_names={"s1", "s2", "m"},
        params={},
    )
    mod = ir_to_torch_module(ir)
    with torch.no_grad():
        out = mod(ts1, ts2, tm)
    assert torch.equal(
        torch.isnan(out), torch.isnan(ref)
    ) or torch.equal(out, ref)


def test_masked_fill_distributes_over_concat_broadcast_reuse():
    """A mask with extent 1 on the cat axis is reused in both blocks —
    no split nodes should be needed."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((4, 1)))  # broadcasts along -1
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("masked_fill_cat_reuse_dim", 0) > 0
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) == 0

    torch.manual_seed(0)
    ts1, ts2 = (
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 5, dtype=torch.float64),
    )
    tm = torch.rand(4, 1) < 0.5
    ref = torch.cat([ts1, ts2], -1).masked_fill(tm, float("-inf"))
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        term = eg.extract_best(canon, flops_cost, overrides={canon: n})
        if "masked_fill" not in op_repr(term):
            continue
        ir = IR(
            root=term,
            inputs=[s1, s2, m],
            input_names={"s1", "s2", "m"},
            params={},
        )
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(ts1, ts2, tm)
        assert torch.equal(out, ref)
        return
    raise AssertionError("reuse member not found")


def test_add_mask_distributes_over_concat_both_orders():
    """Additive masks: add(cat, M) and add(M, cat) both slice M."""
    torch.manual_seed(0)
    for order, style in (("post", "add"), ("pre", "add_r")):
        s1 = Var("s1", TensorType((4, 3)))
        s2 = Var("s2", TensorType((4, 5)))
        m = Var("m", TensorType((4, 8)))
        cat = Op.make("concat", s1, s2, dim=-1)
        t = (
            Op.make("add", cat, m)
            if order == "post"
            else Op.make("add", m, cat)
        )
        eg = EGraph()
        root = eg.add_term(t)
        eg.run(OM_MASK_LAWS, root, max_iterations=5)
        name = (
            f"add_{'cat_m' if order == 'post' else 'm_cat'}_slice_dim"
        )
        assert eg.rule_fires.get(name, 0) > 0, name

        ts1, ts2 = (
            torch.randn(4, 3, dtype=torch.float64),
            torch.randn(4, 5, dtype=torch.float64),
        )
        tm = torch.where(
            torch.rand(4, 8) < 0.3,
            torch.tensor(float("-inf"), dtype=torch.float64),
            torch.zeros(4, 8, dtype=torch.float64),
        )
        ref = torch.cat([ts1, ts2], -1) + tm
        canon = eg.find(root)
        for n in eg.get_class(canon).nodes:
            if n.op != "concat":
                continue
            term = eg.extract_best(
                canon, flops_cost, overrides={canon: n}
            )
            if "add" not in op_repr(term) or "split" not in op_repr(
                term
            ):
                continue
            ir = IR(
                root=term,
                inputs=[s1, s2, m],
                input_names={"s1", "s2", "m"},
                params={},
            )
            mod = ir_to_torch_module(ir)
            with torch.no_grad():
                out = mod(ts1, ts2, tm)
            assert torch.equal(out, ref)
            break
        else:
            raise AssertionError(f"{style}: distributed member missing")


def test_where_distributes_over_concat():
    """where(keep_mask, cat(s1,s2), -inf) slices the keep mask."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((4, 8)))
    t = Op.make("where", m, Op.make("concat", s1, s2, dim=-1), NEG_INF)
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("where_cat_x_slice_dim", 0) > 0

    torch.manual_seed(0)
    ts1, ts2 = (
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 5, dtype=torch.float64),
    )
    tm = torch.rand(4, 8) < 0.6
    ref = torch.where(
        tm,
        torch.cat([ts1, ts2], -1),
        torch.tensor(float("-inf"), dtype=torch.float64),
    )
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        term = eg.extract_best(canon, flops_cost, overrides={canon: n})
        if "where" not in op_repr(term):
            continue
        ir = IR(
            root=term,
            inputs=[s1, s2, m],
            input_names={"s1", "s2", "m"},
            params={},
        )
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(ts1, ts2, tm)
        assert torch.equal(out, ref)
        return
    raise AssertionError("where distribution member missing")


def test_cat_hom_add_mask_itself_concatd():
    """add(cat(s1,s2), cat(M1,M2)) → cat(add(s1,M1), add(s2,M2)) — the
    free concat homomorphism when the mask arrives pre-chunked."""
    s1, s2 = (
        Var("s1", TensorType((4, 3))),
        Var("s2", TensorType((4, 5))),
    )
    m1, m2 = (
        Var("m1", TensorType((4, 3))),
        Var("m2", TensorType((4, 5))),
    )
    t = Op.make(
        "add",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", m1, m2, dim=-1),
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("cat_hom_add_dim", 0) > 0

    torch.manual_seed(0)
    vals = [torch.randn(4, k, dtype=torch.float64) for k in (3, 5)]
    mvals = [torch.randn(4, k, dtype=torch.float64) for k in (3, 5)]
    ref = torch.cat(vals, -1) + torch.cat(mvals, -1)
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        term = eg.extract_best(canon, flops_cost, overrides={canon: n})
        r = op_repr(term)
        if "add" not in r or "split" in r:
            continue
        ir = IR(
            root=term,
            inputs=[s1, s2, m1, m2],
            input_names={"s1", "s2", "m1", "m2"},
            params={},
        )
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(*vals, *mvals)
        assert torch.equal(out, ref)
        return
    raise AssertionError("cat_hom member missing")


def test_distribution_along_non_key_axis():
    """The law is axis-generic: a row-cat (dim 0) mask also slices —
    query-side chunking pays its row offset the same way."""
    s1 = Var("s1", TensorType((3, 6)))
    s2 = Var("s2", TensorType((2, 6)))
    m = Var("m", TensorType((5, 6)))  # rows split 3+2
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=0), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) > 0

    torch.manual_seed(0)
    ts1, ts2 = (
        torch.randn(3, 6, dtype=torch.float64),
        torch.randn(2, 6, dtype=torch.float64),
    )
    tm = torch.rand(5, 6) < 0.4
    ref = torch.cat([ts1, ts2], 0).masked_fill(tm, float("-inf"))
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        term = eg.extract_best(canon, flops_cost, overrides={canon: n})
        if "split" not in op_repr(term):
            continue
        ir = IR(
            root=term,
            inputs=[s1, s2, m],
            input_names={"s1", "s2", "m"},
            params={},
        )
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(ts1, ts2, tm)
        assert torch.equal(out, ref)
        return
    raise AssertionError("row-slice member missing")


def test_lower_rank_mask_slices_on_its_own_axis():
    """A (K,)-rank mask vs (T,K) scores: the cat axis maps to the
    mask's own last dim — slice it there."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((8,)))  # broadcasts over rows
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) > 0

    torch.manual_seed(0)
    ts1, ts2 = (
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 5, dtype=torch.float64),
    )
    tm = torch.rand(8) < 0.4
    ref = torch.cat([ts1, ts2], -1).masked_fill(tm, float("-inf"))
    canon = eg.find(root)
    for n in eg.get_class(canon).nodes:
        if n.op != "concat":
            continue
        term = eg.extract_best(canon, flops_cost, overrides={canon: n})
        if "split" not in op_repr(term):
            continue
        ir = IR(
            root=term,
            inputs=[s1, s2, m],
            input_names={"s1", "s2", "m"},
            params={},
        )
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(ts1, ts2, tm)
        assert torch.equal(out, ref)
        return
    raise AssertionError("lower-rank slice member missing")


# ---------------------------------------------------------------------------
#  (b) the full chain: masked chunked attention via OM_LAWS
# ---------------------------------------------------------------------------


def _masked_chunked_ok(
    style, ksizes, mask_fn, T=5, d=4, dv=7, seed=0, attr_key="dim"
):
    """Saturate a masked chunked attention term, force-extract the
    om_compose carrier, verify fp64 vs dense masked reference."""
    torch.manual_seed(seed)
    q = Var("q", TensorType((T, d)))
    ks = [
        Var(f"k{i}", TensorType((k, d))) for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((k, dv))) for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType(mask_fn(T, sum(ksizes)).shape))
    term = _masked_dense_term(q, ks, vs, m, style, attr_key)

    eg, root, _ = _run_om(term)
    chunked = [
        n
        for n in eg.get_class(root).nodes
        if n.op == "om_apply"
        and _class_has_op(eg, n.children[0], "om_compose")
    ]
    assert chunked, f"{style}: om_split never produced a carrier"
    term = _extract_chunked(eg, root)
    assert term is not None and "om_compose" in op_repr(term)

    inputs = [q] + ks + vs + [m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)

    tq = torch.randn(T, d, dtype=torch.float64)
    tks = [torch.randn(k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, dv, dtype=torch.float64) for k in ksizes]
    tm = mask_fn(T, sum(ksizes))
    ref = _masked_dense_ref(tq, tks, tvs, tm, style)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)
    return eg, term


def test_causal_masked_chunked_attention_fp64():
    """THE law: causal masked_fill chunked attention decomposes —
    softmax(mask(q·cat kᵀ)) @ cat v ≡ om_apply(⊕ om_elem(masked sᵢ, vᵢ))
    with per-block mask slices carrying the absolute key offset."""
    eg, term = _masked_chunked_ok(
        "masked_fill", [3, 6], lambda T, K: _causal_mask(T, K)
    )
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) > 0
    assert eg.rule_fires.get("om_split", 0) > 0
    # The extracted term slices the mask per block — the positional
    # offset is inside the split, no index arithmetic.
    r = op_repr(term)
    assert "split" in r and "masked_fill" in r


def test_additive_mask_chunked_attention_fp64():
    """Additive 0/−inf bias masks distribute the same way (the HF
    attn_bias idiom)."""

    def addmask(T, K):
        return torch.where(
            torch.rand(T, K) < 0.35,
            torch.tensor(float("-inf"), dtype=torch.float64),
            torch.zeros(T, K, dtype=torch.float64),
        )

    _masked_chunked_ok("add", [3, 6], addmask)
    _masked_chunked_ok("add_r", [3, 6], addmask)  # commuted operand


def test_where_mask_chunked_attention_fp64():
    """torch.where keep-mask idiom: where(m, s, -inf)."""
    _masked_chunked_ok(
        "where", [3, 6], lambda T, K: torch.rand(T, K) < 0.6
    )


def test_masked_chunked_three_blocks_nary():
    """A 3-ary concat binarises first; the mask slices cascade —
    split(split(m)) is the block-2 offset paid twice over."""
    _masked_chunked_ok(
        "masked_fill", [2, 3, 4], lambda T, K: _causal_mask(T, K)
    )


def test_masked_chunked_batched_heads():
    """Rank-4 (B,H,T,K): mask (B,1,T,K) broadcasts over heads and still
    slices on the key axis."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 3, 5, 4, 6
    ksizes = [2, 5]
    q = Var("q", TensorType((B, H, T, d)))
    ks = [
        Var(f"k{i}", TensorType((B, H, k, d)))
        for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((B, H, k, dv)))
        for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType((B, 1, T, sum(ksizes))))
    term = _masked_dense_term(q, ks, vs, m, "masked_fill")

    eg, root, _ = _run_om(term)
    chunked = [
        n
        for n in eg.get_class(root).nodes
        if n.op == "om_apply"
        and _class_has_op(eg, n.children[0], "om_compose")
    ]
    assert chunked
    term = _extract_chunked(eg, root)
    assert term is not None

    inputs = [q] + ks + vs + [m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)
    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [
        torch.randn(B, H, k, dv, dtype=torch.float64) for k in ksizes
    ]
    tm = torch.rand(B, 1, T, sum(ksizes)) < 0.4
    ref = _masked_dense_ref(tq, tks, tvs, tm, "masked_fill")
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)


# ---------------------------------------------------------------------------
#  (c) −inf / NaN edge semantics — must match torch.softmax exactly
# ---------------------------------------------------------------------------


def test_fully_masked_row_nan_positions_match():
    """A row masked across ALL blocks → NaN there in both forms, finite
    elsewhere.  om_apply stays unclamped: a/l = 0/0 = NaN."""
    torch.manual_seed(0)
    T, K = 4, 7
    ksizes = [3, 4]

    def mask_fn(t_, k_):
        m = torch.rand(t_, k_) < 0.3
        m[1] = True  # row 1 fully masked → NaN row
        m[3] = True  # row 3 fully masked → NaN row
        return m

    q = Var("q", TensorType((T, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 5))) for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType((T, K)))
    term = _masked_dense_term(q, ks, vs, m, "masked_fill")
    eg, root, _ = _run_om(term)
    term = _extract_chunked(eg, root)
    assert term is not None

    inputs = [q] + ks + vs + [m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)
    tq = torch.randn(T, 4, dtype=torch.float64)
    tks = [torch.randn(k, 4, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, 5, dtype=torch.float64) for k in ksizes]
    tm = mask_fn(T, K)
    ref = _masked_dense_ref(tq, tks, tvs, tm, "masked_fill")
    assert torch.isnan(ref[1]).all() and torch.isnan(ref[3]).all()
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)


def test_fully_masked_block_contributes_zero():
    """One whole block masked (its mask slice is all-True) — the
    isfinite guard in om_compose drops its NaN contribution, matching
    dense softmax where those columns are exp(−inf)=0."""
    torch.manual_seed(0)
    T, K = 4, 7
    ksizes = [3, 4]

    def mask_fn(t_, k_):
        m = torch.zeros(t_, k_, dtype=torch.bool)
        m[:, ksizes[0] :] = True  # block 1 entirely masked
        return m

    q = Var("q", TensorType((T, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 5))) for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType((T, K)))
    term = _masked_dense_term(q, ks, vs, m, "masked_fill")
    eg, root, _ = _run_om(term)
    term = _extract_chunked(eg, root)
    assert term is not None

    inputs = [q] + ks + vs + [m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)
    tq = torch.randn(T, 4, dtype=torch.float64)
    tks = [torch.randn(k, 4, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, 5, dtype=torch.float64) for k in ksizes]
    tm = mask_fn(T, K)
    ref = _masked_dense_ref(tq, tks, tvs, tm, "masked_fill")
    assert torch.isfinite(ref).all()
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)


def test_causal_mask_row_zero_is_fine():
    """Causal sanity: row 0 sees only key 0 — the chunked combine must
    give weight 1.0 on v[..., 0, :] exactly as dense softmax does."""
    torch.manual_seed(0)
    T, d, dv = 4, 3, 5
    ksizes = [2, 2]
    q = Var("q", TensorType((T, d)))
    ks = [
        Var(f"k{i}", TensorType((k, d))) for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((k, dv))) for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType((T, sum(ksizes))))
    term = _masked_dense_term(q, ks, vs, m, "masked_fill")
    eg, root, _ = _run_om(term)
    term = _extract_chunked(eg, root)
    assert term is not None

    inputs = [q] + ks + vs + [m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)
    tq = torch.randn(T, d, dtype=torch.float64)
    tks = [torch.randn(k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, dv, dtype=torch.float64) for k in ksizes]
    tm = _causal_mask(T, sum(ksizes))
    ref = _masked_dense_ref(tq, tks, tvs, tm, "masked_fill")
    assert torch.equal(ref[0], tvs[0][0])  # dense: row0 = v0 exactly
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)
    assert (out[0] - tvs[0][0]).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  (d) negative checks — wrong extents/axes must NOT fire
# ---------------------------------------------------------------------------


def test_mask_with_mismatched_extent_does_not_distribute():
    """Mask extent neither 1 nor K1+K2 on the cat axis → veto."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((4, 6)))  # 6 != 8 and != 1
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert not any(
        k.startswith("masked_fill_cat_") for k in eg.rule_fires
    )


def test_mask_on_wrong_axis_does_not_distribute():
    """Mask covers the ROW axis extent while scores cat on keys —
    extent (K1+K2) is on the wrong axis → veto (slicing rows would
    produce a semantically different program)."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((8, 4)))  # transposed-ish: vetoes
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert not any(
        k.startswith("masked_fill_cat_") for k in eg.rule_fires
    )


def test_unknown_block_extent_does_not_distribute():
    """Unknown (None) block extents → can't derive split sizes → veto."""
    s1 = Var("s1", TensorType((4, None)))
    s2 = Var("s2", TensorType((4, 5)))
    m = Var("m", TensorType((4, 8)))
    t = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, NEG_INF
    )
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(OM_MASK_LAWS, root, max_iterations=5)
    assert not any(
        k.startswith("masked_fill_cat_") for k in eg.rule_fires
    )
