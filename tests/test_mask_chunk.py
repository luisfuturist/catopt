# ruff: noqa: RUF002, RUF003
"""sdpa with an EXPLICIT attn_mask over concatenated keys/values.

SDPA_CAT_LAWS (catopt/om.py) already chunked the is_causal flag and the
unmasked form; the remaining gap was the 4th-operand mask:

    sdpa(q, cat(k_i), cat(v_i), m, ...)
      = om_apply(om_elem(
            add(cat(qs @ k_i.T), attnbias(m)), cat(v_i)))

``attnbias`` materialises torch's own mask coercion — a float mask IS
the additive bias, a bool keep-mask becomes where(m, 0, -inf) — so ONE
law covers both dtypes soundly (the term level carries no dtype).  The
existing ADD_MASK_CAT laws then slice the bias on the key axis
(``split(attnbias(m), (K1,K2), -1, i)`` is block i's mask columns) and
OM_SPLIT splits the carrier — the same machinery as the materialised
mask laws, no new vocabulary beyond the coercion op.

Covered here:

* the law fires on the spellings torch.export emits — bare 4-operand
  ``sdpa(q, cat k, cat v, m)`` and positional ``arg4``/``arg5``/``arg6``
  — plus the kwarg forms, in the canonical concat attr spelling;
* fp64-exact equivalence with ``F.scaled_dot_product_attention`` for
  BOTH mask dtypes — additive float (finite bias and 0/−inf) and bool
  keep-mask — including the ``logical_not(mk)`` operand shape that
  sdpa_fold produces;
* even/uneven and n-ary blocks, batched heads, broadcast masks
  (lower-rank, extent-1 key axis), custom scale;
* the one carrier/kernel divergence, asserted precisely: a
  fully-masked ROW yields NaN (om_apply is deliberately unclamped —
  dense-softmax semantics) where torch's sdpa kernel emits 0, while a
  fully-masked BLOCK still contributes zero via om_compose's isfinite
  guard and matches exactly;
* negative checks: mismatched mask extents, mask extent on the wrong
  axis, dropout_p != 0, and is_causal=True alongside a mask all veto.
"""

import torch
import torch.nn.functional as F

from catopt.cost import flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, TensorType, Var, op_repr
from catopt.om import OM_LAWS
from catopt.torch_bridge import ir_to_torch_module

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _nested_cat(ts, dim, attr_key="dim"):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, **{attr_key: dim})
    return out


def _sdpa_mask_term(
    q, ks, vs, m, attr_key="dim", spelling="export", scale=None
):
    """sdpa(q, cat(k_i), cat(v_i), m) as an IR term.

    ``spelling`` selects the flag encoding:
      "export" — the bare 4-operand form torch.export emits for
                 defaulted flags;
      "pos"    — positional attrs: arg4=dropout_p, arg5=is_causal,
                 arg6=scale;
      "kwarg"  — is_causal=False / scale= named attrs.
    """
    kcat = _nested_cat(ks, -2, attr_key)
    vcat = _nested_cat(vs, -2, attr_key)
    attrs = {}
    if spelling == "pos":
        attrs["arg4"] = 0.0
        attrs["arg5"] = False
        if scale is not None:
            attrs["arg6"] = float(scale)
    elif spelling == "kwarg":
        attrs["is_causal"] = False
        if scale is not None:
            attrs["scale"] = float(scale)
    else:
        if scale is not None:
            attrs["scale"] = float(scale)
    return Op.make("sdpa", q, kcat, vcat, m, **attrs)


def _sdpa_ref(q, ks, vs, m, scale=None):
    kw = {}
    if scale is not None:
        kw["scale"] = scale
    return F.scaled_dot_product_attention(
        q,
        torch.cat(list(ks), dim=-2),
        torch.cat(list(vs), dim=-2),
        attn_mask=m,
        **kw,
    )


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
    cheaper).  Same override protocol as test_om_mask/test_om_causal."""
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
    """NaN-aware fp64 comparison, with the one documented carrier/kernel
    divergence: for a FULLY-masked row torch's sdpa kernel emits 0 while
    the om carrier reproduces dense-softmax semantics (a/l = 0/0 = NaN —
    om_apply is deliberately unclamped).  So: NaN positions of ``out``
    must be exactly the positions where ``ref`` is 0, and everywhere
    ``out`` is finite it must match ``ref`` to tol."""
    nan = torch.isnan(out)
    assert not torch.isnan(ref).any()  # sdpa never NaNs
    assert torch.equal(ref[nan], torch.zeros_like(ref[nan]))
    fin = ~nan
    assert (out[fin] - ref[fin]).abs().max().item() < tol


def _mask_chunked_ok(
    ksizes,
    mask_fn,
    T=5,
    d=4,
    dv=7,
    seed=0,
    attr_key="dim",
    spelling="export",
    scale=None,
    batch=None,
    mask_term=None,
):
    """Saturate an sdpa-over-concats term carrying an explicit
    attn_mask, force-extract the chunked carrier, verify fp64 vs
    F.scaled_dot_product_attention.

    ``mask_fn(T, K)`` builds the mask TENSOR (its shape fixes the mask
    Var's type).  ``mask_term`` optionally wraps the mask operand in
    the IR term (e.g. logical_not for the fold-produced form); the
    reference then applies the same wrap to the tensor.
    """
    torch.manual_seed(seed)
    qshape = (*batch, T, d) if batch else (T, d)

    def kshape(k):
        return (*batch, k, d) if batch else (k, d)

    def vshape(k):
        return (*batch, k, dv) if batch else (k, dv)
    tm_probe = mask_fn(T, sum(ksizes))
    q = Var("q", TensorType(qshape))
    ks = [
        Var(f"k{i}", TensorType(kshape(k)))
        for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType(vshape(k)))
        for i, k in enumerate(ksizes)
    ]
    m = Var("m", TensorType(tuple(tm_probe.shape)))
    mterm = mask_term(m) if mask_term is not None else m
    term = _sdpa_mask_term(
        q,
        ks,
        vs,
        mterm,
        attr_key=attr_key,
        spelling=spelling,
        scale=scale,
    )

    eg, root, _ = _run_om(term)
    chunked = [
        n
        for n in eg.get_class(root).nodes
        if n.op == "om_apply"
        and _class_has_op(eg, n.children[0], "om_compose")
    ]
    assert chunked, (
        "masked sdpa never decomposed into a chunked carrier"
    )
    term = _extract_chunked(eg, root)
    assert term is not None and "om_compose" in op_repr(term)

    inputs = [q, *ks, *vs, m]
    ir = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    mod = ir_to_torch_module(ir)

    tq = torch.randn(*qshape, dtype=torch.float64)
    tks = [torch.randn(*kshape(k), dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(*vshape(k), dtype=torch.float64) for k in ksizes]
    tm = tm_probe
    ref_mask = torch.logical_not(tm) if mask_term is not None else tm
    ref = _sdpa_ref(tq, tks, tvs, ref_mask, scale=scale)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs, tm)
    _assert_close_or_nan(out, ref)
    return eg, term, out, ref, tm


# ---------------------------------------------------------------------------
#  (a) the law fires — every spelling export/hand-built terms produce
# ---------------------------------------------------------------------------


def _fires_mask_law(spelling="export", attr_key="dim", scale=None):
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))
    ]
    m = Var("m", TensorType((5, 9)))
    term = _sdpa_mask_term(
        q, ks, vs, m, attr_key=attr_key, spelling=spelling, scale=scale
    )
    eg, _root, _ = _run_om(term)
    return eg


def test_mask_law_fires_export_spelling():
    """sdpa(q, cat k, cat v, m) — the bare 4-operand form torch.export
    emits — gains the om_apply member."""
    eg = _fires_mask_law()
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires), (
        dict(eg.rule_fires)
    )


def test_mask_law_fires_positional_attrs():
    """arg4=dropout_p, arg5=is_causal=False — the positional flag form."""
    eg = _fires_mask_law(spelling="pos")
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)


def test_mask_law_fires_kwarg_spelling():
    """The hand-built is_causal=False kwarg form fires too."""
    eg = _fires_mask_law(spelling="kwarg")
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)


def test_mask_law_fires_dim_concat_spelling():
    """Concat dims spelled with the canonical ``dim`` match."""
    eg = _fires_mask_law(attr_key="dim")
    assert any(
        k.startswith("sdpa_cat_mask_") and k.endswith("dim")
        for k in eg.rule_fires
    )


def test_mask_law_fires_with_scale():
    """A non-default scale lands as scale= / arg6= and still fires."""
    eg = _fires_mask_law(scale=0.5)
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)
    eg = _fires_mask_law(spelling="pos", scale=0.5)
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)


# ---------------------------------------------------------------------------
#  (b) fp64 equivalence — additive (float) masks
# ---------------------------------------------------------------------------


def _addmask_fn(p=0.35):
    """Additive 0/−inf bias mask — the HF attn_bias idiom."""

    def fn(T, K):
        return torch.where(
            torch.rand(T, K) < p,
            torch.tensor(float("-inf"), dtype=torch.float64),
            torch.zeros(T, K, dtype=torch.float64),
        )

    return fn


def _finite_bias_fn(T, K):
    """A fully-finite additive mask — arbitrary per-position bias."""
    return torch.randn(T, K, dtype=torch.float64)


def test_additive_mask_chunked_fp64():
    """THE law: sdpa(q, cat k, cat v, float_mask) ≡ om tree, fp64.
    The mask slices per block — block i's columns are the split."""
    eg, term, _out, _ref, _ = _mask_chunked_ok([3, 6], _addmask_fn())
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)
    assert eg.rule_fires.get("add_cat_m_slice_dim", 0) > 0
    assert eg.rule_fires.get("om_split", 0) > 0
    r = op_repr(term)
    assert "split" in r and "attnbias" in r and "add" in r


def test_additive_finite_bias_chunked_fp64():
    """Not just 0/−inf: an arbitrary float bias chunks identically."""
    _mask_chunked_ok([3, 6], _finite_bias_fn)


def test_additive_mask_positional_attrs_fp64():
    _mask_chunked_ok([3, 6], _addmask_fn(), spelling="pos")


def test_additive_mask_kwarg_fp64():
    _mask_chunked_ok([3, 6], _addmask_fn(), spelling="kwarg")


def test_additive_mask_uneven_three_blocks():
    """n-ary concat: mask slices cascade — split(split(attnbias(m)))
    carries block 2's column offset [o1+o2, K)."""
    _mask_chunked_ok([2, 3, 4], _addmask_fn())


def test_additive_mask_uneven_two_blocks():
    """Very uneven blocks — block 0 is a single key."""
    _mask_chunked_ok([1, 8], _addmask_fn())


def test_additive_mask_custom_scale():
    """An explicit scale attr rides through fill(q, value=SC); the
    bias adds to the SCALED scores (torch applies the mask after
    scaling)."""
    _mask_chunked_ok([3, 6], _addmask_fn(), spelling="kwarg", scale=0.5)
    _mask_chunked_ok([3, 6], _addmask_fn(), spelling="pos", scale=0.25)


def test_additive_mask_batched_heads():
    """Rank-4 (B,H,T,d) with a (B,1,T,K) mask broadcasting over heads —
    the mask still slices on the key axis."""
    B, H = 2, 3

    def mask_fn(T, K):
        return torch.where(
            torch.rand(B, 1, T, K) < 0.35,
            torch.tensor(float("-inf"), dtype=torch.float64),
            torch.zeros(B, 1, T, K, dtype=torch.float64),
        )

    _mask_chunked_ok([2, 5], mask_fn, batch=(B, H))


# ---------------------------------------------------------------------------
#  (c) fp64 equivalence — bool keep-masks
# ---------------------------------------------------------------------------


def _boolmask_fn(T, K):
    """Bool keep-mask: True = attend (torch's attn_mask convention)."""
    m = torch.rand(T, K) < 0.6
    m[:, 0] = True  # keep row 0-key so rows stay live
    return m


def test_bool_mask_chunked_fp64():
    """attnbias turns the keep-mask into a 0/−inf bias — the same law,
    the same add-form slice, fp64-exact vs sdpa's bool path."""
    eg, term, _out, _ref, _ = _mask_chunked_ok([3, 6], _boolmask_fn)
    assert any(k.startswith("sdpa_cat_mask_") for k in eg.rule_fires)
    assert "split" in op_repr(term) and "attnbias" in op_repr(term)


def test_bool_mask_uneven_three_blocks():
    _mask_chunked_ok([2, 3, 4], _boolmask_fn)


def test_bool_mask_batched():
    B, H = 2, 3

    def mask_fn(T, K):
        m = torch.rand(B, 1, T, K) < 0.6
        m[..., 0] = True
        return m

    _mask_chunked_ok([2, 5], mask_fn, batch=(B, H))


def test_logical_not_mask_operand_fp64():
    """The mask shape sdpa_fold produces: attn_mask = logical_not(mk)
    with mk the bad-mask.  attnbias coerces the bool operand exactly —
    keep where mk is False."""

    def badmask(T, K):
        m = torch.rand(T, K) < 0.4
        m[:, 0] = False  # key 0 always allowed
        return m

    _mask_chunked_ok(
        [3, 6], badmask, mask_term=lambda mv: Op.make("logical_not", mv)
    )


# ---------------------------------------------------------------------------
#  (d) broadcast masks — lower rank, extent-1 key axis
# ---------------------------------------------------------------------------


def test_key_only_mask_fp64():
    """Mask (K,) — broadcast over rows, sliced on its own last dim."""

    def mask_fn(T, K):
        return torch.where(
            torch.rand(K) < 0.4,
            torch.tensor(float("-inf"), dtype=torch.float64),
            torch.zeros(K, dtype=torch.float64),
        )

    _mask_chunked_ok([3, 6], mask_fn)


def test_rowwise_bias_mask_fp64():
    """Mask (T,1) — extent 1 on the key axis broadcasts; ADD_MASK_CAT's
    reuse mode shares the operand across blocks, no split needed."""

    def mask_fn(T, K):
        return torch.randn(T, 1, dtype=torch.float64)

    _eg, _term, out, ref, _ = _mask_chunked_ok([3, 6], mask_fn)
    _assert_close_or_nan(out, ref)


def test_head_broadcast_bool_mask_fp64():
    """Mask (1,1,T,K) under rank-4 scores — leading broadcast dims are
    fine; the key axis is still the last."""

    def mask_fn(T, K):
        m = torch.rand(1, 1, T, K) < 0.6
        m[..., 0] = True
        return m

    _mask_chunked_ok([2, 5], mask_fn, batch=(2, 3))


# ---------------------------------------------------------------------------
#  (e) −inf / NaN edge semantics — must match sdpa exactly
# ---------------------------------------------------------------------------


def test_fully_masked_row_nan_parity():
    """A row masked across ALL blocks: the carrier reproduces dense
    softmax (a/l = 0/0 = NaN) while torch's sdpa kernel emits 0 — the
    one documented divergence, inherited from om_apply's deliberately
    unclamped semantics.  Asserted precisely: NaN rows in out, zeros in
    ref, identical elsewhere."""

    def mask_fn(T, K):
        m = torch.zeros(T, K, dtype=torch.float64)
        m[torch.rand(T, K) < 0.3] = float("-inf")
        m[1] = float("-inf")  # fully-masked row
        m[3] = float("-inf")
        return m

    _eg, _term, out, ref, _ = _mask_chunked_ok([3, 4], mask_fn, T=4)
    assert torch.isnan(out[1]).all() and torch.isnan(out[3]).all()
    assert torch.equal(ref[1], torch.zeros_like(ref[1]))
    assert torch.equal(ref[3], torch.zeros_like(ref[3]))
    _assert_close_or_nan(out, ref)


def test_fully_masked_block_contributes_zero():
    """One whole block masked — its mask slice is all −inf, the carrier
    is dropped by om_compose's isfinite guard, matching sdpa."""

    def mask_fn(T, K):
        m = torch.zeros(T, K, dtype=torch.float64)
        m[:, 3:] = float("-inf")  # block 1 entirely masked
        return m

    _eg, _term, out, ref, _ = _mask_chunked_ok([3, 4], mask_fn, T=4)
    assert torch.isfinite(ref).all()
    _assert_close_or_nan(out, ref)


def test_bool_fully_masked_row_nan_parity():
    """Bool variant: an all-False row is fully masked → NaN carrier,
    zeros in sdpa — same documented divergence as the additive case."""

    def mask_fn(T, K):
        m = torch.rand(T, K) < 0.6
        m[2] = False  # fully-masked row
        return m

    _eg, _term, out, ref, _ = _mask_chunked_ok([3, 4], mask_fn, T=4)
    assert torch.isnan(out[2]).all()
    assert torch.equal(ref[2], torch.zeros_like(ref[2]))
    _assert_close_or_nan(out, ref)


# ---------------------------------------------------------------------------
#  (f) negative checks — ill-typed masks must NOT chunk
# ---------------------------------------------------------------------------


def _no_fire(term):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=8)
    return eg


def test_mask_wrong_extent_does_not_fire():
    """Mask last dim 6 while Tk = 3+6 = 9 — unsplittable → veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))
    ]
    m = Var("m", TensorType((5, 6)))
    eg = _no_fire(_sdpa_mask_term(q, ks, vs, m))
    assert not any(
        k.startswith("sdpa_cat_mask_") for k in eg.rule_fires
    )


def test_mask_extent_on_wrong_axis_does_not_fire():
    """Mask (Tk, Tq) — the key extent sits on the row axis → veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))
    ]
    m = Var("m", TensorType((9, 5)))
    eg = _no_fire(_sdpa_mask_term(q, ks, vs, m))
    assert not any(
        k.startswith("sdpa_cat_mask_") for k in eg.rule_fires
    )


def test_mask_unknown_key_extent_does_not_fire():
    """Unknown (None) block extents → can't derive split sizes → veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var("k0", TensorType((None, 4))),
        Var("k1", TensorType((6, 4))),
    ]
    vs = [
        Var("v0", TensorType((None, 7))),
        Var("v1", TensorType((6, 7))),
    ]
    m = Var("m", TensorType((5, 9)))
    eg = _no_fire(_sdpa_mask_term(q, ks, vs, m))
    assert not any(
        k.startswith("sdpa_cat_mask_") for k in eg.rule_fires
    )


def test_mask_with_dropout_does_not_fire():
    """dropout_p != 0 alongside a mask is not pure math — veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))
    ]
    m = Var("m", TensorType((5, 9)))
    term = Op.make(
        "sdpa",
        q,
        Op.make("concat", ks[0], ks[1], dim=-2),
        Op.make("concat", vs[0], vs[1], dim=-2),
        m,
        arg4=0.5,
        arg5=False,
    )
    eg = _no_fire(term)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_mask_with_is_causal_true_does_not_fire():
    """mask + is_causal=True — torch forbids the combo; no pattern
    matches it (the mask laws only spell is_causal=False)."""
    q = Var("q", TensorType((5, 4)))
    ks = [
        Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))
    ]
    vs = [
        Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))
    ]
    m = Var("m", TensorType((5, 9)))
    term = Op.make(
        "sdpa",
        q,
        Op.make("concat", ks[0], ks[1], dim=-2),
        Op.make("concat", vs[0], vs[1], dim=-2),
        m,
        arg4=0.0,
        arg5=True,
    )
    eg = _no_fire(term)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_mask_non_sequence_concat_does_not_fire():
    """k/v concatenated on the FEATURE axis (dim -1) with a mask —
    same veto as the causal laws."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((3, 4))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((3, 7))) for i in range(2)]
    m = Var("m", TensorType((5, 3)))
    term = Op.make(
        "sdpa",
        q,
        Op.make("concat", ks[0], ks[1], dim=-1),
        Op.make("concat", vs[0], vs[1], dim=-1),
        m,
    )
    eg = _no_fire(term)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)
