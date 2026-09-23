"""Causal sdpa chunking — ``is_causal`` is a flag, not a mask operand.

``sdpa(q, cat(k_i), cat(v_i), is_causal=True)`` has no materialised
mask for OM_MASK_LAWS to slice.  SDPA_CAT_LAWS (catopt/om.py) chunk the
flag itself: they unfold the implicit causal mask into a materialised
``cmask`` generator over the concatenated score matrix,

    sdpa(q, cat k, cat v, is_causal)
      = om_apply(om_elem(
            masked_fill(cat(qs @ k_i.T), cmask(cat s), -inf), cat v))

with qs = q·scale (the kernel's default 1/√E folded into q through the
``fill`` constant generator — derived scalars can only land in attrs).
The existing laws then do all the work: masked_fill_cat_slice puts
block i's positional offset inside the cmask slice, and OM_SPLIT
splits the carrier.

Covered here:

* the law fires on both flag spellings — kwarg ``is_causal=True`` and
  torch.export's positional ``arg4=0.0, arg5=True`` — and on the
  unmasked companion;
* fp64-exact equivalence with ``F.scaled_dot_product_attention`` —
  even/uneven blocks, n-ary concats, batched heads, custom scale,
  T_q=1 decode rows;
* NaN parity: causal rows are never fully masked, so the parity check
  is that neither side produces NaN — while the T_q=1 / uneven-block
  cases exercise om_compose's isfinite guard internally (late blocks
  are fully masked for early rows);
* negative checks: non-sequence concats, dropout_p != 0, explicit
  attn_mask (arity), and is_causal=False all refuse to fire.
"""

import torch
import torch.nn.functional as F

from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, TensorType, op_repr
from catopt.om import OM_LAWS, SDPA_CAT_LAWS
from catopt.cost import flops_cost
from catopt.torch_bridge import ir_to_torch_module


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _nested_cat(ts, dim, attr_key="dim"):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, **{attr_key: dim})
    return out


def _sdpa_cat_term(q, ks, vs, attr_key="dim", causal=True,
                   spelling="export", scale=None):
    """sdpa(q, cat(k_i), cat(v_i), is_causal) as an IR term.

    ``spelling`` selects the flag encoding:
      "export" — positional attrs: arg4=dropout_p, arg5=is_causal
                 (what torch.export actually emits);
      "kwarg"  — is_causal=True / is_causal=False named attrs.
    """
    kcat = _nested_cat(ks, -2, attr_key)
    vcat = _nested_cat(vs, -2, attr_key)
    attrs = {}
    if spelling == "export":
        attrs["arg4"] = 0.0
        attrs["arg5"] = bool(causal)
        if scale is not None:
            attrs["arg6"] = float(scale)
    else:
        attrs["is_causal"] = bool(causal)
        if scale is not None:
            attrs["scale"] = float(scale)
    return Op.make("sdpa", q, kcat, vcat, **attrs)


def _sdpa_ref(q, ks, vs, causal=True, scale=None):
    kw = {"is_causal": causal}
    if scale is not None:
        kw["scale"] = scale
    return F.scaled_dot_product_attention(
        q, torch.cat(list(ks), dim=-2), torch.cat(list(vs), dim=-2),
        **kw)


def _run_om(term, max_iterations=20, max_nodes=200_000):
    eg = EGraph()
    root = eg.add_term(term)
    stats = eg.run(OM_LAWS, root, max_iterations=max_iterations,
                   max_nodes=max_nodes)
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
    cheaper).  Same override protocol as test_om_mask/test_om_monoid."""
    canon = eg.find(root)
    ov = {}
    for cid in list(eg._classes):
        c = eg.find(cid)
        comps = [n for n in eg._classes[c].nodes if n.op == "om_compose"]
        if comps:
            ov.setdefault(c, comps[0])
    for n in eg.get_class(canon).nodes:
        if (n.op == "om_apply"
                and _class_has_op(eg, n.children[0], "om_compose")):
            o = dict(ov)
            o[canon] = n
            t = eg.extract_best(canon, flops_cost, overrides=o)
            if t is not None:
                return t
    return None


def _assert_close_or_nan(out, ref, tol=1e-12):
    """NaN-aware fp64 comparison: NaN positions must match torch sdpa
    exactly, finite positions must agree to tol."""
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    fin = ~torch.isnan(ref)
    assert (out[fin] - ref[fin]).abs().max().item() < tol


def _causal_chunked_ok(ksizes, T=5, d=4, dv=7, seed=0,
                       attr_key="dim", spelling="export", scale=None,
                       causal=True, batch=None):
    """Saturate an sdpa-over-concats term, force-extract the chunked
    carrier, verify fp64 vs F.scaled_dot_product_attention."""
    torch.manual_seed(seed)
    qshape = (batch + (T, d)) if batch else (T, d)
    kshape = lambda k: (batch + (k, d)) if batch else (k, d)
    vshape = lambda k: (batch + (k, dv)) if batch else (k, dv)
    q = Var("q", TensorType(qshape))
    ks = [Var(f"k{i}", TensorType(kshape(k)))
          for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType(vshape(k)))
          for i, k in enumerate(ksizes)]
    term = _sdpa_cat_term(q, ks, vs, attr_key=attr_key,
                          causal=causal, spelling=spelling,
                          scale=scale)

    eg, root, _ = _run_om(term)
    chunked = [n for n in eg.get_class(root).nodes
               if n.op == "om_apply"
               and _class_has_op(eg, n.children[0], "om_compose")]
    assert chunked, "sdpa never decomposed into a chunked carrier"
    term = _extract_chunked(eg, root)
    assert term is not None and "om_compose" in op_repr(term)

    inputs = [q] + ks + vs
    ir = IR(root=term, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = ir_to_torch_module(ir)

    tq = torch.randn(*qshape, dtype=torch.float64)
    tks = [torch.randn(*kshape(k), dtype=torch.float64)
           for k in ksizes]
    tvs = [torch.randn(*vshape(k), dtype=torch.float64)
           for k in ksizes]
    ref = _sdpa_ref(tq, tks, tvs, causal=causal, scale=scale)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    _assert_close_or_nan(out, ref)
    return eg, term, out, ref, tvs


# ---------------------------------------------------------------------------
#  (a) the law fires — both flag spellings, both concat spellings
# ---------------------------------------------------------------------------

def test_causal_law_fires_export_spelling():
    """sdpa(q, cat k, cat v, arg4=0.0, arg5=True) — the exact term
    torch.export emits — gains the om_apply member."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))]
    vs = [Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))]
    term = _sdpa_cat_term(q, ks, vs, spelling="export")
    eg, root, _ = _run_om(term)
    assert any(k.startswith("sdpa_cat_causal_")
               for k in eg.rule_fires), dict(eg.rule_fires)
    assert any(n.op == "om_apply" for n in eg.get_class(root).nodes)


def test_causal_law_fires_kwarg_spelling():
    """The hand-built is_causal=True attr form fires too."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))]
    vs = [Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))]
    term = _sdpa_cat_term(q, ks, vs, spelling="kwarg")
    eg, root, _ = _run_om(term)
    assert any(k.startswith("sdpa_cat_causal_")
               for k in eg.rule_fires)


def test_causal_law_fires_arg1_concat_spelling():
    """Concat dims spelled arg1= (raw positional) also match."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))]
    vs = [Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))]
    term = _sdpa_cat_term(q, ks, vs, attr_key="arg1",
                          spelling="export")
    eg, root, _ = _run_om(term)
    assert any(k.startswith("sdpa_cat_causal_")
               and k.endswith("arg1") for k in eg.rule_fires)


# ---------------------------------------------------------------------------
#  (b) fp64 equivalence — the chunked carrier IS the causal sdpa
# ---------------------------------------------------------------------------

def test_causal_chunked_attention_fp64():
    """THE law: sdpa(q, cat k, cat v, is_causal) ≡ om tree, fp64."""
    eg, term, out, ref, _ = _causal_chunked_ok([3, 6])
    assert eg.rule_fires.get("masked_fill_cat_slice_dim", 0) > 0
    assert eg.rule_fires.get("om_split", 0) > 0
    r = op_repr(term)
    # The materialised causal mask is sliced per block — the offset
    # lives inside the split, exactly as for an explicit mask.
    assert "cmask" in r and "split" in r and "masked_fill" in r


def test_causal_chunked_kwarg_fp64():
    _causal_chunked_ok([3, 6], spelling="kwarg")


def test_causal_chunked_uneven_three_blocks():
    """n-ary concat: the cmask slice-of-slice carries both offsets —
    block 2's mask columns are split(split(cmask)) at [o1, o1+o2)."""
    _causal_chunked_ok([2, 3, 4])


def test_causal_chunked_uneven_two_blocks():
    """Very uneven blocks — block 0 is a single key."""
    _causal_chunked_ok([1, 8])


def test_causal_decode_single_query_row():
    """T_q=1: torch's is_causal is lower-LEFT triangular, so the decode
    row sees only key 0 — the chunked carrier must reproduce exactly
    that (including the footgun), i.e. out == v[...,0,:]."""
    eg, term, out, ref, tvs = _causal_chunked_ok([3, 4], T=1)
    # row 0 attends key 0 with weight 1 in BOTH forms (ref is (1, dv)).
    assert torch.equal(ref, tvs[0][:1])
    _assert_close_or_nan(out, ref)
    assert (out[0] - tvs[0][0]).abs().max().item() < 1e-12


def test_causal_custom_scale():
    """An explicit scale attr rides through fill(q, value=SC)."""
    _causal_chunked_ok([3, 6], spelling="kwarg", scale=0.5)
    _causal_chunked_ok([3, 6], spelling="export", scale=0.25)


def test_causal_batched_heads():
    """Rank-4 (B,H,T,d): cmask broadcasts over heads and still slices
    on the key axis."""
    _causal_chunked_ok([2, 5], batch=(2, 3))


def test_causal_tq_gt_first_block():
    """Rows t >= o_i see block i's earlier columns fully — the split
    mask gives those rows an all-False block slice (full attention)."""
    _causal_chunked_ok([4, 2], T=7)


# ---------------------------------------------------------------------------
#  (c) the unmasked companion law
# ---------------------------------------------------------------------------

def test_unmasked_sdpa_cat_fp64():
    """sdpa(q, cat k, cat v) — no mask at all — chunks to the plain om
    homomorphism over scaled scores."""
    eg, term, out, ref, _ = _causal_chunked_ok([3, 6], causal=False)
    assert any(k.startswith("sdpa_cat_")
               and not k.startswith("sdpa_cat_causal_")
               for k in eg.rule_fires)
    assert "om_compose" in op_repr(term)
    assert "cmask" not in op_repr(term)


def test_unmasked_sdpa_cat_kwarg_false():
    """is_causal=False spelled out is the same program."""
    _causal_chunked_ok([3, 6], causal=False, spelling="kwarg")


# ---------------------------------------------------------------------------
#  (d) NaN parity and the fully-masked-block guard
# ---------------------------------------------------------------------------

def test_fully_masked_row_nan_parity():
    """A causal row is never fully masked (row t sees key 0), so the
    parity check is isnan-equality: no NaN positions on either side —
    while uneven blocks exercise the isfinite guard internally for
    rows t < o_i (fully masked within late blocks)."""
    eg, term, out, ref, _ = _causal_chunked_ok([5, 2], T=3)
    # every row's late-block slice is fully masked for t < 5
    assert not torch.isnan(out).any() and not torch.isnan(ref).any()
    _assert_close_or_nan(out, ref)


def test_fully_masked_row_nan_parity_decode():
    """T_q=1, block 0 size 1: the second block is fully masked for the
    only row — its NaN carrier must drop out (isfinite guard), leaving
    out == v0 exactly, matching sdpa."""
    _, _, out, ref, tvs = _causal_chunked_ok([1, 6], T=1)
    assert not torch.isnan(out).any()
    _assert_close_or_nan(out, ref)


# ---------------------------------------------------------------------------
#  (e) negative checks — the flag must NOT chunk when ill-typed
# ---------------------------------------------------------------------------

def test_non_sequence_concat_does_not_fire():
    """k/v concatenated on the FEATURE axis (dim -1) is a different
    program — veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((3, 4))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((3, 7))) for i in range(2)]
    term = Op.make(
        "sdpa", q,
        Op.make("concat", ks[0], ks[1], dim=-1),
        Op.make("concat", vs[0], vs[1], dim=-1),
        arg4=0.0, arg5=True)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=5)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_dropout_nonzero_does_not_fire():
    """dropout_p != 0 is not a pure function — veto."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))]
    vs = [Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))]
    term = Op.make(
        "sdpa", q,
        Op.make("concat", ks[0], ks[1], dim=-2),
        Op.make("concat", vs[0], vs[1], dim=-2),
        arg4=0.5, arg5=True)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=5)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_explicit_attn_mask_does_not_fire():
    """sdpa with a 4th (attn_mask) operand is a different arity — the
    is_causal law must not touch it (torch forbids mask+is_causal
    anyway)."""
    q = Var("q", TensorType((5, 4)))
    ks = [Var(f"k{i}", TensorType((k, 4))) for i, k in enumerate((3, 6))]
    vs = [Var(f"v{i}", TensorType((k, 7))) for i, k in enumerate((3, 6))]
    m = Var("m", TensorType((5, 9)))
    term = Op.make(
        "sdpa", q,
        Op.make("concat", ks[0], ks[1], dim=-2),
        Op.make("concat", vs[0], vs[1], dim=-2),
        m, is_causal=True)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=5)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_sdpa_no_concat_does_not_fire():
    """Plain sdpa(q, k, v, is_causal) — nothing to chunk, no fire."""
    q = Var("q", TensorType((5, 4)))
    k = Var("k", TensorType((9, 4)))
    v = Var("v", TensorType((9, 7)))
    term = Op.make("sdpa", q, k, v, arg4=0.0, arg5=True)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=5)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)


def test_mismatched_kv_blocks_do_not_fire():
    """k_i's key count must equal v_i's — (3,4) keys vs (5,7) values
    is ill-typed chunking → veto."""
    q = Var("q", TensorType((5, 4)))
    k1 = Var("k1", TensorType((3, 4)))
    k2 = Var("k2", TensorType((6, 4)))
    v1 = Var("v1", TensorType((3, 7)))
    v2 = Var("v2", TensorType((5, 7)))   # != 6
    term = Op.make(
        "sdpa", q,
        Op.make("concat", k1, k2, dim=-2),
        Op.make("concat", v1, v2, dim=-2),
        arg4=0.0, arg5=True)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=5)
    assert not any(k.startswith("sdpa_cat_") for k in eg.rule_fires)
