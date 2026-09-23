"""OM_LAWS — the online-softmax monoid (FlashAttention combine as a law).

The affine-map monoid (SCAN_LAWS) lifted a linear recurrence into a
carrier where associativity alone discovers parallel-scan bracketings.
This is the nonlinear analogue: softmax(qk^T)@v lifts into the
(m, l, a) carrier where the same associativity generates chunked /
blocked attention schedules.

Covered here:

* bindings — om_elem / om_compose / om_apply agree with direct
  softmax@v in fp64, including the FlashAttention rescaling and the
  NaN semantics of fully-masked rows (and fully-masked BLOCKS, which
  the isfinite guard reduces to a zero contribution).
* chunked equivalence — a dense ``softmax(q @ cat(k_i).T) @ cat(v_i)``
  term saturated under OM_LAWS sprouts the om_apply(om_compose(...))
  form in the SAME e-class; force-extracting it lowers to a module that
  is fp64-exact vs the dense computation.
* diverse_classes — the root e-class visibly contains both the
  softmax-form and the om-form.
* negative dim checks — the split/transpose-concat rules refuse
  well-typed-but-wrong cat axes.
"""

import torch

from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, TensorType, op_repr
from catopt.om import (
    OM_LAWS, OM_SPLIT, OM_SPLIT_ARG1, CONCAT_BINARIZE,
    MATMUL_T_CONCAT,
)
from catopt.cost import flops_cost, dag_cost
from catopt.torch_bridge import ir_to_torch_module, _IR_TO_TORCH


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _om_elem(s, v):
    return _IR_TO_TORCH["om_elem"](s, v)


def _om_compose(f, g):
    return _IR_TO_TORCH["om_compose"](f, g)


def _om_apply(f):
    return _IR_TO_TORCH["om_apply"](f)


def _nested_cat(ts, dim):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, dim=dim)
    return out


def _dense_chunked_term(q, ks, vs):
    """softmax(q @ cat(k_i, -2).T) @ cat(v_i, -2) as an IR term."""
    kcat = _nested_cat(ks, -2)
    vcat = _nested_cat(vs, -2)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, arg1=-2, arg2=-1))
    return Op.make(
        "matmul", Op.make("softmax", scores, arg1=-1), vcat)


def _dense_chunked_ref(q, ks, vs):
    s = q @ torch.cat(list(ks), dim=-2).transpose(-2, -1)
    return torch.softmax(s, dim=-1) @ torch.cat(list(vs), dim=-2)


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


def _n_enodes_with_op(eg, opname):
    return sum(1 for n in eg._node_to_class if n.op == opname)


# ---------------------------------------------------------------------------
#  (a) bindings
# ---------------------------------------------------------------------------

def test_om_elem_apply_matches_dense():
    """om_apply(om_elem(s,v)) == softmax(s) @ v, fp64."""
    torch.manual_seed(0)
    s = torch.randn(2, 6, 9, dtype=torch.float64)
    v = torch.randn(2, 9, 5, dtype=torch.float64)
    out = _om_apply(_om_elem(s, v))
    ref = torch.softmax(s, dim=-1) @ v
    assert (out - ref).abs().max().item() < 1e-12


def test_om_compose_matches_dense():
    """elem(s1,v1) ⊕ elem(s2,v2) == elem(cat s, cat v) == softmax@v."""
    torch.manual_seed(0)
    s1 = torch.randn(3, 6, 4, dtype=torch.float64)
    s2 = torch.randn(3, 6, 7, dtype=torch.float64)
    v1 = torch.randn(3, 4, 5, dtype=torch.float64)
    v2 = torch.randn(3, 7, 5, dtype=torch.float64)
    f = _om_compose(_om_elem(s1, v1), _om_elem(s2, v2))
    out = _om_apply(f)
    s = torch.cat([s1, s2], dim=-1)
    v = torch.cat([v1, v2], dim=-2)
    ref = torch.softmax(s, dim=-1) @ v
    assert (out - ref).abs().max().item() < 1e-12


def test_om_compose_associativity_fp64():
    """Both bracketings of a 3-block product agree with dense — ⊕ is
    associative in exact arithmetic modulo rounding (<1e-12)."""
    torch.manual_seed(0)
    ss = [torch.randn(2, 5, k, dtype=torch.float64) for k in (3, 4, 6)]
    vs = [torch.randn(2, k, 7, dtype=torch.float64) for k in (3, 4, 6)]
    es = [_om_elem(s, v) for s, v in zip(ss, vs)]
    left = _om_apply(_om_compose(_om_compose(es[0], es[1]), es[2]))
    right = _om_apply(_om_compose(es[0], _om_compose(es[1], es[2])))
    ref = torch.softmax(torch.cat(ss, -1), dim=-1) @ torch.cat(vs, -2)
    assert (left - ref).abs().max().item() < 1e-12
    assert (right - ref).abs().max().item() < 1e-12


def test_om_compose_commutes_numerically():
    """⊕ is commutative (no comm RULE — but the law must hold, else the
    rewrites it enables would be unsound)."""
    torch.manual_seed(0)
    s1 = torch.randn(4, 5, dtype=torch.float64)
    s2 = torch.randn(4, 3, dtype=torch.float64)
    v1 = torch.randn(5, 6, dtype=torch.float64)
    v2 = torch.randn(3, 6, dtype=torch.float64)
    a = _om_apply(_om_compose(_om_elem(s1, v1), _om_elem(s2, v2)))
    b = _om_apply(_om_compose(_om_elem(s2, v2), _om_elem(s1, v1)))
    ref = torch.softmax(torch.cat([s1, s2], -1), -1) @ torch.cat(
        [v1, v2], -2)
    assert (a - ref).abs().max().item() < 1e-12
    assert (b - ref).abs().max().item() < 1e-12


def test_om_fully_masked_row_nan_semantics():
    """A fully-masked row must produce NaN in BOTH forms — softmax's
    NaN IS the semantics; om_apply stays unclamped."""
    s = torch.full((2, 5), float("-inf"), dtype=torch.float64)
    v = torch.randn(5, 3, dtype=torch.float64)
    dense = torch.softmax(s, dim=-1) @ v
    out = _om_apply(_om_elem(s, v))
    assert torch.isnan(dense).all() and torch.isnan(out).all()
    # NaN-safe comparison: same NaN positions AND equal where finite.
    assert torch.equal(torch.isnan(dense), torch.isnan(out))


def test_om_fully_masked_block_contributes_zero():
    """A fully-masked BLOCK inside an unmasked row contributes nothing:
    the isfinite guard zeroes its (NaN) contribution, matching dense
    softmax which treats those columns as exp(-inf)=0."""
    torch.manual_seed(0)
    s1 = torch.randn(2, 5, dtype=torch.float64)
    s2 = torch.full((2, 4), float("-inf"), dtype=torch.float64)
    v1 = torch.randn(5, 3, dtype=torch.float64)
    v2 = torch.randn(4, 3, dtype=torch.float64)
    out = _om_apply(_om_compose(_om_elem(s1, v1), _om_elem(s2, v2)))
    dense = torch.softmax(torch.cat([s1, s2], -1), -1) @ torch.cat(
        [v1, v2], -2)
    assert torch.isfinite(dense).all()
    assert (out - dense).abs().max().item() < 1e-12


def test_om_packaging_op_is_identity_triple():
    """om(m, l, a) is pure packaging — the identity on the triple."""
    m = torch.randn(4, 1)
    l = torch.randn(4, 1)
    a = torch.randn(4, 3)
    f = _IR_TO_TORCH["om"](m, l, a)
    assert f == (m, l, a)


# ---------------------------------------------------------------------------
#  (b) chunked equivalence via the e-graph
# ---------------------------------------------------------------------------

def _run_om(term, max_iterations=20, max_nodes=200_000):
    eg = EGraph()
    root = eg.add_term(term)
    stats = eg.run(OM_LAWS, root, max_iterations=max_iterations,
                   max_nodes=max_nodes)
    return eg, root, stats


def _extract_chunked(eg, root, cost_fn=flops_cost):
    """Force-extract a maximally-chunked carrier term.

    Per-class greedy extraction can never see the composed form — an
    om_elem over the full concat is locally cheaper than splitting it —
    so we override the root to an om_apply enode and every carrier
    e-class holding an om_compose to that enode.  Returns the term, or
    None if no chunked member materialised.
    """
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
            t = eg.extract_best(canon, cost_fn, overrides=o)
            if t is not None:
                return t
    return None


def test_chunked_attention_emerges_via_om_split():
    """Dense softmax over concat'd keys saturates into the chunked
    om_apply(om_compose(elem, elem, ...)) form — in the SAME e-class —
    and the extracted chunked term is fp64-exact vs dense."""
    torch.manual_seed(0)
    T, d, dv = 5, 4, 7
    ksizes = [3, 4, 6]
    q = Var("q", TensorType((T, d)))
    ks = [Var(f"k{i}", TensorType((k, d))) for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType((k, dv))) for i, k in enumerate(ksizes)]
    term = _dense_chunked_term(q, ks, vs)

    eg, root, stats = _run_om(term)
    root_ops = {n.op for n in eg.get_class(root).nodes}
    # The root class contains BOTH the dense matmul form and the lifted
    # om_apply form — softmax-form ≡ om-form in one e-class.
    assert "matmul" in root_ops
    assert "om_apply" in root_ops

    # At least one om_apply enode reaches an om_compose — i.e. the
    # homomorphism actually split the carrier across blocks.
    chunked = [n for n in eg.get_class(root).nodes
               if n.op == "om_apply"
               and _class_has_op(eg, n.children[0], "om_compose")]
    assert chunked, "OM_SPLIT never produced a composed carrier"

    # Force-extract the chunked form and verify fp64-exact.
    chunked_term = _extract_chunked(eg, root)
    assert chunked_term is not None
    assert "om_compose" in op_repr(chunked_term)

    inputs = [q] + ks + vs
    ir = IR(root=chunked_term, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = ir_to_torch_module(ir)

    tq = torch.randn(T, d, dtype=torch.float64)
    tks = [torch.randn(k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, dv, dtype=torch.float64) for k in ksizes]
    ref = _dense_chunked_ref(tq, tks, tvs)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-12


def test_chunked_attention_batched():
    """Same discovery on rank-4 (B, H, T, K) scores — dims are checked
    on the last axes only, so batching is transparent."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 3, 5, 4, 6
    ksizes = [2, 5]
    q = Var("q", TensorType((B, H, T, d)))
    ks = [Var(f"k{i}", TensorType((B, H, k, d)))
          for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType((B, H, k, dv)))
          for i, k in enumerate(ksizes)]
    term = _dense_chunked_term(q, ks, vs)

    eg, root, _ = _run_om(term)
    chunked = [n for n in eg.get_class(root).nodes
               if n.op == "om_apply"
               and _class_has_op(eg, n.children[0], "om_compose")]
    assert chunked
    chunked_term = _extract_chunked(eg, root)
    assert chunked_term is not None
    inputs = [q] + ks + vs
    ir = IR(root=chunked_term, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = ir_to_torch_module(ir)

    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=torch.float64)
           for k in ksizes]
    ref = _dense_chunked_ref(tq, tks, tvs)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-12


def test_greedy_extraction_also_verifies():
    """Whichever member wins the cost model is still equivalent —
    greedy extraction must verify fp64 against dense too."""
    torch.manual_seed(0)
    q = Var("q", TensorType((4, 4)))
    ks = [Var(f"k{i}", TensorType((3, 4))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((3, 5))) for i in range(2)]
    term = _dense_chunked_term(q, ks, vs)
    eg, root, _ = _run_om(term)
    best = eg.extract_best(root, flops_cost)
    inputs = [q] + ks + vs
    ir = IR(root=best, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = ir_to_torch_module(ir)
    tq = torch.randn(4, 4, dtype=torch.float64)
    tks = [torch.randn(3, 4, dtype=torch.float64) for _ in ks]
    tvs = [torch.randn(3, 5, dtype=torch.float64) for _ in vs]
    ref = _dense_chunked_ref(tq, tks, tvs)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  (c) e-class diversity + alternatives
# ---------------------------------------------------------------------------

def test_diverse_class_shows_softmax_and_om_forms():
    """The root e-class's member sketches include both the softmax
    pipeline (matmul over softmax) and the carrier pipeline
    (om_apply over om_compose)."""
    torch.manual_seed(0)
    q = Var("q", TensorType((4, 4)))
    ks = [Var(f"k{i}", TensorType((3, 4))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((3, 5))) for i in range(2)]
    eg, root, _ = _run_om(_dense_chunked_term(q, ks, vs))
    canon = eg.find(root)
    classes = {d["eid"]: d["members"] for d in eg.diverse_classes()}
    members = classes.get(canon, [])
    assert any(m.startswith("matmul(") for m in members)
    assert any(m.startswith("om_apply(") for m in members)


def test_extract_alternatives_frontier_has_both_forms():
    """extract_alternatives (the [G]-frontier view) surfaces both the
    dense softmax term and a chunked om term."""
    torch.manual_seed(0)
    q = Var("q", TensorType((4, 4)))
    ks = [Var(f"k{i}", TensorType((3, 4))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((3, 5))) for i in range(2)]
    eg, root, _ = _run_om(_dense_chunked_term(q, ks, vs))
    alts = eg.extract_alternatives(root, flops_cost, top_k=8)
    reps = [op_repr(t) for _, t in alts]
    assert any("softmax" in r for r in reps)
    assert any("om_apply" in r for r in reps)
    # The split carrier shows up as a class holding both om_elem and
    # om_compose member sketches (per-class greedy extraction can never
    # surface it at the root: the unsplit elem is locally cheaper).
    assert any("om_compose" in " ".join(d["members"])
               for d in eg.diverse_classes())


# ---------------------------------------------------------------------------
#  Negative dim checks — well-typed but WRONG programs must be refused
# ---------------------------------------------------------------------------

def test_om_split_rejects_wrong_score_axis():
    """scores concat along the ROW axis (dim 0 for (T,K)) is not the
    key axis — the homomorphism does not apply there."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((5, 3)))
    v1 = Var("v1", TensorType((3, 6)))
    v2 = Var("v2", TensorType((3, 6)))
    # scores cat on dim 0 (rows), values cat on dim -2 (keys) —
    # mismatched axes: must not split.
    t = Op.make("om_elem",
                Op.make("concat", s1, s2, dim=0),
                Op.make("concat", v1, v2, dim=-2))
    eg = EGraph()
    root = eg.add_term(t)
    eg.run([OM_SPLIT, OM_SPLIT_ARG1], root, max_iterations=5)
    assert _n_enodes_with_op(eg, "om_compose") == 0


def test_om_split_rejects_wrong_value_axis():
    """values concat along the d axis (dim -1) is not the key axis."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    v1 = Var("v1", TensorType((3, 6)))
    v2 = Var("v2", TensorType((5, 4)))
    t = Op.make("om_elem",
                Op.make("concat", s1, s2, dim=-1),
                Op.make("concat", v1, v2, dim=-1))
    eg = EGraph()
    root = eg.add_term(t)
    eg.run([OM_SPLIT, OM_SPLIT_ARG1], root, max_iterations=5)
    assert _n_enodes_with_op(eg, "om_compose") == 0


def test_om_split_rejects_incompatible_chunks():
    """s_i's key dim must equal v_i's key dim — otherwise elem(cat)
    decomposes into pieces that don't even matmul."""
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    v1 = Var("v1", TensorType((3, 6)))
    v2 = Var("v2", TensorType((7, 6)))  # 7 != 5: wrong pair
    t = Op.make("om_elem",
                Op.make("concat", s1, s2, dim=-1),
                Op.make("concat", v1, v2, dim=-2))
    eg = EGraph()
    root = eg.add_term(t)
    eg.run([OM_SPLIT, OM_SPLIT_ARG1], root, max_iterations=5)
    assert _n_enodes_with_op(eg, "om_compose") == 0


def test_matmul_t_concat_rejects_feature_axis():
    """q @ cat(k1,k2, dim=-1).T — cat along the FEATURE axis does not
    distribute into score blocks."""
    q = Var("q", TensorType((4, 8)))
    k1 = Var("k1", TensorType((3, 8)))
    k2 = Var("k2", TensorType((3, 8)))
    t = Op.make("matmul", q,
                Op.make("transpose",
                        Op.make("concat", k1, k2, dim=-1),
                        arg1=-2, arg2=-1))
    eg = EGraph()
    root = eg.add_term(t)
    eg.run([MATMUL_T_CONCAT], root, max_iterations=5)
    # No concat-of-matmuls may appear; the only concat is the input's.
    for n in eg._node_to_class:
        if n.op == "concat":
            for c in n.children:
                assert not any(ch.op == "matmul"
                               for ch in eg.get_class(c).nodes)


def test_matmul_t_concat_rejects_partial_transpose():
    """A transpose that is not .T on the last two dims must not fire —
    the concat axis would land somewhere other than the score rows."""
    q = Var("q", TensorType((2, 4, 8)))
    k1 = Var("k1", TensorType((2, 3, 8)))
    k2 = Var("k2", TensorType((2, 5, 8)))
    # concat on seq dim (-2) but transpose swaps dims 0 and -1 — the
    # cat axis does not land on scores' last dim.
    t = Op.make("matmul", q,
                Op.make("transpose",
                        Op.make("concat", k1, k2, dim=-2),
                        arg1=0, arg2=-1))
    eg = EGraph()
    root = eg.add_term(t)
    eg.run([MATMUL_T_CONCAT], root, max_iterations=5)
    for n in eg._node_to_class:
        if n.op == "concat":
            for c in n.children:
                assert not any(ch.op == "matmul"
                               for ch in eg.get_class(c).nodes)


# ---------------------------------------------------------------------------
#  concat binarization
# ---------------------------------------------------------------------------

def test_concat_binarize_nary_to_binary():
    """3-ary concat gains a nested-binary equivalent in its e-class,
    and both evaluate identically."""
    a = Var("a", TensorType((4, 3)))
    b = Var("b", TensorType((4, 5)))
    c = Var("c", TensorType((4, 2)))
    t = Op.make("concat", a, b, c, dim=-1)
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(CONCAT_BINARIZE, root, max_iterations=5)
    nodes = eg.get_class(root).nodes
    assert any(len(n.children) == 3 for n in nodes)   # original n-ary
    assert any(len(n.children) == 2 for n in nodes)   # binarized

    ta = torch.randn(4, 3, dtype=torch.float64)
    tb = torch.randn(4, 5, dtype=torch.float64)
    tc = torch.randn(4, 2, dtype=torch.float64)
    inputs = [a, b, c]
    for n in nodes:
        if len(n.children) != 2:
            continue
        canon = eg.find(root)
        term = eg.extract_best(canon, flops_cost,
                               overrides={canon: n})
        ir = IR(root=term, inputs=inputs,
                input_names={v.name for v in inputs}, params={})
        mod = ir_to_torch_module(ir)
        with torch.no_grad():
            out = mod(ta, tb, tc)
        assert torch.equal(out, torch.cat([ta, tb, tc], dim=-1))
        return
    raise AssertionError("no binary concat alternative materialised")


def test_concat_binarize_feeds_om_split():
    """An n-ary concat'd dense attention splits into per-block carriers
    via CONCAT_BINARIZE + OM_SPLIT — no hand-built nesting needed."""
    torch.manual_seed(0)
    T, d, dv = 4, 4, 5
    ksizes = [2, 3, 4]
    q = Var("q", TensorType((T, d)))
    ks = [Var(f"k{i}", TensorType((k, d))) for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType((k, dv))) for i, k in enumerate(ksizes)]
    kcat = Op.make("concat", *ks, dim=-2)          # one n-ary concat
    vcat = Op.make("concat", *vs, dim=-2)
    scores = Op.make("matmul", q,
                     Op.make("transpose", kcat, arg1=-2, arg2=-1))
    term = Op.make("matmul",
                   Op.make("softmax", scores, arg1=-1), vcat)

    eg, root, _ = _run_om(term)
    chunked = [n for n in eg.get_class(root).nodes
               if n.op == "om_apply"
               and _class_has_op(eg, n.children[0], "om_compose")]
    assert chunked, "n-ary concat never reached the chunked carrier"
    chunked_term = _extract_chunked(eg, root)
    assert chunked_term is not None
    inputs = [q] + ks + vs
    ir = IR(root=chunked_term, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = ir_to_torch_module(ir)
    tq = torch.randn(T, d, dtype=torch.float64)
    tks = [torch.randn(k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(k, dv, dtype=torch.float64) for k in ksizes]
    ref = _dense_chunked_ref(tq, tks, tvs)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  cost model sanity — the om ops are priced, and `om` packaging is free
# ---------------------------------------------------------------------------

def test_om_cost_model_shapes_and_flops():
    """_infer_op_shape prices the carrier by its eventual tensor."""
    from catopt.cost import _shape_of
    s = Var("s", TensorType((4, 6)))
    v = Var("v", TensorType((6, 3)))
    e = Op.make("om_elem", s, v)
    assert _shape_of(e) == (4, 3)
    f = Op.make("om_compose", e, e)
    assert _shape_of(f) == (4, 3)
    out = Op.make("om_apply", f)
    assert _shape_of(out) == (4, 3)
    assert flops_cost(out) > 0
    # om packaging is a view op: zero local cost.
    m = Var("m", TensorType((4, 1)))
    l = Var("l", TensorType((4, 1)))
    a = Var("a", TensorType((4, 3)))
    triple = Op.make("om", m, l, a)
    assert _shape_of(triple) == (4, 3)
    assert dag_cost(triple, flops_cost) == 0.0
