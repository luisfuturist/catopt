"""View-op commute laws + the batched dense omd fiber — MHA coverage.

Companion to ``tests/test_xcarrier.py``.  That file establishes the
cross-carrier seam; this one covers what a NATURAL multi-head
attention block does to it: the head split ``v.view(T,nh,hd)
.transpose(0,1)`` wraps the value in view ops, hiding the affine
member one level below the om leaf's value class.  The new laws

    reshape(apply(aff(A,b),h), S)   = apply(aff(reshape(A,S+(i,)),
                                                reshape(b,S)), h)
    transpose(apply(aff(A,b),h), d) = apply(aff(transpose(A,d'),
                                               transpose(b,d')), h)

sink the view into the map coefficients (the map's input axis is
always LAST, so every value axis is a leading axis of A — the sink is
exact for ANY well-typed view on the dense fiber).  The diagonal
variants exist but are restricted to views that keep the feature axis
last (h is not permuted — a feature-moving view is a DIFFERENT map and
is vetoed).

With the apply member surfaced, the om leaf can lift — and
``_check_om_elem_aff`` now accepts rank->=3 batched maps
A (…,K,d,i): a per-head map (nh,K,hd,D) contracts its key axis while
the head axis rides along as a matmul batch dim (the fused
``_om_elem_aff`` and deferred ``_omd_elem``/``_omd_applym`` bindings
all handle it).

The e2e test exports a ScanAttnMH-style module (scan values -> linear
projection -> view+transpose head split -> causal softmax attention ->
output projection), saturates carrier+XC laws, and asserts the
``omd_applym`` member appears and evaluates fp64-exact.

Honest limits, exercised here as vetoes:
  * a broadcast-smaller b (b != A[:-1]) is not viewable — vetoed;
  * reshape on the diagonal fiber only passes when S[-1] stays the
    feature dim d (head-PACKING yes, head-SPLITTING no — that case
    goes through the dense promotion, which is what MHA does anyway);
  * transpose on the diagonal fiber vetoes any dim touching the last
    (feature) axis.
"""

from __future__ import annotations

import torch

import catopt.xcarrier as XC
from catopt import meta
from catopt.egraph import EGraph
from catopt.ir import Op, TensorType, Var

# ---------------------------------------------------------------------------
#  helpers (same conventions as test_xcarrier.py)
# ---------------------------------------------------------------------------


def _V(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(shape, seed_shift: int = 0):
    g = torch.Generator().manual_seed(4321 + seed_shift)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _law(rule, t0, env, tol=1e-10):
    applied = meta.apply_rewrite_at(rule, t0, ())
    assert applied is not None, f"{rule.name} did not fire"
    a = meta._eval_term(t0, env)
    b = meta._eval_term(applied, env)
    assert meta._eval_allclose(a, b, tol=tol), (
        f"{rule.name}: fp64 mismatch {(a - b).abs().max().item():.3e}"
    )
    return applied


def _no_fire(rule, t0):
    assert meta.apply_rewrite_at(rule, t0, ()) is None


def _class_ops(eg: EGraph, eid: int) -> set:
    return {n.op for n in eg.get_class(eid).nodes}


# ---------------------------------------------------------------------------
#  A. The view-commute laws — dense fiber
# ---------------------------------------------------------------------------


def test_reshape_apply_dense_headsplit():
    """The MHA head split: reshape(apply(aff(A,b),h), (T,nh,hd))
    sinks into the map — A's input axis stays last."""
    T, nh, hd, i = 8, 4, 3, 5
    o = nh * hd
    A, b, h = _V("A", (T, o, i)), _V("b", (T, o)), _V("h", (i,))
    env = {
        A: _rand((T, o, i), 1),
        b: _rand((T, o), 2),
        h: _rand((i,), 3),
    }
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, b), h),
        shape=(T, nh, hd),
    )
    out = _law(XC.XC_RESHAPE_APPLY, t0, env)
    # the minted form really is apply(aff(reshape A, reshape b), h)
    assert out.op == "apply" and out.args[0].op == "aff"
    amap = out.args[0].args[0]
    assert amap.op == "reshape" and tuple(amap.attrs["shape"]) == (
        T,
        nh,
        hd,
        i,
    )
    # reverse refolds
    back = _law(XC.XC_RESHAPE_APPLY_REV, out, env)


def test_reshape_apply_dense_merge():
    """Any well-typed value reshape works — also merging axes back:
    (T,nh,hd) -> (T, nh*hd)."""
    T, nh, hd, i = 8, 4, 3, 5
    A, b, h = (
        _V("A", (T, nh, hd, i)),
        _V("b", (T, nh, hd)),
        _V("h", (i,)),
    )
    env = {
        A: _rand((T, nh, hd, i), 4),
        b: _rand((T, nh, hd), 5),
        h: _rand((i,), 6),
    }
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, b), h),
        shape=(T, nh * hd),
    )
    _law(XC.XC_RESHAPE_APPLY, t0, env)


def test_reshape_apply_dense_minus1():
    """Exported views can carry a literal -1 — it resolves against the
    value numel and is carried verbatim onto the map's reshape."""
    T, nh, hd, i = 8, 4, 3, 5
    A, b, h = (
        _V("A", (T, nh * hd, i)),
        _V("b", (T, nh * hd)),
        _V("h", (i,)),
    )
    env = {
        A: _rand((T, nh * hd, i), 7),
        b: _rand((T, nh * hd), 8),
        h: _rand((i,), 9),
    }
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, b), h),
        shape=(T, -1, hd),
    )
    _law(XC.XC_RESHAPE_APPLY, t0, env)


def test_reshape_apply_vetoes():
    """(i) a broadcast-smaller b cannot be reshaped like the value —
    vetoed; (ii) a numel-inconsistent shape is not a view — vetoed."""
    T, o, i = 8, 12, 5
    A, b, h = _V("A", (T, o, i)), _V("b", (T, o)), _V("h", (i,))
    bb = _V("bb", (o,))  # broadcast-smaller b
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, bb), h),
        shape=(T, 4, 3),
    )
    _no_fire(XC.XC_RESHAPE_APPLY, t0)
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, b), h),
        shape=(T, 5, 3),
    )  # 8*12 != 8*5*3
    _no_fire(XC.XC_RESHAPE_APPLY, t0)


def test_transpose_apply_dense():
    """The MHA head swap: transpose(apply(aff(A,b),h), 0, 1) sinks the
    permutation into the map's output axes — value dims are normalised
    mod the value rank so the map's input axis is never touched."""
    nh, T, hd, i = 4, 8, 3, 5
    A, b, h = (
        _V("A", (T, nh, hd, i)),
        _V("b", (T, nh, hd)),
        _V("h", (i,)),
    )
    env = {
        A: _rand((T, nh, hd, i), 10),
        b: _rand((T, nh, hd), 11),
        h: _rand((i,), 12),
    }
    t0 = Op.make(
        "transpose",
        Op.make("apply", Op.make("aff", A, b), h),
        arg1=0,
        arg2=1,
    )
    out = _law(XC.XC_TRANSPOSE_APPLY, t0, env)
    assert out.op == "apply" and out.args[0].op == "aff"
    # negative dims are normalised against the VALUE rank (3), not the
    # map rank (4): transpose(v, -3, -2) == transpose(v, 0, 1).
    t0 = Op.make(
        "transpose",
        Op.make("apply", Op.make("aff", A, b), h),
        arg1=-3,
        arg2=-2,
    )
    _law(XC.XC_TRANSPOSE_APPLY, t0, env)
    # reverse: the pushed-through member re-fuses.
    back = _law(XC.XC_TRANSPOSE_APPLY_REV, out, env)


def test_transpose_apply_rev_veto_input_axis():
    """A transpose that permutes the map's INPUT axis is not a value
    view — the reverse law must not claim it."""
    T, o, i = 8, 12, 5
    # transposing A's axis 2 <-> 3 mixes the input axis: vetoed.
    A, b, h = _V("A", (T, o, i)), _V("b", (T, o)), _V("h", (i,))
    t0 = Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("transpose", A, arg1=1, arg2=2),  # o <-> i !
            Op.make("transpose", b, arg1=0, arg2=1),
        ),
        h,
    )
    _no_fire(XC.XC_TRANSPOSE_APPLY_REV, t0)
    # ...and disagreement between the two coefficient views vetoes too
    t0 = Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("transpose", A, arg1=0, arg2=1),
            Op.make("transpose", b, arg1=1, arg2=0),
        ),
        h,
    )
    # dims {0,1} vs {1,0} are the SAME swap — this one is actually
    # legal; use a genuinely different pair for the veto:
    _law(
        XC.XC_TRANSPOSE_APPLY_REV,
        t0,
        {
            A: _rand((T, o, i), 13),
            b: _rand((T, o), 14),
            h: _rand((i,), 15),
        },
    )


# ---------------------------------------------------------------------------
#  B. The view-commute laws — diagonal fiber (restricted)
# ---------------------------------------------------------------------------


def test_transpose_applyd_head_swap():
    """Diagonal maps commute with head-axis transposes: the feature
    axis stays last, h is not permuted."""
    nh, T, d = 4, 7, 3
    a, b, h = _V("a", (T, nh, d)), _V("b", (T, nh, d)), _V("h", (d,))
    env = {
        a: _rand((T, nh, d), 20),
        b: _rand((T, nh, d), 21),
        h: _rand((d,), 22),
    }
    t0 = Op.make(
        "transpose",
        Op.make("applyd", Op.make("aff_diag", a, b), h),
        arg1=0,
        arg2=1,
    )
    out = _law(XC.XC_TRANSPOSE_APPLYD, t0, env)
    _law(XC.XC_TRANSPOSE_APPLYD_REV, out, env)


def test_transpose_applyd_veto_feature_axis():
    """Moving the feature axis is a DIFFERENT map (h's components would
    have to permute) — vetoed."""
    nh, T, d = 4, 7, 3
    a, b, h = _V("a", (nh, T, d)), _V("b", (nh, T, d)), _V("h", (d,))
    base = Op.make("applyd", Op.make("aff_diag", a, b), h)
    _no_fire(
        XC.XC_TRANSPOSE_APPLYD,
        Op.make("transpose", base, arg1=1, arg2=2),
    )
    _no_fire(
        XC.XC_TRANSPOSE_APPLYD,
        Op.make("transpose", base, arg1=-1, arg2=0),
    )


def test_reshape_applyd_feature_last():
    """Diagonal reshape is sound only when the resolved last axis
    stays d — head-PACKING (T,d)->(T,1,d) passes, head-SPLITTING
    (T,d)->(T,nh,hd) does NOT (d splits across axes and h can no
    longer broadcast — that path is the dense promotion's job)."""
    T, d, nh, hd = 8, 12, 4, 3
    a, b, h = _V("a", (T, d)), _V("b", (T, d)), _V("h", (d,))
    env = {
        a: _rand((T, d), 30),
        b: _rand((T, d), 31),
        h: _rand((d,), 32),
    }
    base = Op.make("applyd", Op.make("aff_diag", a, b), h)
    # packing / regrouping that keeps the feature axis intact
    t0 = Op.make("reshape", base, shape=(T, 1, d))
    out = _law(XC.XC_RESHAPE_APPLYD, t0, env)
    _law(XC.XC_RESHAPE_APPLYD_REV, out, env)
    t0 = Op.make("reshape", base, shape=(2, 4, d))
    _law(XC.XC_RESHAPE_APPLYD, t0, env)
    # UNSOUND forms — vetoed
    _no_fire(
        XC.XC_RESHAPE_APPLYD,
        Op.make("reshape", base, shape=(T, nh, hd)),
    )  # split d
    _no_fire(
        XC.XC_RESHAPE_APPLYD,
        Op.make("reshape", base, shape=(nh * hd, T)),
    )  # mix axes
    _no_fire(
        XC.XC_RESHAPE_APPLYD, Op.make("reshape", base, shape=(T * d,))
    )  # merge all


# ---------------------------------------------------------------------------
#  C. Batched dense om/omd — the rank-4 (per-head) map
# ---------------------------------------------------------------------------


def test_om_elem_aff_batched_heads():
    """om_elem(s, apply(aff(A,b),h)) with a PER-HEAD map A (nh,K,d,i):
    the head axis is a matmul batch dim in both the concrete and the
    fused element — fp64-identical."""
    nh, Tq, K, d, i = 4, 5, 7, 3, 6
    s, A, b, h = (
        _V("s", (nh, Tq, K)),
        _V("A", (nh, K, d, i)),
        _V("b", (nh, K, d)),
        _V("h", (i,)),
    )
    env = {
        s: _rand((nh, Tq, K), 40),
        A: _rand((nh, K, d, i), 41),
        b: _rand((nh, K, d), 42),
        h: _rand((i,), 43),
    }
    t0 = Op.make(
        "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
    )
    applied = meta.apply_rewrite_at(XC.XC_OM_ELEM_AFF, t0, ())
    assert applied is not None, "batched om_elem_aff vetoed"
    a = meta._eval_term(t0, env)
    bb = meta._eval_term(applied, env)
    assert all(
        meta._eval_allclose(x, y, tol=1e-12) for x, y in zip(a, bb)
    )
    # unfold direction too
    _law(XC.XC_OM_ELEM_AFF_REV, Op.make("om_elem_aff", s, A, b, h), env)


def test_om_elem_aff_veto_bad_batch():
    """Head counts that don't broadcast between s and A must veto."""
    nh, Tq, K, d, i = 4, 5, 7, 3, 6
    s, A, b, h = (
        _V("s", (nh, Tq, K)),
        _V("A", (nh + 1, K, d, i)),
        _V("b", (nh + 1, K, d)),
        _V("h", (i,)),
    )
    t0 = Op.make(
        "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
    )
    _no_fire(XC.XC_OM_ELEM_AFF, t0)


def test_omd_lift_dense_batched():
    """The deferred carrier over a per-head map: omd_applym of an
    omd_elem(s, A, b) with A (nh,K,d,i) — h contracts the last axis,
    heads ride along."""
    nh, Tq, K, d, i = 4, 5, 7, 3, 6
    s, A, b, h = (
        _V("s", (nh, Tq, K)),
        _V("A", (nh, K, d, i)),
        _V("b", (nh, K, d)),
        _V("h", (i,)),
    )
    env = {
        s: _rand((nh, Tq, K), 50),
        A: _rand((nh, K, d, i), 51),
        b: _rand((nh, K, d), 52),
        h: _rand((i,), 53),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
        ),
    )
    out = _law(XC.XC_OMD_LIFT_DENSE, t0, env, tol=1e-12)
    assert out.op == "omd_applym"
    # the deferred numerator evaluates to softmax(s) @ (A@h+b)
    assert meta._eval_allclose(
        meta._eval_term(out, env), meta._eval_term(t0, env), tol=1e-12
    )
    _law(XC.XC_OMD_UNLIFT_DENSE, out, env)


def test_omd_compose_batched_pair():
    """Two head-batched deferred elems compose under omd_compose and
    the application stays exact — the MHA chunked-attention shape."""
    nh, Tq, K1, K2, d, i = 2, 4, 3, 5, 3, 6
    s1, s2 = _V("s1", (nh, Tq, K1)), _V("s2", (nh, Tq, K2))
    A1, b1 = _V("A1", (nh, K1, d, i)), _V("b1", (nh, K1, d))
    A2, b2 = _V("A2", (nh, K2, d, i)), _V("b2", (nh, K2, d))
    h = _V("h", (i,))
    env = {
        s1: _rand((nh, Tq, K1), 60),
        s2: _rand((nh, Tq, K2), 61),
        A1: _rand((nh, K1, d, i), 62),
        b1: _rand((nh, K1, d), 63),
        A2: _rand((nh, K2, d, i), 64),
        b2: _rand((nh, K2, d), 65),
        h: _rand((i,), 66),
    }
    defer = Op.make(
        "omd_applym",
        Op.make(
            "omd_compose",
            Op.make("omd_elem", s1, A1, b1),
            Op.make("omd_elem", s2, A2, b2),
        ),
        h,
    )
    conc = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make(
                "om_elem",
                s1,
                Op.make("apply", Op.make("aff", A1, b1), h),
            ),
            Op.make(
                "om_elem",
                s2,
                Op.make("apply", Op.make("aff", A2, b2), h),
            ),
        ),
    )
    a = meta._eval_term(conc, env)
    bb = meta._eval_term(defer, env)
    assert meta._eval_allclose(a, bb, tol=1e-12)


# ---------------------------------------------------------------------------
#  D. End-to-end — ScanAttnMH: view+transpose head split over scanned
#     values, output projection.  The pre-fix gap: the om leaf's value
#     class held only reshape/transpose members, so _elem_affine_options
#     was empty and no omd member existed anywhere.
# ---------------------------------------------------------------------------


class _ScanAttnMH(torch.nn.Module):
    """h_t = a_t⊙h + x_t; v/q/k = W·(stack h) split into heads via
    view+transpose; causal scaled softmax attention; output proj.

    Small dims keep the e-graph tiny — T=8, D=16, nh=2, hd=8."""

    def __init__(self, T: int, D: int, nh: int, hd: int):
        super().__init__()
        self.nh, self.hd = nh, hd
        self.a = torch.nn.Parameter(torch.randn(T, D) * 0.1)
        self.h0 = torch.nn.Parameter(torch.randn(D) * 0.1)
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, nh * hd, bias=False)
        self.wv = torch.nn.Linear(D, nh * hd, bias=False)
        self.wo = torch.nn.Linear(nh * hd, D, bias=False)
        # A registered buffer, not a Python float — torch.export
        # serialises a bare float attribute through fp32 (a uniform
        # ~1e-8 output deviation; an export artifact, not a catopt
        # issue — same note as bench_omd2.ScanAttnMQA).
        self.register_buffer(
            "sq", torch.tensor(hd**-0.5, dtype=torch.float64)
        )
        mask = torch.zeros(T, T)
        mask.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
            float("-inf"),
        )
        self.register_buffer("cm", mask)

    def forward(self, x):
        T = x.shape[0]
        h = self.h0
        outs = []
        for t in range(x.shape[0]):
            h = self.a[t] * h + x[t]
            outs.append(h)
        hs = torch.stack(outs)
        v = self.wv(hs).view(T, self.nh, self.hd).transpose(0, 1)
        q = self.wq(x).view(T, self.nh, self.hd).transpose(0, 1)
        k = self.wk(x).view(T, self.nh, self.hd).transpose(0, 1)
        s = q @ k.transpose(-1, -2) * self.sq + self.cm
        o = torch.softmax(s, dim=-1) @ v
        return self.wo(o.transpose(0, 1).reshape(T, self.nh * self.hd))


def _find_path(t, pred, path=()):
    if isinstance(t, Op):
        if pred(t):
            return path
        for i, a in enumerate(t.args):
            r = _find_path(a, pred, path + (i,))
            if r is not None:
                return r
    return None


def _replace_path(t, path, sub):
    if not path:
        return sub
    i = path[0]
    args = list(t.args)
    args[i] = _replace_path(args[i], path[1:], sub)
    return Op.make(t.op, *args, **dict(t.attrs))


def test_mha_omd_fires_and_is_exact():
    """After the view-commute laws, the om leaf's value class gains an
    ``apply`` member under the reshape/transpose head split, and
    ``omd_applym`` lands in the om_apply class — nested (wo follows
    the attention, so omd is NOT at the root — same as the chunked-MH
    variant).  The extracted member evaluates fp64-exact."""
    from catopt.ir import IR
    from catopt.regime import default_rules
    from catopt.torch_bridge import export_to_ir, ir_to_torch_module
    from catopt.xcarrier import (
        XC_LAWS,
        _elem_affine_options,
        gather_apply_stack,
        gather_applyd_stack,
        omd_tree_lift,
    )

    torch.manual_seed(0)
    T, D, nh, hd = 8, 16, 2, 8
    m = _ScanAttnMH(T, D, nh, hd).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, src = export_to_ir(m, x)

    eg = EGraph()
    root = eg.add_term(ir.root)
    # bounded saturation: core carriers -> non-local lifts -> XC tier
    # -> lifts again (same shape as bench_omd2.bounded_build_xc).
    eg.run(default_rules(), root, max_iterations=6, max_nodes=300_000)
    lifts = (
        gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )
    if lifts:
        eg.rebuild()
    eg.run(XC_LAWS, root, max_iterations=5, max_nodes=300_000)
    more = (
        gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )
    if more:
        eg.rebuild()

    # ---- structural: the om leaf's value class gained an `apply` ----
    omd_classes = {}
    saw_affine_value = False
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for n in ec.nodes:
            if n.op == "om_elem" and len(n.children) == 2:
                opts = _elem_affine_options(eg, n.children[1])
                if opts:
                    saw_affine_value = True
            if n.op in ("omd_apply", "omd_applym"):
                omd_classes.setdefault(c, []).append(n)

    assert saw_affine_value, (
        "no om leaf sees an affine value member — the view-commute "
        "laws did not surface the apply"
    )
    assert omd_classes, "no omd member minted on ScanAttnMH"

    # ---- exactness: splice the omd member for the om_apply it
    # replaces, inside the original root term --------------------------
    mm_path = _find_path(
        ir.root,
        lambda t: (
            t.op == "matmul"
            and isinstance(t.args[0], Op)
            and t.args[0].op == "softmax"
        ),
    )
    assert mm_path is not None, "softmax@value matmul not in root term"

    cls = next(iter(omd_classes))
    node = sorted(omd_classes[cls], key=repr)[0]
    omd_term = Op.make(
        node.op,
        *(eg.any_term(eg.find(ch)) for ch in node.children),
        **dict(node.attrs),
    )
    assert node.op == "omd_applym"  # dense fiber — promoted Wv map

    new_root = _replace_path(ir.root, mm_path, omd_term)
    mod = ir_to_torch_module(
        IR(root=new_root, inputs=ir.inputs, params=ir.params), src
    )
    with torch.no_grad():
        ref = m(x)
        out = mod(x)
    err = (out - ref).abs().max().item()
    assert err < 1e-12, f"omd member deviates: max|Δ|={err:.2e}"


def test_mha_omd_value_member_is_viewed_apply():
    """The surfaced member has the exact MHA shape: the value class of
    the om leaf contains apply(aff(map,h)) whose A part is a
    transpose/reshape view of the projected scan coefficients —
    rank-4 (nh,T,hd,D)."""
    from catopt.cost import _shape_of
    from catopt.regime import default_rules
    from catopt.torch_bridge import export_to_ir
    from catopt.xcarrier import (
        XC_LAWS,
        _elem_affine_options,
        gather_apply_stack,
        gather_applyd_stack,
        omd_tree_lift,
    )

    torch.manual_seed(0)
    T, D, nh, hd = 8, 16, 2, 8
    m = _ScanAttnMH(T, D, nh, hd).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, src = export_to_ir(m, x)

    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(default_rules(), root, max_iterations=6, max_nodes=300_000)
    lifts = (
        gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )
    if lifts:
        eg.rebuild()
    eg.run(XC_LAWS, root, max_iterations=5, max_nodes=300_000)
    eg.rebuild()

    found_rank4 = False
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for n in ec.nodes:
            if n.op == "om_elem" and len(n.children) == 2:
                for h_eid, opts in _elem_affine_options(
                    eg, n.children[1]
                ).items():
                    for kind, a_eid, b_eid in opts:
                        if kind != "dense":
                            continue
                        at = eg.any_term(a_eid)
                        s = _shape_of(at)
                        if (
                            isinstance(s, tuple)
                            and len(s) == 4
                            and s[0] == nh
                            and s[-1] == D
                        ):
                            found_rank4 = True
    assert found_rank4, (
        "expected a per-head dense map (nh,T,hd,D) in the value "
        "class — the view laws did not produce it"
    )
