"""Fused level-steps for the online-softmax carriers (om_fused).

:mod:`catopt_carriers.om_fused` provides the om/omd analogues of the
scan fused bodies: ``fused_om_levels`` / ``fused_omd_levels`` /
``fused_omdm_levels`` re-bracket a stacked leaf carrier sequence into
the canonical adjacent-pair reduction (identity-padded to a power of
two) — the shape Inductor fuses wholesale.  These tests pin fp64
equivalence with the serial IRModule and the batched executors on the
same terms, the pad/mask/DAG-occurrence edges, the compiled path, and
an end-to-end run on an egraph-extracted om term.
"""

import catopt_carriers.xcarrier  # noqa: F401 — omd_* torch bindings
import pytest
import torch
from catopt.cost import flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.om import OM_LAWS
from catopt.om_lower import build_om_plan, to_batched_om_module
from catopt.omd_lower import build_omd_plan
from catopt.torch_bridge import ir_to_torch_module
from catopt_carriers.om_fused import (
    fused_om_levels,
    fused_omd_levels,
    fused_omdm_levels,
    occurrence_slots,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _compose(opname, leaves):
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        opname,
        _compose(opname, leaves[:mid]),
        _compose(opname, leaves[mid:]),
    )


def _qk_leaf(q, k, v):
    """om_elem(q @ k.T, v) — the leaf shape OM_SPLIT extraction emits."""
    s = Op.make("matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1))
    return Op.make("om_elem", s, v)


def _om_ir(q, ks, vs):
    leaves = [_qk_leaf(q, k, v) for k, v in zip(ks, vs, strict=True)]
    root = Op.make("om_apply", _compose("om_compose", leaves))
    inputs = [q, *ks, *vs]
    return IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )


def _env(mod, xs):
    x = xs[0] if xs else None
    env = {"self": x}
    for i, inp in enumerate(mod._inputs):
        env[inp.name] = xs[i] if i < len(xs) else x
    return x, env


def _fused_om_apply(ir, xs, pv=None):
    """The om_lower fused wiring, standalone: eval each leaf carrier
    serially, stack components in occurrence order, reduce through
    ``fused_om_levels``."""
    mod = ir_to_torch_module(ir, param_values=pv)
    plan = build_om_plan(mod._root)
    assert plan is not None
    x, env = _env(mod, xs)
    memo = {}
    trips = [mod._eval(lf, env, x, memo) for lf in plan["leaves"]]
    m = torch.stack([t[0] for t in trips])
    l_ = torch.stack([t[1] for t in trips])
    a = torch.stack([t[2] for t in trips])
    occ = occurrence_slots(plan["f"], plan["leaves"])
    assert occ is not None
    if occ != list(range(len(occ))):
        idx = torch.tensor(occ, dtype=torch.long)
        m, l_, a = m[idx], l_[idx], a[idx]
    return fused_om_levels(m, l_, a)


def _fused_omd_apply(ir, xs, pv=None):
    """The omd analogue: leaf 4-tuples → fused_omd[m]_levels."""
    mod = ir_to_torch_module(ir, param_values=pv)
    plan = build_omd_plan(mod._root)
    assert plan is not None
    x, env = _env(mod, xs)
    memo = {}
    quads = [mod._eval(lf, env, x, memo) for lf in plan["omd_leaves"]]
    parts = [torch.stack([t[i] for t in quads]) for i in range(4)]
    occ = occurrence_slots(mod._root.args[0], plan["omd_leaves"])
    assert occ is not None
    if occ != list(range(len(occ))):
        idx = torch.tensor(occ, dtype=torch.long)
        parts = [p[idx] for p in parts]
    h = mod._eval(mod._root.args[1], env, x, memo)
    fn = (
        fused_omdm_levels
        if plan["apply_op"] == "omd_applym"
        else fused_omd_levels
    )
    return fn(*parts, h)


# ---------------------------------------------------------------------------
#  om carrier — fused_om_levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_blocks", [1, 2, 4, 5, 8])
def test_fused_om_matches_serial(n_blocks):
    """fp64: fused_om_levels agrees with the tuple-passing IRModule and
    the dense softmax reference — including non-pow2 leaf counts that
    exercise the (-inf, 0, 0) identity pad."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 2, 16, 8, 8
    ksizes = [8] * n_blocks
    q = _v("q", B, H, T, d)
    ks = [_v(f"k{i}", B, H, k, d) for i, k in enumerate(ksizes)]
    vs = [_v(f"v{i}", B, H, k, dv) for i, k in enumerate(ksizes)]
    ir = _om_ir(q, ks, vs)

    serial = ir_to_torch_module(ir).eval()
    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [
        torch.randn(B, H, k, dv, dtype=torch.float64) for k in ksizes
    ]
    xs = (tq, *tks, *tvs)
    with torch.no_grad():
        out = _fused_om_apply(ir, xs)
        ref_ser = serial(*xs)
    assert (out - ref_ser).abs().max().item() < 1e-12
    ref = torch.softmax(
        tq @ torch.cat(tks, -2).transpose(-2, -1), dim=-1
    ) @ torch.cat(tvs, -2)
    assert (out - ref).abs().max().item() < 1e-12


def test_fused_om_matches_batched_module():
    """Same term through BatchedOMModule (itself a canonical
    reduction) — cross-schedule agreement at fp64."""
    from catopt.om_lower import to_batched_om_module

    torch.manual_seed(0)
    B, H, T, d, dv = 1, 2, 8, 4, 6
    ksizes = [4] * 6
    q = _v("q", B, H, T, d)
    ks = [_v(f"k{i}", B, H, k, d) for i, k in enumerate(ksizes)]
    vs = [_v(f"v{i}", B, H, k, dv) for i, k in enumerate(ksizes)]
    ir = _om_ir(q, ks, vs)
    bat = to_batched_om_module(ir).eval()
    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [
        torch.randn(B, H, k, dv, dtype=torch.float64) for k in ksizes
    ]
    xs = (tq, *tks, *tvs)
    with torch.no_grad():
        diff = (_fused_om_apply(ir, xs) - bat(*xs)).abs().max().item()
    assert diff < 1e-12


def test_fused_om_masked_rows_nan():
    """A row masked in EVERY block is NaN (dense-softmax semantics); a
    fully-masked leaf contributes exactly the identity."""
    torch.manual_seed(0)
    B, H, T, dv, n, K = 1, 2, 8, 4, 4, 4
    ss = [_v(f"s{i}", B, H, T, K) for i in range(n)]
    vs = [_v(f"v{i}", B, H, K, dv) for i in range(n)]
    leaves = [
        Op.make("om_elem", s, v) for s, v in zip(ss, vs, strict=True)
    ]
    ir = IR(
        root=Op.make("om_apply", _compose("om_compose", leaves)),
        inputs=ss + vs,
        input_names={v.name for v in ss + vs},
        params={},
    )
    serial = ir_to_torch_module(ir).eval()
    tss = [
        torch.randn(B, H, T, K, dtype=torch.float64) for _ in range(n)
    ]
    tvs = [
        torch.randn(B, H, K, dv, dtype=torch.float64) for _ in range(n)
    ]
    for t in tss:
        t[..., 5, :] = float("-inf")  # row 5 fully masked everywhere
    tss[1][..., 0, :] = float("-inf")  # row 0 masked in block 1 only
    with torch.no_grad():
        out = _fused_om_apply(ir, (*tss, *tvs))
        ref = serial(*tss, *tvs)
        dense = torch.softmax(torch.cat(tss, -1), -1) @ torch.cat(
            tvs, -2
        )
    assert torch.equal(torch.isnan(out), torch.isnan(dense))
    assert torch.isnan(out[..., 5, :]).all()
    diff = (out - ref).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    assert diff.max().item() < 1e-12


def test_fused_om_dag_shared_leaf():
    """``om_compose(leaf, leaf)`` — one leaf slot, two occurrences; the
    fused reduction gathers by occurrence order."""
    torch.manual_seed(0)
    B, H, T, K, dv = 1, 1, 4, 3, 5
    s, v = _v("s", B, H, T, K), _v("v", B, H, K, dv)
    leaf = Op.make("om_elem", s, v)
    ir = IR(
        root=Op.make("om_apply", Op.make("om_compose", leaf, leaf)),
        inputs=[s, v],
        input_names={"s", "v"},
        params={},
    )
    mod = ir_to_torch_module(ir).eval()
    plan = build_om_plan(mod._root)
    assert len(plan["leaves"]) == 1
    assert occurrence_slots(plan["f"], plan["leaves"]) == [0, 0]
    ts = torch.randn(B, H, T, K, dtype=torch.float64)
    tv = torch.randn(B, H, K, dv, dtype=torch.float64)
    with torch.no_grad():
        out = _fused_om_apply(ir, (ts, tv))
        ref = mod(ts, tv)
    # doubled block = softmax over the scores with v counted twice —
    # i.e. softmax over duplicated columns.
    dense = torch.softmax(torch.cat([ts, ts], -1), -1) @ torch.cat(
        [tv, tv], -2
    )
    assert (out - ref).abs().max().item() < 1e-12
    assert (out - dense).abs().max().item() < 1e-12


def test_fused_om_compiled_fullgraph():
    """``torch.compile(fullgraph=True)`` over fused_om_levels — the
    deployment shape — agrees with both the eager body and serial."""
    torch.manual_seed(0)
    n, T, dv = 6, 4, 5
    m = torch.randn(n, T, 1, dtype=torch.float64)
    l_ = torch.rand(n, T, 1, dtype=torch.float64) + 0.5
    a = torch.randn(n, T, dv, dtype=torch.float64)
    fused_c = torch.compile(fused_om_levels, fullgraph=True)
    out_c = fused_c(m, l_, a)
    out_e = fused_om_levels(m, l_, a)
    assert (out_c - out_e).abs().max().item() < 1e-12
    # serial fold reference over the same leaf triples
    m1, l1, a1 = m[0], l_[0], a[0]
    for i in range(1, n):
        m2, l2, a2 = m[i], l_[i], a[i]
        mx = torch.maximum(m1, m2)
        e1 = torch.exp(m1 - mx)
        e2 = torch.exp(m2 - mx)
        m1, l1, a1 = mx, l1 * e1 + l2 * e2, a1 * e1 + a2 * e2
    assert (out_c - a1 / l1).abs().max().item() < 1e-12


def test_fused_om_extracted_term():
    """End-to-end: saturate a dense chunked-attention term with
    OM_LAWS, force-extract the om member, run the fused path."""

    def _class_has_op(eg, eid, opname, seen=None):
        seen = set() if seen is None else seen
        eid = eg.find(eid)
        if eid in seen:
            return False
        seen.add(eid)
        for n in eg.get_class(eid).nodes:
            if n.op == opname:
                return True
            if any(
                _class_has_op(eg, c, opname, seen) for c in n.children
            ):
                return True
        return False

    torch.manual_seed(0)
    B, H, T, d, dv = 1, 1, 8, 4, 6
    ksizes = [4, 4]
    q = _v("q", B, H, T, d)
    ks = [_v(f"k{i}", B, H, k, d) for i, k in enumerate(ksizes)]
    vs = [_v(f"v{i}", B, H, k, dv) for i, k in enumerate(ksizes)]
    kcat = Op.make("concat", ks[0], ks[1], dim=-2)
    vcat = Op.make("concat", vs[0], vs[1], dim=-2)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, dim0=-2, dim1=-1)
    )
    term = Op.make("matmul", Op.make("softmax", scores, dim=-1), vcat)

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=20, max_nodes=200_000)
    canon = eg.find(root)
    ov = {}
    for cid in list(eg._classes):
        c = eg.find(cid)
        comps = [
            n for n in eg._classes[c].nodes if n.op == "om_compose"
        ]
        if comps:
            ov.setdefault(c, comps[0])
    chunked = None
    for n in eg.get_class(canon).nodes:
        if n.op == "om_apply" and _class_has_op(
            eg, n.children[0], "om_compose"
        ):
            o = dict(ov)
            o[canon] = n
            chunked = eg.extract_best(canon, flops_cost, overrides=o)
            if chunked is not None:
                break
    assert chunked is not None

    inputs = [q, *ks, *vs]
    ir = IR(
        root=chunked,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    serial = ir_to_torch_module(ir).eval()
    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [
        torch.randn(B, H, k, dv, dtype=torch.float64) for k in ksizes
    ]
    xs = (tq, *tks, *tvs)
    with torch.no_grad():
        out = _fused_om_apply(ir, xs)
        ref = serial(*xs)
    dense = torch.softmax(
        tq @ torch.cat(tks, -2).transpose(-2, -1), dim=-1
    ) @ torch.cat(tvs, -2)
    assert (out - ref).abs().max().item() < 1e-10
    assert (out - dense).abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
#  omd carrier — fused_omd_levels (diag) / fused_omdm_levels (dense)
# ---------------------------------------------------------------------------


def _omd_ir(n, Tq, K, d, dense=False, di=None):
    """omd_apply[m](omd_compose tree over omd_elem(s_i, a_i, b_i), h)."""
    di = d if di is None else di
    ss = [_v(f"s{i}", Tq, K) for i in range(n)]
    if dense:
        aa = [_p(f"a{i}", K, d, di) for i in range(n)]
    else:
        aa = [_p(f"a{i}", K, d) for i in range(n)]
    bb = [_p(f"b{i}", K, d) for i in range(n)]
    h = _p("h", di)
    leaves = [
        Op.make("omd_elem", s, a, b)
        for s, a, b in zip(ss, aa, bb, strict=True)
    ]
    op = "omd_applym" if dense else "omd_apply"
    root = Op.make(op, _compose("omd_compose", leaves), h)
    ir = IR(
        root=root,
        inputs=ss,
        input_names={v.name for v in ss},
        params={},
    )
    return ir, ss, aa, bb, h


@pytest.mark.parametrize("n_blocks", [1, 2, 4, 5])
def test_fused_omd_diag_matches_serial(n_blocks):
    """Diag fiber: (fa⊙h + fb)/l agrees with the serial evaluator and
    the dense softmax reference at fp64."""
    torch.manual_seed(0)
    Tq, K, d = 4, 3, 6
    ir, ss, aa, bb, _h = _omd_ir(n_blocks, Tq, K, d)
    tss = [torch.randn(Tq, K, dtype=torch.float64) for _ in ss]
    pv = {
        **{a.name: torch.randn(K, d, dtype=torch.float64) for a in aa},
        **{b.name: torch.randn(K, d, dtype=torch.float64) for b in bb},
        "h": torch.randn(d, dtype=torch.float64),
    }
    serial = ir_to_torch_module(ir, param_values=pv).eval()
    with torch.no_grad():
        out = _fused_omd_apply(ir, tuple(tss), pv)
        ref_ser = serial(*tss)
    # value per key: v_k = a_k ⊙ h + b_k
    a_cat = torch.cat([pv[a.name] for a in aa], dim=0)
    b_cat = torch.cat([pv[b.name] for b in bb], dim=0)
    v = a_cat * pv["h"] + b_cat
    ref = torch.softmax(torch.cat(tss, -1), -1) @ v
    assert (out - ref_ser).abs().max().item() < 1e-12
    assert (out - ref).abs().max().item() < 1e-12


@pytest.mark.parametrize("n_blocks", [2, 5])
def test_fused_omdm_dense_matches_serial(n_blocks):
    """Dense fiber: (fa@h + fb)/l — the reshape-broadcast on fa's
    extra trailing axis — agrees with serial at fp64."""
    torch.manual_seed(0)
    Tq, K, do, di = 4, 3, 5, 6
    ir, ss, aa, bb, _h = _omd_ir(n_blocks, Tq, K, do, dense=True, di=di)
    tss = [torch.randn(Tq, K, dtype=torch.float64) for _ in ss]
    pv = {
        **{
            a.name: torch.randn(K, do, di, dtype=torch.float64) * 0.3
            for a in aa
        },
        **{b.name: torch.randn(K, do, dtype=torch.float64) for b in bb},
        "h": torch.randn(di, dtype=torch.float64),
    }
    serial = ir_to_torch_module(ir, param_values=pv).eval()
    with torch.no_grad():
        out = _fused_omd_apply(ir, tuple(tss), pv)
        ref_ser = serial(*tss)
    a_cat = torch.cat([pv[a.name] for a in aa], dim=0)  # (nK, do, di)
    b_cat = torch.cat([pv[b.name] for b in bb], dim=0)
    v = a_cat @ pv["h"] + b_cat
    ref = torch.softmax(torch.cat(tss, -1), -1) @ v
    assert (out - ref_ser).abs().max().item() < 1e-12
    assert (out - ref).abs().max().item() < 1e-12


def test_fused_omd_masked_block():
    """A fully-masked leaf (m = -inf, NaN payloads) drops out of the
    product — the hoisted sanitisation matches the serial where()."""
    torch.manual_seed(0)
    Tq, K, d = 4, 3, 5
    n = 4
    ir, ss, aa, bb, _h = _omd_ir(n, Tq, K, d)
    tss = [torch.randn(Tq, K, dtype=torch.float64) for _ in ss]
    tss[1][:, :] = float("-inf")  # block 1 fully masked
    pv = {
        **{a.name: torch.randn(K, d, dtype=torch.float64) for a in aa},
        **{b.name: torch.randn(K, d, dtype=torch.float64) for b in bb},
        "h": torch.randn(d, dtype=torch.float64),
    }
    serial = ir_to_torch_module(ir, param_values=pv).eval()
    with torch.no_grad():
        out = _fused_omd_apply(ir, tuple(tss), pv)
        ref_ser = serial(*tss)
    a_cat = torch.cat(
        [pv[a.name] for i, a in enumerate(aa) if i != 1], dim=0
    )
    b_cat = torch.cat(
        [pv[b.name] for i, b in enumerate(bb) if i != 1], dim=0
    )
    s_cat = torch.cat([t for i, t in enumerate(tss) if i != 1], -1)
    v = a_cat * pv["h"] + b_cat
    ref = torch.softmax(s_cat, -1) @ v
    assert not torch.isnan(out).any()
    assert (out - ref_ser).abs().max().item() < 1e-12
    assert (out - ref).abs().max().item() < 1e-12


def test_fused_omd_dag_shared_leaf():
    """omd_compose(leaf, leaf): occurrence-expanded reduction."""
    torch.manual_seed(0)
    Tq, K, d = 3, 2, 4
    s, a, b, h = (
        _v("s", Tq, K),
        _p("a", K, d),
        _p("b", K, d),
        _p("h", d),
    )
    leaf = Op.make("omd_elem", s, a, b)
    root = Op.make("omd_apply", Op.make("omd_compose", leaf, leaf), h)
    ir = IR(root=root, inputs=[s], input_names={"s"}, params={})
    pv = {
        "a": torch.randn(K, d, dtype=torch.float64),
        "b": torch.randn(K, d, dtype=torch.float64),
        "h": torch.randn(d, dtype=torch.float64),
    }
    mod = ir_to_torch_module(ir, param_values=pv).eval()
    plan = build_omd_plan(mod._root)
    assert len(plan["omd_leaves"]) == 1
    assert occurrence_slots(mod._root.args[0], plan["omd_leaves"]) == [
        0,
        0,
    ]
    ts = torch.randn(Tq, K, dtype=torch.float64)
    with torch.no_grad():
        out = _fused_omd_apply(ir, (ts,), pv)
        ref = mod(ts)
    assert (out - ref).abs().max().item() < 1e-12


def test_fused_omd_compiled_fullgraph():
    """Compile both omd apply flavours under fullgraph=True."""
    torch.manual_seed(0)
    n, Tq, d, di = 6, 4, 5, 3
    m = torch.randn(n, Tq, 1, dtype=torch.float64)
    l_ = torch.rand(n, Tq, 1, dtype=torch.float64) + 0.5
    fa = torch.randn(n, Tq, d, dtype=torch.float64)
    fb = torch.randn(n, Tq, d, dtype=torch.float64)
    h = torch.randn(d, dtype=torch.float64)
    out_c = torch.compile(fused_omd_levels, fullgraph=True)(
        m, l_, fa, fb, h
    )
    out_e = fused_omd_levels(m, l_, fa, fb, h)
    assert (out_c - out_e).abs().max().item() < 1e-12

    fam = torch.randn(n, Tq, d, di, dtype=torch.float64)
    hm = torch.randn(di, dtype=torch.float64)
    out_cm = torch.compile(fused_omdm_levels, fullgraph=True)(
        m, l_, fam, fb, hm
    )
    out_em = fused_omdm_levels(m, l_, fam, fb, hm)
    assert (out_cm - out_em).abs().max().item() < 1e-12
    # root readout shape: (Tq, d) / (Tq, do)
    assert out_c.shape == (Tq, d)
    assert out_cm.shape == (Tq, d)


def test_fused_om_single_leaf_direct():
    """n=1: the reduction never enters the loop; apply is a/l."""
    torch.manual_seed(0)
    T, dv = 4, 3
    m = torch.randn(1, T, 1, dtype=torch.float64)
    l_ = torch.rand(1, T, 1, dtype=torch.float64) + 0.5
    a = torch.randn(1, T, dv, dtype=torch.float64)
    out = fused_om_levels(m, l_, a)
    assert torch.allclose(out, a[0] / l_[0])


# ---------------------------------------------------------------------------
#  Module wiring — to_batched_om_module / to_batched_omd_module fused=
# ---------------------------------------------------------------------------


def _om_small():
    """A small chunked-attention om IR + inputs."""
    torch.manual_seed(0)
    q = Var("q", _T(1, 2, 4, 8))
    ks = [Var(f"k{i}", _T(1, 2, 8, 8)) for i in range(4)]
    vs = [Var(f"v{i}", _T(1, 2, 8, 8)) for i in range(4)]
    ir = _om_ir(q, ks, vs)
    xs = [
        torch.randn(*v.typ.shape, dtype=torch.float64)
        for v in [q, *ks, *vs]
    ]
    return ir, xs


def test_batched_om_module_fused_matches_serial():
    """BatchedOMModule(fused=...) matches serial fp64-exactly in all
    modes — the wiring path, not just the standalone function."""
    ir, xs = _om_small()
    serial = ir_to_torch_module(ir)
    for fused in (False, "eager", True):
        mod = to_batched_om_module(ir, fused=fused)
        assert mod.is_batched
        with torch.no_grad():
            diff = (mod(*xs) - serial(*xs)).abs().max().item()
        assert diff < 1e-12


def test_batched_om_module_fused_validation_and_fallbacks():
    ir, xs = _om_small()
    # invalid fused value
    with pytest.raises(ValueError, match="fused must be"):
        to_batched_om_module(ir, fused="bogus")
    # fused on a non-om root: identity — generic path, no fused state
    plain = IR(
        root=Op.make("neg", Var("z", _T(4))),
        inputs=[Var("z", _T(4))],
        input_names={"z"},
        params={},
    )
    mod = to_batched_om_module(plain, fused=True)
    assert not mod.is_batched and mod._fused is None
    # compile-failure fallback: eager fused body still correct
    mod2 = to_batched_om_module(ir, fused=True)
    mod2._fused_c = lambda *a: (_ for _ in ()).throw(RuntimeError("x"))
    with torch.no_grad():
        assert torch.allclose(mod2(*xs), ir_to_torch_module(ir)(*xs))
    assert mod2._fused_compile_failed


def test_batched_omd_module_fused_matches_serial():
    """BatchedOmdModule(fused=...) on both fibers, fp64-exact."""
    from catopt_carriers.omd_lower import to_batched_omd_module

    for dense in (False, True):
        ir, ss, aa, bb, h = _omd_ir(4, Tq=4, K=3, d=6, dense=dense)
        xs = [
            torch.randn(*s.typ.shape, dtype=torch.float64) for s in ss
        ]
        pv = {
            a.name: torch.randn(*a.typ.shape, dtype=torch.float64)
            for a in aa
        }
        pv.update(
            {
                b.name: torch.randn(*b.typ.shape, dtype=torch.float64)
                for b in bb
            }
        )
        pv["h"] = torch.randn(*h.typ.shape, dtype=torch.float64)
        serial = ir_to_torch_module(ir, param_values=pv)
        for fused in (False, "eager", True):
            mod = to_batched_omd_module(
                ir, param_values=pv, fused=fused
            )
            assert mod.is_batched
            with torch.no_grad():
                diff = (mod(*xs) - serial(*xs)).abs().max().item()
            assert diff < 1e-12


def test_batched_omd_module_fused_edges():
    """fused validation, non-plan decline, and the occ-decline
    fallback path are all covered."""
    from catopt_carriers.omd_lower import to_batched_omd_module

    ir, ss, aa, bb, h = _omd_ir(4, Tq=4, K=3, d=6)
    pv = {
        a.name: torch.randn(*a.typ.shape, dtype=torch.float64)
        for a in aa
    }
    pv.update(
        {
            b.name: torch.randn(*b.typ.shape, dtype=torch.float64)
            for b in bb
        }
    )
    pv["h"] = torch.randn(*h.typ.shape, dtype=torch.float64)
    xs = [torch.randn(*s.typ.shape, dtype=torch.float64) for s in ss]
    with pytest.raises(ValueError, match="fused must be"):
        to_batched_omd_module(ir, param_values=pv, fused="bogus")
    # fused on a non-omd root: no plan → fused stays off
    plain = IR(
        root=Op.make("neg", Var("z", _T(4))),
        inputs=[Var("z", _T(4))],
        input_names={"z"},
        params={},
    )
    mod = to_batched_omd_module(plain, fused=True)
    assert not mod.is_batched and mod._fused is None
    # occ decline → the uniform path still runs correctly via the
    # standard level schedule
    mod2 = to_batched_omd_module(ir, param_values=pv, fused="eager")
    mod2._fused_occ = None  # simulate occurrence_slots decline
    with torch.no_grad():
        diff = (
            (mod2(*xs) - ir_to_torch_module(ir, param_values=pv)(*xs))
            .abs()
            .max()
            .item()
        )
    assert diff < 1e-12
    assert mod2.fallbacks == 1  # the occ decline counted it
