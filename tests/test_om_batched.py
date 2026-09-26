"""Level-batched lowering for the online-softmax monoid.

The e-graph discovers that ``softmax(q @ cat(k_i).T) @ cat(v_i)`` is a
product in the (m, l, a) carrier and extracts a chunked
``om_apply(om_compose-tree of om_elem)`` term.  The generic IRModule
evaluates that tree serially via tuple passing;
:class:`catopt.om_lower.BatchedOMModule` instead batches every om_elem
leaf group into one amax/exp/sum/bmm sequence and every compose level
into one batched FlashAttention combine — a few kernels per level
instead of per node.
"""

import math

import pytest
import torch

from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, TensorType, op_repr
from catopt.om import OM_LAWS
from catopt.cost import flops_cost
from catopt.om_lower import (
    BatchedOMModule,
    build_om_plan,
    is_om_apply_term,
    to_batched_om_module,
)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.models import SwiGLU


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _dense_ref(q, ks, vs):
    s = q @ torch.cat(list(ks), dim=-2).transpose(-2, -1)
    return torch.softmax(s, dim=-1) @ torch.cat(list(vs), dim=-2)


def _qk_leaf(q, k, v):
    """om_elem(q @ k.T, v) — the leaf shape OM_SPLIT extraction emits."""
    s = Op.make("matmul", q,
                Op.make("transpose", k, arg1=-2, arg2=-1))
    return Op.make("om_elem", s, v)


def _compose_tree(leaves):
    """Balanced binary om_compose bracketing (chunk order preserved)."""
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make("om_compose",
                   _compose_tree(leaves[:mid]),
                   _compose_tree(leaves[mid:]))


def _om_ir(q, ks, vs, leaf_fn=_qk_leaf):
    """om_apply(balanced compose tree over per-block om_elem leaves)."""
    leaves = [leaf_fn(q, k, v) for k, v in zip(ks, vs)]
    root = Op.make("om_apply", _compose_tree(leaves))
    inputs = [q] + list(ks) + list(vs)
    return IR(root=root, inputs=inputs,
              input_names={v.name for v in inputs}, params={})


def _qk_vars(B, H, T, d, dv, ksizes):
    q = Var("q", TensorType((B, H, T, d)))
    ks = [Var(f"k{i}", TensorType((B, H, k, d)))
          for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType((B, H, k, dv)))
          for i, k in enumerate(ksizes)]
    return q, ks, vs


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


def _nested_cat(ts, dim):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, dim=dim)
    return out


def _dense_chunked_term(q, ks, vs):
    kcat = _nested_cat(ks, -2)
    vcat = _nested_cat(vs, -2)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, arg1=-2, arg2=-1))
    return Op.make(
        "matmul", Op.make("softmax", scores, arg1=-1), vcat)


def _extract_chunked(eg, root, cost_fn=flops_cost):
    """Force-extract a chunked carrier term (same approach as
    test_om_monoid: per-class greedy never picks the split form)."""
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


# ---------------------------------------------------------------------------
#  Numerical equivalence vs dense softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_blocks", [4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batched_om_matches_dense(n_blocks, dtype):
    """om_apply(compose over om_elem(q@k_i.T, v_i)) lowered level-batched
    equals dense softmax(q @ cat(k).T) @ cat(v), B=2 H=4 T=256 d=64."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 4, 256, 64, 64
    K = T // (n_blocks if T % n_blocks == 0 else 1)
    ksizes = [K] * n_blocks
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    ir = _om_ir(q, ks, vs)

    mod = to_batched_om_module(ir)
    assert mod.is_batched
    assert mod.n_blocks == n_blocks
    assert mod.n_levels <= math.ceil(math.log2(n_blocks)) + 1

    tq = torch.randn(B, H, T, d, dtype=dtype)
    tks = [torch.randn(B, H, k, d, dtype=dtype) for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=dtype) for k in ksizes]
    ref = _dense_ref(tq, tks, tvs)
    mod.eval()
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    tol = 1e-10 if dtype == torch.float64 else 2e-5
    assert (out - ref).abs().max().item() < tol


def test_batched_om_matches_serial_lowering():
    """Batched execution agrees with the tuple-passing IRModule on the
    identical term (guards against a slot-ordering slip)."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 4, 64, 32, 48
    ksizes = [16] * 4
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    ir = _om_ir(q, ks, vs)

    serial = ir_to_torch_module(ir).eval()
    batched = to_batched_om_module(ir).eval()
    assert batched.is_batched
    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=torch.float64)
           for k in ksizes]
    with torch.no_grad():
        diff = (serial(tq, *tks, *tvs)
                - batched(tq, *tks, *tvs)).abs().max().item()
    assert diff < 1e-12


def test_mixed_block_sizes():
    """Non-uniform key-block sizes: leaves split into per-shape groups
    (some singletons) — still exact."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 2, 32, 16, 24
    ksizes = [8, 16, 8, 32]
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    ir = _om_ir(q, ks, vs)
    mod = to_batched_om_module(ir)
    assert mod.is_batched
    # two shape groups: (K=8) x2, plus singletons for K=16 and K=32
    sizes = sorted(len(g["members"]) for g in mod._plan["leaf_groups"])
    assert sizes == [1, 1, 2]

    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=torch.float64)
           for k in ksizes]
    ref = _dense_ref(tq, tks, tvs)
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-10


def test_score_vars_directly():
    """Leaves may carry raw score tensors (om_elem(s_i, v_i)) — the
    elem level still batches by stacking them."""
    torch.manual_seed(0)
    B, H, T, dv = 2, 3, 48, 16
    ksizes = [16] * 6
    ss = [Var(f"s{i}", TensorType((B, H, T, k)))
          for i, k in enumerate(ksizes)]
    vs = [Var(f"v{i}", TensorType((B, H, k, dv)))
          for i, k in enumerate(ksizes)]
    leaves = [Op.make("om_elem", s, v) for s, v in zip(ss, vs)]
    root = Op.make("om_apply", _compose_tree(leaves))
    inputs = ss + vs
    ir = IR(root=root, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched
    grp = mod._plan["leaf_groups"][0]
    assert grp["s_mode"] == "stack" and len(grp["members"]) == 6

    tss = [torch.randn(B, H, T, k, dtype=torch.float64)
           for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=torch.float64)
           for k in ksizes]
    ref = torch.softmax(torch.cat(tss, -1), -1) @ torch.cat(tvs, -2)
    with torch.no_grad():
        out = mod(*tss, *tvs)
    assert (out - ref).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  Operand-gather fast paths
# ---------------------------------------------------------------------------

def test_chunked_kv_dense_score_path():
    """k_i = chunk(K,-2,i), v_i = chunk(V,-2,i): the elem level must
    take the dense_qk path — ONE matmul q@K.T, scores re-chunked by a
    view — and the value stack is a view too."""
    torch.manual_seed(0)
    B, H, T, d, dv, n, K = 2, 4, 64, 32, 24, 4, 16
    q = Var("q", TensorType((B, H, T, d)))
    Kb = Var("K", TensorType((B, H, n * K, d)))
    Vb = Var("V", TensorType((B, H, n * K, dv)))

    def leaf(_, i):
        ki = Op.make("chunk", Kb, arg1=n, arg2=-2, index=i)
        vi = Op.make("chunk", Vb, arg1=n, arg2=-2, index=i)
        return _qk_leaf(q, ki, vi)

    inputs = [q, Kb, Vb]
    root = Op.make("om_apply", _compose_tree(
        [leaf(None, i) for i in range(n)]))
    ir = IR(root=root, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched
    grp = mod._plan["leaf_groups"][0]
    assert grp["s_mode"] == "dense_qk"
    assert grp["v_gather"] is not None

    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tK = torch.randn(B, H, n * K, d, dtype=torch.float64)
    tV = torch.randn(B, H, n * K, dv, dtype=torch.float64)
    ref = torch.softmax(tq @ tK.transpose(-2, -1), -1) @ tV
    with torch.no_grad():
        out = mod(tq, tK, tV)
    assert (out - ref).abs().max().item() < 1e-12


def test_chunked_scores_slice_path():
    """s_i = chunk(S,-1,i): the stacked score tensor is a view of S —
    the 'slice' s_mode."""
    torch.manual_seed(0)
    B, H, T, dv, n, K = 2, 2, 32, 16, 4, 8
    Sb = Var("S", TensorType((B, H, T, n * K)))
    Vb = Var("V", TensorType((B, H, n * K, dv)))
    leaves = [
        Op.make("om_elem",
                Op.make("chunk", Sb, arg1=n, arg2=-1, index=i),
                Op.make("chunk", Vb, arg1=n, arg2=-2, index=i))
        for i in range(n)
    ]
    root = Op.make("om_apply", _compose_tree(leaves))
    inputs = [Sb, Vb]
    ir = IR(root=root, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched
    grp = mod._plan["leaf_groups"][0]
    assert grp["s_mode"] == "slice"

    tS = torch.randn(B, H, T, n * K, dtype=torch.float64)
    tV = torch.randn(B, H, n * K, dv, dtype=torch.float64)
    ref = torch.softmax(tS, -1) @ tV
    with torch.no_grad():
        out = mod(tS, tV)
    assert (out - ref).abs().max().item() < 1e-12


def test_om_packaging_leaf():
    """A raw om(m,l,a) leaf mixes into the tree as a serial leaf —
    evaluated once, spliced into the stacked slots."""
    torch.manual_seed(0)
    B, H, T, dv = 1, 2, 16, 8
    K = 8
    ss = [Var(f"s{i}", TensorType((B, H, T, K))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((B, H, K, dv))) for i in range(2)]
    m3 = Var("m3", TensorType((B, H, T, 1)))
    l3 = Var("l3", TensorType((B, H, T, 1)))
    a3 = Var("a3", TensorType((B, H, T, dv)))
    tree = Op.make(
        "om_compose",
        Op.make("om_compose",
                Op.make("om_elem", ss[0], vs[0]),
                Op.make("om", m3, l3, a3)),
        Op.make("om_elem", ss[1], vs[1]))
    root = Op.make("om_apply", tree)
    inputs = ss + vs + [m3, l3, a3]
    ir = IR(root=root, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched

    tss = [torch.randn(B, H, T, K, dtype=torch.float64)
           for _ in range(2)]
    tvs = [torch.randn(B, H, K, dv, dtype=torch.float64)
           for _ in range(2)]
    # Build the third block's carrier the way om_elem would.
    s3 = torch.randn(B, H, T, K, dtype=torch.float64)
    v3 = torch.randn(B, H, K, dv, dtype=torch.float64)
    mm = s3.amax(-1, keepdim=True)
    e3 = torch.exp(s3 - mm)
    tm, tl = mm, e3.sum(-1, keepdim=True)
    ta = e3 @ v3
    ref = torch.softmax(torch.cat(tss + [s3], -1), -1) \
        @ torch.cat(tvs + [v3], -2)
    with torch.no_grad():
        out = mod(*tss, *tvs, tm, tl, ta)
    assert (out - ref).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  NaN semantics
# ---------------------------------------------------------------------------

def test_fully_masked_row_nan_matches_dense():
    """A row masked in EVERY block is NaN, matching dense softmax —
    om_apply stays unclamped in the batched path too."""
    torch.manual_seed(0)
    B, H, T, dv, n, K = 2, 2, 32, 8, 4, 8
    ss = [Var(f"s{i}", TensorType((B, H, T, K))) for i in range(n)]
    vs = [Var(f"v{i}", TensorType((B, H, K, dv))) for i in range(n)]
    root = Op.make("om_apply", _compose_tree(
        [Op.make("om_elem", s, v) for s, v in zip(ss, vs)]))
    inputs = ss + vs
    ir = IR(root=root, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched

    tss = [torch.randn(B, H, T, K, dtype=torch.float64)
           for _ in range(n)]
    tvs = [torch.randn(B, H, K, dv, dtype=torch.float64)
           for _ in range(n)]
    for t in tss:
        t[..., 5, :] = float("-inf")      # row 5 fully masked
    tss[1][..., 0, :] = float("-inf")     # row 0 masked in block 1 only
    with torch.no_grad():
        out = mod(*tss, *tvs)
    dense = torch.softmax(torch.cat(tss, -1), -1) @ torch.cat(tvs, -2)
    assert torch.equal(torch.isnan(out), torch.isnan(dense))
    assert torch.isnan(out[..., 5, :]).all()
    diff = (out - dense).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    assert diff.max().item() < 1e-12


# ---------------------------------------------------------------------------
#  Detection on the real extracted term (EGraph + OM_LAWS)
# ---------------------------------------------------------------------------

def test_detects_extracted_om_term():
    """The module recognises the term OM_LAWS extraction emits:
    saturate the dense chunked-attention term, force-extract the
    om_apply(om_compose...) member, lower — batched and fp64-exact."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 3, 8, 4, 6
    ksizes = [4, 4]
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    term = _dense_chunked_term(q, ks, vs)

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=20, max_nodes=200_000)
    chunked = _extract_chunked(eg, root)
    assert chunked is not None
    assert is_om_apply_term(chunked), op_repr(chunked)
    assert "om_compose" in op_repr(chunked)

    inputs = [q] + ks + vs
    ir = IR(root=chunked, inputs=inputs,
            input_names={v.name for v in inputs}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched

    tq = torch.randn(B, H, T, d, dtype=torch.float64)
    tks = [torch.randn(B, H, k, d, dtype=torch.float64) for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, dtype=torch.float64)
           for k in ksizes]
    ref = _dense_ref(tq, tks, tvs)
    mod.eval()
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
#  Plan structure / API surface
# ---------------------------------------------------------------------------

def test_plan_levels_are_independent():
    """build_om_plan groups composes by depth; leaves = om_elem nodes."""
    torch.manual_seed(0)
    B, H, T, d, dv = 1, 1, 16, 8, 8
    ksizes = [4] * 8
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    ir = _om_ir(q, ks, vs)
    plan = build_om_plan(ir.root)
    assert plan is not None
    assert len(plan["leaves"]) == 8
    assert all(l.op == "om_elem" for l in plan["leaves"])
    # balanced tree over 8 leaves: 4 + 2 + 1 composes on 3 levels
    assert [len(lv) for lv in plan["levels"]] == [4, 2, 1]
    assert len(plan["leaf_groups"]) == 1  # one uniform shape group


def test_plan_rejects_non_om_root():
    x = torch.randn(4, 4)
    m = SwiGLU(4, hidden_mult=2).eval()
    ir, _ = export_to_ir(m, x)
    assert build_om_plan(ir.root) is None
    assert not is_om_apply_term(ir.root)


def test_single_elem_apply_is_batched():
    """om_apply(om_elem(s,v)) — the OM_LIFT form, no compose — still
    takes the batched path (one leaf group, zero levels)."""
    s = Var("s", TensorType((4, 6)))
    v = Var("v", TensorType((6, 3)))
    root = Op.make("om_apply", Op.make("om_elem", s, v))
    ir = IR(root=root, inputs=[s, v],
            input_names={"s", "v"}, params={})
    mod = to_batched_om_module(ir)
    assert mod.is_batched and mod.n_levels == 0 and mod.n_blocks == 1
    ts = torch.randn(4, 6, dtype=torch.float64)
    tv = torch.randn(6, 3, dtype=torch.float64)
    with torch.no_grad():
        out = mod(ts, tv)
    assert (out - torch.softmax(ts, -1) @ tv).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  CUDA graph replay (optional fast path)
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
def test_cuda_graph_replay_matches_eager():
    """capture_cuda_graph replays the identical computation; changed
    inputs are picked up via the static input buffers; drop restores
    the eager path."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 4, 64, 32, 32
    ksizes = [16] * 4
    q, ks, vs = _qk_vars(B, H, T, d, dv, ksizes)
    ir = _om_ir(q, ks, vs)

    mod = to_batched_om_module(ir).eval().cuda()
    tq = torch.randn(B, H, T, d, device="cuda")
    tks = [torch.randn(B, H, k, d, device="cuda") for k in ksizes]
    tvs = [torch.randn(B, H, k, dv, device="cuda") for k in ksizes]
    mod.capture_cuda_graph(tq, *tks, *tvs)
    assert mod.is_graph_captured

    with torch.no_grad():
        out1 = mod(tq, *tks, *tvs).clone()
        tq2 = torch.randn(B, H, T, d, device="cuda")
        tks2 = [torch.randn(B, H, k, d, device="cuda") for k in ksizes]
        tvs2 = [torch.randn(B, H, k, dv, device="cuda")
                for k in ksizes]
        out2 = mod(tq2, *tks2, *tvs2).clone()
        ref1 = _dense_ref(tq, tks, tvs)
        ref2 = _dense_ref(tq2, tks2, tvs2)
        assert (ref1 - out1).abs().max().item() < 1e-4
        assert (ref2 - out2).abs().max().item() < 1e-4

        mod.drop_cuda_graph()
        assert not mod.is_graph_captured
        out3 = mod(tq2, *tks2, *tvs2)
        assert (ref2 - out3).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
#  Fallback: non-om IR keeps working
# ---------------------------------------------------------------------------

def test_fallback_matches_plain_lowering():
    """Non-om IR: BatchedOMModule delegates to serial evaluation."""
    torch.manual_seed(0)
    m = SwiGLU(16, hidden_mult=2).eval()
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(m, x)

    mod = to_batched_om_module(ir, param_values=source)
    assert isinstance(mod, BatchedOMModule)
    assert not mod.is_batched
    ref = ir_to_torch_module(ir, param_values=source)
    mod.eval(); ref.eval()
    with torch.no_grad():
        assert torch.equal(mod(x.clone()), ref(x.clone()))
        assert (m(x.clone()) - mod(x.clone())).abs().max().item() < 1e-6


def test_fallback_on_om_ops_outside_apply():
    """A bare om_compose root (no om_apply wrapper) is not om-shaped —
    the module falls back to serial tuple-passing eval, returning the
    (m, l, a) triple."""
    torch.manual_seed(0)
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    v1 = Var("v1", TensorType((3, 6)))
    v2 = Var("v2", TensorType((5, 6)))
    bare = Op.make("om_compose",
                   Op.make("om_elem", s1, v1),
                   Op.make("om_elem", s2, v2))
    ir = IR(root=bare, inputs=[s1, s2, v1, v2],
            input_names={"s1", "s2", "v1", "v2"}, params={})
    mod = to_batched_om_module(ir)
    assert not mod.is_batched
    ts1 = torch.randn(4, 3, dtype=torch.float64)
    ts2 = torch.randn(4, 5, dtype=torch.float64)
    tv1 = torch.randn(3, 6, dtype=torch.float64)
    tv2 = torch.randn(5, 6, dtype=torch.float64)
    with torch.no_grad():
        out = mod(ts1, ts2, tv1, tv2)
    assert isinstance(out, tuple) and len(out) == 3
    s = torch.cat([ts1, ts2], -1)
    e = torch.exp(s - s.amax(-1, keepdim=True))
    assert torch.allclose(out[2] / out[1],
                          torch.softmax(s, -1) @ torch.cat([tv1, tv2], -2))
