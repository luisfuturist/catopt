"""Streaming (bounded-working-set) lowering for the om monoid.

:class:`catopt.om_lower.StreamingOMModule` evaluates the same
``om_apply(<om_compose tree over om_elem>)`` term as
:class:`BatchedOMModule`, but as a LEFT FOLD: each leaf's score/value
block is evaluated, composed into a running ``(m, l, a)`` carrier, and
released — never materialising the full q×T_kv score matrix or holding
all K/V blocks.  The module-level ``om_step``/``om_step_qk`` helpers
cover the incremental decode regime: existing carrier + one new block
→ updated carrier in O(block) work.
"""

import gc
import math

import pytest
import torch
import torch.nn.functional as F

import catopt.om_lower as om_lower
import catopt.torch_bridge as torch_bridge
from catopt.ir import IR, Op, TensorType, Var
from catopt.models import SwiGLU
from catopt.om_lower import (
    StreamingOMModule,
    om_apply_state,
    om_empty_state,
    om_step,
    om_step_qk,
    to_batched_om_module,
    to_streaming_om_module,
)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _dense_ref(q, ks, vs):
    s = q @ torch.cat(list(ks), dim=-2).transpose(-2, -1)
    return torch.softmax(s, dim=-1) @ torch.cat(list(vs), dim=-2)


def _qk_leaf(q, k, v):
    s = Op.make("matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1))
    return Op.make("om_elem", s, v)


def _compose_tree(leaves):
    """Balanced binary bracketing (chunk order preserved)."""
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        "om_compose",
        _compose_tree(leaves[:mid]),
        _compose_tree(leaves[mid:]),
    )


def _compose_left(leaves):
    """Left-leaning bracketing — the shape the fold itself computes."""
    t = leaves[0]
    for lf in leaves[1:]:
        t = Op.make("om_compose", t, lf)
    return t


def _qk_vars(B, H, T, d, dv, ksizes):
    q = Var("q", TensorType((B, H, T, d)))
    ks = [
        Var(f"k{i}", TensorType((B, H, k, d)))
        for i, k in enumerate(ksizes)
    ]
    vs = [
        Var(f"v{i}", TensorType((B, H, k, dv)))
        for i, k in enumerate(ksizes)
    ]
    return q, ks, vs


def _om_ir(q, ks, vs, tree=_compose_tree):
    leaves = [_qk_leaf(q, k, v) for k, v in zip(ks, vs)]
    root = Op.make("om_apply", tree(leaves))
    inputs = [q] + list(ks) + list(vs)
    return IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )


def _chunked_kv_ir(B, H, Tq, d, dv, Tkv, C):
    """om_apply(compose over om_elem(q @ chunk(K,-2,i).T, chunk(V,-2,i)))
    — the canonical chunked-attention term (dense_qk operand gather)."""
    n = Tkv // C
    q = Var("q", TensorType((B, H, Tq, d)))
    Kb = Var("K", TensorType((B, H, n * C, d)))
    Vb = Var("V", TensorType((B, H, n * C, dv)))

    def leaf(i):
        ki = Op.make("chunk", Kb, arg1=n, arg2=-2, index=i)
        vi = Op.make("chunk", Vb, arg1=n, arg2=-2, index=i)
        return _qk_leaf(q, ki, vi)

    root = Op.make(
        "om_apply", _compose_tree([leaf(i) for i in range(n)])
    )
    return IR(
        root=root,
        inputs=[q, Kb, Vb],
        input_names={"q", "K", "V"},
        params={},
    )


def _peak_bytes(fn):
    """Peak bytes allocated DURING fn, beyond pre-existing tensors.

    CUDA: torch.cuda.max_memory_allocated minus live-at-entry.
    CPU:  cumulative sum of the autograd profiler's per-op allocation
    deltas (top-level events only — children would double-count);
    tracks live tensor bytes, so its max is the peak working set.
    """
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base
    from torch.autograd.profiler import profile

    with profile(profile_memory=True) as prof:
        fn()
    live = peak = 0
    for e in prof.function_events:
        if getattr(e, "cpu_parent", None) is not None:
            continue
        live += getattr(e, "cpu_memory_usage", 0)
        peak = max(peak, live)
    return peak


# ---------------------------------------------------------------------------
#  (a) equivalence: streaming == dense softmax == BatchedOMModule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_blocks", [4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("tree", [_compose_tree, _compose_left])
def test_streaming_matches_dense_and_batched(n_blocks, dtype, tree):
    """The fold result equals dense softmax and the level-batched
    executor — regardless of the extracted tree's bracketing."""
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 4, 256, 64, 64
    K = T // n_blocks
    q, ks, vs = _qk_vars(B, H, T, d, dv, [K] * n_blocks)
    ir = _om_ir(q, ks, vs, tree=tree)

    mod = to_streaming_om_module(ir).eval()
    bmod = to_batched_om_module(ir).eval()
    assert mod.is_streaming and mod.n_blocks == n_blocks
    assert isinstance(mod, StreamingOMModule)

    tq = torch.randn(B, H, T, d, dtype=dtype)
    tks = [
        torch.randn(B, H, K, d, dtype=dtype) for _ in range(n_blocks)
    ]
    tvs = [
        torch.randn(B, H, K, dv, dtype=dtype) for _ in range(n_blocks)
    ]
    ref = _dense_ref(tq, tks, tvs)
    tol = 1e-10 if dtype == torch.float64 else 2e-5
    with torch.no_grad():
        out = mod(tq, *tks, *tvs)
        out_b = bmod(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < tol
    assert (out - out_b).abs().max().item() < tol


def test_streaming_chunked_kv_matches_dense():
    """The benchmark's canonical term: chunk(K)/chunk(V) leaves.
    Streaming must NOT take the dense q@Kᵀ shortcut (that would
    materialise the full score matrix) — it evaluates q@k_i.T per
    block."""
    torch.manual_seed(0)
    B, H, Tq, d, dv, n, C = 2, 4, 64, 32, 24, 4, 16
    ir = _chunked_kv_ir(B, H, Tq, d, dv, n * C, C)
    mod = to_streaming_om_module(ir).eval()
    assert mod.is_streaming

    tq = torch.randn(B, H, Tq, d, dtype=torch.float64)
    tK = torch.randn(B, H, n * C, d, dtype=torch.float64)
    tV = torch.randn(B, H, n * C, dv, dtype=torch.float64)
    ref = torch.softmax(tq @ tK.transpose(-2, -1), -1) @ tV
    with torch.no_grad():
        out = mod(tq, tK, tV)
    assert (out - ref).abs().max().item() < 1e-12


def test_streaming_score_vars_and_om_leaf():
    """Raw score leaves plus a packaged om(m,l,a) leaf fold in-order."""
    torch.manual_seed(0)
    B, H, T, dv, K = 1, 2, 16, 8, 8
    ss = [Var(f"s{i}", TensorType((B, H, T, K))) for i in range(2)]
    vs = [Var(f"v{i}", TensorType((B, H, K, dv))) for i in range(2)]
    m3 = Var("m3", TensorType((B, H, T, 1)))
    l3 = Var("l3", TensorType((B, H, T, 1)))
    a3 = Var("a3", TensorType((B, H, T, dv)))
    tree = Op.make(
        "om_compose",
        Op.make(
            "om_compose",
            Op.make("om_elem", ss[0], vs[0]),
            Op.make("om", m3, l3, a3),
        ),
        Op.make("om_elem", ss[1], vs[1]),
    )
    ir = IR(
        root=Op.make("om_apply", tree),
        inputs=ss + vs + [m3, l3, a3],
        input_names={v.name for v in ss + vs + [m3, l3, a3]},
        params={},
    )
    mod = to_streaming_om_module(ir).eval()
    assert mod.is_streaming

    tss = [
        torch.randn(B, H, T, K, dtype=torch.float64) for _ in range(2)
    ]
    tvs = [
        torch.randn(B, H, K, dv, dtype=torch.float64) for _ in range(2)
    ]
    s3 = torch.randn(B, H, T, K, dtype=torch.float64)
    v3 = torch.randn(B, H, K, dv, dtype=torch.float64)
    mm = s3.amax(-1, keepdim=True)
    e3 = torch.exp(s3 - mm)
    tm, tl = mm, e3.sum(-1, keepdim=True)
    ta = e3 @ v3
    ref = torch.softmax(torch.cat(tss + [s3], -1), -1) @ torch.cat(
        tvs + [v3], -2
    )
    with torch.no_grad():
        out = mod(*tss, *tvs, tm, tl, ta)
    assert (out - ref).abs().max().item() < 1e-12


def test_streaming_shared_leaf_multiplicity():
    """A DAG-shared leaf contributes once per occurrence — the fold
    honours the plan's multiplicities."""
    torch.manual_seed(0)
    B, H, T, dv, K = 1, 2, 16, 8, 8
    s = Var("s", TensorType((B, H, T, K)))
    v = Var("v", TensorType((B, H, K, dv)))
    leaf = Op.make("om_elem", s, v)
    # compose(leaf, leaf): the SAME node twice — block counted twice.
    tree = Op.make("om_compose", leaf, leaf)
    ir = IR(
        root=Op.make("om_apply", tree),
        inputs=[s, v],
        input_names={"s", "v"},
        params={},
    )
    mod = to_streaming_om_module(ir).eval()
    assert mod.is_streaming

    ts = torch.randn(B, H, T, K, dtype=torch.float64)
    tv = torch.randn(B, H, K, dv, dtype=torch.float64)
    with torch.no_grad():
        out = mod(ts, tv)
    # Dense equivalent: keys duplicated.
    ref = torch.softmax(torch.cat([ts, ts], -1), -1) @ torch.cat(
        [tv, tv], -2
    )
    assert (out - ref).abs().max().item() < 1e-12


def test_streaming_masked_nan_matches_dense():
    """Fully-masked rows stay NaN through the fold, matching dense."""
    torch.manual_seed(0)
    B, H, T, dv, n, K = 2, 2, 32, 8, 4, 8
    ss = [Var(f"s{i}", TensorType((B, H, T, K))) for i in range(n)]
    vs = [Var(f"v{i}", TensorType((B, H, K, dv))) for i in range(n)]
    ir = IR(
        root=Op.make(
            "om_apply",
            _compose_tree(
                [Op.make("om_elem", s, v) for s, v in zip(ss, vs)]
            ),
        ),
        inputs=ss + vs,
        input_names={v.name for v in ss + vs},
        params={},
    )
    mod = to_streaming_om_module(ir).eval()

    tss = [
        torch.randn(B, H, T, K, dtype=torch.float64) for _ in range(n)
    ]
    tvs = [
        torch.randn(B, H, K, dv, dtype=torch.float64) for _ in range(n)
    ]
    for t in tss:
        t[..., 5, :] = float("-inf")
    tss[1][..., 0, :] = float("-inf")
    with torch.no_grad():
        out = mod(*tss, *tvs)
    dense = torch.softmax(torch.cat(tss, -1), -1) @ torch.cat(tvs, -2)
    assert torch.equal(torch.isnan(out), torch.isnan(dense))
    diff = (out - dense).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    assert diff.max().item() < 1e-12


# ---------------------------------------------------------------------------
#  (b) bounded working set — O(block), not O(total)
# ---------------------------------------------------------------------------


def test_streaming_block_working_set_is_bounded(monkeypatch):
    """No leaf ever sees a score tensor wider than its block — the
    fold never materialises the full q×T_kv matrix — and the compose
    count is n-1, one per folded block."""
    torch.manual_seed(0)
    B, H, Tq, d, dv, n, C = 1, 4, 16, 16, 16, 8, 64
    ir = _chunked_kv_ir(B, H, Tq, d, dv, n * C, C)
    mod = to_streaming_om_module(ir).eval()

    s_widths = []
    n_compose = 0
    orig_elem = torch_bridge._om_elem
    # forward_state calls om_lower's own binding import; the leaf
    # evaluator dispatches torch_bridge's.  Spy on each.
    orig_compose = om_lower._om_compose

    def spy_elem(s, v):
        s_widths.append(s.shape[-1])
        return orig_elem(s, v)

    def spy_compose(f, g):
        nonlocal n_compose
        n_compose += 1
        return orig_compose(f, g)

    monkeypatch.setattr(torch_bridge, "_om_elem", spy_elem)
    monkeypatch.setattr(om_lower, "_om_compose", spy_compose)

    tq = torch.randn(B, H, Tq, d)
    tK = torch.randn(B, H, n * C, d)
    tV = torch.randn(B, H, n * C, dv)
    with torch.no_grad():
        mod(tq, tK, tV)
    assert len(s_widths) == n
    assert max(s_widths) == C  # never n*C
    assert n_compose == n - 1  # left fold: one ⊕ per block


@pytest.mark.requires_cuda
def test_streaming_peak_memory_flat_in_Tkv():
    """Peak allocated bytes during the fold is ~independent of the
    number of blocks (and far below the batched executor, which stacks
    all block scores).  Same shapes on CUDA or CPU."""
    torch.manual_seed(0)
    B, H, Tq, d, dv, C = 1, 8, 32, 64, 64, 2048
    dev = torch.device(DEV)

    def run(Tkv):
        ir = _chunked_kv_ir(B, H, Tq, d, dv, Tkv, C)
        smod = to_streaming_om_module(ir).eval()
        bmod = to_batched_om_module(ir).eval()
        q = torch.randn(B, H, Tq, d, device=dev)
        K = torch.randn(B, H, Tkv, d, device=dev)
        V = torch.randn(B, H, Tkv, dv, device=dev)
        with torch.no_grad():
            sp = _peak_bytes(lambda: smod(q, K, V))
            bp = _peak_bytes(lambda: bmod(q, K, V))
        del q, K, V, smod, bmod
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()
        return sp, bp

    s1, b1 = run(16384)  # 8 blocks
    s2, b2 = run(32768)  # 16 blocks — inputs double, peaks should not

    # streaming peak tracks BLOCK size, not Tkv: flat across doubling.
    assert s2 < s1 * 1.5 + 4 * 2**20, (s1, s2)
    # and far below the batched schedule (which materialises all blocks)
    assert s1 * 4 < b1 and s2 * 4 < b2, (s1, b1, s2, b2)


@pytest.mark.requires_cuda
def test_streaming_peak_cuda_where_dense_cannot():
    """The headline regime: T_kv where the full score matrix is ~GiB
    but streaming needs only ~block.  Batched/naive must OOM or blow
    far past streaming's flat ~tens-of-MiB transient."""
    torch.manual_seed(0)
    B, H, Tq, d, dv, C = 1, 8, 64, 64, 64, 4096
    Tkv = 262144  # scores alone = 64*262144*4 = 64MiB/row-set
    ir = _chunked_kv_ir(B, H, Tq, d, dv, Tkv, C)
    mod = to_streaming_om_module(ir).eval()
    q = torch.randn(B, H, Tq, d, device="cuda")
    K = torch.randn(B, H, Tkv, d, device="cuda")
    V = torch.randn(B, H, Tkv, dv, device="cuda")
    # Full score matrix would be B*H*Tq*Tkv*4 = 512MiB of pure scores
    # (per copy — batched holds several); streaming works in ~C blocks.
    with torch.no_grad():
        peak = _peak_bytes(lambda: mod(q, K, V))
    score_bytes = B * H * Tq * Tkv * 4
    assert peak < score_bytes // 8, (peak, score_bytes)


# ---------------------------------------------------------------------------
#  (c) incremental / decode mode
# ---------------------------------------------------------------------------


def test_incremental_step_equals_recompute():
    """state(prefix) ⊕ om_elem(new block) ≡ fold of the whole stream —
    and om_step_qk's q@k.T form matches too."""
    torch.manual_seed(0)
    B, H, Tq, d, dv, n, C = 2, 4, 8, 32, 24, 8, 16
    ir = _chunked_kv_ir(B, H, Tq, d, dv, n * C, C)
    mod = to_streaming_om_module(ir).eval()
    tq = torch.randn(B, H, Tq, d, dtype=torch.float64)
    tK = torch.randn(B, H, n * C, d, dtype=torch.float64)
    tV = torch.randn(B, H, n * C, dv, dtype=torch.float64)

    with torch.no_grad():
        full = mod.forward_state(tq, tK, tV)
        # stream the first half, then step block-by-block
        cut = n // 2 * C
        st = om_step_qk(None, tq, tK[:, :, :cut], tV[:, :, :cut])
        for i in range(cut, n * C, C):
            st = om_step_qk(
                st, tq, tK[:, :, i : i + C], tV[:, :, i : i + C]
            )
        # also the raw-score step form
        st2 = om_step(
            None, tq @ tK[:, :, :cut].transpose(-2, -1), tV[:, :, :cut]
        )
        for i in range(cut, n * C, C):
            st2 = om_step(
                st2,
                tq @ tK[:, :, i : i + C].transpose(-2, -1),
                tV[:, :, i : i + C],
            )

    for a, b in zip(st, full):
        assert (a - b).abs().max().item() < 1e-12
    assert (
        om_apply_state(st2) - om_apply_state(full)
    ).abs().max().item() < 1e-12


@pytest.mark.requires_cuda
def test_incremental_growing_cache_matches_sdpa():
    """Decode regime: a fixed query streams over a cache that grows one
    block per step — each step O(block), output equals sdpa on the
    cache seen so far."""
    torch.manual_seed(0)
    B, H, Tq, d, dv = 1, 8, 4, 64, 64
    T0, steps, C = 512, 8, 64
    q = torch.randn(
        B, H, Tq, d, device=DEV, dtype=torch.float64
    ) / math.sqrt(d)
    K = torch.randn(
        B, H, T0 + steps * C, d, device=DEV, dtype=torch.float64
    )
    V = torch.randn(
        B, H, T0 + steps * C, dv, device=DEV, dtype=torch.float64
    )

    st = om_step_qk(None, q, K[:, :, :T0], V[:, :, :T0])
    for t in range(steps):
        st = om_step_qk(
            st,
            q,
            K[:, :, T0 + t * C : T0 + (t + 1) * C],
            V[:, :, T0 + t * C : T0 + (t + 1) * C],
        )
        out = om_apply_state(st)
        ref = F.scaled_dot_product_attention(
            q,
            K[:, :, : T0 + (t + 1) * C],
            V[:, :, : T0 + (t + 1) * C],
            scale=1.0,
        )
        assert (out - ref).abs().max().item() < 1e-12


def test_empty_state_is_identity():
    """om_empty_state is the ⊕ identity: compose(x, empty) = x, and a
    decode loop seeded with it matches one seeded with None."""
    torch.manual_seed(0)
    B, H, Tq, d, dv = 1, 2, 4, 16, 8
    q = torch.randn(B, H, Tq, d, dtype=torch.float64)
    k = torch.randn(B, H, 32, d, dtype=torch.float64)
    v = torch.randn(B, H, 32, dv, dtype=torch.float64)

    st_none = om_step_qk(None, q, k, v)
    st_empty = om_step_qk(
        om_empty_state((B, H, Tq), dv, dtype=torch.float64), q, k, v
    )
    for a, b in zip(st_none, st_empty):
        assert torch.equal(a, b)
    # composing the empty state on the RIGHT is also the identity
    from catopt.torch_bridge import _om_compose

    st2 = _om_compose(
        st_none, om_empty_state((B, H, Tq), dv, dtype=torch.float64)
    )
    for a, b in zip(st_none, st2):
        assert torch.equal(a, b)


def test_incremental_through_module_statics():
    """The decode API is reachable on the module class itself."""
    torch.manual_seed(0)
    B, H, Tq, d, dv = 1, 2, 4, 16, 8
    s = Var("s", TensorType((B, H, Tq, 32)))
    v = Var("v", TensorType((B, H, 32, dv)))
    ir = IR(
        root=Op.make("om_apply", Op.make("om_elem", s, v)),
        inputs=[s, v],
        input_names={"s", "v"},
        params={},
    )
    mod = to_streaming_om_module(ir)
    st = mod.step(
        None,
        torch.randn(B, H, Tq, 32, dtype=torch.float64),
        torch.randn(B, H, 32, dv, dtype=torch.float64),
    )
    out = mod.apply(st)
    assert out.shape == (B, H, Tq, dv)


# ---------------------------------------------------------------------------
#  (d) fallback for non-om terms
# ---------------------------------------------------------------------------


def test_fallback_non_om_matches_serial():
    """Non-om IR: the streaming module delegates to serial eval."""
    torch.manual_seed(0)
    m = SwiGLU(16, hidden_mult=2).eval()
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(m, x)

    mod = to_streaming_om_module(ir, param_values=source).eval()
    assert not mod.is_streaming
    ref = ir_to_torch_module(ir, param_values=source).eval()
    with torch.no_grad():
        assert torch.equal(mod(x.clone()), ref(x.clone()))
        assert (m(x.clone()) - mod(x.clone())).abs().max().item() < 1e-6
        # forward_state returns whatever serial eval produces
        assert torch.equal(mod.forward_state(x.clone()), ref(x.clone()))


def test_fallback_bare_om_tree_returns_carrier():
    """A bare om_compose root (no om_apply) isn't om-shaped: forward
    falls back to serial tuple-passing and returns the (m,l,a) triple."""
    torch.manual_seed(0)
    s1 = Var("s1", TensorType((4, 3)))
    s2 = Var("s2", TensorType((4, 5)))
    v1 = Var("v1", TensorType((3, 6)))
    v2 = Var("v2", TensorType((5, 6)))
    bare = Op.make(
        "om_compose",
        Op.make("om_elem", s1, v1),
        Op.make("om_elem", s2, v2),
    )
    ir = IR(
        root=bare,
        inputs=[s1, s2, v1, v2],
        input_names={"s1", "s2", "v1", "v2"},
        params={},
    )
    mod = to_streaming_om_module(ir).eval()
    assert not mod.is_streaming
    ts1 = torch.randn(4, 3, dtype=torch.float64)
    ts2 = torch.randn(4, 5, dtype=torch.float64)
    tv1 = torch.randn(3, 6, dtype=torch.float64)
    tv2 = torch.randn(5, 6, dtype=torch.float64)
    with torch.no_grad():
        out = mod(ts1, ts2, tv1, tv2)
    assert isinstance(out, tuple) and len(out) == 3
    s = torch.cat([ts1, ts2], -1)
    assert torch.allclose(
        out[2] / out[1],
        torch.softmax(s, -1) @ torch.cat([tv1, tv2], -2),
    )
