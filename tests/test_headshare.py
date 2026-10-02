"""HeadShare — ``share_duplicate_attention_heads`` + cache recheck.

The pass folds ``sdpa`` sites whose per-head (Wq, Wk, Wv) blocks are
*bitwise-equal* into a gather-``sdpa``-gather member that computes
each unique head once.  These tests pin: the offers fire only under
provable equality, the member is bitwise-identical (fp64
``torch.equal``), the offer competes through the cost model, the
opt-in flag gates it in the pipeline, and the compositional-cache
recheck replays the byte-equality assumptions on new blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.egraph.types import ENode
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.laws.headshare import (
    _Ctx,
    _joint_sig,
    _piece,
    _resolve,
    headshare_keys_hold,
    share_duplicate_attention_heads,
)
from catopt_orchestrator.optimize import (
    Optimizer,
    _cache_replay,
    _term_param_names,
)
from catopt_orchestrator import SearchResult
from catopt_torch.backend import TorchBackend
from catopt_torch.torch_bridge import IRModule, export_to_ir

B, T, H, D, I = 2, 5, 4, 4, 16  # batch, seq, heads, head-dim, model-dim


def _t(*shape):
    return TensorType(tuple(shape))


def _w(seed=0, h=H, d=D, i=I):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(h * d, i, generator=g, dtype=torch.float64)


def _dup_rows(w, *pairs):
    """Copy row block ``src`` onto ``dst`` — bitwise-equal heads."""
    w = w.clone()
    for src, dst in pairs:
        w[dst * D : (dst + 1) * D] = w[src * D : (src + 1) * D]
    return w


def _chain(x, w, b=None):
    """transpose(reshape(linear(x, w[, b]), (B,T,H,D)), 1, 2)."""
    lin = (
        Op.make("linear", x, w, b) if b is not None
        else Op.make("linear", x, w)
    )
    return Op.make(
        "transpose",
        Op.make("reshape", lin, shape=(B, T, H, D)),
        dim0=1,
        dim1=2,
    )


def _sdpa(x, wq, wk, wv, **attrs):
    return Op.make(
        "sdpa", _chain(x, wq), _chain(x, wk), _chain(x, wv), **attrs
    )


def _graph(term):
    eg = EGraph()
    return eg, eg.add_term(term)


def _mha(h=H, dim=I, *, bias=False, dup=((0, 2),)):
    """Canonical multi-head attention with row-duplicated blocks."""

    class MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.dh = h, dim // h
            self.wq = nn.Linear(dim, dim, bias=bias)
            self.wk = nn.Linear(dim, dim, bias=bias)
            self.wv = nn.Linear(dim, dim, bias=bias)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            b_, t_, c = x.shape
            q = self.wq(x).view(b_, t_, self.h, self.dh).transpose(1, 2)
            k = self.wk(x).view(b_, t_, self.h, self.dh).transpose(1, 2)
            v = self.wv(x).view(b_, t_, self.h, self.dh).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v)
            return self.wo(o.transpose(1, 2).reshape(b_, t_, c))

    m = MHA().double()
    with torch.no_grad():
        d = m.dh
        for mod in (m.wq, m.wk, m.wv):
            for s, t in dup:
                mod.weight[t * d : (t + 1) * d] = mod.weight[
                    s * d : (s + 1) * d
                ]
                if mod.bias is not None:
                    mod.bias[t * d : (t + 1) * d] = mod.bias[
                        s * d : (s + 1) * d
                    ]
    return m


def _export(m, *shape):
    return export_to_ir(m, (torch.randn(*shape, dtype=torch.float64),))


def _lower(term, ir, leaves, xv):
    mod = IRModule(
        IR(
            root=term,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params={},
        ),
        param_values=dict(leaves),
    )
    with torch.no_grad():
        return mod(xv)


# ---------------------------------------------------------------------------
#  Fires — canonical spellings, bitwise-verified
# ---------------------------------------------------------------------------


def test_shared_heads_offer_extract_and_bitwise():
    m = _mha()
    ir, leaves = _export(m, B, T, I)
    eg, root = _graph(ir.root)
    offers = share_duplicate_attention_heads(eg, leaves)
    assert len(offers) == 1
    rec = offers[0]
    assert (rec["heads"], rec["unique"], rec["axis"]) == (H, 3, 1)
    assert rec["index_map"] == (0, 1, 0, 2)
    assert rec["uniq"] == (0, 1, 3)
    assert set(rec["params"]) == {"p_wq_weight", "p_wk_weight",
                                "p_wv_weight"}
    best = eg.extract_best(root, flops_cost)
    assert "index_select" in repr(best)  # gathered re-expansion wins
    xv = torch.randn(B, T, I, dtype=torch.float64)
    with torch.no_grad():
        ref = m(xv)
    out = _lower(best, ir, leaves, xv)
    assert torch.equal(ref, out)  # bitwise — not just close


def test_flops_cost_actually_prefers_the_shared_member():
    """The offer wins strictly on cost — smaller sdpa, views are free."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, root = _graph(_sdpa(x, q, k, v))
    w = _w()
    tensors = {
        "p_q": _dup_rows(w, (0, 2)),
        "p_k": _dup_rows(_w(1), (0, 2)),
        "p_v": _dup_rows(_w(2), (0, 2)),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    best = eg.extract_best(root, flops_cost)
    assert dag_cost(best, flops_cost) < dag_cost(
        _sdpa(x, q, k, v), flops_cost
    )


def test_no_bitwise_duplicate_heads_no_offer():
    m = _mha(dup=())  # all weights distinct
    ir, leaves = _export(m, B, T, I)
    eg, root = _graph(ir.root)
    assert share_duplicate_attention_heads(eg, leaves) == []


def test_near_equal_but_not_bitwise_no_offer():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, root = _graph(_sdpa(x, q, k, v))
    wq = _w()
    wk, wv = _w(1), _w(2)
    wq2 = wq.clone()
    wq2[2 * D : 3 * D] = wq[0:D]
    wk2 = wk.clone()
    # ``wq`` head 2 == head 0 by *value* (well, bitwise) but k differs
    # by one ulp — the joint head signatures must NOT merge.
    wk2[2 * D : 3 * D] = wk[0:D] + torch.finfo(torch.float64).eps
    wv2 = wv.clone()
    wv2[2 * D : 3 * D] = wv[0:D]
    tensors = {"p_q": wq2, "p_k": wk2, "p_v": wv2}
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_only_q_shared_no_offer():
    """All three operands must share jointly — q alone is not enough."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, root = _graph(_sdpa(x, q, k, v))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 2)),
        "p_k": _w(1),
        "p_v": _w(2),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_all_heads_identical_folds_to_one():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, root = _graph(_sdpa(x, q, k, v))
    base = _w()
    one = torch.cat([base[0:D]] * H)
    tensors = {"p_q": one, "p_k": _dup_rows(_w(1), *[(0, j) for j in
                                                   range(1, H)]),
               "p_v": _dup_rows(_w(2), *[(0, j) for j in range(1, H)])}
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["unique"] == 1
    assert offers[0]["index_map"] == (0, 0, 0, 0)


def test_biased_projection_equal_bias_fires():
    m = _mha(bias=True)
    ir, leaves = _export(m, B, T, I)
    eg, root = _graph(ir.root)
    assert len(share_duplicate_attention_heads(eg, leaves)) == 1


def test_biased_projection_unequal_bias_declines():
    m = _mha(bias=True)
    with torch.no_grad():
        m.wv.bias[2 * D : 3 * D] += 1.0  # v-head 2's bias now differs
    ir, leaves = _export(m, B, T, I)
    eg, root = _graph(ir.root)
    assert share_duplicate_attention_heads(eg, leaves) == []


def test_gqa_expand_spelling_fires():
    class GQA(nn.Module):
        def __init__(self):
            super().__init__()
            self.nh, self.nkv = H, 2
            self.hd = I // H
            self.wq = nn.Linear(I, self.nh * self.hd, bias=False)
            self.wk = nn.Linear(I, self.nkv * self.hd, bias=False)
            self.wv = nn.Linear(I, self.nkv * self.hd, bias=False)
            self.wo = nn.Linear(I, I, bias=False)

        def _exp(self, t, b_, t_):
            g = self.nh // self.nkv
            return (
                t[:, :, None, :, :]
                .expand(b_, self.nkv, g, t_, self.hd)
                .reshape(b_, self.nh, t_, self.hd)
            )

        def forward(self, x):
            b_, t_, c = x.shape
            q = self.wq(x).view(b_, t_, self.nh, self.hd).transpose(1, 2)
            k = self.wk(x).view(b_, t_, self.nkv, self.hd).transpose(1, 2)
            v = self.wv(x).view(b_, t_, self.nkv, self.hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q, self._exp(k, b_, t_), self._exp(v, b_, t_),
                is_causal=True,
            )
            return self.wo(o.transpose(1, 2).reshape(b_, t_, c))

    m = GQA().double()
    with torch.no_grad():
        d = m.hd
        # group-aligned q duplication: heads (0,1) and (2,3)
        m.wq.weight[d : 2 * d] = m.wq.weight[0:d]
        m.wq.weight[3 * d : 4 * d] = m.wq.weight[2 * d : 3 * d]
    ir, leaves = _export(m, B, T, I)
    eg, root = _graph(ir.root)
    offers = share_duplicate_attention_heads(eg, leaves)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 1, 1)
    best = eg.extract_best(root, flops_cost)
    xv = torch.randn(B, T, I, dtype=torch.float64)
    with torch.no_grad():
        ref = m(xv)
    assert torch.equal(ref, _lower(best, ir, leaves, xv))


def test_gqa_misaligned_q_dup_declines():
    """q ties must align with the kv broadcast groups to merge."""
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    kv = Param("p_kv", _t(2 * D, I))
    kc = Op.make(  # expand(unsqueeze(transpose(reshape(linear))))
        "reshape",
        Op.make(
            "expand",
            Op.make(
                "unsqueeze",
                Op.make(
                    "transpose",
                    Op.make(
                        "reshape",
                        Op.make("linear", x, kv),
                        shape=(B, T, 2, D),
                    ),
                    dim0=1,
                    dim1=2,
                ),
                dim=2,
            ),
            shape=(B, 2, 2, T, D),
        ),
        shape=(B, H, T, D),
    )
    kc2 = Op.make(
        "reshape",
        Op.make(
            "expand",
            Op.make(
                "unsqueeze",
                Op.make(
                    "transpose",
                    Op.make(
                        "reshape",
                        Op.make("linear", x, Param("p_kv2", _t(2 * D, I))),
                        shape=(B, T, 2, D),
                    ),
                    dim0=1,
                    dim1=2,
                ),
                dim=2,
            ),
            shape=(B, 2, 2, T, D),
        ),
        shape=(B, H, T, D),
    )
    eg, root = _graph(Op.make("sdpa", _chain(x, q), kc, kc2))
    wq = _w()
    wq[2 * D : 3 * D] = wq[0:D]  # heads 0==2 — spans kv groups
    tensors = {"p_q": wq, "p_kv": _w(1, 2, D), "p_kv2": _w(2, 2, D)}
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_fused_qkv_split_offsets():
    """One packed (3·h·d, i) weight split on the feature axis."""
    x = Var("x", _t(B, T, I))
    wqkv = Param("p_qkv", _t(3 * H * D, I))
    lin = Op.make("linear", x, wqkv)

    def section(idx):
        return Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make(
                    "split", lin, sizes=(H * D, H * D, H * D),
                    dim=-1, index=idx,
                ),
                shape=(B, T, H, D),
            ),
            dim0=1,
            dim1=2,
        )

    eg, root = _graph(Op.make("sdpa", section(0), section(1),
                              section(2)))
    g = torch.Generator().manual_seed(0)
    w = torch.randn(3 * H * D, I, generator=g, dtype=torch.float64)
    # fuse layout: rows [0:hd)=q, [hd:2hd)=k, [2hd:3hd)=v; dup head 2
    w2 = w.clone()
    for sec in range(3):
        off = sec * H * D
        w2[off + 2 * D : off + 3 * D] = w2[off : off + D]
    offers = share_duplicate_attention_heads(eg, {"p_qkv": w2})
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 1, 0, 2)


def test_matmul_orientation():
    """``matmul(x, W)`` — columns orientation — same fold."""
    x = Var("x", _t(B, T, I))

    def mm_chain(w):
        return Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make("matmul", x, w),
                shape=(B, T, H, D),
            ),
            dim0=1,
            dim1=2,
        )

    q, k, v = (Param(f"p_{n}", _t(I, H * D)) for n in "qkv")
    eg, root = _graph(Op.make("sdpa", mm_chain(q), mm_chain(k),
                              mm_chain(v)))
    g = torch.Generator().manual_seed(3)
    wq = torch.randn(I, H * D, generator=g, dtype=torch.float64)
    wk = torch.randn(I, H * D, generator=g, dtype=torch.float64)
    wv = torch.randn(I, H * D, generator=g, dtype=torch.float64)
    for w in (wq, wk, wv):
        w[:, 3 * D : 4 * D] = w[:, 0:D]  # column head 3 == head 0
    offers = share_duplicate_attention_heads(
        eg, {"p_q": wq, "p_k": wk, "p_v": wv}
    )
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 1, 2, 0)


def test_head_uniform_factor_keeps_resolution():
    """A RoPE-style table broadcast along heads must not break the fold."""
    x = Var("x", _t(B, T, I))
    tab = Param("p_tab", _t(1, 1, T, 1))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")

    def roped(w):
        ch = _chain(x, w)
        return Op.make(
            "mul",
            ch,
            Op.make("expand", tab, shape=(B, H, T, 1)),
        )

    eg, root = _graph(Op.make("sdpa", roped(q), _chain(x, k),
                              _chain(x, v)))
    g = torch.Generator().manual_seed(4)
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_tab": torch.randn(1, 1, T, 1, generator=g,
                             dtype=torch.float64),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 1, 2)


def test_per_head_factor_declines():
    """A table that genuinely varies per head: equal weights ≠ equal heads."""
    x = Var("x", _t(B, T, I))
    tab = Param("p_tab", _t(1, H, T, 1))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")

    def roped(w):
        return Op.make("mul", _chain(x, w), tab)

    eg, _ = _graph(Op.make("sdpa", roped(q), _chain(x, k),
                           _chain(x, v)))
    g = torch.Generator().manual_seed(5)
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_tab": torch.randn(1, H, T, 1, generator=g,
                             dtype=torch.float64),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_two_sites_merge_under_add():
    """``add(linear(x,Wq), linear(x,Uq))`` — both weights' blocks key."""
    x = Var("x", _t(B, T, I))
    uq = Param("p_uq", _t(H * D, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")

    def added(w, u):
        lin = Op.make(
            "add", Op.make("linear", x, w), Op.make("linear", x, u)
        )
        return Op.make(
            "transpose",
            Op.make("reshape", lin, shape=(B, T, H, D)),
            dim0=1,
            dim1=2,
        )

    eg, _ = _graph(
        Op.make("sdpa", added(q, uq), _chain(x, k), _chain(x, v))
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 2)),
        "p_uq": _dup_rows(_w(7), (0, 2)),
        "p_k": _dup_rows(_w(1), (0, 2)),
        "p_v": _dup_rows(_w(2), (0, 2)),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # one of the two factors differing at a shared head breaks the
    # joint signature — no offer.
    tensors["p_uq"][2 * D, 0] += 1.0
    eg2, _ = _graph(
        Op.make("sdpa", added(q, uq), _chain(x, k), _chain(x, v))
    )
    assert share_duplicate_attention_heads(eg2, tensors) == []


def test_scalar_factor_is_uniform():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    c = Param("p_s", TensorType(()))
    scaled = Op.make("mul", _chain(x, q), c)
    eg, _ = _graph(Op.make("sdpa", scaled, _chain(x, k),
                           _chain(x, v)))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_s": torch.tensor(1.5, dtype=torch.float64),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


# ---------------------------------------------------------------------------
#  Declines — attr vetoes and unresolvable chains
# ---------------------------------------------------------------------------


def test_sdpa_attr_and_arity_vetoes():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    mask = Param("p_mask", _t(B, H, T, T))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_mask": torch.randn(B, H, T, T, dtype=torch.float64),
    }
    for term in (
        Op.make("sdpa", _chain(x, q), _chain(x, k), _chain(x, v),
                dropout_p=0.1),
        Op.make("sdpa", _chain(x, q), _chain(x, k), _chain(x, v),
                enable_gqa=True),
        Op.make("sdpa", _chain(x, q), _chain(x, k), _chain(x, v),
                mask, validate=False),
    ):
        eg, _ = _graph(term)
        assert share_duplicate_attention_heads(eg, tensors) == []


def test_unknown_op_in_chain_declines():
    """``roll`` (not whitelisted) leaves the operand unresolvable."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    rolled = Op.make("roll", _chain(x, q), shifts=1, dim=-2)
    eg, _ = _graph(Op.make("sdpa", rolled, _chain(x, k),
                           _chain(x, v), validate=False))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_missing_tensors_decline():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, _ = _graph(_sdpa(x, q, k, v))
    assert share_duplicate_attention_heads(eg, {}) == []


def test_witness_flag_threaded():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, root = _graph(_sdpa(x, q, k, v))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    offers = share_duplicate_attention_heads(eg, tensors,
                                             witness=False)
    assert len(offers) == 1
    best = eg.extract_best(root, flops_cost)
    assert best.op == "index_select"


# ---------------------------------------------------------------------------
#  Resolver branches — hand-built chains
# ---------------------------------------------------------------------------


def _heads_term(x, w, *, mid=None):
    """A heads-shaped chain with an optional per-node decorator."""
    base = Op.make(
        "transpose",
        Op.make(
            "reshape", Op.make("linear", x, w), shape=(B, T, H, D)
        ),
        dim0=1,
        dim1=2,
    )
    return mid(base) if mid else base


def test_gather_on_head_axis_remaps():
    """index_select on the head axis reorders — and may itself dedup."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    gathered = Op.make(
        "index_select", _chain(x, q), dim=1, index=(0, 0, 1, 2)
    )
    eg, _ = _graph(
        Op.make("sdpa", gathered, _chain(x, k), _chain(x, v))
    )
    tensors = {
        "p_q": _w(),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # q operand heads = qbase heads (0,0,1,2): the gather itself
    # duplicates q0, and k1==k0 / v1==v0 close the joint signature —
    # heads (0,1) share without any dup inside Wq.
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 1, 2)


def test_index_select_bad_index_and_off_axis():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    bad = Op.make(
        "index_select", _chain(x, q), dim=1, index=(0, 9)
    )
    eg, _ = _graph(
        Op.make("sdpa", bad, _chain(x, k), _chain(x, v),
                validate=False)
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []
    # off-axis gather passes the head structure through untouched —
    # the shared-head fold still applies.
    off = Op.make("index_select", _chain(x, q), dim=2, index=(0, 1))
    eg, _ = _graph(
        Op.make("sdpa", off, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_slice_off_head_axis_and_split_chunk_on_head():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    # slice along T keeps the head axis — all operands get T=3
    def half(c):
        return Op.make("slice", c, dim=2, start=0, end=3)

    eg, _ = _graph(
        Op.make("sdpa", half(_chain(x, q)), half(_chain(x, k)),
                half(_chain(x, v)), validate=False)
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # slice ON the head axis of one operand — head count drops to 2
    # but the others still have 4 → arity mismatch, no offer.
    eg, _ = _graph(
        Op.make(
            "sdpa",
            Op.make("slice", _chain(x, q), dim=1, start=0, end=2),
            _chain(x, k),
            _chain(x, v),
            validate=False,
        )
    )
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_narrow_and_step_slice():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # narrow along T keeps the head axis — the fold still applies
    nar = Op.make("narrow", _chain(x, q), dim=2, start=0, length=3)
    eg, _ = _graph(
        Op.make("sdpa", nar, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # a strided head-axis slice is not a contiguous block — declines
    stepped = Op.make(
        "slice", _chain(x, q), dim=1, start=0, end=4, step=2
    )
    eg, _ = _graph(
        Op.make("sdpa", stepped, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []
    # narrowing the head axis of one operand leaves a count mismatch
    eg, _ = _graph(
        Op.make(
            "sdpa",
            Op.make("narrow", _chain(x, q), dim=1, start=0, length=2),
            _chain(x, k),
            _chain(x, v),
            validate=False,
        )
    )
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_chunk_on_head_axis():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    # split the head axis in half; sdpa over the first 2-head chunk —
    # chunks of the other operands need matching 2-head slices.
    def ch(c, idx):
        return Op.make("chunk", c, chunks=2, dim=1, index=idx)

    eg, _ = _graph(
        Op.make("sdpa", ch(_chain(x, q), 0), ch(_chain(x, k), 0),
                ch(_chain(x, v), 0), validate=False)
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0)


def test_unsqueeze_squeeze_and_reshape_regroup():
    """Head-axis bookkeeping through extra axes."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # squeeze a leading size-1 axis
    sq = Op.make("squeeze", Op.make("unsqueeze", _chain(x, q), dim=0),
                 dim=0)
    eg, _ = _graph(Op.make("sdpa", sq, _chain(x, k), _chain(x, v)))
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # reshape (B,h,T,d) -> (B,h,T*d) keeps head structure via regroup
    flat = Op.make("reshape", _chain(x, q), shape=(B, H, T * D))
    assert _resolve(_Ctx(eg, {}), eg.find(eg.add_term(flat))) == ()
    # regroup on a heads state: (B,h,T,d) -> (B,h,T,d) in two parts
    deep = Op.make(
        "reshape",
        Op.make("reshape", _chain(x, q), shape=(B, H, T * D)),
        shape=(B, H, T, D),
    )
    eg, _ = _graph(
        Op.make("sdpa", deep, _chain(x, k), _chain(x, v))
    )
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_select_nonhead_and_stack_pieces():
    """Per-head ``stack`` spellings compare param-erased skeletons."""
    x = Var("x", _t(B, T, I))
    ws = [Param(f"p_s{i}", _t(I, D)) for i in range(H)]

    def piece(w):
        return Op.make("matmul", x, w)  # (B,T,d) per-head

    stacked = Op.make("stack", *(piece(w) for w in ws), dim=2)
    hq = Op.make("permute", stacked, dims=(0, 2, 1, 3))  # (B,h,T,d)
    k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "kv")
    eg, _ = _graph(
        Op.make("sdpa", hq, _chain(x, k), _chain(x, v),
                validate=False)
    )
    g = torch.Generator().manual_seed(9)
    ws_t = [
        torch.randn(I, D, generator=g, dtype=torch.float64)
        for _ in range(H)
    ]
    ws_t[3] = ws_t[1].clone()  # heads 1 and 3 piecewise-equal
    tensors = {f"p_s{i}": t for i, t in enumerate(ws_t)}
    tensors["p_k"] = _dup_rows(_w(1), (1, 3))
    tensors["p_v"] = _dup_rows(_w(2), (1, 3))
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 1, 2, 1)


def test_stack_of_selects_uses_base_sig():
    """``select`` on a head axis lifts the base head's signature."""
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    base = _chain(x, q)  # (B,H,T,D)
    pieces = [
        Op.make("select", base, dim=1, index=i) for i in range(H)
    ]
    restack = Op.make("stack", *pieces, dim=1)
    k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "kv")
    eg, _ = _graph(
        Op.make("sdpa", restack, _chain(x, k), _chain(x, v))
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 2)),
        "p_k": _dup_rows(_w(1), (0, 2)),
        "p_v": _dup_rows(_w(2), (0, 2)),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 1, 0, 2)


def test_concat_on_and_off_head_axis():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # concat along T (off head axis) — merges signatures pairwise
    cat_t = Op.make("concat", _chain(x, q), _chain(x, q), dim=2)
    eg, _ = _graph(
        Op.make("sdpa", cat_t, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []
    # concat along the head axis of a 2-head half with itself — heads
    # (0,1) then (0,1) again → 4 heads, sigs (q0,q1,q0,q1)
    wq2 = Param("p_q2", _t(2 * D, I))
    half = Op.make(
        "transpose",
        Op.make(
            "reshape", Op.make("linear", x, wq2), shape=(B, T, 2, D)
        ),
        dim0=1,
        dim1=2,
    )
    doubled = Op.make("concat", half, half, dim=1)
    eg, _ = _graph(
        Op.make("sdpa", doubled, _chain(x, k), _chain(x, v))
    )
    tensors["p_q2"] = _w(9, 2, D)
    assert share_duplicate_attention_heads(eg, tensors) == []
    # now make k/v heads 2==0 and 3==1 so the doubled halves align
    tensors["p_k"][2 * D : 3 * D] = tensors["p_k"][0:D]
    tensors["p_k"][3 * D : 4 * D] = tensors["p_k"][D : 2 * D]
    tensors["p_v"][2 * D : 3 * D] = tensors["p_v"][0:D]
    tensors["p_v"][3 * D : 4 * D] = tensors["p_v"][D : 2 * D]
    tensors["p_q"] = torch.cat([tensors["p_q2"], tensors["p_q2"]])
    eg2, _ = _graph(
        Op.make("sdpa", doubled, _chain(x, k), _chain(x, v))
    )
    offers = share_duplicate_attention_heads(eg2, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 1, 0, 1)


def test_softmax_and_passthrough_unary():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # softmax over positions — head-uniform
    sm = Op.make("softmax", _chain(x, q), dim=-1)
    cont = Op.make("contiguous", _chain(x, k))
    eg, _ = _graph(
        Op.make("sdpa", sm, cont, _chain(x, v), validate=False)
    )
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # softmax over the head axis — not head-uniform
    smh = Op.make("softmax", _chain(x, q), dim=1)
    eg, _ = _graph(
        Op.make("sdpa", smh, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_elemwise_on_site_level():
    """``mul`` of a site by a (…,1) table keeps block attribution."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    gain = Param("p_gain", _t(B, T, 1))

    def scaled(w):
        lin = Op.make(
            "mul", Op.make("linear", x, w), gain
        )
        return Op.make(
            "transpose",
            Op.make("reshape", lin, shape=(B, T, H, D)),
            dim0=1,
            dim1=2,
        )

    eg, _ = _graph(
        Op.make("sdpa", scaled(q), scaled(k), scaled(v))
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_gain": torch.randn(B, T, 1, dtype=torch.float64),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_fullfeature_factor_declines_site():
    """A (B,T,o) non-uniform factor poisons per-head attribution."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    ff = Param("p_ff", _t(B, T, I))

    def scaled(w):
        lin = Op.make("mul", Op.make("linear", x, w), ff)
        return Op.make(
            "transpose",
            Op.make("reshape", lin, shape=(B, T, H, D)),
            dim0=1,
            dim1=2,
        )

    eg, _ = _graph(
        Op.make("sdpa", scaled(q), _chain(x, k), _chain(x, v))
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_ff": torch.randn(B, T, I, dtype=torch.float64),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_leaf_param_as_preattention():
    """A leaf (B,T,o) parameter reshaped to heads is a site too."""
    x = Var("x", _t(B, T, I))
    qleaf = Param("p_qleaf", _t(B, T, H * D))
    k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "kv")
    qheads = Op.make(
        "transpose",
        Op.make("reshape", qleaf, shape=(B, T, H, D)),
        dim0=1,
        dim1=2,
    )
    eg, _ = _graph(
        Op.make("sdpa", qheads, _chain(x, k), _chain(x, v))
    )
    wq = torch.randn(B, T, H * D, dtype=torch.float64)
    wq[:, :, D : 2 * D] = wq[:, :, 0:D]
    tensors = {
        "p_qleaf": wq,
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_movedim_perm_variants():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    md = Op.make(
        "movedim",
        Op.make(
            "reshape", Op.make("linear", x, q), shape=(B, T, H, D)
        ),
        source=2,
        destination=1,
    )
    eg, _ = _graph(Op.make("sdpa", md, _chain(x, k), _chain(x, v)))
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    pm = Op.make(
        "permute",
        Op.make(
            "reshape", Op.make("linear", x, q), shape=(B, T, H, D)
        ),
        dims=(0, 2, 1, 3),
    )
    eg, _ = _graph(Op.make("sdpa", pm, _chain(x, k), _chain(x, v)))
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_bad_attrs_and_shapes_decline_cleanly():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    eg, _ = _graph(_sdpa(x, q, k, v))
    ctx = _Ctx(eg, tensors)
    # enodes with unusable attrs resolve to nothing
    bad_reshape = eg.add_enode(
        "reshape", (eg.find(eg.add_term(_chain(x, q))),), {}
    )
    assert _resolve(ctx, bad_reshape) == ()
    bad_unsq = eg.add_enode("unsqueeze", (0,), {})
    assert _resolve(ctx, bad_unsq) == ()
    bad_tr = eg.add_enode("transpose", (0,), {})
    assert _resolve(ctx, bad_tr) == ()
    # a resolver busy-cycle yields nothing
    ctx.busy.add(0)
    assert _resolve(ctx, 0) == ()
    assert _piece(ctx, 0) is None
    ctx.busy.discard(0)


def test_joint_sig_none_and_key_bytes_edges():
    eg, _ = _graph(Var("x", _t(2)))
    ctx = _Ctx(eg, {})
    # missing tensor → the joint signature fails closed
    assert _joint_sig(ctx, [("blk", (("gone", "rows", 0, D),))]) is None
    # a resolvable key contributes its bytes
    ctx.src["p"] = torch.eye(2, dtype=torch.float64)
    sig = _joint_sig(ctx, [("blk", (("p", "rows", 0, 1),))])
    assert sig == (("blk", (sig[0][1][0],)),)


# ---------------------------------------------------------------------------
#  headshare_keys_hold — the compositional-cache recheck
# ---------------------------------------------------------------------------


def _recheck_for(names=("p",), cols=False):
    kind = "cols" if cols else "rows"
    keys = tuple(
        tuple((n, kind, j * D, (j + 1) * D) for n in names)
        for j in range(H)
    )
    return {"imap": (0, 1, 0, 2), "keys": keys}


def test_recheck_holds_and_breaks():
    w = _dup_rows(_w(), (0, 2))
    site = _recheck_for()
    assert headshare_keys_hold([site], lambda n: n, {"p": w})
    broken = w.clone()
    broken[2 * D, 0] += 1.0
    assert not headshare_keys_hold([site], lambda n: n,
                                   {"p": broken})
    # name mapping, missing names and tensors all decline closed
    assert headshare_keys_hold(
        [site], lambda n: "q" if n == "p" else n, {"q": w}
    )
    assert not headshare_keys_hold([site], lambda n: "q", {"p": w})
    assert not headshare_keys_hold(
        [site], lambda n: n, {"p": None}
    )
    # column-orientation keys slice the other way
    wc = w.t().contiguous()
    site2 = _recheck_for(cols=True)
    assert headshare_keys_hold([site2], lambda n: n, {"p": wc})
    assert not headshare_keys_hold(
        [site2], lambda n: n, {"p": wc.t().contiguous()}
    )  # same bytes transposed — cols slices read rows


def test_recheck_structural_mismatches():
    w = _dup_rows(_w(), (0, 2))
    site = _recheck_for()
    # key-count mismatch between equal-marked heads
    bad = {
        "imap": (0, 1, 0, 2),
        "keys": (site["keys"][0], site["keys"][1], (),
                 site["keys"][3]),
    }
    assert not headshare_keys_hold([bad], lambda n: n, {"p": w})
    # kind mismatch between heads
    keys = [list(k) for k in site["keys"]]
    keys[2] = [("p", "vec", 0, D)]
    bad2 = {"imap": (0, 1, 0, 2), "keys": tuple(tuple(k)
                                               for k in keys)}
    assert not headshare_keys_hold([bad2], lambda n: n, {"p": w})
    # empty site list vacuously holds
    assert headshare_keys_hold([], lambda n: n, {})


def test_cache_replay_headshare_gate():
    """The compositional cache replays a shared-head term only when
    the hit block's own tensors keep the head equality."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    term = _sdpa(x, q, k, v)
    ir = IR(
        root=term,
        inputs=[x],
        input_names={"x"},
        params={"p_q": q, "p_k": k, "p_v": v},
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 2)),
        "p_k": _dup_rows(_w(1), (0, 2)),
        "p_v": _dup_rows(_w(2), (0, 2)),
    }
    eg, _ = _graph(term)
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    recheck = offers[0]["recheck"]
    res = SearchResult(
        ir=ir,
        eg=eg,
        root_eid=0,
        term=term,
        param_values=tensors,
        stats={"param_sharing": {"headshare": [recheck]}},
    )
    from catopt_orchestrator.optimize import _cache_entry

    entry = _cache_entry(res)
    assert entry["headshare"] == [recheck]
    # a block whose weights honour the ties replays
    out = _cache_replay(
        entry, ir, dict(tensors), sink=object(), source=object(),
        model=None,
    )
    assert out is not None
    # one byte flipped → the assumption no longer holds → decline
    bad = dict(tensors)
    bad["p_q"] = tensors["p_q"].clone()
    bad["p_q"][2 * D, 0] += 1.0
    assert _cache_replay(
        entry, ir, bad, sink=object(), source=object(), model=None
    ) is None


# ---------------------------------------------------------------------------
#  Pipeline integration — the opt-in flag
# ---------------------------------------------------------------------------


def test_opt_in_flag_gates_the_pass():
    m = _mha()
    x = torch.randn(B, T, I, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    off = opt.search(m, x)
    assert (off.stats.get("param_sharing") or {}).get(
        "headshare"
    ) == []
    on = opt.search(m, x, detect_headshare=True)
    recs = on.stats["param_sharing"]["headshare"]
    assert len(recs) == 1 and recs[0]["imap"] == (0, 1, 0, 2)
    lr = opt.lower(on, x)
    with torch.no_grad():
        assert torch.equal(lr.module(x), m(x))


def test_optimizer_default_untouched_on_distinct_heads():
    m = _mha(dup=())
    x = torch.randn(B, T, I, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(m, x, detect_headshare=True)
    assert (res.stats.get("param_sharing") or {}).get(
        "headshare", []
    ) == []
    lr = opt.lower(res, x)
    with torch.no_grad():
        assert torch.equal(lr.module(x), m(x))


# ---------------------------------------------------------------------------
#  Resolver/handler branch coverage — synthetic states and crafted terms
# ---------------------------------------------------------------------------

from catopt_core.laws.headshare import (  # noqa: E402
    _bias_sources,
    _bounds,
    _cls_shape,
    _const_along,
    _h_concat,
    _h_cut,
    _h_dimop,
    _h_gather,
    _h_perm,
    _h_proj,
    _h_split,
    _h_squeeze,
    _h_stack,
    _h_unsq,
    _heads_merge,
    _heads_regroup,
    _interpret,
    _merge_one,
    _merge_sites,
    _piece_select,
    _prod,
    _site_regroup,
    _site_to_heads,
    _skel,
)


class _StubEG:
    """Duck-typed e-graph stand-in for impossible-to-build states."""

    def __init__(self, term=None):
        self.term = term

    def find(self, cid):
        return cid

    def _any_term_cached(self, cid):
        return self.term


def test_pure_helpers():
    assert _prod((2, 3)) == 6
    assert _prod((2, None)) is None
    assert _cls_shape(_StubEG(), 0) is None


def test_site_to_heads_branches():
    src = (("p", "rows", 0),)
    site = ("s", (B, T, H * D), src, frozenset())
    st = _site_to_heads(site, (B, T, H, D))
    assert st[0] == "h" and len(st[3]) == H
    assert _site_to_heads(site, (B * T,)) is None  # too few dims
    assert _site_to_heads(("s", (), src, frozenset()), (B, T)) is None
    node_none = ("s", (B, T, None), src, frozenset())
    assert _site_to_heads(node_none, (B, T, H, D)) is None
    assert _site_to_heads(site, (B, T, H, 5)) is None  # hb*d != width
    assert _site_to_heads(site, (B, T + 1, H, D)) is None  # prefix


def test_site_regroup_branches():
    src = (("p", "rows", 0),)
    site = ("s", (B, T, H * D), src, frozenset())
    out = _site_regroup(site, (B * T, H * D))
    assert out is not None and out[1] == (B * T, H * D)
    assert _site_regroup(site, ()) is None
    assert _site_regroup(("s", (), src, frozenset()), (1,)) is None
    assert _site_regroup(site, (B, T, H * D - 1)) is None
    # a site whose last axis is a broadcast keeps the mark
    sb = ("s", (B, T, 1), src, frozenset({2}))
    assert _site_regroup(sb, (B * T, 1))[3] == frozenset({1})


def test_heads_merge_branches():
    sigs = tuple(("e", j) for j in range(H))
    base = ("h", 1, (B, H, 2, T, D), sigs, frozenset({2}))
    m = _heads_merge(base, (B, H * 2, T, D))
    assert m[0] == "h" and m[3][2] == sigs[1]  # head 2 reads base 1
    assert m[4] == frozenset()  # repeat axis consumed
    # length mismatch / head axis not second-to-last-ish
    assert _heads_merge(base, (B, H, 2, T, D)) is None
    tail = ("h", 3, (B, T, D, H, 2), sigs, frozenset())
    assert _heads_merge(tail, (B, T, D, H * 2)) is None
    # non-int extents
    nn = ("h", 1, (B, H, None, T, D), sigs, frozenset({2}))
    assert _heads_merge(nn, (B, H * 2, T, D)) is None
    # repeat axis not provably broadcast
    nb = ("h", 1, (B, H, 2, T, D), sigs, frozenset())
    assert _heads_merge(nb, (B, H * 2, T, D)) is None
    # extent / prefix / suffix mismatches
    assert _heads_merge(base, (B, H * 2 + 1, T, D)) is None
    assert _heads_merge(base, (B + 1, H * 2, T, D)) is None
    assert _heads_merge(base, (B, H * 2, T, D + 1)) is None
    # g == 1 merges need no broadcast proof
    g1 = ("h", 1, (B, H, 1, T, D), sigs, frozenset())
    assert _heads_merge(g1, (B, H, T, D)) is not None
    # later broadcast axes shift down through the merge
    bc = ("h", 1, (B, H, 2, T, D), sigs, frozenset({2, 4}))
    assert _heads_merge(bc, (B, H * 2, T, D))[4] == frozenset({3})


def test_heads_regroup_branches():
    sigs = tuple(("e", j) for j in range(H))
    base = ("h", 1, (B, H, T, D), sigs, frozenset())
    out = _heads_regroup(base, (B, H, T * D))
    assert out is not None and out[1] == 1 and out[2] == (B, H, T * D)
    out2 = _heads_regroup(base, (B, H, T, 2, D // 2))
    assert out2 is not None and out2[1] == 1
    # unknown head extent
    nn = ("h", 1, (B, None, T, D), sigs, frozenset())
    assert _heads_regroup(nn, (B, 4, T, D)) is None
    npd = ("h", 1, (B, H, None, D), sigs, frozenset())
    assert _heads_regroup(npd, (B, H, T, D)) is None
    # no factor boundary matches
    assert _heads_regroup(base, (B * H, T, D)) is None
    # a broadcast head axis keeps its mark through regroup
    bt = ("h", 1, (B, H, T, D), sigs, frozenset({1}))
    assert _heads_regroup(bt, (B, H, T * D))[4] == frozenset({1})


def test_axis_perm_branches():
    x = Var("x", _t(B, T, I))
    lin = Op.make("linear", x, Param("p_w", _t(H * D, I)))
    eg = EGraph()
    le = eg.add_term(lin)
    node = next(
        n for n in eg._classes[eg.find(le)].nodes if n.op == "linear"
    )
    # transpose with missing attrs
    tn = eg.add_enode("transpose", (le,), {})
    ctx = _Ctx(eg, {"p_w": _w()})
    assert _resolve(ctx, tn) == ()
    # permute: wrong arity / non-int dims
    pm_bad = eg.add_term(
        Op.make("permute", lin, dims=(0, 1), validate=False)
    )
    assert _resolve(ctx, pm_bad) == ()
    pm_dup = eg.add_term(
        Op.make("permute", lin, dims=(0, 0, 1, 3), validate=False)
    )
    assert _resolve(ctx, pm_dup) == ()
    # movedim: missing / mismatched source-destination
    md = eg.add_term(Op.make("movedim", lin, validate=False))
    assert _resolve(ctx, md) == ()
    md2 = eg.add_term(
        Op.make("movedim", lin, source=(0, 1), destination=2,
                validate=False)
    )
    assert _resolve(ctx, md2) == ()
    # _axis_perm on an unknown axis op
    fake = Op.make("frobnicate", lin, validate=False)
    fe = eg.add_enode("frobnicate", (le,), {})
    fn = next(iter(eg._classes[fe].nodes))
    assert _interpret(ctx, fn, (B, T, I)) == []
    _ = fake


def test_site_transpose_keeps_and_drops():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    lin = Op.make("linear", x, w)
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    tr_keep = eg.add_term(Op.make("transpose", lin, dim0=0, dim1=1))
    assert _resolve(ctx, tr_keep)[0][0] == "s"
    tr_drop = eg.add_term(Op.make("transpose", lin, dim0=1, dim1=2))
    assert _resolve(ctx, tr_drop) == ()


def test_bias_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    bu = Param("p_bu", _t(1))  # broadcastable scalar-ish bias
    bv = Var("bv", _t(H * D))
    bx = Param("p_bx", _t(H * D))
    eg = EGraph()
    ctx = _Ctx(
        eg,
        {"p_w": _w(), "p_bu": torch.randn(1, dtype=torch.float64)},
    )
    # scalar Const bias — contributes nothing
    t1 = eg.add_term(Op.make("linear", x, w, Const(0.5)))
    assert _resolve(ctx, t1)[0][0] == "s"
    # all-dims-1 bias — uniform
    t2 = eg.add_term(Op.make("linear", x, w, bu))
    assert _resolve(ctx, t2)[0][0] == "s"
    # Var bias — unattributable, vetoes the site
    t3 = eg.add_term(Op.make("linear", x, w, bv, validate=False))
    assert _resolve(ctx, t3) == ()
    # computed bias (non-leaf enode in the class) — vetoes
    t4 = eg.add_term(
        Op.make("linear", x, w, Op.make("add", bx, bx),
                validate=False)
    )
    assert _resolve(ctx, t4) == ()
    # missing bias tensor — vetoes
    t5 = eg.add_term(Op.make("linear", x, w, Param("p_gone", _t(H * D))))
    assert _resolve(ctx, t5) == ()
    # 3-D weight and an absent weight both decline
    t6 = eg.add_term(
        Op.make("linear", x, Param("p_3d", _t(2, 2, 2)),
                validate=False)
    )
    assert _resolve(ctx, t6) == ()
    t7 = eg.add_term(Op.make("linear", x, Param("p_absent", _t(H * D, I))))
    assert _resolve(ctx, t7) == ()
    # _h_proj directly: class-shape / weight-width mismatch declines
    lin = eg.add_term(Op.make("linear", x, w))
    node = next(
        n for n in eg._classes[eg.find(lin)].nodes if n.op == "linear"
    )
    assert _h_proj(ctx, node, (B, T, H * D + 1)) == []
    assert _h_proj(ctx, node, None) == []
    # an over-ary linear enode hits the arity guard
    wide = eg.add_enode("linear", (0, lin, lin, lin), {})
    assert _resolve(ctx, wide) == ()


def test_unsq_squeeze_expand_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    lin = Op.make("linear", x, w)
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    le = eg.add_term(lin)
    # unsqueeze past the feature axis displaces it — no site
    assert _resolve(ctx, eg.add_term(
        Op.make("unsqueeze", lin, dim=-1))) == ()
    # unsqueeze a batch axis keeps the site
    st = _resolve(ctx, eg.add_term(Op.make("unsqueeze", lin, dim=0)))
    assert st[0][0] == "s"
    # dim-less or non-int-dim ops decline
    assert _resolve(ctx, eg.add_term(
        Op.make("unsqueeze", lin, validate=False))) == ()
    assert _resolve(ctx, eg.add_term(
        Op.make("squeeze", lin, validate=False))) == ()
    # squeeze on the feature axis declines; off-axis keeps the site
    assert _resolve(ctx, eg.add_term(
        Op.make("squeeze", lin, dim=-1))) == ()
    st = _resolve(ctx, eg.add_term(Op.make("squeeze", lin, dim=0)))
    assert st[0][0] == "s"
    # expand: bad shape attr, shrinking, non-broadcastable axes
    e_bad = eg.add_term(
        Op.make("expand", lin, shape="nope", validate=False)
    )
    assert _resolve(ctx, e_bad) == ()
    e_small = eg.add_term(Op.make("expand", lin, shape=(T, I)))
    assert _resolve(ctx, e_small) == ()
    e_mis = eg.add_term(Op.make("expand", lin, shape=(B, T, I + 1)))
    assert _resolve(ctx, e_mis) == ()
    # expanding a non-feature axis keeps the site and marks it bcast
    e_ok = eg.add_term(
        Op.make("expand", lin, shape=(2, B, T, I))
    )
    st = _resolve(ctx, e_ok)
    assert st[0][0] == "s" and st[0][1] == (2, B, T, I)
    e_keep = eg.add_term(
        Op.make("expand", lin, shape=(B, T, -1), validate=False)
    )
    assert _resolve(ctx, e_keep)[0][0] == "s"


def test_expand_head_axis_copies():
    """A single-head operand broadcast to H heads — degenerate share."""
    x = Var("x", _t(B, T, I))
    w1 = Param("p_w1", _t(D, I))
    k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "kv")
    one = Op.make(
        "expand",
        Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make("linear", x, w1),
                shape=(B, T, 1, D),
            ),
            dim0=1,
            dim1=2,
        ),
        shape=(B, H, T, D),
    )
    eg, _ = _graph(Op.make("sdpa", one, _chain(x, k), _chain(x, v)))
    tensors = {
        "p_w1": _w(11, 1, D),
        "p_k": torch.cat([_w(12, 1, D, I)] * H),
        "p_v": torch.cat([_w(13, 1, D, I)] * H),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 0, 0)
    # a broadcast head axis survives a following regroup (bcast mark)
    hstate = ("h", 1, (B, H, T, D), (("e", 0),) * H, frozenset({1}))
    assert _heads_regroup(hstate, (B, H, T * D)) is not None


def test_cut_split_gather_edge_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    lin = Op.make("linear", x, w)
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    # select on the feature axis — a site loses its last dim
    assert _resolve(ctx, eg.add_term(
        Op.make("select", lin, dim=-1, index=0))) == ()
    # select off the feature axis keeps the site (rank drops)
    st = _resolve(
        ctx, eg.add_term(Op.make("select", lin, dim=0, index=0))
    )
    assert st[0][0] == "s" and st[0][1] == (T, H * D)
    # slice on the feature axis shifts offsets
    st = _resolve(
        ctx,
        eg.add_term(
            Op.make("slice", lin, dim=-1, start=D, end=3 * D)
        ),
    )
    assert st[0][0] == "s" and st[0][2] == (("p_w", "rows", D),)
    # empty / inverted / non-int slices decline
    assert _resolve(ctx, eg.add_term(
        Op.make("slice", lin, dim=-1, start=8, end=4))) == ()
    assert _resolve(ctx, eg.add_term(
        Op.make("slice", lin, dim=-1, start=None, end=4,
                validate=False))) == ()
    # index_select on the feature axis declines
    assert _resolve(ctx, eg.add_term(
        Op.make("index_select", lin, dim=-1, index=(0,)))) == ()
    # non-canonical dims/attrs — exercised at handler level because
    # typing crashes before the resolver can see them
    le = eg.add_term(lin)
    ctx._en = lambda op, attrs: ENode(op, (le,), tuple(attrs.items()))
    assert _h_cut(ctx, ctx._en("narrow", {"dim": -1}),
                  (T, H * D)) == []
    assert _h_cut(ctx, ctx._en("slice", {"dim": "d"}),
                  (T, H * D)) == []
    assert _h_split(ctx, ctx._en("split", {"dim": 1.5}),
                    (T, H * D)) == []
    assert _h_gather(ctx, ctx._en("index_select", {"dim": None}),
                     (T, H * D)) == []
    # shape=None guards on every axis handler
    for fn in (_h_cut, _h_split, _h_gather, _h_dimop):
        assert fn(ctx, ctx._en("x", {"dim": 0}), None) == []


def test_split_chunk_edge_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    lin = Op.make("linear", x, w)
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    # off-axis split keeps the site
    st = _resolve(
        ctx,
        eg.add_term(
            Op.make("split", lin, sizes=(1, 1), dim=0, index=0)
        ),
    )
    assert st[0][0] == "s" and st[0][1] == (1, T, H * D)
    # sizes not summing to the extent → decline
    assert _resolve(ctx, eg.add_term(
        Op.make("split", lin, sizes=(1, 1), dim=-1, index=0))) == ()
    # index out of range → decline
    assert _resolve(ctx, eg.add_term(
        Op.make("split", lin, sizes=(8, 8), dim=-1, index=4))) == ()
    # uneven chunk → decline
    assert _resolve(ctx, eg.add_term(
        Op.make("chunk", lin, chunks=3, dim=-1, index=0))) == ()
    # chunk index out of range → decline
    assert _resolve(ctx, eg.add_term(
        Op.make("chunk", lin, chunks=4, dim=-1, index=9,
                validate=False))) == ()
    # feature-axis split offsets tracked
    st = _resolve(
        ctx,
        eg.add_term(
            Op.make(
                "split", lin, sizes=(D, D, D, D), dim=-1, index=1
            )
        ),
    )
    assert st[0][0] == "s" and st[0][2] == (("p_w", "rows", D),)
    # missing index → _part_span declines
    assert _resolve(ctx, eg.add_term(
        Op.make("split", lin, sizes=(8, 8), dim=-1,
                validate=False))) == ()
    # non-int extent — a reshape to an unknown-width split
    # heads-level split on a non-head axis passes through
    q = Param("p_q", _t(H * D, I))
    tensors = {"p_w": _w(), "p_q": _w()}
    ctx2 = _Ctx(eg, tensors)
    sp = eg.add_term(
        Op.make("split", _chain(x, q), sizes=(2, 3), dim=2, index=0)
    )
    assert _resolve(ctx2, sp)[0][0] == "h"


def test_dimop_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    lin = Op.make("linear", x, w)
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    # softmax on the feature axis mixes head blocks — decline
    assert _resolve(ctx, eg.add_term(
        Op.make("softmax", lin, dim=-1))) == ()
    # softmax off the feature axis is uniform
    st = _resolve(ctx, eg.add_term(Op.make("softmax", lin, dim=0)))
    assert st[0][0] == "s"
    # no dim → decline
    assert _resolve(ctx, eg.add_term(
        Op.make("softmax", lin, validate=False))) == ()


def test_merge_one_and_heads_branches():
    x = Var("x", _t(B, T, I))
    a = Param("p_a", _t(H, T, D))
    b = Param("p_b", _t(H, T, D))
    mul = Op.make("mul", a, b)
    eg = EGraph()
    me = eg.add_term(mul)
    ctx = _Ctx(
        eg,
        {
            "p_a": torch.randn(H, T, D, dtype=torch.float64),
            "p_b": torch.randn(H, T, D, dtype=torch.float64),
        },
    )
    node = next(n for n in eg._classes[me].nodes if n.op == "mul")
    h0 = ("h", 0, (H, T, D), (("e", j) for j in range(H)), frozenset())
    h0 = ("h", 0, (H, T, D), tuple(("e", j) for j in range(H)),
          frozenset())
    # head axis misaligned
    h1 = ("h", 1, (T, H, D), tuple(("e", j) for j in range(H)),
          frozenset())
    assert _merge_one(ctx, node, (H, T, D), h0, h1) is None
    # extent-1 heads replicate against a wider partner
    h1x = ("h", 0, (H, T, D), (("e", 0),), frozenset())
    got = _merge_one(ctx, node, (H, T, D), h0, h1x)
    assert got is not None and len(got[3]) == H
    # mismatched extents decline
    h2 = ("h", 0, (2, T, D), (("e", 0), ("e", 1)), frozenset())
    assert _merge_one(ctx, node, (H, T, D), h0, h2) is None
    # _const_along: extent-1 axis constant, wider partner not
    pc_e = eg.add_term(Param("p_c", _t(1, T, D)))
    ctx.src["p_c"] = torch.randn(1, T, D, dtype=torch.float64)
    assert _const_along(ctx, pc_e, 0, 3)
    assert not _const_along(ctx, eg.add_term(b), 0, 3)
    # a child whose class shape is bigger than the parent's declines
    assert not _const_along(ctx, me, 0, 2)


def test_elemwise_left_heads_right_uniform():
    """heads ⨯ scalar and scalar ⨯ heads both keep attribution."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    c = Param("p_c", TensorType(()))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_c": torch.tensor(2.0, dtype=torch.float64),
    }
    for term in (
        Op.make("mul", _chain(x, q), c),
        Op.make("mul", c, _chain(x, q)),
        Op.make("sub", _chain(x, q), c),
        Op.make("pow", _chain(x, q), c),
        Op.make("div", _chain(x, q), c),
    ):
        eg, _ = _graph(Op.make("sdpa", term, _chain(x, k),
                               _chain(x, v)))
        assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_two_heads_children_merge():
    """``mul(heads, heads)`` — merged signature lists."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    q2 = Param("p_q2", _t(H * D, I))
    both = Op.make("mul", _chain(x, q), _chain(x, q2))
    eg, _ = _graph(Op.make("sdpa", both, _chain(x, k), _chain(x, v)))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_q2": _dup_rows(_w(7), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    tensors["p_q2"][0, 0] += 1.0  # break head-0 == head-1 in q2
    eg, _ = _graph(Op.make("sdpa", both, _chain(x, k), _chain(x, v)))
    assert share_duplicate_attention_heads(eg, tensors) == []
    # a single-head heads-child replicates against a 4-head one
    w1 = Param("p_w1", _t(D, I))
    one = Op.make(
        "transpose",
        Op.make(
            "reshape",
            Op.make("linear", x, w1),
            shape=(B, T, 1, D),
        ),
        dim0=1,
        dim1=2,
    )
    merged = Op.make("mul", _chain(x, q), one)
    eg, _ = _graph(
        Op.make("sdpa", merged, _chain(x, k), _chain(x, v))
    )
    tensors["p_q"] = _dup_rows(_w(), (0, 1))
    tensors["p_w1"] = _w(8, 1, D)
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1


def test_heads_off_axis_select_and_slice():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # select the batch axis — heads survive at the shifted position
    sel = Op.make("select", _chain(x, q), dim=0, index=0)
    kk = Op.make("select", _chain(x, k), dim=0, index=0)
    vv = Op.make("select", _chain(x, v), dim=0, index=0)
    eg, _ = _graph(Op.make("sdpa", sel, kk, vv))
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    # head-axis select demotes to a piece — no heads state to merge
    piece = Op.make("select", _chain(x, q), dim=1, index=0)
    eg, _ = _graph(
        Op.make("sdpa", piece, kk, vv, validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_concat_branches():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg = EGraph()
    ctx = _Ctx(eg, {"p_q": _w()})
    # concat of unresolvable children / unshapeable result → ()
    bad = Op.make(
        "concat",
        Op.make("frobnicate", x, validate=False),
        x,
        dim=0,
        validate=False,
    )
    assert _resolve(ctx, eg.add_term(bad)) == ()
    # non-int dim attr — handler-level, typing rejects it earlier
    le = eg.add_term(Op.make("linear", x, Param("p_q", _t(H * D, I))))
    dn = ENode("concat", (le, le), (("dim", "d"),))
    assert _h_concat(ctx, dn, (2 * B, T, H * D)) == []
    # heads children with different ranks — concat can't align them
    h3 = Op.make(
        "reshape",
        Op.make("linear", Var("x2", _t(B, I)), q),
        shape=(B, H, D),
    )
    bad3 = eg.add_term(
        Op.make("concat", _chain(x, q), h3, dim=0, validate=False)
    )
    assert _resolve(ctx, bad3) == ()
    # different head counts on a non-head concat — signatures differ
    h2 = Op.make(
        "transpose",
        Op.make(
            "reshape",
            Op.make("linear", x, Param("p_h2", _t(2 * D, I))),
            shape=(B, T, 2, D),
        ),
        dim0=1,
        dim1=2,
    )
    bad4 = eg.add_term(
        Op.make("concat", _chain(x, q), h2, dim=0, validate=False)
    )
    assert _resolve(ctx, bad4) == ()


def test_stack_branches():
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(I, D))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w(3, 1, D)})
    p = Op.make("matmul", x, w)
    # empty children
    e0 = eg.add_enode("stack", (), {"dim": 0})
    assert _resolve(ctx, e0) == ()
    # non-int dim — handler level (typing rejects it earlier)
    pe = eg.add_term(p)
    dn = ENode("stack", (pe, pe), (("dim", "d"),))
    assert _h_stack(ctx, dn, (2, B, T, D)) == []
    # an unresolvable piece vetoes the whole stack
    bad2 = eg.add_term(
        Op.make(
            "stack",
            p,
            Op.make("frobnicate", x, validate=False),
            dim=0,
            validate=False,
        )
    )
    # frobnicate still resolves to a skeleton piece — stack survives
    st = _resolve(ctx, bad2)
    assert st[0][0] == "h"
    # same piece twice — memoised _piece path
    eg2 = EGraph()
    ctx2 = _Ctx(eg2, {"p_w": _w(3, 1, D)})
    dd = eg2.add_term(Op.make("stack", p, p, dim=0))
    st2 = _resolve(ctx2, dd)
    assert st2[0][0] == "h" and st2[0][3][0] == st2[0][3][1]


def test_piece_select_variants():
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    base = _chain(x, q)
    eg = EGraph()
    be = eg.add_term(base)
    ctx = _Ctx(eg, {"p_q": _w()})
    # a select on a non-head axis falls back to the skeleton path
    sel_t = Op.make("select", base, dim=2, index=0)
    se = eg.add_term(sel_t)
    piece = _piece(ctx, se)
    assert piece[0][0] == "skel"  # non-head select → skeleton
    # a bare select (missing attrs) falls to skeleton too
    sel_b = eg.add_term(Op.make("select", base))
    assert _piece(ctx, sel_b)[0][0] == "skel"
    # single-element index_select on the head axis → sel entry
    isel = eg.add_term(
        Op.make("index_select", base, dim=1, index=(2,))
    )
    p_isel = _piece(ctx, isel)
    assert p_isel[0] == ("sel", "blk")
    # multi-index index_select → skeleton
    isel2 = eg.add_term(
        Op.make("index_select", base, dim=1, index=(0, 2))
    )
    assert _piece(ctx, isel2)[0][0] == "skel"
    # a leaf piece is a whole-tensor skeleton
    pl = eg.add_term(q)
    leaf_piece = _piece(ctx, pl)
    assert leaf_piece == (("skel", ("p",)), (("p_q", "whole", 0, 0),))
    # _piece on a class whose canon term is absent (stub)
    ctx2 = _Ctx(_StubEG(), {})
    assert _piece(ctx2, 0) is None
    # _piece_select direct: non-Op terms and non-select ops decline
    assert _piece_select(ctx, Param("p", _t(1))) is None
    assert _piece_select(ctx, Op.make("mul", x, x)) is None


def test_skel_fallback_branches():
    keys = []
    assert _skel(Const(2.5), keys) == ("c", 2.5)
    assert _skel("rawkey", keys) == ("l", "'rawkey'")
    # a piece skeleton embeds consts and vars
    x = Var("x", _t(B, T, I))
    w = Param("p", _t(I, D))
    keys = []
    sk = _skel(
        Op.make("mul", Op.make("matmul", x, w), Const(2.0)), keys
    )
    assert sk[0] == "op" and keys == [("p", "whole", 0, 0)]


def test_offer_edge_branches():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    # operands of rank < 3 cannot carry a batch head axis
    q2d = Op.make("reshape", Param("p_q2", _t(H * D,)), shape=(H, D))
    eg, _ = _graph(
        Op.make("sdpa", q2d, _chain(x, k), _chain(x, v),
                validate=False)
    )
    tensors["p_q2"] = torch.randn(H * D, dtype=torch.float64)
    assert share_duplicate_attention_heads(eg, tensors) == []
    # head axis at rank-2 (…, h, d tail) — sharing it would corrupt
    # attention; the pass declines.
    no_tr = Op.make(
        "reshape", Op.make("linear", x, q), shape=(B, T, H, D)
    )
    eg, _ = _graph(
        Op.make("sdpa", no_tr, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []
    # mismatched head axes across operands decline
    q_at2 = Op.make(
        "permute", _chain(x, q), dims=(0, 2, 1, 3)
    )  # (B,T,H,D) — heads at axis 2
    eg, _ = _graph(
        Op.make("sdpa", q_at2, _chain(x, k), _chain(x, v),
                validate=False)
    )
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_offer_joint_sig_failure():
    """A key unresolvable at signature time fails closed."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, _ = _graph(_sdpa(x, q, k, v))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    ctx = _Ctx(eg, dict(tensors))
    # resolve the operand classes first, then pull the tensor
    sdpa_node = None
    for cid in eg._classes:
        for n in eg._classes[cid].nodes:
            if n.op == "sdpa":
                sdpa_node = (cid, n)
    for c in sdpa_node[1].children:
        _resolve(ctx, c)
    del ctx.src["p_q"]
    from catopt_core.laws.headshare import _offer_sdpa

    assert _offer_sdpa(ctx, eg.find(sdpa_node[0]), sdpa_node[1],
                     True) is None


def test_second_run_noop_offer():
    """A repeat offer is already merged — union reports False."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, _ = _graph(_sdpa(x, q, k, v))
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert len(share_duplicate_attention_heads(eg, tensors)) == 1
    again = share_duplicate_attention_heads(eg, tensors)
    assert again == []


def test_two_sdpa_sites_share_bytes_cache():
    """Two attention sites over the same params both offer."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    term = Op.make(
        "add",
        _sdpa(x, q, k, v),
        _sdpa(x, q, k, v, is_causal=True),
        validate=False,
    )
    eg, _ = _graph(term)
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 2


def test_shape_none_class_declines():
    """An unshapeable child class declines every check cleanly."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    mystery = Op.make("frobnicate", x, validate=False)
    mul = Op.make("mul", _chain(x, q), mystery)
    eg, _ = _graph(
        Op.make("sdpa", mul, _chain(x, k), _chain(x, v),
                validate=False)
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


# ---------------------------------------------------------------------------
#  Remaining decline/edge branches — resolver corner cases
# ---------------------------------------------------------------------------


def test_bias_source_nonuniform_param_vetoes():
    """A bias leaf that is neither ``(o,)`` nor all-ones vetoes."""
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    eg = EGraph()
    ctx = _Ctx(
        eg,
        {
            "p_w": _w(),
            "p_b2": torch.randn(2, 2, dtype=torch.float64),
            "p_b5": torch.randn(5, dtype=torch.float64),
        },
    )
    # a 2-D bias is neither per-output nor uniform → None (veto)
    b2 = eg.add_term(Param("p_b2", _t(2, 2)))
    assert _bias_sources(ctx, b2, H * D) is None
    # a 1-D bias of the wrong width — same veto
    b5 = eg.add_term(Param("p_b5", _t(5)))
    assert _bias_sources(ctx, b5, H * D) is None
    # … and the enclosing linear site declines with it
    lin = eg.add_term(
        Op.make("linear", x, w, Param("p_b2", _t(2, 2)),
                validate=False)
    )
    assert _resolve(ctx, lin) == ()


def test_bare_leaf_enode_and_proj_edge_arms():
    """Attr-less ``leaf`` and wrong-arity/non-matrix ``_h_proj`` arms."""
    x = Var("x", _t(B, T, I))
    w3d = Param("p_3d", _t(2, 2, 2))
    eg = EGraph()
    ctx = _Ctx(
        eg, {"p_3d": torch.randn(2, 2, 2, dtype=torch.float64)}
    )
    # a leaf enode carrying no attrs contributes no site
    bare = eg.add_enode("leaf", (), {})
    assert _resolve(ctx, bare) == ()
    # matmul is a 2-argument site — a 3-child enode misses both arms
    le = eg.add_term(x)
    assert _h_proj(
        ctx, ENode("matmul", (le, le, le), ()), (B, T, I)
    ) == []
    # a weight param whose tensor is not a matrix is skipped
    t = eg.add_term(Op.make("linear", x, w3d, validate=False))
    assert _resolve(ctx, t) == ()


def test_axis_handler_none_shapes_and_bad_permute_dims():
    """Shapeless classes and non-permutation dims fail closed."""
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(H * D, I))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w()})
    le = eg.add_term(Op.make("linear", x, w))
    # permute dims that are not a list of ints → no axis map
    for dims in (("a", "b"), "nope"):
        en = ENode("permute", (le,), (("dims", dims),))
        assert _h_perm(ctx, en, (B, T, I)) == []
    # every axis-op handler declines a class with no inferred shape
    tr = ENode("transpose", (le,), (("dim0", 0), ("dim1", 1)))
    assert _h_perm(ctx, tr, None) == []
    un = ENode("unsqueeze", (le,), (("dim", 0),))
    assert _h_unsq(ctx, un, None) == []
    sq = ENode("squeeze", (le,), (("dim", 0),))
    assert _h_squeeze(ctx, sq, None) == []
    cat = ENode("concat", (le, le), (("dim", 0),))
    assert _h_concat(ctx, cat, None) == []
    assert _h_concat(ctx, cat, ()) == []


def test_squeeze_on_head_axis_and_uneven_head_chunk():
    """Cuts on the head axis that are not contiguous blocks decline."""
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_q": _w()})
    # squeezing the head axis itself is not head-safe
    sq = eg.add_term(
        Op.make("squeeze", _chain(x, q), dim=1, validate=False)
    )
    assert _resolve(ctx, sq) == ()
    # a head-axis chunk that does not split evenly has no span
    ck = eg.add_term(
        Op.make(
            "chunk", _chain(x, q), chunks=3, dim=1, index=0,
            validate=False,
        )
    )
    assert _resolve(ctx, ck) == ()


def test_expand_growing_feature_axis_declines():
    """A site may not grow its flat feature axis — blocks would mix."""
    eg = EGraph()
    ctx = _Ctx(
        eg, {"p_1": torch.randn(B, T, 1, dtype=torch.float64)}
    )
    e = eg.add_term(
        Op.make(
            "expand",
            Param("p_1", _t(B, T, 1)),
            shape=(B, T, 4),
            validate=False,
        )
    )
    assert _resolve(ctx, e) == ()


def test_index_select_off_feature_axis_keeps_site():
    """Gathering a leading axis of a flat-feature site is head-safe."""
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_q": _w()})
    g = eg.add_term(
        Op.make(
            "index_select", Op.make("linear", x, q), dim=0, index=(0,)
        )
    )
    st = _resolve(ctx, g)
    assert st[0][0] == "s" and st[0][1] == (1, T, H * D)


def test_bounds_rejects_unknown_extent():
    """Slice/narrow bounds need a concrete integer extent."""
    node = ENode("slice", (), ())
    assert _bounds({"dim": -1}, node, None) is None
    assert _bounds({"dim": -1}, node, "x") is None


def test_elemwise_heads_misaligned_under_broadcast():
    """Head axes landing on different parent axes cannot merge."""
    x = Var("x", _t(B, T, I))
    x3 = Var("x3", _t(H, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    w3 = Param("p_w3", _t(T * D, I))
    # (H,T,D) heads at axis 1 — broadcast against (B,H,T,D) shifts it
    # onto the parent's T axis, so _merge_one finds no common axis.
    h3 = Op.make(
        "reshape", Op.make("linear", x3, w3), shape=(H, T, D)
    )
    mul = Op.make("mul", _chain(x, q), h3)
    eg = EGraph()
    mc = eg.add_term(mul)
    eg.add_term(Op.make("sdpa", mul, _chain(x, k), _chain(x, v)))
    ctx = _Ctx(eg, {"p_q": _w(), "p_w3": _w(5, T, D)})
    assert _resolve(ctx, mc) == ()
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_w3": _w(5, T, D),
    }
    assert share_duplicate_attention_heads(eg, tensors) == []


def test_width_one_left_site_merges_right_sources():
    """``mul((…,1) site, full site)`` — the wide site alone keys."""
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    p1 = Param("p_1", _t(B, T, 1))
    scaled = Op.make(
        "transpose",
        Op.make(
            "reshape",
            Op.make("mul", p1, Op.make("linear", x, q)),
            shape=(B, T, H, D),
        ),
        dim0=1,
        dim1=2,
    )
    eg, _ = _graph(
        Op.make("sdpa", scaled, _chain(x, k), _chain(x, v))
    )
    tensors = {
        "p_q": _dup_rows(_w(), (0, 1)),
        "p_k": _dup_rows(_w(1), (0, 1)),
        "p_v": _dup_rows(_w(2), (0, 1)),
        "p_1": torch.randn(B, T, 1, dtype=torch.float64),
    }
    offers = share_duplicate_attention_heads(eg, tensors)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 1, 2)


def test_concat_offhead_sig_count_mismatch_skips():
    """Off-axis concat of different head counts cannot merge sigs."""
    x = Var("x", _t(B, T, I))
    q = Param("p_q", _t(H * D, I))
    w2 = Param("p_w2", _t(2 * D, I))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_q": _w(), "p_w2": _w(9, 2, D)})
    c4 = eg.add_term(_chain(x, q))
    c2 = eg.add_term(
        Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make("linear", x, w2),
                shape=(B, T, 2, D),
            ),
            dim0=1,
            dim1=2,
        )
    )
    en = ENode("concat", (c4, c2), (("dim", 0),))
    assert _h_concat(ctx, en, (2 * B, H, T, D)) == []


def test_stack_busy_piece_and_resolve_busy_class():
    """Cycle guards: a busy piece vetoes ``stack``; a busy class → ()."""
    x = Var("x", _t(B, T, I))
    w = Param("p_w", _t(I, D))
    eg = EGraph()
    ctx = _Ctx(eg, {"p_w": _w(3, 1, D)})
    pe = eg.add_term(Op.make("matmul", x, w))
    ctx.busy.add(eg.find(pe))
    en = ENode("stack", (pe,), (("dim", 0),))
    assert _h_stack(ctx, en, (1, B, T, D)) == []
    # _resolve on a busy class it has never memoised — the DFS-cycle arm
    pz = eg.add_term(Param("p_z", _t(1)))
    ctx.busy.add(eg.find(pz))
    assert _resolve(ctx, pz) == ()
    assert eg.find(pz) not in ctx.memo


def test_merge_sites_incompatible_widths_decline():
    """Sites of different non-uniform widths cannot share blocks."""
    x = Var("x", _t(B, T, I))
    eg = EGraph()
    ctx = _Ctx(eg, {})
    xe = eg.add_term(x)
    pe = eg.add_term(Param("p", _t(B, T, I)))
    node = ENode("mul", (xe, pe), ())
    s4 = ("s", (B, T, 4), (("p_a", "vec", 0),), frozenset())
    s8 = ("s", (B, T, 8), (("p_b", "vec", 0),), frozenset())
    # w0 != w1, neither width-1 — the pair is dropped by every arm
    assert _merge_sites(ctx, node, (B, T, I), (s4,), (s8,)) == []
    assert _merge_sites(ctx, node, (B, T, I), (s8,), (s4,)) == []
