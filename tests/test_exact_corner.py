"""Regression tests pinning the exact-structure corner's measured claim:

    "~0% on dense LLMs, real on structured ones."

Measured by /tmp/exact_probe.py across model archetypes under the storage
cost axis (``param_bytes_cost_for``), fp64 outputs checked every run:

  * dense (real stories15M + synthetic 2-layer MHA/FFN transformer)
      -> ~0% (real file: 0 duplicate tensors/slices; synth: 0.08% — one
         norm-scale vector absorbed into the head weight).
  * GQA 8q/2kv with repeat_kv materialised -> 37.5% (slice dedup).
  * tied embedding/classifier stored twice -> 50% (whole-tensor tying).
  * adapter-merged (W + B·A stored, biased) -> 11.0% (param-only fold).
  * MoE with weight-tied routed experts -> 75% (whole-tensor tying).
  * dead param -> the unused tensor drops out of the weights file.

Honest limits pinned too:

  * the UNmerged adapter form (base(x) + B·A·x) currently REGRESSES in
    stored bytes — pair_shared_input_linears' forced extraction
    concatenates phantom weight classes that the dag_cost leaf billing
    prices as free; the lowered module materialises them anyway.
  * weight-tied experts on a SHARED input save only 37.5% (not the
    routed 75%) — pairing re-materialises the up-projections.
  * composed dense linears w2(w1·x) fold exactly but net 0 B — the
    stacked-concat member stores both spellings.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from catopt.cost import param_bytes_cost_for
from catopt.optimize import optimize_model, param_report

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/tmp/stories15M.bin"

BYTES = 8          # fp64


def _randn(shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _opt(model, example, **kw):
    model = model.eval().double()
    if example.is_floating_point():
        example = example.double()
    low, stats = optimize_model(model, example,
                                cost_fn=param_bytes_cost_for(),
                                verbose=False, **kw)
    with torch.no_grad():
        ref = model(example.clone())
        out = low(example.clone())
    rel = ((out - ref).abs().max().item()
           / (ref.abs().max().item() + 1e-8))
    return low, stats, param_report(model, low), rel


# ---------------------------------------------------------------------------
#  Archetype builders (kept in lock-step with /tmp/exact_probe.py)
# ---------------------------------------------------------------------------

class _DenseBlock(nn.Module):
    def __init__(self, d=64, n_heads=4, hidden=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.h = d, n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.scale = nn.Parameter(_randn((d,), g))
        with torch.no_grad():
            for lin in (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2):
                lin.weight.copy_(_randn(lin.weight.shape, g))

    def forward(self, x):
        B, T, d = x.shape
        hd = d // self.h
        q = self.wq(x).view(B, T, self.h, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, d))
        x = x + self.w2(F.gelu(self.w1(x)))
        return x * self.scale


class _DenseLM(nn.Module):
    def __init__(self, n_layers=2, d=64, vocab=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [_DenseBlock(d=d, seed=i + 1) for i in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class _GQAProj(nn.Module):
    DIM, NH, NKV = 64, 8, 2
    HD = DIM // NH

    def __init__(self, kv_map, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.wq = nn.Linear(self.DIM, self.DIM, bias=False)
        self.wk = nn.Linear(self.DIM, self.DIM, bias=False)
        self.wv = nn.Linear(self.DIM, self.DIM, bias=False)
        self.wo = nn.Linear(self.DIM, self.DIM, bias=False)
        with torch.no_grad():
            for lin in (self.wq, self.wo):
                lin.weight.copy_(_randn(lin.weight.shape, g))
            for lin in (self.wk, self.wv):
                uniq = [_randn((self.HD, self.DIM), g)
                        for _ in range(max(kv_map) + 1)]
                lin.weight.copy_(torch.cat([uniq[i] for i in kv_map], dim=0))

    def forward(self, x):
        q = self.wq(x[..., :self.DIM])
        k = self.wk(x[..., self.DIM:2 * self.DIM])
        v = self.wv(x[..., 2 * self.DIM:3 * self.DIM])
        return self.wo(q + k + v)


class _AdapterMerged(nn.Module):
    I, O, R = 128, 128, 8

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(_randn((self.O, self.I), g))
        self.A = nn.Parameter(_randn((self.R, self.I), g))
        self.B = nn.Parameter(_randn((self.O, self.R), g))
        self.bias = nn.Parameter(_randn((self.O,), g))

    def forward(self, x):
        return F.linear(x, self.W + self.B @ self.A, self.bias)


class _AdapterUnmerged(nn.Module):
    I, O, R = 128, 128, 8

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = nn.Linear(self.I, self.O, bias=True)
        self.la = nn.Linear(self.I, self.R, bias=False)
        self.lb = nn.Linear(self.R, self.O, bias=False)
        with torch.no_grad():
            for lin in (self.base, self.la, self.lb):
                lin.weight.copy_(_randn(lin.weight.shape, g))

    def forward(self, x):
        return self.base(x) + self.lb(self.la(x))


class _TiedTwice(nn.Module):
    def __init__(self, vocab=512, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight.data = self.embed.weight.data.clone()

    def forward(self, idx):
        return self.head(self.embed(idx))


class _MoERouted(nn.Module):
    def __init__(self, n=4, d=64, hidden=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = _randn((hidden, d), g)
        w2 = _randn((d, hidden), g)
        self.experts = nn.ModuleList()
        for _ in range(n):
            e = nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(),
                            nn.Linear(hidden, d, bias=False))
            with torch.no_grad():
                e[0].weight.copy_(w1)
                e[2].weight.copy_(w2)
            self.experts.append(e)

    def forward(self, x):
        return torch.stack([e(x[i]) for i, e in enumerate(self.experts)])


class _MoEShared(nn.Module):
    """Same tied experts, ALL consuming the same tokens — the textbook
    top-k shared-input layout.  Honest middle case: the tying IS found,
    but shared-input pairing re-materialises part of it as a fused GEMM."""

    def __init__(self, n=4, d=64, hidden=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = _randn((hidden, d), g)
        w2 = _randn((d, hidden), g)
        self.experts = nn.ModuleList()
        for _ in range(n):
            e = nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(),
                            nn.Linear(hidden, d, bias=False))
            with torch.no_grad():
                e[0].weight.copy_(w1)
                e[2].weight.copy_(w2)
            self.experts.append(e)
        self.gate = nn.Parameter(torch.full((n,), 1.0 / n,
                                            dtype=torch.float64))

    def forward(self, x):
        return sum(self.gate[i] * e(x) for i, e in enumerate(self.experts))


class _ComposedChain(nn.Module):
    """Two stacked dense square projections, NO nonlinearity between —
    the folded composed-linears case.  assoc_linear offers the single
    fused weight; measured: paired extraction stores both spellings,
    so the weights file does not shrink."""

    def __init__(self, d=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Linear(d, d, bias=False)
        self.w2 = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            for lin in (self.w1, self.w2):
                lin.weight.copy_(_randn(lin.weight.shape, g))

    def forward(self, x):
        return self.w2(self.w1(x))


class _DeadParam(nn.Module):
    def __init__(self, d=64, unused=4096):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        self.unused = nn.Parameter(_randn(
            (unused, d), torch.Generator().manual_seed(0)))

    def forward(self, x):
        return self.lin(x)


# ---------------------------------------------------------------------------
#  Dense archetype — "~0%"
# ---------------------------------------------------------------------------

def test_dense_transformer_saves_nearly_zero():
    """A trained-style dense MHA+FFN LM (all heads distinct, untied
    embedding) saves ~0%: the only win is the norm-affine gain folded
    into the head weight — measured 512 B of 656,384 B (0.08%)."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_DenseLM(n_layers=2),
                              torch.randint(0, 128, (2, 8)),
                              max_iterations=12, max_enodes=150_000)
    assert rel < 1e-12                                    # fp64-exact
    assert r["bytes_saved"] <= r["original_bytes"] // 100  # < 1%
    assert 0 <= r["bytes_saved"]
    # whatever was saved came from a param-only fold, never from
    # weight sharing — there is no exact structure to share.
    assert not any("__heads" in n for n in low.state_dict())


@pytest.mark.skipif(not os.path.exists(CKPT),
                    reason="stories15M.bin not downloaded")
def test_real_stories15m_checkpoint_offers_zero():
    """The real dense checkpoint: share_duplicate_params and
    share_duplicate_param_slices find ZERO exploitable structure —
    0 bytes, the honest headline for dense LLMs."""
    import numpy as np
    sys.path.insert(0, REPO)
    from measure_weights import load_llama2c
    from catopt.egraph import EGraph
    from catopt.ir import Param, TensorType
    from catopt.rules import (share_duplicate_params,
                              share_duplicate_param_slices)

    w = load_llama2c(CKPT)
    src = {}
    for name, t in w.items():
        if t.ndim == 3:
            for i in range(t.shape[0]):
                src[f"{name}_{i}"] = torch.from_numpy(
                    np.array(t[i])).clone()
        elif name != "_tail":
            src[name] = torch.from_numpy(np.array(t)).clone()
    eg = EGraph()
    for n, t in src.items():
        eg.add_term(Param(n, TensorType(tuple(t.shape))))
    assert share_duplicate_params(eg, src) == []
    assert share_duplicate_param_slices(eg, src) == []


# ---------------------------------------------------------------------------
#  Structured archetypes — "real on structured ones"
# ---------------------------------------------------------------------------

def test_gqa_materialised_repeat_kv_37pct():
    """8 query / 2 kv heads, kv replication baked into wk/wv:
    each kv weight drops to 2 of 8 head-blocks (-75%); the module
    saves exactly 2*(8-2)*8*64*8 = 49,152 B = 37.5%, fp64-exact."""
    torch.manual_seed(0)
    m = _GQAProj(kv_map=[0, 0, 0, 0, 1, 1, 1, 1])
    low, stats, r, rel = _opt(m, torch.randn(4, 3 * _GQAProj.DIM))
    assert rel < 1e-12
    expect = 2 * (_GQAProj.NH - _GQAProj.NKV) * _GQAProj.HD \
        * _GQAProj.DIM * BYTES
    assert r["bytes_saved"] == expect                      # 49,152 B
    assert abs(r["bytes_saved"] / r["original_bytes"] - 0.375) < 1e-9
    sd = low.state_dict()
    assert tuple(sd["p_wk_weight__heads8"].shape) == (2, 8, 64)
    assert tuple(sd["p_wv_weight__heads8"].shape) == (2, 8, 64)
    # bitwise-exact reconstruction of the eliminated weights
    recon = torch.index_select(sd["p_wk_weight__heads8"], 0,
                               torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.equal(recon.reshape(64, 64), m.wk.weight.detach())


def test_tied_embedding_classifier_stored_twice_50pct():
    """share_duplicate_params discovers the tie bitwise — one of the two
    512×64 tables drops out of the file: exactly 50%, zero output diff."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_TiedTwice(), torch.randint(0, 512, (8,)))
    assert rel == 0.0
    assert r["optimized_bytes"] == r["original_bytes"] // 2
    assert len(r["eliminated"]) == 1


def test_adapter_merged_fold_11pct():
    """Checkpoint stores W + A + B with the merge computed in forward;
    the W + B@A param-only subtree folds to one stored tensor — A and B
    eliminated: r*(i+o)*8 = 16,384 B = 11.03%, bitwise-exact output."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_AdapterMerged(), torch.randn(4, 128))
    assert rel == 0.0
    expect = _AdapterMerged.R * (_AdapterMerged.I + _AdapterMerged.O) \
        * BYTES
    assert r["bytes_saved"] == expect                      # 16,384 B
    assert {"p_A", "p_B"} <= set(r["eliminated"])
    sd = low.state_dict()
    fused = [v for k, v in sd.items() if k.startswith("fused_")]
    assert fused and fused[0].shape == (128, 128)


def test_adapter_unmerged_currently_regresses():
    """HONEST LIMIT pinned: base(x) + B·A·x through the full pipeline
    grows the file — pair_shared_input_linears' forced extraction
    concatenates the phantom composed weight class B@A next to A, and
    lowering materialises the concatenation (dag_cost bills leaf names
    and cannot see it).  fp64-exact output, negative storage delta.
    If a future change makes this non-negative, UPDATE the claim."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_AdapterUnmerged(), torch.randn(4, 128))
    assert rel < 1e-12
    assert r["bytes_saved"] < 0


def test_moe_tied_routed_experts_75pct():
    """Four bitwise-identical expert stacks, token-routed to distinct
    inputs: share_duplicate_params merges all copies — 3 of 4 expert
    weight sets drop out: exactly 75%, zero output diff."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_MoERouted(n=4), torch.randn(4, 4, 64))
    assert rel == 0.0
    assert r["bytes_saved"] == (4 - 1) * 2 * 64 * 64 * BYTES  # 196,608 B
    assert r["optimized_bytes"] * 4 == r["original_bytes"]
    assert len(r["eliminated"]) == 6


def test_moe_tied_shared_input_partial():
    """Same four tied experts, all reading the SAME tokens: tying is
    found, but the shared-input pairing law forces a fused concat that
    re-materialises the four up-projection copies — measured saving is
    only the three spare DOWN weights: 3·64·64·8 = 98,304 B (37.5%),
    not the routed case's 75%.  Bitwise-exact output.  If extraction
    ever learns to keep the dedup under pairing, UPDATE this pin."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_MoEShared(n=4), torch.randn(4, 64))
    assert rel == 0.0
    # 3 of 4 down-projections dedup'd; up-projections re-materialised
    # inside a fused_* concat parameter (measured, not intended).
    assert r["bytes_saved"] == 3 * 64 * 64 * BYTES            # 98,304 B
    assert 0 < r["bytes_saved"] < 0.75 * r["original_bytes"]
    assert any(n.startswith("fused_") for n in low.state_dict())


def test_composed_linears_fold_zero_bytes():
    """Two stacked dense projections, no nonlinearity: assoc_linear
    offers the single fused weight W2@W1 and the fold IS exact — but
    pair_shared_input_linears' coordinated extraction prefers the
    stacked-concat GEMM, which stores BOTH spellings (W1 next to
    W2·W1).  Net: 0 B saved — the fold exists, the file does not
    shrink.  fp64-exact output (reassociation noise only)."""
    torch.manual_seed(0)
    low, stats, r, rel = _opt(_ComposedChain(d=128), torch.randn(4, 128))
    assert rel < 1e-12
    assert r["bytes_saved"] == 0
    assert {"p_w1_weight", "p_w2_weight"} <= set(r["eliminated"])
    assert any(n.startswith("fused_") for n in low.state_dict())


def test_dead_param_dropped():
    """A registered parameter never referenced by the forward drops out
    of the optimized state_dict — the trivially-exact corner case."""
    torch.manual_seed(0)
    m = _DeadParam(d=64, unused=4096)
    low, stats, r, rel = _opt(m, torch.randn(4, 64))
    assert rel == 0.0
    assert "p_unused" in r["eliminated"]
    assert r["bytes_saved"] == 4096 * 64 * BYTES           # 2,097,152 B
    assert "p_unused" not in low.state_dict()
