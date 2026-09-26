"""Batched-input verification guards.

Regression coverage for the intermittent batched failure reported by
``bench/decode_bench.py``: an optimized member carrying a reshape
specialized to a *different* cell's ``(B, T)`` — e.g. evaluating a
``(16, 256, 768)`` input through a ``reshape`` attr ``(16, 128, 12,
64)`` baked for ``T=128`` — surfaced as

    RuntimeError: shape '[16, 128, 12, 64]' is invalid for input of
    size 3145728

Two layers are pinned here:

* **Minted-member invariant** — no reshape enode in a saturated
  e-graph may declare a shape whose numel disagrees with its input.
  ``cost._shape_of`` now flags such members ``_INVALID`` so they can
  never be preferred by extraction (a reshape is a *free* view — an
  unguarded ill-typed one is cheap AND wrong).

* **Caller-model integrity** — ``optimize_compositional`` must never
  leave the *caller's* module grafted with shape-specialized
  IRModules.  When ``copy.deepcopy(model)`` fails (e.g. CUDA OOM on a
  tight card — exactly what hits the large cells) the current
  ``in_place`` fallback mutates ``model``; the next invocation at a
  different ``(B, T)`` then crashes on the previous cell's baked
  reshape.  That path is marked xfail until the recompose fallback
  keeps ``model`` pristine.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from catopt.cost import _INVALID, _shape_of, flops_cost
from catopt.egraph import EGraph
from catopt.ir import Op, TensorType, Var
from catopt.optimize import optimize_compositional, optimize_model

# ---------------------------------------------------------------------------
#  A batched rope-style block — mirrors bench/decode_bench.py's BatchedBlock
# ---------------------------------------------------------------------------


class RopeBlock(nn.Module):
    """The reported failure shape: a rank-5 ``stack`` -> ``reshape``
    rope, plus the qkv/mlp GEMM groups the pairing pass fuses."""

    def __init__(
        self,
        dim: int = 96,
        hidden: int = 192,
        nh: int = 4,
        hd: int = 24,
    ) -> None:
        super().__init__()
        self.nh, self.hd = nh, hd
        self.wq = nn.Linear(dim, nh * hd, bias=False)
        self.wk = nn.Linear(dim, nh * hd, bias=False)
        self.wv = nn.Linear(dim, nh * hd, bias=False)
        self.wo = nn.Linear(nh * hd, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, self.nh, self.hd)
        x1, x2 = x[..., ::2], x[..., 1::2]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1)
        return out.reshape(B, T, self.nh * self.hd)

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T = h.shape[0], h.shape[1]
        q = self.rope(self.wq(h), cos, sin)
        k = self.rope(self.wk(h), cos, sin)
        v = self.wv(h)
        q = q.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        h = h + self.wo(y)
        return h + self.w2(F.silu(self.w1(h)) * self.w3(h))


class RopeStories(nn.Module):
    """emb -> ModuleList(RopeBlock) -> head, like the bench model."""

    def __init__(
        self,
        dim: int = 96,
        hidden: int = 192,
        nh: int = 4,
        hd: int = 24,
        depth: int = 2,
        seq: int = 1024,
        vocab: int = 256,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            RopeBlock(dim, hidden, nh, hd) for _ in range(depth)
        )
        self.head = nn.Linear(dim, vocab, bias=False)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
        outer = torch.outer(torch.arange(seq).float(), freqs)
        self.register_buffer("cos", outer.cos())
        self.register_buffer("sin", outer.sin())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[-1]
        h = self.emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            h = b(h, cos, sin)
        return self.head(h)


def _V(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d if isinstance(d, int) and d > 0 else 1
    return n


def _all_terms(term, seen=None):
    if seen is None:
        seen = set()
    if id(term) in seen:
        return
    seen.add(id(term))
    yield term
    if isinstance(term, Op):
        for a in term.args:
            yield from _all_terms(a, seen)


# ---------------------------------------------------------------------------
#  Shape-inference guard: an ill-typed reshape is _INVALID, never a shape
# ---------------------------------------------------------------------------


def test_reshape_inference_flags_numel_mismatch():
    """``reshape(x, S)`` where ``numel(S) != numel(x)`` is ill-typed:
    cost must say ``_INVALID`` (never extractable), not report S."""
    B, T, nh, hd = 16, 256, 12, 64
    x = _V("x", (B, T, nh * hd))
    bad = Op.make("reshape", x, shape=(B, T // 2, nh, hd))
    assert _shape_of(bad) is _INVALID
    # the exact reported flavour: T halved onto the head axes
    bad2 = Op.make("reshape", x, shape=(16, 128, 12, 64))
    assert _shape_of(bad2) is _INVALID
    # a consistent reshape still reports its attr
    good = Op.make("reshape", x, shape=(B, T, nh, hd))
    assert _shape_of(good) == (B, T, nh, hd)


def test_extract_best_never_picks_illtyped_reshape():
    """An e-class containing a well-typed member AND a free-view
    reshape with a numel-mismatched attr must extract the good member:
    the ill-typed view would otherwise win on cost and crash eval."""
    torch.manual_seed(0)
    from catopt.ir import IR
    from catopt.torch_bridge import ir_to_torch_module

    B, T, D = 16, 256, 768
    x = _V("x", (B, T, D))
    eg = EGraph()
    root = eg.add_term(Op.make("mul", x, x))  # a real (costly) member
    # hand-mint the reported failure shape as a sibling member of the
    # root class — a reshape view that halved the wrong axis
    x_eid = eg.add_leaf(repr(x))
    bad_eid = eg.add_enode(
        "reshape", (x_eid,), {"shape": (B, T // 2, 12, 64)}
    )
    eg.union(root, bad_eid)  # one class, two members

    term = eg.extract_best(root, flops_cost)
    # must be the well-typed member — never the ill-typed free view
    assert isinstance(term, Op) and term.op == "mul", term
    # and it must actually evaluate
    mod = ir_to_torch_module(IR(root=term, inputs=[x]), param_values={})
    with torch.no_grad():
        out = mod(torch.randn(B, T, D))
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
#  Minted-member invariant on the real batched-rope e-graph
# ---------------------------------------------------------------------------


def test_batched_rope_egraph_reshapes_welltyped():
    """Replicate ``optimize_model``'s saturation on a batched-rope
    block and audit EVERY reshape enode: attr numel must equal the
    child class's inferred numel — the reported minted shape may never
    appear ill-typed."""
    torch.manual_seed(0)
    from catopt.optimize import _EXPANSIVE_RULES
    from catopt.rules import all_rules
    from catopt.torch_bridge import export_to_ir

    blk = RopeBlock().eval()
    B, T, D = 16, 128, 96
    h = torch.randn(B, T, D)
    cos, sin = torch.randn(T, 12), torch.randn(T, 12)
    ir, _st = export_to_ir(blk, (h, cos, sin))

    eg = EGraph()
    root = eg.add_term(ir.root)
    _SUB = {
        "swiglu_fuse",
        "parallel_mul_fuse",
        "qkv_fuse",
        "qkv_fuse_asym",
    }
    rules = [r for r in all_rules() if r.name not in _SUB]
    budgets = {n: 2048 for n in _EXPANSIVE_RULES}
    eg.run(rules, root, rule_budgets=budgets)

    bad = []
    for en in eg._node_to_class:
        if en.op != "reshape":
            continue
        s = dict(en.attrs).get("shape")
        if not isinstance(s, tuple) or -1 in s:
            continue
        child = eg.any_term(en.children[0])
        cs = _shape_of(child)
        if (
            isinstance(cs, tuple)
            and all(isinstance(d, int) for d in cs)
            and _numel(cs) != _numel(s)
        ):
            bad.append((s, cs))
    assert bad == [], f"ill-typed reshape enodes minted: {bad}"


# ---------------------------------------------------------------------------
#  Caller-model integrity across sequential optimize_compositional cells
# ---------------------------------------------------------------------------


def test_compositional_leaves_caller_model_pristine():
    """Sequential cells (the decode_bench sweep pattern): each call
    must leave ``model`` untouched — every module still its original
    class, and ``model`` still evals at both shapes afterwards."""
    torch.manual_seed(0)
    model = RopeStories(depth=2).eval()
    for T in (64, 128):
        idx = torch.randint(0, 256, (4, T))
        opt, rep = optimize_compositional(model, idx, verbose=False)
        assert rep["in_place"] is False
        # the caller's model is never the recomposed one
        assert opt is not model
        assert all(isinstance(b, RopeBlock) for b in model.blocks)
        # and it still evals at any input shape
        with torch.no_grad():
            model(torch.randint(0, 256, (4, 32)))
            model(idx)


def test_compositional_inplace_fallback_preserves_model():
    """REGRESSION for the reported bug: when ``copy.deepcopy(model)``
    fails (CUDA OOM after a heavy cell is exactly how the report hit
    it), the in-place fallback currently grafts shape-specialized
    IRModules into the caller's ``model`` — so the NEXT cell crashes
    on the previous cell's baked reshape attrs.

    The fix (catopt/optimize.py ``optimize_compositional``): on
    deepcopy failure do NOT graft into ``model`` — return it
    unmodified — so the caller's module is never corrupted.
    """
    torch.manual_seed(0)
    model = RopeStories(depth=2).eval()
    idx64 = torch.randint(0, 256, (4, 64))

    real_deepcopy = copy.deepcopy

    def flaky(obj, *a, **k):
        # fail ONLY the recompose deepcopy — per-block exports etc.
        # still work, mirroring an OOM that hits the model clone
        if obj is model:
            raise RuntimeError("simulated OOM during deepcopy(model)")
        return real_deepcopy(obj, *a, **k)

    orig = copy.deepcopy
    copy.deepcopy = flaky
    try:
        opt, rep = optimize_compositional(model, idx64, verbose=False)
    finally:
        copy.deepcopy = orig
    assert rep["in_place"] is True

    # The bug: ``model`` was silently mutated with T=64-specialized
    # IRModules — calling it at a different T crashes with the stale
    # baked shape (the reported 'shape [B, T1, nh, hd] invalid').
    with torch.no_grad():
        model(torch.randint(0, 256, (4, 128)))  # must not raise
        model(idx64)  # still fine at T=64


def test_batched_rope_optimize_model_multi_shape():
    """A rope-style block optimized at two different (B,T) inputs
    produces modules that each eval correctly at THEIR shape — the
    per-cell specialization itself is sound; it must never leak into
    another cell's program."""
    torch.manual_seed(0)
    blk = RopeBlock().eval()
    for B, T in ((4, 64), (4, 128)):
        h = torch.randn(B, T, 96)
        cos = torch.randn(T, 12)
        sin = torch.randn(T, 12)
        opt, _stats = optimize_model(blk, (h, cos, sin), verbose=False)
        with torch.no_grad():
            ref = blk(h, cos, sin)
            out = opt(h, cos, sin)
        rd = ((ref - out).abs().max() / (ref.abs().max() + 1e-8)).item()
        assert rd < 1e-4
        # no member of the optimized program is an ill-typed reshape
        for t in _all_terms(opt._root):
            if isinstance(t, Op) and t.op == "reshape":
                s = t.attrs.get("shape")
                if isinstance(s, tuple) and -1 not in s:
                    assert _shape_of(t) is not _INVALID
