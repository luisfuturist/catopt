#!/usr/bin/env python3
# ruff: noqa: RUF001
"""exact_probe.py — measure the exact-structure corner across model archetypes.

ARCHIVED — weight-space probe, falsified axis (see ../REPORT.md).
Lives under retros/ as evidence for the report's exact-corner pins;
the llama2c loader it imports is bench/llama2c.py on main.

The corner's legitimate scope is EXACT structure only:
  * share_duplicate_params        — whole-tensor tying (bitwise-equal params)
  * share_duplicate_param_slices  — intra-tensor head-block dedup (GQA/MoE
                                    replication baked into a checkpoint)
  * param-only folding            — composed linears / adapter merges
                                    materialised as one stored tensor by
                                    IRModule._fold_weight_chains
  * dead params                   — unreferenced leaves dropped by _build_params

Question sharpened here: "~0% on dense LLMs, real on structured ones" —
measured, per archetype, under the storage cost axis
(param_bytes_cost_for), with fp64 output equality checked every time.

Usage:  python3 /tmp/exact_probe.py            (prints a markdown table)
        python3 -m pytest tests/test_exact_corner.py   (pinned results)
"""

from __future__ import annotations

import os
import sys
import time

REPO = "/home/luis/Desktop/catopt"
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from catopt.cost import param_bytes_cost_for  # noqa: E402
from catopt.optimize import optimize_model, param_report  # noqa: E402

CKPT = "/tmp/stories15M.bin"

# ----------------------------------------------------------------------
#  Archetype builders — every model is built in fp64 so 'exact' means
#  bitwise-or-rounding-noise output equality, not a tolerance story.
# ----------------------------------------------------------------------


def _randn(shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


class DenseBlock(nn.Module):
    """Dense MHA + FFN block: all heads distinct, all weights trained-style
    random — the 'no exact structure' case."""

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
        # a norm-affine gain — the one piece of foldable weight structure
        # a trained dense model legitimately carries.
        self.scale = nn.Parameter(_randn((d,), g))
        with torch.no_grad():
            for lin in (
                self.wq,
                self.wk,
                self.wv,
                self.wo,
                self.w1,
                self.w2,
            ):
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


class DenseLM(nn.Module):
    def __init__(self, n_layers=2, d=64, vocab=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [DenseBlock(d=d, seed=i + 1) for i in range(n_layers)]
        )
        self.head = nn.Linear(d, vocab, bias=False)  # untied

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class GQAProj(nn.Module):
    """8 query heads, 2 kv heads with repeat_kv MATERIALISED into wk/wv —
    the llama-7B-export pattern: wk/wv each carry 8 head-blocks of which
    only 2 are unique (index map [0,0,0,0,1,1,1,1])."""

    DIM, NH, NKV = 64, 8, 2
    HD = DIM // NH

    def __init__(self, kv_map=(), seed=0):
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
                uniq = [
                    _randn((self.HD, self.DIM), g)
                    for _ in range(max(kv_map) + 1)
                ]
                lin.weight.copy_(
                    torch.cat([uniq[i] for i in kv_map], dim=0)
                )

    def forward(self, x):
        # distinct input slices isolate the slice pass from qkv pairing
        q = self.wq(x[..., : self.DIM])
        k = self.wk(x[..., self.DIM : 2 * self.DIM])
        v = self.wv(x[..., 2 * self.DIM : 3 * self.DIM])
        return self.wo(q + k + v)


class AdapterMerged(nn.Module):
    """Checkpoint stores base W + LoRA factors A, B; forward computes the
    merged weight W + B@A (the merge_and_unload() pattern).  The bias keeps
    the exported linear 3-ary so weight_distribute/pairing cannot make
    phantom siblings of the merged member — measured separately below."""

    def __init__(self, i=128, o=128, r=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(_randn((o, i), g))
        self.A = nn.Parameter(_randn((r, i), g))
        self.B = nn.Parameter(_randn((o, r), g))
        self.bias = nn.Parameter(_randn((o,), g))

    def forward(self, x):
        return F.linear(x, self.W + self.B @ self.A, self.bias)


class AdapterUnmerged(nn.Module):
    """The UNmerged variant — base(x) + B(A(x)) with all three stored —
    kept to document the pipeline's honest regression (see report)."""

    def __init__(self, i=128, o=128, r=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = nn.Linear(i, o, bias=True)
        self.la = nn.Linear(i, r, bias=False)
        self.lb = nn.Linear(r, o, bias=False)
        with torch.no_grad():
            for lin in (self.base, self.la, self.lb):
                lin.weight.copy_(_randn(lin.weight.shape, g))

    def forward(self, x):
        return self.base(x) + self.lb(self.la(x))


class TiedTwice(nn.Module):
    """Embedding and classifier tied but stored TWICE — the HF-export
    pattern where wcls is materialised as a second tensor."""

    def __init__(self, vocab=512, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight.data = self.embed.weight.data.clone()

    def forward(self, idx):
        return self.head(self.embed(idx))


class MoERouted(nn.Module):
    """N weight-tied experts, token-routed: expert i sees token-slice i.
    Identical expert weights = the 'experts share weights' MoE variant
    (tied branches materialised per-slot in the checkpoint)."""

    def __init__(self, n=4, d=64, hidden=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = _randn((hidden, d), g)
        w2 = _randn((d, hidden), g)
        self.experts = nn.ModuleList()
        for _ in range(n):
            e = nn.Sequential(
                nn.Linear(d, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, d, bias=False),
            )
            with torch.no_grad():
                e[0].weight.copy_(w1)
                e[2].weight.copy_(w2)
            self.experts.append(e)

    def forward(self, x):  # x: (n_experts, T, d)
        return torch.stack(
            [e(x[i]) for i, e in enumerate(self.experts)]
        )


class MoEShared(nn.Module):
    """Same tied experts but ALL consuming the same tokens (textbook
    top-k routing input).  Honest middle case: the tying is found, but
    shared-input pairing re-materialises part of it as a fused GEMM."""

    def __init__(self, n=4, d=64, hidden=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = _randn((hidden, d), g)
        w2 = _randn((d, hidden), g)
        self.experts = nn.ModuleList()
        for _ in range(n):
            e = nn.Sequential(
                nn.Linear(d, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, d, bias=False),
            )
            with torch.no_grad():
                e[0].weight.copy_(w1)
                e[2].weight.copy_(w2)
            self.experts.append(e)
        self.gate = nn.Parameter(
            torch.full((n,), 1.0 / n, dtype=torch.float64)
        )

    def forward(self, x):
        return sum(
            self.gate[i] * e(x) for i, e in enumerate(self.experts)
        )


class ComposedChain(nn.Module):
    """Folded composed-linears: two stacked dense square projections,
    NO nonlinearity between (projection chains some exports emit).
    assoc_linear offers the single fused weight — measured: the paired
    pass stores both spellings, so the file does not shrink."""

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


class DeadParam(nn.Module):
    """A registered-but-unreferenced parameter — the dead-weights case."""

    def __init__(self, d=64, unused=4096):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        self.unused = nn.Parameter(
            _randn((unused, d), torch.Generator().manual_seed(0))
        )

    def forward(self, x):
        return self.lin(x)


# ----------------------------------------------------------------------
#  Measurement driver
# ----------------------------------------------------------------------


def _mechanism(report, stats):
    """Attribute the byte change to the pass that produced it."""
    mech = []
    der = report["derived"]
    elim = report["eliminated"]
    if any("__heads" in n for n in der):
        mech.append("share_duplicate_param_slices (head dedup)")
    if elim and not der:
        mech.append("share_duplicate_params (tying) / dead params")
    if any(n.startswith("fused_") for n in der):
        mech.append("param-only fold (composed/scale weights)")
    if report["bytes_saved"] == 0 and not mech:
        mech.append("none — no exact structure found")
    return "; ".join(mech) if mech else "paired-GEMM realisation"


def probe(
    model,
    example,
    name,
    *,
    max_iterations=12,
    max_enodes=150_000,
    quiet=True,
):
    """optimize_model under the storage cost axis; verify fp64 equality;
    return the param_report row."""
    model = model.eval().double()
    if example.is_floating_point():
        example = example.double()
    t0 = time.time()
    low, stats = optimize_model(
        model,
        example,
        cost_fn=param_bytes_cost_for(),
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        verbose=False,
    )
    dt = time.time() - t0
    with torch.no_grad():
        ref = model(example.clone())
        out = low(example.clone())
    abs_diff = (out - ref).abs().max().item()
    rel_diff = abs_diff / (ref.abs().max().item() + 1e-8)
    r = param_report(model, low)
    r.update(
        name=name,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        mechanism=_mechanism(r, stats),
        seconds=dt,
        stats=stats,
    )
    if not quiet:
        print(
            f"[{name}] {r['original_bytes']} -> {r['optimized_bytes']} B "
            f"(-{r['bytes_saved']} B, "
            f"{100 * r['bytes_saved'] / r['original_bytes']:.2f}%) "
            f"max|d|={abs_diff:.2e} rel={rel_diff:.2e} ({dt:.1f}s)"
        )
        print(f"    mechanism: {r['mechanism']}")
        print(f"    eliminated: {r['eliminated']}")
        print(f"    derived:    {r['derived']}")
    return r


def probe_stories15m():
    """The real dense checkpoint, measured directly: run BOTH share passes
    on its tensors and count what they would offer.  (An nn.Module at this
    size is outside the bench budget; the pass-level measurement IS the
    honest zero — the passes see the same tensors either way.)"""
    if not os.path.exists(CKPT):
        return None
    import numpy as np

    from catopt.egraph import EGraph
    from catopt.ir import Param, TensorType
    from catopt.rules import (
        share_duplicate_param_slices,
        share_duplicate_params,
    )
    from retros.measure_weights import load_llama2c

    w = load_llama2c(CKPT)
    src = {}
    for name, t in w.items():
        if t.ndim == 3:
            for i in range(t.shape[0]):
                src[f"{name}_{i}"] = torch.from_numpy(
                    np.array(t[i])
                ).clone()
        elif name != "_tail":
            src[name] = torch.from_numpy(np.array(t)).clone()
    total = sum(t.numel() * t.element_size() for t in src.values())
    eg = EGraph()
    for n, t in src.items():
        eg.add_term(Param(n, TensorType(tuple(t.shape))))
    g_whole = share_duplicate_params(eg, src)
    g_slice = share_duplicate_param_slices(eg, src)
    return {
        "name": "stories15M.bin (real, dense)",
        "tensors": len(src),
        "total_bytes": total,
        "whole_groups": g_whole,
        "slice_offers": g_slice,
        "bytes_saved": 0,
        "pct": 0.0,
    }


# ----------------------------------------------------------------------
#  Main — build every archetype, measure, print the REPORT-ready table.
# ----------------------------------------------------------------------


def run_all(quiet=False):
    rows = []

    real = probe_stories15m()
    rows.append(
        real if real else {"name": "stories15M.bin", "skipped": True}
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            DenseLM(n_layers=2),
            torch.randint(0, 128, (2, 8)),
            "dense transformer (synth, 2L)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            GQAProj(kv_map=[0, 0, 0, 0, 1, 1, 1, 1]),
            torch.randn(4, 3 * GQAProj.DIM),
            "GQA 8q/2kv (repeat_kv materialised)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            AdapterMerged(),
            torch.randn(4, 128),
            "adapter-merged (W + B·A stored)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            AdapterUnmerged(),
            torch.randn(4, 128),
            "adapter UNmerged (honest limit)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            ComposedChain(),
            torch.randn(4, 128),
            "composed linears w2(w1 x), no act",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            TiedTwice(),
            torch.randint(0, 512, (8,)),
            "tied embed/cls stored twice",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            MoERouted(),
            torch.randn(4, 4, 64),
            "MoE: 4 weight-tied experts (routed)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            MoEShared(),
            torch.randn(4, 64),
            "MoE: 4 weight-tied experts (shared input)",
            quiet=quiet,
        )
    )

    torch.manual_seed(0)
    rows.append(
        probe(
            DeadParam(),
            torch.randn(4, 64),
            "dead param (unused 4096×64)",
            quiet=quiet,
        )
    )
    return rows


def fmt_table(rows):
    out = [
        "| archetype | orig params (B) | optimized (B) | saved | % | "
        "max|Δout| (fp64) | mechanism |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("skipped"):
            out.append(
                f"| {r['name']} | – | – | – | – | – | "
                "checkpoint not present |"
            )
            continue
        if "total_bytes" in r:  # stories15M direct
            out.append(
                f"| {r['name']} | {r['total_bytes']:,} | "
                f"{r['total_bytes']:,} | 0 B | 0.00% | – | "
                "no duplicate tensors / head-slices "
                f"({r['tensors']} tensors probed) |"
            )
            continue
        pct = 100 * r["bytes_saved"] / max(r["original_bytes"], 1)
        out.append(
            f"| {r['name']} | {r['original_bytes']:,} | "
            f"{r['optimized_bytes']:,} | {r['bytes_saved']:,} B | "
            f"{pct:.2f}% | {r['abs_diff']:.1e} | {r['mechanism']} |"
        )
    return "\n".join(out)


def main():
    print(
        "# exact_probe: the exact-structure corner across archetypes\n"
    )
    rows = run_all(quiet=False)
    print("\n## REPORT-ready table (param bytes, fp64 models)\n")
    print(fmt_table(rows))


if __name__ == "__main__":
    main()
