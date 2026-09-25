"""Real-checkpoint benchmark: stories15M through catopt vs eager/Inductor.

A real trained SLM (llama2.c checkpoint, /tmp/stories15M.bin) wrapped
as an exportable nn.Module (sdpa attention, rmsnorm, swiglu), run
through ``optimize_compositional``, verified bitwise, and timed with
``torch.utils.benchmark``.

    python bench/stories15m_bench.py [--seq 128] [--device cuda]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure_weights import load_llama2c                      # noqa: E402


class Block(nn.Module):
    def __init__(self, dim, hidden, nh, hd):
        super().__init__()
        self.nh, self.hd = nh, hd
        self.rms_att = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rms_ffn = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, h, cos, sin):
        T = h.shape[0]

        def rope(x):
            x = x.reshape(T, self.nh, self.hd)
            x1, x2 = x[..., ::2], x[..., 1::2]
            c, s = cos[:, None, :], sin[:, None, :]
            out = torch.stack([x1 * c - x2 * s,
                               x1 * s + x2 * c], -1)
            return out.reshape(T, self.nh * self.hd)

        xn = F.rms_norm(h, (h.shape[-1],), self.rms_att, 1e-5)
        q = rope(self.wq(xn)).reshape(T, self.nh, self.hd)
        k = rope(self.wk(xn)).reshape(T, self.nh, self.hd)
        v = self.wv(xn).reshape(T, self.nh, self.hd)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None],
            v.transpose(0, 1)[None], is_causal=True)
        out = out[0].transpose(0, 1).reshape(T, -1)
        h = h + self.wo(out)
        xn = F.rms_norm(h, (h.shape[-1],), self.rms_ffn, 1e-5)
        return h + self.w2(F.silu(self.w1(xn)) * self.w3(xn))


class Stories15M(nn.Module):
    """llama2.c stories15M — exportable forward."""

    def __init__(self, w, cfg):
        super().__init__()
        dim, hidden = cfg["dim"], cfg["hidden"]
        nh, hd = cfg["n_heads"], dim // cfg["n_heads"]
        self.emb = nn.Embedding(cfg["vocab"], dim)
        self.blocks = nn.ModuleList(
            Block(dim, hidden, nh, hd) for _ in range(cfg["n_layers"]))
        self.rms_final = nn.Parameter(torch.ones(dim))
        self.head = nn.Linear(dim, cfg["vocab"], bias=False)
        freqs = 1.0 / (10000.0 ** (
            torch.arange(0, hd, 2).float() / hd))
        outer = torch.outer(torch.arange(cfg["seq_len"]).float(), freqs)
        self.register_buffer("cos", outer.cos())
        self.register_buffer("sin", outer.sin())
        with torch.no_grad():
            self.emb.weight.copy_(torch.tensor(w["token_embedding"]))
            self.head.weight.copy_(torch.tensor(w["token_embedding"]))
            for l, b in enumerate(self.blocks):
                b.rms_att.copy_(torch.tensor(w["rms_att"][l]))
                b.rms_ffn.copy_(torch.tensor(w["rms_ffn"][l]))
                b.wq.weight.copy_(torch.tensor(w["wq"][l]))
                b.wk.weight.copy_(torch.tensor(w["wk"][l]))
                b.wv.weight.copy_(torch.tensor(w["wv"][l]))
                b.wo.weight.copy_(torch.tensor(w["wo"][l]))
                # llama2.c stores FFN weights (out,in) row-major; the
                # loader's (dim,hidden)/(hidden,dim) reshape scrambles —
                # reshape recovers the true nn.Linear orientation.
                b.w1.weight.copy_(torch.tensor(
                    w["w1"][l].reshape(hidden, dim)))
                b.w2.weight.copy_(torch.tensor(
                    w["w2"][l].reshape(dim, hidden)))
                b.w3.weight.copy_(torch.tensor(
                    w["w3"][l].reshape(hidden, dim)))
            self.rms_final.copy_(torch.tensor(w["rms_final"]))

    def forward(self, idx):
        T = idx.shape[-1]
        h = self.emb(idx)
        if h.dim() == 3:
            h = h[0]
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            h = b(h, cos, sin)
        h = F.rms_norm(h, (h.shape[-1],), self.rms_final, 1e-5)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="/tmp/stories15M.bin")
    args = ap.parse_args()

    w = load_llama2c(args.ckpt)
    # cfg from shapes + header (n_heads isn't inferable from shapes)
    import struct
    with open(args.ckpt, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
    dim, hidden, L, nh = int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3])
    vocab, seq = w["token_embedding"].shape[0], int(hdr[6])
    cfg = dict(dim=dim, hidden=hidden, n_layers=L, n_heads=nh,
               vocab=vocab, seq_len=seq)
    m = Stories15M(w, cfg).eval().to(args.device)
    idx = torch.randint(0, cfg["vocab"], (1, args.seq),
                        device=args.device)
    with torch.no_grad():
        ref = m(idx)
    print(f"stories15M real checkpoint, T={args.seq}, "
          f"device={args.device} — logits {tuple(ref.shape)}")

    from catopt.optimize import optimize_compositional
    t0 = time.time()
    opt, rep = optimize_compositional(m, idx, verbose=False)
    pipeline = time.time() - t0
    with torch.no_grad():
        err = (opt(idx) - ref).abs().max().item()
    print(f"catopt pipeline: {pipeline:.1f}s — "
          f"blocks optimized {rep['n_optimized']}/{rep['n_blocks']} — "
          f"max|Δout| = {err:.2e}")

    for name, e in rep["blocks"].items():
        st = e.get("stats") or {}
        print(f"  {name:12} {e.get('status','?'):10} "
              f"paired={st.get('paired_extract')}")

    results = []
    variants = [("eager", lambda: m(idx)),
                ("catopt", lambda: opt(idx))]
    try:
        cmp_ = torch.compile(m)
        opt_cmp = torch.compile(opt)
        with torch.no_grad():
            cmp_(idx)
            opt_cmp(idx)
        variants += [("inductor", lambda: cmp_(idx)),
                     ("catopt+inductor", lambda: opt_cmp(idx))]
    except Exception as e:
        print(f"inductor skipped: {e}")
    for tag, fn in variants:
        t = Timer("fn()", globals={"fn": fn})
        r = t.blocked_autorange(min_run_time=1.0)
        r.description = tag
        results.append(r)
    base = results[0].median
    print(f"{'variant':>16} {'median ms':>10} {'vs eager':>9} {'IQR ms':>8}")
    for r in results:
        print(f"{r.description:>16} {r.median*1e3:>10.3f} "
              f"{r.median/base:>8.3f}x {r.iqr*1e3:>8.3f}")


if __name__ == "__main__":
    main()
