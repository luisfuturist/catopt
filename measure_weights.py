"""Phase 0 — falsification harness for weight-space structure.

Loads a llama2.c-format .bin checkpoint and measures, per 2D weight
matrix, the quantities that decide whether "the weight file is a
program" is a breakthrough or a dead end:

  * numerical rank at several tolerances + 90/95/99% energy cutoffs
    (is there a low-rank core? — decides whether UV factors and
    spectral-certified rewrites exist)
  * displacement rank  rank(W - shift(W))   (Toeplitz-likeness —
    <= ~2*bandwidth means a generator representation exists)
  * stable rank  ||W||_F^2 / sigma_max^2    (intrinsic dimensionality)
  * symmetry defect  ||W - W^T|| / ||W||
  * stored-value ratios implied by each representation

Go/no-go:  if relative numerical rank ~ 1.0 at fp32 tolerances and
displacement rank ~ n, weight programs are dead.  If either is << n,
there is exploitable structure.

Usage:  python3 measure_weights.py /path/to/checkpoint.bin
"""
from __future__ import annotations

import sys

import numpy as np


def load_llama2c(path: str) -> dict[str, np.ndarray]:
    """Parse a llama2.c v2-style .bin: 7-int32 header
    (dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq) followed by
    fp32 weights in the documented order.  stories15M ties wcls to the
    embedding table."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
        dim, hidden, L, _h, _kv, vocab, _seq = (int(v) for v in hdr)
        data = np.frombuffer(f.read(), dtype=np.float32)

    w: dict[str, np.ndarray] = {}
    off = 0

    def take(name, shape):
        nonlocal off
        n = int(np.prod(shape))
        w[name] = data[off:off + n].reshape(shape)
        off += n

    take("token_embedding", (vocab, dim))
    take("rms_att", (L, dim))
    for i, nm in enumerate(("wq", "wk", "wv", "wo")):
        take(nm, (L, dim, dim))
    take("rms_ffn", (L, dim))
    take("w1", (L, dim, hidden))
    take("w2", (L, hidden, dim))
    take("w3", (L, dim, hidden))
    take("rms_final", (dim,))
    if off < len(data):
        w["_tail"] = data[off:]
    return w


def analyze(W: np.ndarray) -> dict:
    m, n = W.shape
    sv = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    smax = sv[0]
    if smax == 0:
        return {"skip": "zero"}
    s2 = sv ** 2
    energy = np.cumsum(s2) / s2.sum()

    def energy_rank(p):
        return int(np.searchsorted(energy, p) + 1)

    rel = {
        f"rank@{t:g}": int((sv > t * smax).sum())
        for t in (1e-1, 1e-2, 1e-3, 1e-4)
    }
    # displacement rank (Toeplitz generator check)
    Z = np.zeros_like(W)
    D = W.copy()
    D[1:, :] -= W[:-1, :]           # row-shift displacement
    dsv = np.linalg.svd(D, compute_uv=False)
    disp = int((dsv > 1e-3 * dsv[0]).sum()) if dsv[0] > 0 else 0
    Dc = W.copy()
    Dc[:, 1:] -= W[:, :-1]          # col-shift displacement
    csv = np.linalg.svd(Dc, compute_uv=False)
    dispc = int((csv > 1e-3 * csv[0]).sum()) if csv[0] > 0 else 0
    sym = (np.linalg.norm(W - W.T) / np.linalg.norm(W)
           if m == n else float("nan"))
    return {
        "shape": f"{m}x{n}",
        **rel,
        "r90%": energy_rank(0.90),
        "r95%": energy_rank(0.95),
        "r99%": energy_rank(0.99),
        "stable_rk": round(float(s2.sum() / smax ** 2), 1),
        "disp_row": disp,
        "disp_col": dispc,
        "sym_defect": round(float(sym), 3),
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stories15M.bin"
    w = load_llama2c(path)
    rows = []
    for name, W in w.items():
        if W.ndim == 2:
            rows.append((name, W))
        elif W.ndim == 3:
            for i in range(W.shape[0]):
                rows.append((f"{name}[{i}]", W[i]))

    print(f"{'matrix':<22} {'shape':>9} {'r@1e-2':>6} {'r@1e-3':>6} "
          f"{'r90%':>5} {'r99%':>5} {'stbl':>6} {'dispR':>5} "
          f"{'dispC':>5} {'sym':>6}")
    print("-" * 92)
    for name, W in rows:
        a = analyze(W)
        if "skip" in a:
            continue
        n = min(W.shape)
        print(f"{name:<22} {a['shape']:>9} "
              f"{a['rank@0.01']:>4}/{n:<2} "
              f"{a['rank@0.001']:>4}/{n:<2} "
              f"{a['r90%']:>5} {a['r99%']:>5} "
              f"{a['stable_rk']:>6} {a['disp_row']:>5} "
              f"{a['disp_col']:>5} {a['sym_defect']:>6}")

    # ---- aggregate verdict -------------------------------------------------
    mats = [W for _, W in rows]
    tot = sum(W.size for W in mats)
    r99_total = sum(analyze(W)["r99%"] * (W.shape[0] + W.shape[1])
                    for _, W in rows)
    print("-" * 92)
    print(f"total matrix params: {tot:,}")
    print(f"low-rank storage at 99% energy: {r99_total:,} "
          f"({r99_total/tot:.1%} of params)")


if __name__ == "__main__":
    main()
