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


def _best_kron(W: np.ndarray) -> dict:
    """Kronecker-product test: W (m×n) is A⊗B with A (m1×n1),
    B (m2×n2) iff the rearranged matrix R (m1·n1 × m2·n2) is rank-1.
    Try a few balanced factorisations; report the best rank-1 energy
    share — the decisive *product*-structure probe."""
    m, n = W.shape
    best = None
    for m1 in range(2, int(m ** 0.5) + 2):
        if m % m1:
            continue
        m2 = m // m1
        for n1 in range(2, int(n ** 0.5) + 2):
            if n % n1:
                continue
            n2 = n // n1
            R = (W.reshape(m1, m2, n1, n2)
                   .transpose(0, 2, 1, 3)
                   .reshape(m1 * n1, m2 * n2))
            sv = np.linalg.svd(R, compute_uv=False)
            e1 = float(sv[0] ** 2 / (sv ** 2).sum())
            if best is None or e1 > best["e1"]:
                best = {"e1": e1, "factors": f"{m1}x{n1}⊗{m2}x{n2}",
                        "r95": int(np.searchsorted(
                            np.cumsum(sv ** 2) / (sv ** 2).sum(),
                            0.95) + 1)}
    return best or {"e1": 0.0, "factors": "-", "r95": -1}


def _hmat_rank(W: np.ndarray) -> dict:
    """H-matrix probe: split W into 2×2 blocks recursively; a
    hierarchical-low-rank matrix has numerically-low-rank off-diagonal
    blocks at every level.  Report the worst relative rank of
    off-diagonal blocks at the finest split."""
    m, n = W.shape
    lvl = 1
    worst = 0.0
    while min(m // (2 ** lvl), n // (2 ** lvl)) >= 16 and lvl <= 4:
        bs_m, bs_n = m // (2 ** lvl), n // (2 ** lvl)
        ranks = []
        for i in range(2 ** lvl):
            for j in range(2 ** lvl):
                if i == j:
                    continue
                B = W[i * bs_m:(i + 1) * bs_m, j * bs_n:(j + 1) * bs_n]
                sv = np.linalg.svd(B, compute_uv=False)
                if sv[0] > 0:
                    ranks.append(int((sv > 1e-2 * sv[0]).sum())
                                 / min(B.shape))
        if ranks:
            worst = max(worst, max(ranks))
        lvl += 1
    return {"h_offdiag_rel": round(worst, 3)}


def _sparsity(W: np.ndarray) -> dict:
    """Sparse+low-rank probe: what fraction of Frobenius energy sits in
    the largest 10%/1% of entries (a sparse top plus a small residual
    is itself a program: W ≈ sparse(S) + lowrank)."""
    a = np.abs(W).ravel()
    a.sort()
    a = a[::-1]
    e = a ** 2
    tot = e.sum()
    return {"top10%": round(float(e[:len(a) // 10].sum() / tot), 3),
            "top1%": round(float(e[:max(1, len(a) // 100)].sum() / tot),
                           3)}


def analyze2(W: np.ndarray) -> dict:
    """The structured-algebra family: beyond Toeplitz displacement."""
    out = _best_kron(W)
    out.update(_hmat_rank(W))
    out.update(_sparsity(W))
    return out


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

    # ---- the structured-algebra family -------------------------------------
    print("\n=== Phase 0b: structured algebras (Kronecker / H-matrix / "
          "sparse) ===")
    print(f"{'matrix':<22} {'shape':>9} {'kron_e1':>8} {'kron95':>7} "
          f"{'H-rel':>6} {'top10%':>7} {'top1%':>7}")
    print("-" * 70)
    for name, W in rows:
        if max(W.shape) > 4096:
            continue  # skip the embedding table — SVD cost dominates
        a2 = analyze2(W)
        print(f"{name:<22} {W.shape[0]}x{W.shape[1]:<5} "
              f"{a2['e1']:>8.3f} {a2['r95']:>7} "
              f"{a2['h_offdiag_rel']:>6} {a2['top10%']:>7} "
              f"{a2['top1%']:>7}")


if __name__ == "__main__":
    main()
