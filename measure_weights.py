# ruff: noqa: RUF002, RUF003
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
        w[name] = data[off : off + n].reshape(shape)
        off += n

    take("token_embedding", (vocab, dim))
    take("rms_att", (L, dim))
    for _i, nm in enumerate(("wq", "wk", "wv", "wo")):
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
    s2 = sv**2
    energy = np.cumsum(s2) / s2.sum()

    def energy_rank(p):
        return int(np.searchsorted(energy, p) + 1)

    rel = {
        f"rank@{t:g}": int((sv > t * smax).sum())
        for t in (1e-1, 1e-2, 1e-3, 1e-4)
    }
    # displacement rank (Toeplitz generator check)
    _Z = np.zeros_like(W)
    D = W.copy()
    D[1:, :] -= W[:-1, :]  # row-shift displacement
    dsv = np.linalg.svd(D, compute_uv=False)
    disp = int((dsv > 1e-3 * dsv[0]).sum()) if dsv[0] > 0 else 0
    Dc = W.copy()
    Dc[:, 1:] -= W[:, :-1]  # col-shift displacement
    csv = np.linalg.svd(Dc, compute_uv=False)
    dispc = int((csv > 1e-3 * csv[0]).sum()) if csv[0] > 0 else 0
    sym = (
        np.linalg.norm(W - W.T) / np.linalg.norm(W)
        if m == n
        else float("nan")
    )
    return {
        "shape": f"{m}x{n}",
        **rel,
        "r90%": energy_rank(0.90),
        "r95%": energy_rank(0.95),
        "r99%": energy_rank(0.99),
        "stable_rk": round(float(s2.sum() / smax**2), 1),
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
    for m1 in range(2, int(m**0.5) + 2):
        if m % m1:
            continue
        m2 = m // m1
        for n1 in range(2, int(n**0.5) + 2):
            if n % n1:
                continue
            n2 = n // n1
            R = (
                W.reshape(m1, m2, n1, n2)
                .transpose(0, 2, 1, 3)
                .reshape(m1 * n1, m2 * n2)
            )
            sv = np.linalg.svd(R, compute_uv=False)
            e1 = float(sv[0] ** 2 / (sv**2).sum())
            if best is None or e1 > best["e1"]:
                best = {
                    "e1": e1,
                    "factors": f"{m1}x{n1}⊗{m2}x{n2}",
                    "r95": int(
                        np.searchsorted(
                            np.cumsum(sv**2) / (sv**2).sum(), 0.95
                        )
                        + 1
                    ),
                }
    return best or {"e1": 0.0, "factors": "-", "r95": -1}


def _hmat_rank(W: np.ndarray) -> dict:
    """H-matrix probe: split W into 2×2 blocks recursively; a
    hierarchical-low-rank matrix has numerically-low-rank off-diagonal
    blocks at every level.  Report the worst relative rank of
    off-diagonal blocks at the finest split."""
    m, n = W.shape
    lvl = 1
    worst = 0.0
    while min(m // (2**lvl), n // (2**lvl)) >= 16 and lvl <= 4:
        bs_m, bs_n = m // (2**lvl), n // (2**lvl)
        ranks = []
        for i in range(2**lvl):
            for j in range(2**lvl):
                if i == j:
                    continue
                B = W[
                    i * bs_m : (i + 1) * bs_m, j * bs_n : (j + 1) * bs_n
                ]
                sv = np.linalg.svd(B, compute_uv=False)
                if sv[0] > 0:
                    ranks.append(
                        int((sv > 1e-2 * sv[0]).sum()) / min(B.shape)
                    )
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
    e = a**2
    tot = e.sum()
    return {
        "top10%": round(float(e[: len(a) // 10].sum() / tot), 3),
        "top1%": round(
            float(e[: max(1, len(a) // 100)].sum() / tot), 3
        ),
    }


def analyze2(W: np.ndarray) -> dict:
    """The structured-algebra family: beyond Toeplitz displacement."""
    out = _best_kron(W)
    out.update(_hmat_rank(W))
    out.update(_sparsity(W))
    return out


def _monarch_als(
    W: np.ndarray, block: int, iters: int = 25, seed: int = 0
) -> tuple[float, int]:
    """Two-factor monarch/butterfly:  W ≈ L · P · R  where L,R are
    block-diagonal (block×block) and P is the fixed transpose-reshape
    permutation — the atom of butterfly factorisations.

    Storage: 2·k·block² = 2·n·block for an n×n matrix.
    Returns (relative Frobenius residual, stored values)."""
    n, m = W.shape
    if n != m or n % block:
        return float("nan"), -1
    rng = np.random.default_rng(seed)
    k = n // block
    # P: the (k,block)-transpose permutation — index i = a*block+b ->
    # p(i) = b*k + a  (swap the two axes of the (k,block) grid)
    perm = np.arange(n).reshape(k, block).T.ravel()
    R = np.stack(
        [rng.standard_normal((block, block)) for _ in range(k)]
    )  # (k, b, b)
    for _ in range(iters):
        # fix R -> solve for L.  L block-diag means product row-block j
        # is  W[rows j] = L_j @ Q[cols j]  with Q = P·R.
        PR = np.zeros((n, n))
        for j in range(k):
            PR[
                j * block : (j + 1) * block, j * block : (j + 1) * block
            ] = R[j]  # block-diag
        Q = PR[perm, :]  # P·R
        L = np.zeros((k, block, block))
        for j in range(k):
            Wj = W[j * block : (j + 1) * block, :]  # (b, n)
            Qj = Q[j * block : (j + 1) * block, :]  # (b, n)
            Lj, *_ = np.linalg.lstsq(Qj.T, Wj.T, rcond=None)
            L[j] = Lj.T  # (b, b)
        # fix L -> solve R' = L⁺W densely, then project to the
        # block-diagonal component under P
        Ld = np.zeros((n, n))
        for j in range(k):
            Ld[
                j * block : (j + 1) * block, j * block : (j + 1) * block
            ] = L[j]
        Rp = np.linalg.pinv(Ld) @ W  # dense
        Rq = Rp[perm, :]  # R = P⁻¹R'
        for j in range(k):
            R[j] = Rq[
                j * block : (j + 1) * block, j * block : (j + 1) * block
            ]
    # residual
    PR = np.zeros((n, n))
    for j in range(k):
        PR[j * block : (j + 1) * block, j * block : (j + 1) * block] = (
            R[j]
        )
    Q = PR[perm, :]
    Ld = np.zeros((n, n))
    for j in range(k):
        Ld[j * block : (j + 1) * block, j * block : (j + 1) * block] = (
            L[j]
        )
    err = float(np.linalg.norm(W - Ld @ Q) / np.linalg.norm(W))
    return err, 2 * k * block * block


def _inr_probe(
    W: np.ndarray, hidden: int = 64, steps: int = 400, seed: int = 0
) -> tuple[float, int]:
    """Nonlinear probe: W ≈ g(i,j) where g is a small MLP over
    positional features — the 'weights as generated objects' test.
    Storage = net params; error = final relative Frobenius."""
    import torch

    torch.manual_seed(seed)
    m, n = W.shape
    ii, jj = np.meshgrid(
        np.linspace(-1, 1, m), np.linspace(-1, 1, n), indexing="ij"
    )
    feats = np.stack(
        [
            ii,
            jj,
            np.sin(np.pi * ii),
            np.sin(np.pi * jj),
            np.cos(np.pi * ii),
            np.cos(np.pi * jj),
        ],
        -1,
    )
    F = feats.shape[-1]
    X = torch.tensor(feats.reshape(-1, F), dtype=torch.float64)
    Y = torch.tensor(W.reshape(-1), dtype=torch.float64)
    g = torch.nn.Sequential(
        torch.nn.Linear(F, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, 1),
    ).double()
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((g(X).squeeze(-1) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = g(X).squeeze(-1).numpy().reshape(W.shape)
    err = float(np.linalg.norm(W - pred) / np.linalg.norm(W))
    n_params = sum(p.numel() for p in g.parameters())
    return err, n_params


def _svd_storage(W: np.ndarray, energy: float) -> int:
    sv = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    e = np.cumsum(sv**2) / (sv**2).sum()
    r = int(np.searchsorted(e, energy) + 1)
    return r * (W.shape[0] + W.shape[1])


def _kron_storage(W: np.ndarray, energy: float) -> tuple[int, str]:
    """Sum-of-Kronecker at given energy: K terms of best balanced
    factorisation."""
    m, n = W.shape
    best = (10**18, "-")
    for m1 in range(2, int(m**0.5) + 2):
        if m % m1:
            continue
        m2 = m // m1
        for n1 in range(2, int(n**0.5) + 2):
            if n % n1:
                continue
            n2 = n // n1
            R = (
                W.reshape(m1, m2, n1, n2)
                .transpose(0, 2, 1, 3)
                .reshape(m1 * n1, m2 * n2)
            )
            sv = np.linalg.svd(R, compute_uv=False)
            e = np.cumsum(sv**2) / (sv**2).sum()
            K = int(np.searchsorted(e, energy) + 1)
            stored = K * (m1 * n1 + m2 * n2)
            if stored < best[0]:
                best = (stored, f"({m1}x{n1})x({m2}x{n2}) K={K}")
    return best


def _sparse_storage(W: np.ndarray, energy: float) -> int:
    """Keep the largest entries until they hold `energy` of the
    Frobenius mass; a CSR-ish entry costs ~2 words (index+value)."""
    a = np.abs(W).ravel()
    a.sort()
    a = a[::-1]
    e = np.cumsum(a**2) / (a**2).sum()
    k = int(np.searchsorted(e, energy) + 1)
    return 2 * k


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

    print(
        f"{'matrix':<22} {'shape':>9} {'r@1e-2':>6} {'r@1e-3':>6} "
        f"{'r90%':>5} {'r99%':>5} {'stbl':>6} {'dispR':>5} "
        f"{'dispC':>5} {'sym':>6}"
    )
    print("-" * 92)
    for name, W in rows:
        a = analyze(W)
        if "skip" in a:
            continue
        n = min(W.shape)
        print(
            f"{name:<22} {a['shape']:>9} "
            f"{a['rank@0.01']:>4}/{n:<2} "
            f"{a['rank@0.001']:>4}/{n:<2} "
            f"{a['r90%']:>5} {a['r99%']:>5} "
            f"{a['stable_rk']:>6} {a['disp_row']:>5} "
            f"{a['disp_col']:>5} {a['sym_defect']:>6}"
        )

    # ---- aggregate verdict -------------------------------------------------
    mats = [W for _, W in rows]
    tot = sum(W.size for W in mats)
    r99_total = sum(
        analyze(W)["r99%"] * (W.shape[0] + W.shape[1]) for _, W in rows
    )
    print("-" * 92)
    print(f"total matrix params: {tot:,}")
    print(
        f"low-rank storage at 99% energy: {r99_total:,} "
        f"({r99_total / tot:.1%} of params)"
    )

    # ---- the structured-algebra family -------------------------------------
    print(
        "\n=== Phase 0b: structured algebras (Kronecker / H-matrix / "
        "sparse) ==="
    )
    print(
        f"{'matrix':<22} {'shape':>9} {'kron_e1':>8} {'kron95':>7} "
        f"{'H-rel':>6} {'top10%':>7} {'top1%':>7}"
    )
    print("-" * 70)
    for name, W in rows:
        if max(W.shape) > 4096:
            continue  # skip the embedding table — SVD cost dominates
        a2 = analyze2(W)
        print(
            f"{name:<22} {W.shape[0]}x{W.shape[1]:<5} "
            f"{a2['e1']:>8.3f} {a2['r95']:>7} "
            f"{a2['h_offdiag_rel']:>6} {a2['top10%']:>7} "
            f"{a2['top1%']:>7}"
        )

    # ---- rate–distortion gate: does ANYTHING beat plain SVD? --------
    # storage in 32-bit words at matched relative-Frobenius accuracy.
    print(
        "\n=== Phase 0b gate: stored values at 95% / 99% energy "
        "(vs SVD) ==="
    )
    print(
        f"{'matrix':<10} {'svd95':>8} {'kron95':>8} {'sprs95':>8} "
        f"{'mon95':>8} {'inr':>8} | {'svd99':>8} {'kron99':>8} "
        f"{'sprs99':>8}"
    )
    print("-" * 78)
    agg = {}
    for name, W in rows:
        if W.shape[0] != W.shape[1] or W.shape[0] % 32:
            continue  # monarch needs square, multiple-of-32
        s95 = _svd_storage(W, 0.95)
        k95, _kf = _kron_storage(W, 0.95)
        sp95 = _sparse_storage(W, 0.95)
        s99 = _svd_storage(W, 0.99)
        k99, _ = _kron_storage(W, 0.99)
        sp99 = _sparse_storage(W, 0.99)
        merr, mstore = _monarch_als(W, block=32, iters=20)
        ierr, istore = _inr_probe(W, hidden=48, steps=300)
        print(
            f"{name:<10} {s95:>8} {k95:>8} {sp95:>8} "
            f"{mstore:>8} {istore:>8} | {s99:>8} {k99:>8} "
            f"{sp99:>8}   monarch_err={merr:.3f} inr_err={ierr:.3f}"
        )
        for k_, v in (
            ("svd95", s95),
            ("kron95", k95),
            ("sprs95", sp95),
            ("svd99", s99),
            ("kron99", k99),
            ("sprs99", sp99),
        ):
            agg[k_] = agg.get(k_, 0) + v
    if agg:
        print("-" * 78)
        print(
            f"{'TOTAL':<10} {agg['svd95']:>8} {agg['kron95']:>8} "
            f"{agg['sprs95']:>8} {'':>8} {'':>8} | "
            f"{agg['svd99']:>8} {agg['kron99']:>8} {agg['sprs99']:>8}"
        )


# ---------------------------------------------------------------------------
# Relational probes — EXACT cross-layer structure (the last untested
# class).  Probes 1-3 are gates (any bitwise/exact hit = real lossless
# win); 4-5 are diagnostics (approximate redundancy evidence).
# ---------------------------------------------------------------------------


def _row_hashes(A: np.ndarray) -> dict:
    """{row_bytes: [indices]} — bitwise duplicate detection."""
    out: dict[bytes, list[int]] = {}
    fb = np.ascontiguousarray(A.reshape(len(A), -1))
    for i in range(len(fb)):
        out.setdefault(fb[i].tobytes(), []).append(i)
    return {k: v for k, v in out.items() if len(v) > 1}


def _cross_layer_dups(w: dict) -> dict:
    """Probe 1: bitwise row/col/head-block matches ACROSS layers."""
    hits = []
    for fam in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        if fam not in w:
            continue
        A = w[fam]  # (L, out, in)
        L, _O, _I = A.shape
        # rows (out-features), cols (in-features), head-blocks (48-row)
        rows = {i: {} for i in range(L)}
        cols = {i: {} for i in range(L)}
        for lyr in range(L):
            rows[lyr] = _row_hashes(A[lyr])
            cols[lyr] = _row_hashes(A[lyr].T)
        for i in range(L):
            for j in range(i + 1, L):
                shared_r = set(rows[i]) & set(rows[j])
                shared_c = set(cols[i]) & set(cols[j])
                for k in shared_r:
                    hits.append(
                        (fam, "row", i, j, rows[i][k], rows[j][k])
                    )
                for k in shared_c:
                    hits.append(
                        (fam, "col", i, j, cols[i][k], cols[j][k])
                    )
    return {"hits": hits, "n": len(hits)}


def _exact_rank(A: np.ndarray, tol_frac: float = 1e-6) -> int:
    """Numerical rank at strict tolerance — σ_i < tol·σ_max means
    linearly dependent to float precision."""
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s >= s[0] * tol_frac).sum())


def _shared_subspace(w: dict) -> dict:
    """Probe 2: stack all layers' rows; rank(stack) < Σ rank(layer)
    means layers share an exact subspace (store basis + coeffs)."""
    out = {}
    for fam in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        if fam not in w:
            continue
        A = w[fam]
        L = A.shape[0]
        stacked = A.reshape(L * A.shape[1], -1)
        r_stack = _exact_rank(stacked)
        r_sum = sum(_exact_rank(A[lyr]) for lyr in range(L))
        # The gate is rank(stack) < min(#rows, #cols) — a stacked
        # matrix that is FULL column rank shares no subspace; the
        # Σ-layer comparison alone is degenerate (ambient dim caps
        # the stack regardless).
        ambient = min(stacked.shape)
        out[fam] = {
            "stack_rank": r_stack,
            "sum_layer_rank": r_sum,
            "ambient": ambient,
            "saving_rows": ambient - r_stack,
        }
    return out


def _procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: Q minimizing ‖X − YQ‖_F.  Q = UVᵀ of
    XᵀY — implemented directly (no scipy)."""
    M = Y.T.astype(np.float64) @ X.astype(np.float64)
    U, _, Vt = np.linalg.svd(M)
    return (U @ Vt).T


def _greedy_assign(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Greedy max-|inner-product| assignment of B's rows to A's —
    approximation of Hungarian; enough for a diagnostic."""
    An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    C = np.abs(An @ Bn.T)
    assign = np.full(len(A), -1)
    order = np.argsort(-C.max(1))
    used = set()
    for i in order:
        j = int(
            np.argmax(
                np.where(
                    np.isin(np.arange(len(B)), list(used), invert=True),
                    C[i],
                    -1,
                )
            )
        )
        assign[i] = j
        used.add(j)
    return assign


def _alignment_probe(w: dict) -> dict:
    """Probe 3: align layer j to layer i (orthogonal Procrustes on the
    weight matrices + greedy row permutation), then re-check
    duplicates / shared subspace."""
    out = {}
    for fam in ("wq", "wv", "w1"):
        if fam not in w:
            continue
        A = w[fam].astype(np.float64)
        L = A.shape[0]
        aligned_dups = 0
        for j in range(1, L):
            Q = _procrustes(A[0], A[j])
            Aj = A[j] @ Q
            # bitwise near-duplicates after alignment (exact rel
            # tolerance at fp32 rounding level)
            scale = np.abs(A[0]).max() or 1.0
            close = np.abs(A[0] - Aj) <= scale * 1e-6
            aligned_dups += int(close.all(-1).any(-1).sum())
            # shared subspace after alignment
            st = np.vstack([A[0], Aj])
            rs = _exact_rank(st)
            r_sum = _exact_rank(A[0]) + _exact_rank(Aj)
            out.setdefault(fam, []).append(
                {
                    "vs": (0, j),
                    "aligned_dup_rows": int(
                        close.all(-1).any(-1).sum()
                    ),
                    "stack_rank": rs,
                    "sum_rank": r_sum,
                }
            )
        out[fam + "_total_aligned_dups"] = aligned_dups
    return out


def _stacked_spectrum(w: dict) -> dict:
    """Probe 4 (diagnostic): rank of the stacked normalized matrix at
    90/95/99% energy vs Σ dims — approximate cross-layer redundancy."""
    out = {}
    for fam in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        if fam not in w:
            continue
        A = w[fam]
        L = A.shape[0]
        st = A.reshape(L * A.shape[1], -1)
        st = st / np.linalg.norm(st, axis=1, keepdims=True).clip(1e-12)
        s = np.linalg.svd(st.astype(np.float64), compute_uv=False)
        e = np.cumsum(s**2) / (s**2).sum()
        out[fam] = {
            f"r{int(p * 100)}": int((e < p).sum()) + 1
            for p in (0.9, 0.95, 0.99)
        }
        out[fam]["dims"] = st.shape[1]
    return out


def _shared_dictionary(w: dict, k: int = 64) -> dict:
    """Probe 5 (diagnostic): cross-layer shared dictionary via SVD
    basis of the stacked matrix + greedy sparse codes."""
    out = {}
    for fam in ("wq", "wk", "wv", "wo"):
        if fam not in w:
            continue
        A = w[fam]
        L = A.shape[0]
        st = A.reshape(L * A.shape[1], -1).astype(np.float64)
        _U, _S, Vt = np.linalg.svd(st, full_matrices=False)
        D = Vt[:k]  # dictionary (k,in)
        # codes: st ≈ C @ D — least squares per row
        C = st @ D.T  # (rows, k)
        resid = st - C @ D
        rel = float(np.linalg.norm(resid) / np.linalg.norm(st))
        stored = D.size + C.size
        out[fam] = {
            "k": k,
            "rel_resid": round(rel, 4),
            "storage_ratio": round(stored / st.size, 4),
        }
    return out


def relational(
    path: str = "/tmp/stories15M.bin",
    path2: str | None = "/tmp/stories110M.bin",
) -> None:
    for p in [path] + ([path2] if path2 else []):
        try:
            w = load_llama2c(p)
        except Exception as e:
            print(f"{p}: load failed: {e}")
            continue
        print(f"\n{'=' * 64}\n{p} — relational probes\n{'=' * 64}")
        d = _cross_layer_dups(w)
        print(f"[1] exact cross-layer duplicates: {d['n']} hits")
        for h in d["hits"][:10]:
            print(f"    {h}")
        ss = _shared_subspace(w)
        for fam, v in ss.items():
            flag = " SHARED" if v["saving_rows"] > 0 else " full"
            print(
                f"[2] {fam}: stack_rank={v['stack_rank']}/"
                f"{v['ambient']} Σlayer_rank={v['sum_layer_rank']}"
                f"{flag}"
            )
        al = _alignment_probe(w)
        for fam in ("wq", "wv", "w1"):
            tot = al.get(fam + "_total_aligned_dups", 0)
            print(
                f"[3] {fam}: aligned dup rows total={tot}  "
                f"per-layer={al.get(fam)}"
            )
        sp = _stacked_spectrum(w)
        for fam, v in sp.items():
            print(
                f"[4] {fam}: dims={v['dims']} r90={v['r90']} "
                f"r95={v['r95']} r99={v['r99']}"
            )
        dc = _shared_dictionary(w)
        for fam, v in dc.items():
            print(
                f"[5] {fam}: dict k={v['k']} resid={v['rel_resid']} "
                f"storage={v['storage_ratio']}x"
            )


# ---------------------------------------------------------------------------
# Symmetry probes — the remaining untested hypothesis classes for
# EXACT structure.  Probe A (equivariance) and C (polynomial identity)
# are cheap; B (learned displacement operators) is the real bet.
# Exact hit = zero residual / small exact rank; approximate does not
# count (Phase 5).
# ---------------------------------------------------------------------------


def _cyclic_matrix(n: int, shift: int = 1) -> np.ndarray:
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i + shift) % n] = 1.0
    return P


def _reversal_matrix(n: int) -> np.ndarray:
    return np.eye(n)[::-1]


def _equivariance(W: np.ndarray) -> dict:
    """Probe A: search group families for G with WG = GW exactly.

    Cyclic shifts C^k (circulant case = C^1), the reversal R
    (dihedral), and block-cyclic shifts.  Exact hit = residual ~0."""
    n = min(W.shape)
    Wf = W[:n, :n].astype(np.float64)
    wnorm = np.linalg.norm(Wf)
    if wnorm == 0:
        return {"skip": "zero"}
    out = {}

    def resid(G):
        return float(np.linalg.norm(Wf @ G - G @ Wf) / wnorm)

    cres = {}
    for k in range(1, n):
        if n % k == 0 or k == 1:
            cres[k] = resid(_cyclic_matrix(n, k))
    best_c = min(cres, key=cres.get)
    out["cyclic"] = {
        "best_shift": best_c,
        "resid": cres[best_c],
        "circulant_resid": cres[1],
        "all_under_1e3": [k for k, r in cres.items() if r < 1e-3],
    }
    out["reversal"] = resid(_reversal_matrix(n))
    bres = {}
    for b in (2, 4, 8, 16):
        if n % b:
            continue
        nb = n // b
        Pb = np.zeros((n, n))
        for i in range(nb):
            for j in range(b):
                Pb[i * b + j, ((i + 1) % nb) * b + j] = 1.0
        bres[b] = resid(Pb)
    out["block_shift"] = bres
    out["n"] = n
    return out


def _poly_identity(W: np.ndarray, max_deg: int = 6) -> dict:
    """Probe C: minimal-polynomial evidence.

    (a) Krylov dimension: rank of [vec I, vec W, vec W², …] — degree
    at which it saturates bounds the minimal poly degree.
    (b) explicit low-degree fits p(W)=0 via least squares; exact hit
    = resid ≈ 0.  (c) distinct-eigenvalue cluster count (loose bound
    on minimal poly degree)."""
    n = min(W.shape)
    Wf = W[:n, :n].astype(np.float64)
    out = {}
    vecs = [np.eye(n).ravel()]
    P = np.eye(n)
    kry_rank = []
    for d in range(max_deg + 1):
        if d:
            P = P @ Wf
            vecs.append(P.ravel())
        kry_rank.append(_exact_rank(np.stack(vecs)))
    out["krylov_rank_by_degree"] = kry_rank
    fits = {}
    for d in range(1, min(max_deg, n) + 1):
        P = np.eye(n)
        basis = [np.eye(n).ravel()]
        for _k in range(1, d):
            P = P @ Wf
            basis.append(P.ravel())
        P = P @ Wf  # W^d
        A = np.stack(basis).T  # (n², d)
        b = -P.ravel()
        c, _res, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ c
        fits[d] = float(np.linalg.norm(pred - b)) / (
            np.linalg.norm(b) or 1.0
        )
    out["poly_fits"] = fits
    try:
        ev = np.linalg.eigvals(Wf)
        tol = np.abs(ev).max() * 1e-4
        clusters = []
        for e in sorted(ev, key=lambda z: (z.real, z.imag)):
            if not clusters or abs(e - clusters[-1][0]) > tol:
                clusters.append([e])
            else:
                clusters[-1].append(e)
        out["n_eig_clusters_1e4"] = len(clusters)
        out["n_eigs"] = len(ev)
    except Exception:
        out["eig_error"] = True
    return out


def _learned_displacement(
    W: np.ndarray,
    fam: str = "diag",
    iters: int = 40,
    inits: int = 4,
    seed: int = 0,
) -> dict:
    """Probe B: minimize rank(AW − WB) over LEARNED structured A, B —
    diagonal (Stein-lite) and circulant parameterizations.  Gradient
    descent, multi-init, divergence reported honestly."""
    n = min(W.shape)
    Wf = W[:n, :n].astype(np.float64)
    wnorm = np.linalg.norm(Wf) or 1.0
    rng = np.random.default_rng(seed)

    def params_to_ops(p):
        if fam == "diag":
            return np.diag(p[:n]), np.diag(p[n:])
        a = np.zeros((n, n))
        b = np.zeros((n, n))
        for k in range(n):
            a += p[k] * _cyclic_matrix(n, k)
            b += p[n + k] * _cyclic_matrix(n, k)
        return a, b

    def resid_rank(A, B):
        R = A @ Wf - Wf @ B
        rn = np.linalg.norm(R)
        tol = rn * 1e-3 if rn > 0 else 1e-12
        return rn / wnorm, int(np.linalg.matrix_rank(R, tol=tol))

    best = {"resid": np.inf}
    for init in range(inits):
        p = rng.standard_normal(2 * n) * 0.1
        hist, diverged = [], False
        for _it in range(iters):
            A, B = params_to_ops(p)
            R = A @ Wf - Wf @ B
            res = float(np.linalg.norm(R)) / wnorm
            hist.append(res)
            if not np.isfinite(res) or res > 1e6:
                diverged = True
                break
            gA = R @ Wf.T
            gB = -Wf.T @ R
            if fam == "diag":
                grad = np.concatenate([np.diag(gA), np.diag(gB)])
            else:
                grad = np.concatenate(
                    [
                        [
                            np.trace(_cyclic_matrix(n, k).T @ gA)
                            for k in range(n)
                        ],
                        [
                            np.trace(_cyclic_matrix(n, k).T @ gB)
                            for k in range(n)
                        ],
                    ]
                )
            step = 0.01 / (np.linalg.norm(grad) + 1e-12)
            p = p - step * grad
        A, B = params_to_ops(p)
        res, rr = resid_rank(A, B)
        if res < best["resid"]:
            best = {
                "resid": float(res),
                "rank_resid": rr,
                "n": n,
                "hist": [round(x, 4) for x in hist],
                "diverged": diverged,
                "init": init,
            }
    return best


def symmetries(path: str = "/tmp/stories15M.bin") -> None:
    w = load_llama2c(path)
    print(
        f"{'=' * 64}\n{path} — symmetry probes (EXACT only)\n{'=' * 64}"
    )
    mats = {"token_embedding": w["token_embedding"]}
    for fam in ("wq", "wv", "w1", "w2"):
        if fam in w:
            mats[fam + "[0]"] = w[fam][0]
    for name, W in mats.items():
        print(f"\n--- {name} {W.shape} ---")
        e = _equivariance(W)
        if "skip" not in e:
            print(
                f"[A] circulant resid={e['cyclic']['circulant_resid']:.4f}"
                f" best_cyclic=shift{e['cyclic']['best_shift']}:"
                f"{e['cyclic']['resid']:.4f}"
                f" reversal={e['reversal']:.4f}"
                f" block={e['block_shift']}"
            )
        p = _poly_identity(W)
        print(f"[C] krylov rank by deg: {p['krylov_rank_by_degree']}")
        print(
            f"    poly fits (resid): "
            f"{ {d: round(r, 4) for d, r in p['poly_fits'].items()} }"
        )
        if "n_eig_clusters_1e4" in p:
            print(
                f"    distinct eigenvalues (1e-4): "
                f"{p['n_eig_clusters_1e4']}/{p['n_eigs']}"
            )
        for fam2 in ("diag", "circ"):
            b = _learned_displacement(W, fam=fam2)
            print(
                f"[B] learned {fam2}: resid={b['resid']:.4f} "
                f"disp_rank≈{b['rank_resid']}/{b['n']} "
                f"diverged={b['diverged']}"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "relational":
        relational(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "symmetries":
        symmetries(*(sys.argv[2:] or ["/tmp/stories15M.bin"]))
    else:
        main()
