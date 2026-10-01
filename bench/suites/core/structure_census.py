"""Structural census of REAL trained checkpoints.

The decisive empirical question behind catopt's non-local passes:
how much of the structure they exploit — low-rank factors, dead and
duplicate weight slices, diagonal/block-diagonal weights, tied
parameters — actually exists in trained weights?

For each available llama2.c checkpoint (``bench/fetch.py``) this
reports, per weight matrix (normalised to ``(out, in)``):

* effective rank (SVD energy at 90%/99%) and the SVD-certified
  minimum rank at each relative-Frobenius tolerance — a *rigorous
  lower bound*: no rank-``r`` factorisation can reconstruct the
  weight with residual ``<= tol·‖W‖_F`` below that rank, so if the
  SVD rank already fails catopt's break-even ``r·(i+o) < i·o``, no
  low-rank offer can pay, regardless of the basis construction;
* dead / near-dead row & column fractions (``max|·|`` thresholds);
* bitwise and near (``max|Δ| <= 1e-3``) duplicate fractions;
* diagonal-ness, exact block-diagonal partitions, identity/zero.

Then, which catopt offers WOULD fire — the exact decision logic of
``offer_weight_specials`` (identity/diagonal/zero/elide/block_diag,
``error_bound = 0``) and its ``budget=`` members
(``elide_bounded``/``zero_bounded``, ``bound_norm="max_abs"``), plus
``offer_low_rank_factors`` (Gram-Schmidt rank + residual gate +
flop break-even) and the whole-model tie passes
(``share_duplicate_params`` / ``share_duplicate_param_slices``).

Method honesty: the slice analyses are faithful vectorized replicas
of ``specials._analyse`` / ``specials._bounded_elide`` (same bitwise
signatures, same greedy first-occurrence clustering, same
thresholds).  ``specials._blocks`` itself is imported and run on the
computed spans, so the block-diagonal decision is the real code.
The low-rank decision uses the SVD lower bound plus a numpy port of
``factored._row_basis``'s modified Gram-Schmidt.  All replicas are
validated against the real functions on probe sites; the agreement
record is written into the artifact (``probe_validation``).

The payoff: for every firing offer the flop/param delta it would
produce per token is summed into a per-model "exploitable slack"
table (GEMM flops only — attention/rope excluded; percentages are
explicitly scoped).  Written to ``bench/results/structure_census.json``.

    python bench/structure_census.py            # both checkpoints
    python bench/structure_census.py --ckpt ~/.cache/catopt/stories15M.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO))

from bench.common.llama2c import load_llama2c  # noqa: E402

DEAD_THRESHS = (0.0, 1e-4, 1e-3, 1e-2)
BUDGETS = (1e-4, 1e-3, 1e-2)
NEAR_DUP_TOL = 1e-3
LR_TOLS = (1e-8, 1e-4, 1e-3, 1e-2)


# ---------------------------------------------------------------------------
#  Checkpoint → sites (each weight normalised to wn (out, in), fp32)
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    """``$XDG_CACHE_HOME/catopt``, falling back to ``~/.cache/catopt``."""
    root = os.environ.get("XDG_CACHE_HOME") or str(
        Path.home() / ".cache"
    )
    return Path(root) / "catopt"


def header(path: str) -> dict:
    """Read the 7-int32 llama2.c header."""
    with open(path, "rb") as f:
        h = np.frombuffer(f.read(28), dtype=np.int32)
    keys = ("dim", "hidden", "n_layers", "n_heads", "n_kv", "vocab", "seq")
    return dict(zip(keys, (int(v) for v in h), strict=True))


def build_sites(w: dict, cfg: dict) -> list[dict]:
    """Enumerate weight sites as ``(out, in)`` fp32 matrices.

    llama2.c stores every projection ``(out, in)`` row-major — the
    ``linear`` orientation.  ``w1``/``w3`` come out of the loader
    labelled ``(dim, hidden)`` and ``w2`` ``(hidden, dim)``; the flat
    storage is ``(out, in)`` so a reshape recovers it (same as
    ``stories15m_bench.Stories15M``).  The LM head consumes the tied
    embedding values — a real GEMM site, marked ``shared`` so its
    parameter savings are not double-counted against the table.
    A separately-stored ``wcls`` (untied checkpoint) is picked up from
    ``_tail`` after the freq_cis block.
    """
    dim, hidden, L, vocab = (
        cfg["dim"],
        cfg["hidden"],
        cfg["n_layers"],
        cfg["vocab"],
    )
    emb = np.asarray(w["token_embedding"], dtype=np.float32)
    sites = [
        {
            "name": "token_embedding",
            "ltype": "embed",
            "wn": emb,
            "gemm": False,  # index_select site — not a catopt GEMM offer
            "shared": False,
        }
    ]
    # Untied wcls would sit in _tail after seq * head_dim floats.
    freq = cfg["seq"] * (dim // cfg["n_heads"])
    tail = w.get("_tail")
    wcls = None
    if tail is not None and tail.size >= freq + vocab * dim:
        wcls = np.asarray(
            tail[freq : freq + vocab * dim], dtype=np.float32
        ).reshape(vocab, dim)
    sites.append(
        {
            "name": "head" if wcls is None else "wcls",
            "ltype": "head",
            "wn": emb if wcls is None else wcls,
            "gemm": True,
            "shared": wcls is None,  # tied: params shared with embed
        }
    )
    for li in range(L):
        for nm, lt in (
            ("wq", "qkv"),
            ("wk", "qkv"),
            ("wv", "qkv"),
            ("wo", "o"),
        ):
            sites.append(
                {
                    "name": f"{nm}_{li}",
                    "ltype": lt,
                    "wn": np.asarray(w[nm][li], np.float32),
                    "gemm": True,
                    "shared": False,
                }
            )
        for nm, shape, lt in (
            ("w1", (hidden, dim), "ffn_up"),
            ("w2", (dim, hidden), "down"),
            ("w3", (hidden, dim), "ffn_up"),
        ):
            sites.append(
                {
                    "name": f"{nm}_{li}",
                    "ltype": lt,
                    "wn": np.asarray(w[nm][li], np.float32).reshape(shape),
                    "gemm": True,
                    "shared": False,
                }
            )
    return sites


# ---------------------------------------------------------------------------
#  Value analyses — vectorized replicas of specials._analyse/_bounded_elide
#  (validated against the real functions on probe sites)
# ---------------------------------------------------------------------------


def _spans(a: np.ndarray) -> list[tuple[int, int] | None]:
    """First/last nonzero column per row — ``specials._row_spans``."""
    nz = a != 0
    has = nz.any(axis=1)
    lo = np.argmax(nz, axis=1)
    hi = a.shape[1] - np.argmax(nz[:, ::-1], axis=1)
    return [
        (int(lo[r]), int(hi[r])) if has[r] else None
        for r in range(a.shape[0])
    ]


def _dedup_rows(a: np.ndarray) -> tuple[list[int], np.ndarray, int]:
    """Bitwise first-occurrence dedup — ``specials._dedup`` semantics."""
    seen: dict[bytes, int] = {}
    firsts: list[int] = []
    imap = np.empty(a.shape[0], np.int64)
    n_dup = 0
    for r in range(a.shape[0]):
        b = a[r].tobytes()
        j = seen.get(b)
        if j is None:
            j = len(firsts)
            seen[b] = j
            firsts.append(r)
        else:
            n_dup += 1
        imap[r] = j
    return firsts, imap, n_dup


def _analyse_np(a: np.ndarray) -> dict:
    """``specials._analyse`` equivalent (budget=None branch)."""
    from catopt_core.laws.specials import _blocks

    o, i = a.shape
    spans = _spans(a)
    firsts, imap, _ = _dedup_rows(a)
    col_any = (a != 0).any(axis=0)
    keep = np.nonzero(col_any)[0]
    diag = o == i and all(s == (j, j + 1) for j, s in enumerate(spans))
    return {
        "o": o,
        "i": i,
        "imap": imap,
        "firsts": firsts,
        "keep": keep,
        "diag": diag,
        "ident": bool(diag and bool((np.diag(a) == 1).all())),
        "zero": all(s is None for s in spans),
        "blocks": _blocks(spans, o, i) if not diag else None,
    }


def _bounded_elide_np(a: np.ndarray, budget: float) -> dict:
    """``specials._bounded_elide`` — same thresholds, same greedy order.

    A column is kept iff some element exceeds ``budget``; a row joins
    the zero group at ``max|·| <= budget`` or the FIRST representative
    within ``max|Δ| <= budget``, else opens a new representative.  The
    zero group reuses a real all-zero row when present, else a
    synthetic one is appended.

    For ``o > 8192`` the rep scan uses a column-0 sliding window —
    ``max|Δ| <= budget`` implies the first coordinate is within
    budget — then verifies candidates in ascending rep order.  The
    decisions are identical to the linear scan, just faster.
    """
    import bisect

    o, i = a.shape
    amax_col = np.abs(a).max(axis=0)
    keep = np.nonzero(amax_col > budget)[0]
    bound = (
        float(amax_col[amax_col <= budget].max())
        if keep.size < i
        else 0.0
    )
    row_max = np.abs(a).max(axis=1)
    rep_mat = np.empty((o, i), a.dtype)
    rep_sorted: list[tuple[float, int]] = []  # (col0, rep_pos)
    windowed = o > 8192
    k = 0
    firsts: list[int] = []
    zero_rows: list[int] = []
    real_zero: int | None = None
    n_near_dup = 0
    imap = np.zeros(o, np.int64)
    for r in range(o):
        ra = float(row_max[r])
        if ra <= budget:
            bound = max(bound, ra)
            zero_rows.append(r)
            if ra == 0.0 and real_zero is None:
                real_zero = r
            continue
        pos = -1
        row = a[r]
        if windowed:
            lo = bisect.bisect_left(
                rep_sorted, (float(row[0]) - budget, -1)
            )
            hi = bisect.bisect_right(
                rep_sorted, (float(row[0]) + budget, 1 << 62)
            )
            for cand in sorted(
                rep_sorted[j][1] for j in range(lo, hi)
            ):
                d = float(np.abs(rep_mat[cand] - row).max())
                if d <= budget:
                    pos = cand
                    n_near_dup += 1
                    bound = max(bound, d)
                    break
        elif k:
            d = np.abs(rep_mat[:k] - row).max(axis=1)
            hit = np.nonzero(d <= budget)[0]
            if hit.size:
                pos = int(hit[0])
                n_near_dup += 1
                bound = max(bound, float(d[pos]))
        if pos < 0:
            pos = k
            rep_mat[k] = row
            bisect.insort(rep_sorted, (float(row[0]), k))
            k += 1
            firsts.append(r)
        imap[r] = pos
    zero_appended = False
    if zero_rows:
        if real_zero is not None:
            firsts.append(real_zero)
            zpos = len(firsts) - 1
        else:
            zero_appended = True
            zpos = len(firsts)
        for r in zero_rows:
            imap[r] = zpos
    return {
        "keep": keep,
        "imap": imap,
        "firsts": firsts,
        "zero_appended": zero_appended,
        "bound": bound,
        "all_dead": len(zero_rows) == o,
        "n_zero_rows": len(zero_rows),
        "n_near_dup": n_near_dup,
    }


def _sv(a: np.ndarray) -> np.ndarray:
    """Singular values (descending) via the smaller Gram matrix."""
    o, i = a.shape
    a64 = a.astype(np.float64)
    g = (a64 @ a64.T) if o < i else (a64.T @ a64)
    ev = np.linalg.eigvalsh(g)
    return np.sqrt(np.clip(ev, 0.0, None))[::-1]


def _rank_table(s: np.ndarray, mn: int) -> dict:
    """Energy ranks and SVD-certified minimum ranks per tolerance."""
    e = s * s
    tot = float(e.sum())
    out = {"full": mn}
    if tot <= 0:
        return {**out, "r90": 0, "r99": 0}
    cum = np.cumsum(e)
    resids = np.concatenate([[tot], tot - cum])  # resids[r] = tail
    out["r90"] = int(np.searchsorted(cum, 0.90 * tot) + 1)
    out["r99"] = int(np.searchsorted(cum, 0.99 * tot) + 1)
    for t in LR_TOLS:
        bound = t * t * tot
        hit = np.nonzero(resids <= bound)[0]
        out[f"r_tol_{t:g}"] = int(hit[0]) if hit.size else mn
    return out


def _gs_rank(a: np.ndarray, tol: float) -> tuple[int, float | None]:
    """``factored._row_basis`` + residual gate, in numpy (same math).

    Modified Gram-Schmidt over the rows; a row whose residual norm
    drops below ``tol * max_row_norm`` is covered.  Returns the
    certified basis size ``r`` and the relative Frobenius
    reconstruction residual ``err / ‖W‖_F`` (the pass's own gate).
    """
    o, i = a.shape
    rn = np.sqrt((a.astype(np.float64) ** 2).sum(axis=1))
    scale = float(rn.max()) if o else 0.0
    thresh = tol * max(scale, 1e-12)
    basis = np.empty((o, i), np.float32)
    r = 0
    mn = min(o, i)
    for row in a:
        v = row.astype(np.float32).copy()
        for j in range(r):
            u = basis[j]
            v -= u * float((u * v).sum())
        n = float(np.sqrt((v.astype(np.float64) ** 2).sum()))
        if n > thresh:
            basis[r] = v / np.float32(n)
            r += 1
            if r >= mn:
                # The pass rejects a basis at full rank outright —
                # the offer can never fire; skip the rest.
                return r, None
    fnorm = float(np.sqrt((a.astype(np.float64) ** 2).sum()))
    if r == 0:
        return 0, 1.0 if fnorm else 0.0
    b64 = basis[:r].astype(np.float64)
    resid = a.astype(np.float64) - (a.astype(np.float64) @ b64.T) @ b64
    err = float(np.sqrt((resid * resid).sum()))
    return r, err / max(fnorm, 1e-30)


# ---------------------------------------------------------------------------
#  Census of one matrix
# ---------------------------------------------------------------------------


def census_matrix(a: np.ndarray) -> tuple[dict, dict, dict]:
    """Structural metrics for one ``(out, in)`` weight.

    Returns ``(census_dict, exact_analysis, bounded_by_budget)`` —
    the analyses double as the offer-decision inputs downstream.
    """
    o, i = a.shape
    c: dict = {
        "o": o,
        "i": i,
        "params": o * i,
        "nan": bool(np.isnan(a).any()),
    }
    aa = np.abs(a)
    row_max = aa.max(axis=1) if o else np.zeros(0)
    col_max = aa.max(axis=0) if i else np.zeros(0)
    c["row_max_abs"] = {
        "min": float(row_max.min()) if o else 0.0,
        "p50": float(np.median(row_max)) if o else 0.0,
        "max": float(row_max.max()) if o else 0.0,
    }
    c["col_max_abs"] = {
        "min": float(col_max.min()) if i else 0.0,
        "p50": float(np.median(col_max)) if i else 0.0,
    }
    for t in DEAD_THRESHS:
        c[f"dead_rows_frac@{t:g}"] = float((row_max <= t).mean())
        c[f"dead_cols_frac@{t:g}"] = float((col_max <= t).mean())
    an = _analyse_np(a)
    c["dup_rows_bitwise"] = int(o - len(an["firsts"]))
    c["dup_rows_bitwise_frac"] = float(1 - len(an["firsts"]) / o)
    at = np.ascontiguousarray(a.T)
    _f_c, _imap_c, n_dup_c = _dedup_rows(at)
    c["dup_cols_bitwise"] = n_dup_c
    c["dup_cols_bitwise_frac"] = float(n_dup_c / i)
    c["is_zero"] = bool(an["zero"])
    c["is_diag"] = bool(an["diag"])
    c["is_ident"] = bool(an["ident"])
    c["blocks"] = len(an["blocks"]) if an["blocks"] else 0
    mn = min(o, i)
    diag_abs = float(np.abs(np.diag(a[:mn, :mn])).sum()) if mn else 0.0
    c["diag_mass_frac"] = float(diag_abs / max(aa.sum(), 1e-30))
    off = aa.copy()
    if mn:
        off[np.arange(mn), np.arange(mn)] = 0.0
    c["offdiag_max_abs"] = float(off.max()) if o and i else 0.0
    s = _sv(a)
    c["sigma_max"] = float(s[0]) if s.size else 0.0
    c["rank"] = _rank_table(s, mn)
    # Bounded analyses at every budget — 1e-3 doubles as the
    # near-duplicate metric.
    bounded = {f"{b:g}": _bounded_elide_np(a, b) for b in BUDGETS}
    b1 = bounded[f"{NEAR_DUP_TOL:g}"]
    c["near_dup_rows@1e-3"] = b1["n_near_dup"]
    c["near_dup_rows_frac@1e-3"] = float(b1["n_near_dup"] / o)
    bt = _bounded_elide_np(at, NEAR_DUP_TOL)
    c["near_dup_cols@1e-3"] = bt["n_near_dup"]
    c["near_dup_cols_frac@1e-3"] = float(bt["n_near_dup"] / i)
    return c, an, bounded


# ---------------------------------------------------------------------------
#  Offer decisions — replicating specials._members / factored's gates
# ---------------------------------------------------------------------------


def _special_members(an: dict, bnd: dict | None) -> list[dict]:
    """Candidate special members with flop/param cost — mirrors
    ``specials._members`` (strongest structure wins; elide +
    elide_bounded + block_diag may coexist)."""
    o, i = an["o"], an["i"]
    if an["ident"]:
        return [{"kind": "identity", "flops": 0, "params": 0, "bound": 0.0}]
    if an["diag"]:
        return [
            {"kind": "diagonal", "flops": o, "params": o, "bound": 0.0}
        ]
    if an["zero"]:
        return [{"kind": "zero", "flops": 0, "params": 0, "bound": 0.0}]
    if bnd is not None and bnd["all_dead"]:
        return [
            {
                "kind": "zero_bounded",
                "flops": 0,
                "params": 0,
                "bound": float(bnd["bound"]),
            }
        ]
    ms = []
    if len(an["keep"]) < i or len(an["firsts"]) < o:
        p = len(an["firsts"]) * len(an["keep"])
        ms.append(
            {"kind": "elide", "flops": 2 * p, "params": p, "bound": 0.0}
        )
    if bnd is not None and bnd["bound"] > 0:
        n_sub = len(bnd["firsts"]) + int(bnd["zero_appended"])
        if not (len(bnd["keep"]) == i and n_sub == o):
            p = n_sub * len(bnd["keep"])
            ms.append(
                {
                    "kind": "elide_bounded",
                    "flops": 2 * p,
                    "params": p,
                    "bound": float(bnd["bound"]),
                }
            )
    if an["blocks"]:
        p = sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in an["blocks"])
        ms.append(
            {
                "kind": "block_diag",
                "flops": 2 * p,
                "params": p,
                "bound": 0.0,
            }
        )
    return ms


def _low_rank_member(
    a: np.ndarray, tol: float, sv_rank: int
) -> dict | None:
    """Would ``offer_low_rank_factors`` fire at ``rel_tol=tol``?

    ``sv_rank`` is the SVD-certified minimum rank for a ``tol``-bounded
    reconstruction — if it already fails the break-even no offer can
    pay (a rigorous no).  Otherwise run the Gram-Schmidt replica for
    the real basis size the pass would certify.
    """
    o, i = a.shape
    if sv_rank >= min(o, i) or sv_rank * (o + i) >= o * i:
        return {
            "kind": "low_rank",
            "fired": False,
            "sv_rank": sv_rank,
            "method": "svd_lower_bound",
        }
    r, rel = _gs_rank(a, tol)
    fired = (
        0 < r < min(o, i)
        and r * (o + i) < o * i
        and rel is not None
        and rel <= tol
    )
    out = {
        "kind": "low_rank",
        "fired": bool(fired),
        "gs_rank": int(r),
        "sv_rank": sv_rank,
        "rel_frobenius": float(rel) if rel is not None else None,
        "method": "gs_replica",
    }
    if fired:
        out["flops"] = 2 * r * (o + i)
        out["params"] = r * (o + i)
    return out


# ---------------------------------------------------------------------------
#  Real-pass probes — validate the replicas against catopt itself
# ---------------------------------------------------------------------------


def _eg_site(wt, prefix="p"):
    """Fresh EGraph holding ``linear(x, W)``; return (eg, tensors)."""
    import torch
    from catopt_core.egraph import EGraph
    from catopt_core.ir import Op, Param, TensorType, Var

    o, i = wt.shape
    eg = EGraph()
    x = Var("x", TensorType((4, i)))
    w = Param(prefix, TensorType((o, i)))
    eg.add_term(Op.make("linear", x, w))
    return eg, {prefix: wt}


def _run_probes(model: str, sites: list[dict]) -> list[dict]:
    """Validate replica decisions against the real catopt passes."""
    import torch
    from catopt_core.laws.factored import offer_low_rank_factors
    from catopt_core.laws.specials import (
        _analyse,
        _bounded_elide,
        offer_weight_specials,
    )

    probes: list[dict] = []
    by_type: dict[str, dict] = {}
    for s in sites:
        if s["gemm"]:
            by_type.setdefault(s["ltype"], s)

    def rep_analysis(a):
        an = _analyse_np(a)
        return {
            "keep": tuple(int(c) for c in an["keep"]),
            "firsts": tuple(int(r) for r in an["firsts"]),
            "diag": an["diag"],
            "ident": an["ident"],
            "zero": an["zero"],
            "blocks": an["blocks"],
        }

    # 1. Exact analysis parity on one site per layer type (the real
    # ``_dedup`` is O(o²) on byte strings — probe it on the 32k-row
    # head too; ~1 min is acceptable validation cost).
    for lt, s in sorted(by_type.items()):
        a = s["wn"]
        o, i = a.shape
        t0 = time.time()
        wt = torch.from_numpy(np.array(a, dtype=np.float32))
        real = _analyse(wt, o, i)
        rep = rep_analysis(a)
        agree = (
            tuple(int(c) for c in real["keep"]) == rep["keep"]
            and tuple(int(r) for r in real["firsts"]) == rep["firsts"]
            and real["diag"] == rep["diag"]
            and real["ident"] == rep["ident"]
            and real["zero"] == rep["zero"]
            and real["blocks"] == rep["blocks"]
        )
        probes.append(
            {
                "model": model,
                "site": s["name"],
                "check": "specials._analyse",
                "agree": bool(agree),
                "secs": round(time.time() - t0, 1),
            }
        )

    # 2. Bounded-elide parity on the two smallest-dim site types.
    for lt in ("qkv", "down"):
        s = by_type.get(lt)
        if s is None:
            continue
        a = s["wn"]
        o, i = a.shape
        wt = torch.from_numpy(np.array(a, dtype=np.float32))
        for b in BUDGETS:
            t0 = time.time()
            real = _bounded_elide(wt, o, i, float(b))
            rep = _bounded_elide_np(a, b)
            agree = (
                tuple(int(c) for c in real["keep"])
                == tuple(int(c) for c in rep["keep"])
                and tuple(int(r) for r in real["firsts"])
                == tuple(int(r) for r in rep["firsts"])
                and tuple(int(v) for v in real["imap"])
                == tuple(int(v) for v in rep["imap"])
                and real["zero_appended"] == rep["zero_appended"]
                and real["all_dead"] == rep["all_dead"]
                and abs(float(real["bound"]) - rep["bound"])
                <= 1e-6 * max(rep["bound"], 1e-30)
            )
            probes.append(
                {
                    "model": model,
                    "site": s["name"],
                    "check": f"specials._bounded_elide@{b:g}",
                    "agree": bool(agree),
                    "secs": round(time.time() - t0, 1),
                }
            )

    # 3. End-to-end offer parity on the qkv site (both passes).
    s = by_type["qkv"]
    a = s["wn"]
    wt = torch.from_numpy(np.array(a, dtype=np.float32))
    for b in (None,) + BUDGETS:
        t0 = time.time()
        eg, tensors = _eg_site(wt)
        offers = offer_weight_specials(eg, tensors, budget=b)
        rep = _special_members(
            _analyse_np(a), _bounded_elide_np(a, b) if b else None
        )
        agree = sorted(r["kind"] for r in offers) == sorted(
            m["kind"] for m in rep
        )
        probes.append(
            {
                "model": model,
                "site": s["name"],
                "check": f"offer_weight_specials@{b if b else 'exact'}",
                "real_kinds": sorted(r["kind"] for r in offers),
                "replica_kinds": sorted(m["kind"] for m in rep),
                "agree": bool(agree),
                "secs": round(time.time() - t0, 1),
            }
        )
    for t in LR_TOLS:
        t0 = time.time()
        eg, tensors = _eg_site(wt)
        offers = offer_low_rank_factors(eg, tensors, rel_tol=t)
        sv_r = s["census"]["rank"][f"r_tol_{t:g}"]
        rep = _low_rank_member(a, t, sv_r)
        real_fired = len(offers) > 0
        agree = real_fired == bool(rep["fired"])
        rec = {
            "model": model,
            "site": s["name"],
            "check": f"offer_low_rank_factors@{t:g}",
            "real_fired": real_fired,
            "replica": rep,
            "agree": bool(agree),
            "secs": round(time.time() - t0, 1),
        }
        if real_fired:
            rec["real_rank"] = offers[0]["rank"]
            rec["rank_match"] = offers[0]["rank"] == rep.get("gs_rank")
        probes.append(rec)
    return probes


def _run_share_passes(src: dict) -> dict:
    """Whole-model tie detection — the real passes on real params."""
    import torch
    from catopt_core.egraph import EGraph
    from catopt_core.ir import Param, TensorType
    from catopt_core.laws import (
        share_duplicate_param_slices,
        share_duplicate_params,
    )

    eg = EGraph()
    tensors = {}
    for name, a in src.items():
        wt = torch.from_numpy(np.array(a, dtype=np.float32))
        tensors[name] = wt
        eg.add_term(Param(name, TensorType(tuple(wt.shape))))
    t0 = time.time()
    groups = share_duplicate_params(eg, tensors)
    t_param = time.time() - t0
    t0 = time.time()
    offers = share_duplicate_param_slices(eg, tensors)
    t_slices = time.time() - t0
    return {
        "share_duplicate_params_groups": groups,
        "share_duplicate_params_secs": round(t_param, 2),
        "share_duplicate_param_slices_offers": [
            {
                "param": r["param"],
                "heads": r["heads"],
                "unique": r["unique"],
                "stored_before": r["stored_before"],
                "stored_after": r["stored_after"],
            }
            for r in offers
        ],
        "share_duplicate_param_slices_secs": round(t_slices, 2),
    }


# ---------------------------------------------------------------------------
#  Model census
# ---------------------------------------------------------------------------


def census_model(path: str, probes: bool) -> dict:
    """Full census + offer/slack accounting for one checkpoint."""
    cfg = header(path)
    w = load_llama2c(path)
    sites = build_sites(w, cfg)
    n_stored = sum(int(np.asarray(v).size) for k, v in w.items() if k != "_tail")
    tail = w.get("_tail")
    freq = cfg["seq"] * (cfg["dim"] // cfg["n_heads"])
    if tail is not None and tail.size > freq:
        n_stored += int(tail.size - freq)  # untied wcls
    model = os.path.basename(path)
    print(f"\n=== {model}: {cfg}  stored params={n_stored:,}")

    dense_flops = 0
    dense_params = 0
    for s in sites:
        t0 = time.time()
        a = s["wn"]
        c, an, bounded = census_matrix(a)
        s["census"] = c
        s["analysis_exact"] = an
        s["bounded"] = bounded
        if s["gemm"]:
            s["dense_flops"] = 2 * c["o"] * c["i"]
            dense_flops += s["dense_flops"]
            if not s["shared"]:
                s["dense_params"] = c["o"] * c["i"]
                dense_params += s["dense_params"]
            else:
                s["dense_params"] = 0
        # low-rank decisions per tolerance (SVD bound first)
        s["low_rank"] = {}
        if s["gemm"]:
            for t in LR_TOLS:
                sv_r = c["rank"][f"r_tol_{t:g}"]
                s["low_rank"][f"{t:g}"] = _low_rank_member(a, t, sv_r)
        print(
            f"  {s['name']:>16} [{s['ltype']:>7}] "
            f"{c['o']}x{c['i']}  r90={c['rank']['r90']} "
            f"r99={c['rank']['r99']}/{c['rank']['full']} "
            f"dead%@1e-3={100 * c['dead_rows_frac@0.001']:.2f} "
            f"dup%={100 * c['dup_rows_bitwise_frac']:.2f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    # ---- offer decisions + slack per budget --------------------------
    budgets = ("exact",) + tuple(f"{b:g}" for b in BUDGETS)
    slack = {b: {"flops_saved": 0, "params_saved": 0, "hits": {}}
             for b in budgets}
    for s in sites:
        if not s["gemm"]:
            s["offers"] = {}
            continue
        an = s["analysis_exact"]
        s["offers"] = {"exact": _special_members(an, None)}
        for b in BUDGETS:
            s["offers"][f"{b:g}"] = _special_members(
                an, s["bounded"][f"{b:g}"]
            )
        for tag in budgets:
            bnd_key = tag if tag != "exact" else None
            members = list(s["offers"][bnd_key or "exact"])
            if tag != "exact":
                lr = s["low_rank"].get(tag)
                if lr and lr.get("fired"):
                    members.append(
                        {
                            "kind": "low_rank",
                            "flops": lr["flops"],
                            "params": lr["params"],
                            "bound": lr["rel_frobenius"],
                        }
                    )
            s[f"best_{tag}"] = members
            if not members:
                continue
            df = s["dense_flops"]
            dp = s["dense_params"]
            bf = min(m["flops"] for m in members)
            bp = min(m["params"] for m in members)
            slack[tag]["flops_saved"] += max(0, df - bf)
            # Signed: a derived weight on a SHARED site (tied head)
            # adds storage — the tied embedding cannot shrink — so
            # the delta is honestly negative there.
            slack[tag]["params_saved"] += dp - bp
            for m in members:
                k = m["kind"]
                slack[tag]["hits"][k] = slack[tag]["hits"].get(k, 0) + 1

    res = {
        "model": model,
        "path": path,
        "header": cfg,
        "stored_params": n_stored,
        "gemm_flops_per_token": dense_flops,
        "gemm_params": dense_params,
        "slack": {},
        "probes": _run_probes(model, sites) if probes else [],
        # Only tensors actually STORED in the checkpoint — the tied
        # head aliases token_embedding and must not fake a tie group.
        "share": _run_share_passes(
            {s["name"]: s["wn"] for s in sites if not s["shared"]}
            | {
                k: v
                for k, v in w.items()
                if k in ("rms_att", "rms_ffn", "rms_final")
            }
        ),
        "sites": [],
    }
    for tag in budgets:
        sl = slack[tag]
        res["slack"][tag] = {
            "flops_saved_per_token": sl["flops_saved"],
            "flops_saved_pct": 100.0 * sl["flops_saved"] / dense_flops,
            "params_saved": sl["params_saved"],
            "params_saved_pct_of_stored": (
                100.0 * sl["params_saved"] / n_stored
            ),
            "params_saved_pct_of_gemm": (
                100.0 * sl["params_saved"] / dense_params
                if dense_params
                else 0.0
            ),
            "hits_by_kind": sl["hits"],
        }
    for s in sites:
        c = dict(s["census"])
        c["name"] = s["name"]
        c["ltype"] = s["ltype"]
        c["gemm"] = s["gemm"]
        c["shared"] = s["shared"]
        c["dense_flops_per_token"] = s.get("dense_flops", 0)
        c["low_rank"] = s["low_rank"]
        c["offers_exact"] = [
            {k: v for k, v in m.items()} for m in s["offers"].get("exact", [])
        ]
        for b in BUDGETS:
            c[f"offers@{b:g}"] = [
                {k: v for k, v in m.items()}
                for m in s["offers"].get(f"{b:g}", [])
            ]
            bb = s["bounded"][f"{b:g}"]
            c[f"bounded@{b:g}"] = {
                "bound": float(bb["bound"]),
                "all_dead": bb["all_dead"],
                "n_zero_rows": bb["n_zero_rows"],
                "n_near_dup": bb["n_near_dup"],
                "n_reps": len(bb["firsts"]),
                "n_keep_cols": len(bb["keep"]),
            }
        res["sites"].append(c)
    return res


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

LTYPES = ("embed", "head", "qkv", "o", "ffn_up", "down")


def _pct(x):
    return f"{100 * x:6.3f}"


def print_report(res: dict) -> None:
    """Per-model layer-type table + slack table."""
    h = res["header"]
    print(
        f"\n{'=' * 78}\n{res['model']}  dim={h['dim']} "
        f"hidden={h['hidden']} L={h['n_layers']} vocab={h['vocab']}  "
        f"params={res['stored_params']:,}  "
        f"gemm flops/tok={res['gemm_flops_per_token'] / 1e6:.1f}M\n{'=' * 78}"
    )
    print(
        f"{'type':>8} {'n':>3} {'r90/min':>9} {'r99/min':>9} "
        f"{'dead_r@1e-3':>11} {'dead_c@1e-3':>11} {'dup_r':>6} "
        f"{'ndup_r@1e-3':>11} {'ndup_c@1e-3':>11} {'blk':>4}"
    )
    for lt in LTYPES:
        ss = [s for s in res["sites"] if s["ltype"] == lt]
        if not ss:
            continue
        fr = lambda k: np.median([s[k] for s in ss])
        r90 = np.median(
            [s["rank"]["r90"] / s["rank"]["full"] for s in ss]
        )
        r99 = np.median(
            [s["rank"]["r99"] / s["rank"]["full"] for s in ss]
        )
        print(
            f"{lt:>8} {len(ss):>3} {r90:>9.3f} {r99:>9.3f} "
            f"{_pct(fr('dead_rows_frac@0.001')):>11} "
            f"{_pct(fr('dead_cols_frac@0.001')):>11} "
            f"{_pct(fr('dup_rows_bitwise_frac')):>6} "
            f"{_pct(fr('near_dup_rows_frac@1e-3')):>11} "
            f"{_pct(fr('near_dup_cols_frac@1e-3')):>11} "
            f"{max(s['blocks'] for s in ss):>4}"
        )
    print("\nexploitable slack (GEMM flops / stored params):")
    print(
        f"{'budget':>8} {'flops%':>8} {'params%stored':>14} "
        f"{'params%gemm':>12}  hits"
    )
    for tag, sl in res["slack"].items():
        hits = ", ".join(
            f"{k}×{v}" for k, v in sorted(sl["hits_by_kind"].items())
        ) or "—"
        print(
            f"{tag:>8} {sl['flops_saved_pct']:>7.3f}% "
            f"{sl['params_saved_pct_of_stored']:>13.3f}% "
            f"{sl['params_saved_pct_of_gemm']:>11.3f}%  {hits}"
        )
    sp = res["share"]
    print(
        f"\ntie passes: share_duplicate_params groups="
        f"{len(sp['share_duplicate_params_groups'])}  "
        f"share_duplicate_param_slices offers="
        f"{len(sp['share_duplicate_param_slices_offers'])}"
    )
    bad = [p for p in res["probes"] if not p.get("agree", True)]
    print(
        f"probe validation: {len(res['probes']) - len(bad)}/"
        f"{len(res['probes'])} agree"
        + (f"  DISAGREE: {bad}" if bad else "")
    )


def _jsonable(o):
    """Recursively convert numpy types for json.dump."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="checkpoint path(s); default discovers stories15M.bin "
        "and stories110M.bin in $XDG_CACHE_HOME/catopt, "
        "~/.cache/catopt or /tmp",
    )
    ap.add_argument(
        "--out",
        default=str(BENCH_DIR / "results" / "structure_census.json"),
    )
    ap.add_argument(
        "--no-probes",
        action="store_true",
        help="skip real-pass probe validation (replicas only)",
    )
    args = ap.parse_args()

    paths = list(args.ckpt or [])
    if not paths:
        for name in ("stories15M.bin", "stories110M.bin"):
            for p in (cache_dir() / name, Path("/tmp") / name):
                if p.is_file():
                    paths.append(str(p))
                    break
    if not paths:
        sys.exit(
            "no checkpoints found — run `python bench/fetch.py` or "
            "pass --ckpt"
        )

    results = []
    for p in paths:
        if not Path(p).is_file():
            alt = cache_dir() / Path(p).name
            if alt.is_file():
                p = str(alt)
            else:
                print(f"skip missing checkpoint: {p}")
                continue
        results.append(census_model(p, probes=not args.no_probes))

    for res in results:
        print_report(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": (
            "value census over (out,in)-normalised weight matrices; "
            "exact/bounded offer decisions replicate "
            "catopt_core.laws.specials._analyse/_bounded_elide/_members "
            "and catopt_core.laws.factored's Gram-Schmidt + break-even; "
            "low-rank fire/no-fire additionally gated by the "
            "SVD-certified minimum rank (rigorous lower bound); "
            "replicas validated against the real passes on probe "
            "sites (see probes). GEMM flops = 2·out·in per token per "
            "site; attention/rope/norm excluded. Head params are tied "
            "to the embedding and counted once."
        ),
        "budgets": list(BUDGETS),
        "near_dup_tol": NEAR_DUP_TOL,
        "lr_tols": list(LR_TOLS),
        "dead_thresholds": list(DEAD_THRESHS),
        "models": [_jsonable(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")

    # Honest headline
    for res in results:
        best = max(
            sl["flops_saved_pct"] for sl in res["slack"].values()
        )
        verdict = (
            "thin (<2%) — real weights carry little exploitable "
            "structure"
            if best < 2.0
            else f"concentrated win — {best:.1f}% slack"
        )
        print(f"verdict {res['model']}: {verdict}")


if __name__ == "__main__":
    main()
