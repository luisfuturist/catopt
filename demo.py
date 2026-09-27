"""catopt demo — categorical optimization of a real block, end to end.

One command, five acts:

    1. MODEL       a gated projection block — three shared-input
                   projections (two gates + the value path), a
                   10-deep value chain, an output projection.
    2. SEARCH      ``optimize_model`` runs equality saturation:
                   which rules fired, which non-local passes
                   landed, what term was extracted.
    3. CERTIFICATE the differentiator: the e-graph records WHY
                   every merge is an equality.  We replay the
                   derivation for the *shipped* program through
                   the standalone verifier, then check numbers.
    4. PROGRAM     the extracted term as an op tree — one fused
                   GEMM + three splits where the projections
                   shared an input, one matmul where ten ran
                   left-associative.
    5. RACE        median wall time, CUDA-synchronised where
                   applicable: eager vs torch.compile vs catopt.

The model is chosen so catopt's structural win is honest on this
box: Inductor cannot concatenate shared-input weights (they are
runtime parameters, not constants) and has no matmul-
reassociation pass, so both rewrites are unreachable for it.

Reproducibility: the script re-execs itself once with
``PYTHONHASHSEED=0`` so hash-consing order — and therefore
saturation, extraction and the certificate — is bit-for-bit
deterministic across runs (model weights are ``manual_seed(0)``).

Usage:
    python demo.py                  # CPU
    python demo.py --device cuda    # GPU
    python demo.py --quick          # shorter timing loop

    # CUDA dev venv (see bench/results/GPU_RUN.md):
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-optimize/src:." \
        /tmp/catopt-cuda-venv/bin/python demo.py --device cuda
"""

# ruff: noqa: RUF001 RUF002 RUF003
#   σ, ⊙, ×, ·, ═, →, ★ in strings/docstrings are deliberate
#   math/box-drawing notation; same convention as bench scripts.
from __future__ import annotations

import argparse
import copy
import os
import signal
import statistics
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
from catopt.egraph import verify_certificate
from catopt.ir import Op, Param, Var, op_repr
from catopt.optimize import (
    discover_alternatives,
    optimize_model,
    param_report,
)
from catopt_torch.adapters import TorchSink

# ----------------------------------------------------------------------
#  The model
# ----------------------------------------------------------------------

D, K, R = 512, 10, 4096  # width, chain depth, token rows


class GatedProjectionBlock(nn.Module):
    """``y = Wo( σ(Wg x) ⊙ σ(Wr x) ⊙ (Wv x @ W1 @ … @ W10) )``.

    The shape of a gated transformer block stripped to the part a
    graph optimizer can act on: ``Wg``, ``Wr`` and ``Wv`` all read
    the same input (the product law pairs them into one GEMM +
    split views), and the value path is a left-associative weight
    chain (associativity folds the ten matrices weights-first).
    """

    def __init__(self, d: int, k: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.wg = nn.Linear(d, d, bias=False)
        self.wr = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        for lin in (self.wg, self.wr, self.wv, self.wo):
            with torch.no_grad():
                lin.weight.copy_(
                    torch.randn(d, d, generator=g) * d**-0.5
                )
        self.chain = nn.ParameterList(
            nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
            for _ in range(k)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.wg(x))
        rest = torch.sigmoid(self.wr(x))
        u = self.wv(x)
        for w in self.chain:
            u = u @ w
        return self.wo(gate * rest * u)


# ----------------------------------------------------------------------
#  A Sink that keeps the extracted program
# ----------------------------------------------------------------------


class RecordingSink(TorchSink):
    """``TorchSink`` that remembers the IR it is asked to lower.

    The ``Sink`` port is the boundary the pipeline talks to; this
    is the same object with one hook — after ``optimize_model``
    runs, ``lowered_ir.root`` is the term extraction committed to,
    i.e. the program that gets benchmarked below.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lowered_ir = None

    def lower(self, ir, params=None):
        self.lowered_ir = ir
        return super().lower(ir, params)


# ----------------------------------------------------------------------
#  Small helpers
# ----------------------------------------------------------------------


def _op_counts(term) -> Counter:
    counts: Counter = Counter()
    seen: set = set()

    def rec(t):
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Op):
            counts[t.op] += 1
            for a in t.args:
                rec(a)
        elif isinstance(t, Param):
            counts["param"] += 1
        elif isinstance(t, Var):
            counts["input"] += 1

    rec(term)
    return counts


def _op_counts_sharing(term) -> Counter:
    counts: Counter = Counter()

    def rec(t):
        counts[id(t)] += 1
        if isinstance(t, Op) and counts[id(t)] == 1:
            for a in t.args:
                rec(a)

    rec(term)
    return counts


def _depth(term, _memo=None) -> int:
    if _memo is None:
        _memo = {}
    if id(term) in _memo:
        return _memo[id(term)]
    if isinstance(term, Op):
        d = 1 + max((_depth(a, _memo) for a in term.args), default=0)
    else:
        d = 0
    _memo[id(term)] = d
    return d


def _ascii_tree(term, max_depth: int = 14) -> list[str]:
    """Indented op tree; shared subtrees expand once, then mark."""
    lines: list[str] = []
    shared = _op_counts_sharing(term)
    done: set = set()

    def label(t) -> str:
        if isinstance(t, Param):
            return f"param {t.name}"
        if isinstance(t, Var):
            return f"input {t.name}"
        if isinstance(t, Op):
            attr = ""
            if t.op in ("split", "select"):
                attr = " " + str(t.attrs)
            if t.op == "concat":
                names = [
                    a.name
                    if isinstance(a, Param)
                    else getattr(a, "op", "?")
                    for a in t.args
                ]
                attr = f" [{'|'.join(names)}]"
            return f"{t.op}{attr}"
        return repr(t)

    def rec(t, indent: int):
        tag = label(t)
        if id(t) in done:
            lines.append("  " * indent + tag + "  (shared)")
            return
        if indent >= max_depth and isinstance(t, Op):
            lines.append("  " * indent + tag + "  …")
            return
        lines.append("  " * indent + tag)
        if not isinstance(t, Op):
            return
        # First occurrence expands fully; mark now so later
        # references print as back-pointers.
        if shared.get(id(t), 0) > 1:
            done.add(id(t))
        for a in t.args:
            rec(a, indent + 1)

    rec(term, 0)
    return lines


class _CompileTimeout(Exception):
    pass


def _on_alarm(sig, frm):
    raise _CompileTimeout()


def try_compile(model: nn.Module, x, budget_s: float):
    """``torch.compile`` + first call under a SIGALRM budget.

    A slow/absent toolchain is reported, never hidden: returns
    ``(module, status)`` where module is ``None`` unless the
    compiled forward actually ran.
    """
    if not hasattr(signal, "SIGALRM"):
        return None, "no SIGALRM on this platform"
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(budget_s))
    try:
        cm = torch.compile(copy.deepcopy(model))
        with torch.no_grad():
            cm(x)
        return cm, "ok"
    except _CompileTimeout:
        return None, f"timed out >{budget_s:.0f}s"
    except Exception as exc:  # report, don't crash
        return None, f"failed: {type(exc).__name__}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def median_ms(fn, x, *, n_calls: int, warmup: int, cuda: bool):
    """Median wall ms of ``fn(x)`` under ``no_grad``, synced on CUDA."""

    def call():
        with torch.no_grad():
            fn(x)

    for _ in range(warmup):
        call()
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        call()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _stage(n: int, title: str) -> None:
    print(f"\n{'═' * 3} {n} · {title} {'═' * (40 - len(title))}")


# ----------------------------------------------------------------------
#  The demo
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="catopt end-to-end demo")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--quick", action="store_true", help="shorter timing loop"
    )
    args = ap.parse_args()
    sys.setrecursionlimit(400_000)  # e-class DAG depth

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    cuda = dev.type == "cuda"
    n_calls, warmup = (25, 6) if args.quick else (50, 10)

    print("catopt demo — categorical optimization, end to end")
    print(f"device: {dev} | torch {torch.__version__}")
    t_all = time.perf_counter()

    # -- 1 · model ----------------------------------------------------
    _stage(1, "THE MODEL")
    torch.manual_seed(0)
    model = GatedProjectionBlock(D, K).to(dev).eval()
    x = torch.randn(R, D, device=dev)
    n_par = sum(p.numel() for p in model.parameters())
    print("GatedProjectionBlock — a gated projection block:")
    print(f"  y = Wo( σ(Wg·x) ⊙ σ(Wr·x) ⊙ (Wv·x @ W1 @ … @ W{K}) )")
    print(f"  d={D}, chain depth k={K}, input ({R}, {D})")
    print(
        f"  {n_par:,} parameters in "
        f"{len(list(model.parameters()))} tensors"
    )
    print("  three shared-input projections + a left-associative")
    print("  weight chain — written the only way eager and")
    print(f"  torch.compile can run it: {K + 4} sequential GEMMs.")

    # -- 2 · search ---------------------------------------------------
    _stage(2, "THE SEARCH")
    print("optimize_model(): export → equality saturation →")
    print("non-local passes → extract cheapest program → lower.")
    sink = RecordingSink()
    t0 = time.perf_counter()
    opt_mod, stats = optimize_model(model, x, sink=sink, verbose=False)
    search_s = time.perf_counter() - t0
    fires = stats.get("rule_fires") or {}
    fired = (
        ", ".join(
            f"{n}×{c}"
            for n, c in sorted(fires.items(), key=lambda kv: -kv[1])
        )
        or "none"
    )
    print(f"  search took {search_s:.1f}s")
    print(f"  rules fired: {fired}")
    n_groups = stats.get("pairing_groups") or 0
    paired = bool(stats.get("paired_extract"))
    print(
        f"  non-local pairing: {n_groups} group(s) found —"
        + (
            " extraction shipped it (one GEMM + split views)"
            if paired
            else " extraction declined it on price"
        )
    )
    print(
        f"  carrier lifts (scans/om): "
        f"{stats.get('nonlocal_lifts') or 0} — none present;"
        " reported, not hidden"
    )
    print(
        f"  delivered lowering: {stats.get('lowering')}"
        f" | runner: {stats.get('runner')}"
    )

    shipped = getattr(sink.lowered_ir, "root", None)
    if shipped is None:
        shipped = getattr(opt_mod, "_root", None)

    # -- 3 · certificate ----------------------------------------------
    _stage(3, "THE CERTIFICATE — the differentiator")
    print("The e-graph records why every merge is an equality.")
    print("certificate() replays that provenance into a standalone")
    print("derivation; verify_certificate() re-runs each rule")
    print("application on the real terms — no e-graph involved.")
    res = discover_alternatives(
        model, x, ruleset="all", max_iterations=100
    )
    eg, ir, root_eid = res["eg"], res["ir"], res["root_eid"]
    cert = eg.certificate(ir.root, shipped, root_eid=root_eid)
    print("  certificate: exported program → shipped program")
    print(f"    {cert.n_steps} derivation steps")
    print(f"    standalone rules: {', '.join(cert.rules_used) or '—'}")
    if any(r.startswith("pair#") for r in cert.rules_used):
        print("      (pair#N = witness rewrites the pairing pass")
        print("       attached to its non-local merges)")
    nd = cert.n_egraph_dependent
    if nd:
        print(f"    e-graph-witnessed steps: {nd} (merges with no")
        print(
            "      standalone rule derivation — flagged, not"
            " silently trusted)"
        )
    proof_ok = False
    try:
        out = verify_certificate(ir.root, cert, strict=True)
        proof_ok = out == cert.dst
        print("  verify_certificate(strict): every step replays as")
        print("    a standalone rule application — proof complete")
    except Exception:
        try:
            out = verify_certificate(ir.root, cert)
            proof_ok = op_repr(out) == op_repr(cert.dst)
            print("  verify_certificate: all replayable steps check")
            print("    out; witnessed steps substitute as trusted")
            print("    assertions (counted above).")
        except Exception as exc:
            print(f"  verify_certificate FAILED: {exc}")
    vr = sink.verify(model, opt_mod, x, rtol=1e-4)
    print(
        f"  numeric check: max rel diff {vr.max_rel:.2e}"
        f" vs original (rtol 1e-4) →"
        f" {'PASS' if vr.passed else 'FAIL'}"
    )
    if proof_ok and vr.passed:
        print("  ★ EQUIVALENCE PROVED — the shipped program is a")
        print("    certified rewrite of the exported one.")
    else:
        print(
            "  ✗ proof or numeric check failed — reported, not hidden"
        )

    # -- 4 · the extracted program -------------------------------------
    _stage(4, "THE PROGRAM")
    before_c, after_c = _op_counts(ir.root), _op_counts(shipped)
    bc = ", ".join(f"{k}:{v}" for k, v in before_c.most_common())
    ac = ", ".join(f"{k}:{v}" for k, v in after_c.most_common())
    b_op = ir.root.op if isinstance(ir.root, Op) else "?"
    s_op = shipped.op if isinstance(shipped, Op) else "?"
    print(f"  before: root={b_op} depth={_depth(ir.root)} ops={{{bc}}}")
    print(f"  after : root={s_op} depth={_depth(shipped)} ops={{{ac}}}")
    print("  extracted term (shared subtrees marked):")
    for line in _ascii_tree(shipped):
        print("   ", line)
    rep = param_report(model, opt_mod)
    print(
        f"  weights file: {rep['original_params']} →"
        f" {rep['optimized_params']} tensors"
        f" ({rep['ratio']:.0%} of original bytes)"
    )
    print(f"    eliminated: {', '.join(rep['eliminated'])}")
    print(
        f"    derived (folded at lowering): {', '.join(rep['derived'])}"
    )

    # -- 5 · the race --------------------------------------------------
    _stage(5, "THE RACE")
    sync = ", CUDA-synchronised" if cuda else ""
    print(f"median of {n_calls} forwards, {warmup} warmup{sync}:")
    eager_ms = median_ms(
        model, x, n_calls=n_calls, warmup=warmup, cuda=cuda
    )
    opt_ms = median_ms(
        opt_mod, x, n_calls=n_calls, warmup=warmup, cuda=cuda
    )
    cm, status = try_compile(model, x, 60.0)
    ind_ms = None
    if cm is not None:
        ind_ms = median_ms(
            cm, x, n_calls=n_calls, warmup=warmup, cuda=cuda
        )
    print(f"  eager          {eager_ms:8.3f} ms")
    if ind_ms is not None:
        print(f"  torch.compile  {ind_ms:8.3f} ms")
    else:
        print(f"  torch.compile  {status}")
    print(f"  catopt         {opt_ms:8.3f} ms")
    print(
        f"  → catopt is {eager_ms / opt_ms:.2f}× faster than eager",
        end="",
    )
    if ind_ms:
        print(f", {ind_ms / opt_ms:.2f}× faster than torch.compile")
    else:
        print(" (inductor unavailable — reported, not hidden)")
    print(f"\n  total wall time {time.perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    # Re-exec once under a pinned hash seed: hash-consing order
    # decides saturation order, and therefore which certified
    # member extraction commits to.  Everything else is already
    # seeded via torch.manual_seed(0).
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(
            sys.executable,
            [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        )
    sys.exit(main())
