#!/usr/bin/env python
# ruff: noqa: E402, RUF002
"""bench_omd2 — the omd executor on a *realistic* attention stack.

Follow-up to /tmp/bench_omd.py (toy _ScanAttn).  The question: does the
cross-carrier omd lift survive contact with a transformer-shaped
attention — real q/k/v projections, multi-head, causal mask — and is
BatchedOmdModule (catopt/omd_lower.py) still fast when it does?

VARIANTS
    mqa   — ScanAttnMQA: the firing case.  Multi-QUERY attention
            (nh=4 query heads sharing ONE k/v head — the MQA pattern
            real LLMs use).  The value side is a genuine projection
            v = wv(stack h_i) of a diagonal gated scan, scores get a
            1/sqrt(hd) scale and a materialised causal mask.  The
            exported graph is reshape+transpose on q (opaque to the
            carrier — scores need no affine structure), single-head
            (T,dv) values, so the promoted dense map stays rank-3 and
            ``omd_applym`` lands AT THE ROOT.  fp64-exact.

    mha   — ScanAttnMH: natural multi-head (view+transpose head split,
            output projection).  omd does NOT appear, anywhere.
            Structural report only.

    mha-chunk — ScanAttnChunkMH: same math with heads packed via
            chunk + per-head attention loops (the formulation the
            carrier grammar can see).  Per-head ``omd_applym`` members
            DO appear — nested, one per head — but the root is
            ``wo(cat heads)``: no single omd at the root.

    sdpa  — fused F.scaled_dot_product_attention: exports as one
            ``sdpa`` enode; the om carrier never even lifts
            (the sdpa laws only fire on concat'd k/v operands).

For each firing size: build the e-graph (bounded saturation — same
approach as the toy bench, plus a bounded XC tier, which the projected
value needs: ``linear(applyd ...)`` only becomes an ``apply`` member
through XC_LINEAR_APPLYD), extract eager / omd / best members, and
time lower-quartile ms for

    torch-eager   the raw nn.Module (reference)
    eager-IR      ir.root through the generic IRModule evaluator
    best          flops_cost extraction, generic evaluator
    omd           the forced omd_applym member, generic evaluator
    omd-batched   the same term through BatchedOmdModule
    omd-graph     omd-batched after capture_cuda_graph (CUDA only)
    inductor      torch.compile of the source module (timeout-guarded)
    omd-direct    hand-rolled floor: cumprod/cumsum prefix maps + the
                  omd_elem/omd_applym torch bindings (~7 kernels;
                  numerically fragile, NOT a certified form)

CAVEATS carried over from the toy bench: the exported scan unrolls to
O(T) step nodes and the post-saturation e-graph grows combinatorially,
so builds are bounded (core sat 4 iters, XC 4 iters, 300k-node cap —
the cap *stops* saturation at T>=128; the non-local lifts still run
afterwards and still produce the omd member).  ``lift_scan_to_trace``
is OFF by default here (build_egraph runs it): trace members are the
JSV carrier family — they never feed the om/omd path — and the
minted block-matrix terms are what blows the graph to ~3M enodes at
T=256.  When enabled it is also crash-guarded: scan_lower.
build_scan_plan raises AttributeError on aff_diag leaves whose b-part
is a bare Param (``term.op`` without an isinstance check — existing
bug, out of scope).
"""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.setrecursionlimit(400_000)

import torch

from catopt.cost import _shape_of, flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op
from catopt.omd_lower import to_batched_omd_module
from catopt.regime import default_rules
from catopt.torch_bridge import (
    _IR_TO_TORCH,
    export_to_ir,
    ir_to_torch_module,
)
from catopt.trace_lift import lift_scan_to_trace
from catopt.xcarrier import (
    XC_LAWS,
    _elem_affine_options,
    gather_apply_stack,
    gather_applyd_stack,
    omd_tree_lift,
)

OMD_ROOT_OPS = ("omd_apply", "omd_applym")


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------


class _CausalScan(torch.nn.Module):
    """Shared machinery: diagonal gated scan + causal score mask."""

    def __init__(self, T, D):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(T, D) * 0.1)
        self.h0 = torch.nn.Parameter(torch.randn(D) * 0.1)
        mask = torch.zeros(T, T)
        mask.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
            float("-inf"),
        )
        self.register_buffer("cm", mask)

    def scan(self, x):
        h = self.h0
        outs = []
        for t in range(x.shape[0]):
            h = self.a[t] * h + x[t]
            outs.append(h)
        return torch.stack(outs)


class ScanAttnMQA(_CausalScan):
    """Multi-query attention over scanned values — THE FIRING CASE.

    h_t = a_t⊙h + x_t ; v = wv(stack h) (T,dv) shared across nh query
    heads;  s = q @ kᵀ /√hd + causal-mask ;  out = softmax(s) @ v
    returns the (nh,T,dv) pre-projection output.  Single-head V keeps
    the promoted dense map rank-3 → omd_applym at the root.
    """

    def __init__(self, T, D, nh=4, hd=16, dv=24):
        super().__init__(T, D)
        self.nh, self.hd, self.dv = nh, hd, dv
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, hd, bias=False)
        self.wv = torch.nn.Linear(D, dv, bias=False)
        # NOTE: the scale is a registered BUFFER, not a Python float —
        # torch.export serializes a bare float attribute through a
        # fp32 rounding (measured: 32**-0.5 exports as
        # 0.1767766952966369 vs the double 0.17677669529663687 —
        # a uniform ~1e-8 output deviation, an export artifact, not a
        # catopt issue; invisible at hd=16 where scale=0.25 is
        # fp32-exact).  A buffer exports as a double Param — exact.
        self.register_buffer(
            "sq", torch.tensor(hd**-0.5, dtype=torch.float64)
        )

    def forward(self, x):
        T = x.shape[0]
        v = self.wv(self.scan(x))  # (T, dv)
        q = self.wq(x).view(T, self.nh, self.hd).transpose(0, 1)
        k = self.wk(x)  # (T, hd)
        s = q @ k.transpose(-1, -2) * self.sq + self.cm
        return torch.softmax(s, dim=-1) @ v  # (nh,T,dv)


class ScanAttnMH(_CausalScan):
    """Natural multi-head attention over scanned values + output proj.

    The head split is view+transpose (what torch code actually writes).
    The om leaf's value class then contains only reshape/transpose
    enodes — the affine member sits one view-op *below* it and no
    XC law commutes reshape/transpose through apply — so omd never
    fires, anywhere.
    """

    def __init__(self, T, D, nh=4, hd=16):
        super().__init__(T, D)
        self.nh, self.hd = nh, hd
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, nh * hd, bias=False)
        self.wv = torch.nn.Linear(D, nh * hd, bias=False)
        self.wo = torch.nn.Linear(nh * hd, D, bias=False)
        self.scale = hd**-0.5

    def forward(self, x):
        T = x.shape[0]
        v = (
            self.wv(self.scan(x))
            .view(T, self.nh, self.hd)
            .transpose(0, 1)
        )
        q = self.wq(x).view(T, self.nh, self.hd).transpose(0, 1)
        k = self.wk(x).view(T, self.nh, self.hd).transpose(0, 1)
        s = q @ k.transpose(-1, -2) * self.scale + self.cm
        o = torch.softmax(s, dim=-1) @ v
        return self.wo(o.transpose(0, 1).reshape(T, self.nh * self.hd))


class ScanAttnChunkMH(_CausalScan):
    """Same MHA math, heads packed as chunk slices + per-head loop —
    the formulation the carrier grammar can see.  XC_CHUNK_APPLY
    pushes chunk through the dense apply, so each head's value class
    DOES gain an apply member (rank-3 map (K,hd,D)) and per-head
    omd_applym members appear — but nested under cat+wo, never at
    root.
    """

    def __init__(self, T, D, nh=4, hd=16):
        super().__init__(T, D)
        self.nh, self.hd = nh, hd
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, nh * hd, bias=False)
        self.wv = torch.nn.Linear(D, nh * hd, bias=False)
        self.wo = torch.nn.Linear(nh * hd, D, bias=False)
        self.scale = hd**-0.5

    def forward(self, x):
        vf = self.wv(self.scan(x)).chunk(self.nh, dim=-1)
        qf = self.wq(x).chunk(self.nh, dim=-1)
        kf = self.wk(x).chunk(self.nh, dim=-1)
        outs = []
        for i in range(self.nh):
            s = qf[i] @ kf[i].transpose(-1, -2) * self.scale + self.cm
            outs.append(torch.softmax(s, dim=-1) @ vf[i])
        return self.wo(torch.cat(outs, dim=-1))


class ScanAttnSDPA(_CausalScan):
    """Fused kernel form — exports as a single ``sdpa`` enode."""

    def __init__(self, T, D, nh=4, hd=16):
        super().__init__(T, D)
        self.nh, self.hd = nh, hd
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, nh * hd, bias=False)
        self.wv = torch.nn.Linear(D, nh * hd, bias=False)

    def forward(self, x):
        T = x.shape[0]
        v = (
            self.wv(self.scan(x))
            .view(T, self.nh, self.hd)
            .transpose(0, 1)
        )
        q = self.wq(x).view(T, self.nh, self.hd).transpose(0, 1)
        k = self.wk(x).view(T, self.nh, self.hd).transpose(0, 1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )


# ---------------------------------------------------------------------------
#  Helpers (same conventions as /tmp/bench_omd.py)
# ---------------------------------------------------------------------------


def census(t, c=None):
    if c is None:
        c = {}
    if isinstance(t, Op):
        c[t.op] = c.get(t.op, 0) + 1
        for a in t.args:
            census(a, c)
    return c


def distinct_ops(t):
    seen = set()

    def rec(u):
        if isinstance(u, Op) and id(u) not in seen:
            seen.add(id(u))
            for a in u.args:
                rec(a)

    rec(t)
    return len(seen)


def bounded_build_xc(
    m,
    x,
    max_iter=4,
    xc_iter=4,
    max_nodes=300_000,
    xc_rounds=1,
    trace=False,
):
    """build_egraph minus the explosive re-saturations.

    core sat → non-local lifts → rebuild → [bounded XC tier → gather/
    omd lifts → rebuild → short core pass] × xc_rounds.  The XC tier
    is required here (unlike the toy): the projected value
    ``linear(applyd ...)`` only gains its dense ``apply`` member
    through XC_LINEAR_APPLYD.  One XC round suffices for every omd
    member observed in this benchmark (round 2 re-runs the gather
    passes on the enlarged graph — 100+ extra offers, minutes of
    build time, no new omd members).

    ``trace`` controls ``lift_scan_to_trace``.  It defaults OFF here
    (build_egraph has it on): trace members live in the JSV carrier
    family — they never feed the om/omd path — and the minted
    nilpotent block-matrix terms are what blows the e-graph to ~3M
    enodes at T=256 (measured: 512 offers, enodes 2.9M, full-graph
    extract_best OOM-killed).  The omd member is identical either
    way — it needs only applyd members (SCAN_DIAG_LAWS), the gather
    passes, the XC promotion, and omd_tree_lift.

    Returns (eg, root, ir, src, stats).
    """
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    t0 = time.perf_counter()
    stats = eg.run(
        default_rules(),
        root,
        max_iterations=max_iter,
        max_nodes=max_nodes,
    )
    stats["sat_s"] = time.perf_counter() - t0

    def _lifts(with_trace=True):
        crashed = False
        tl = []
        if with_trace and trace:
            try:
                tl = lift_scan_to_trace(eg)
            except AttributeError:
                # scan_lower.build_scan_plan: 'Param' object has no
                # .op — existing robustness bug on aff_diag leaves
                # with a bare Param b-part; the omd path does not
                # depend on trace members.
                crashed = True
        return (
            tl
            + gather_applyd_stack(eg)
            + gather_apply_stack(eg)
            + omd_tree_lift(eg),
            crashed,
        )

    t0 = time.perf_counter()
    lifts, crash = _lifts()
    if lifts:
        eg.rebuild()
    stats["nonlocal_lifts"] = len(lifts)
    stats["trace_lift_crashed"] = crash
    stats["lifts_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(xc_rounds):
        before = eg.n_enodes
        eg.run(
            XC_LAWS, root, max_iterations=xc_iter, max_nodes=max_nodes
        )
        grew = eg.n_enodes != before
        more, crash2 = _lifts(with_trace=False)
        stats["trace_lift_crashed"] |= crash2
        if more:
            eg.rebuild()
            stats["nonlocal_lifts"] += len(more)
        if not grew and not more:
            break
        # NOTE: deliberately no trailing core re-saturation (unlike
        # build_egraph): one costs ~6min at T=64 (full CARRIER_LAWS
        # e-match over the enlarged graph) and is unnecessary for the
        # omd member, which lands in the root class via
        # omd_tree_lift's union directly.
    stats["xc_s"] = time.perf_counter() - t0
    return eg, root, ir, src, stats


def extract_omd_term(eg, root):
    """Locate an omd_apply/omd_applym enode in the root class; build
    its term via forced extraction (min-flops children)."""
    rc = eg.find(root)
    nodes = [n for n in eg.get_class(rc).nodes if n.op in OMD_ROOT_OPS]
    if not nodes:
        return None, 0
    node = sorted(nodes, key=repr)[0]
    term = eg.extract_best(rc, flops_cost, overrides={rc: node})
    if term is None:
        args = [eg.any_term(eg.find(c)) for c in node.children]
        if any(a is None for a in args):
            return None, len(nodes)
        term = Op.make(node.op, *args, **dict(node.attrs))
    return term, len(nodes)


def carrier_analysis(eg, root):
    """Structural gap report: which carrier members exist where."""
    rc = eg.find(root)
    info = {
        "root_ops": sorted({n.op for n in eg.get_class(rc).nodes}),
        "om_apply_any": 0,
        "omd_any": 0,
        "omd_at_root": sum(
            1 for n in eg.get_class(rc).nodes if n.op in OMD_ROOT_OPS
        ),
        "elems": [],
    }
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for n in ec.nodes:
            if n.op == "om_apply":
                info["om_apply_any"] += 1
            elif n.op in OMD_ROOT_OPS:
                info["omd_any"] += 1
            if n.op == "om_elem" and len(n.children) == 2:
                vcid = eg.find(n.children[1])
                vops = sorted(
                    {nn.op for nn in eg.get_class(vcid).nodes}
                )
                aff = _elem_affine_options(eg, n.children[1])
                mapshape = None
                for nn in eg.get_class(vcid).nodes:
                    if nn.op in ("apply", "applyd"):
                        t = eg.any_term(eg.find(nn.children[0]))
                        mapshape = _shape_of(t)
                        break
                info["elems"].append(
                    {
                        "class": c,
                        "v_ops": vops,
                        "aff_h_eids": sorted(aff),
                        "map_shape": mapshape,
                    }
                )
    return info


def bench(fn, x, warmup=5, reps=30, budget_s=15.0):
    dev = x.device
    with torch.no_grad():
        for _ in range(min(warmup, 3)):
            fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        est = time.perf_counter() - t0
        for _ in range(max(0, warmup - 4)):
            fn(x)
        n = int(budget_s / max(est, 1e-6))
        n = max(3, min(reps, n))
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 4], n


class _CompileTimeout(Exception):
    pass


def _on_alarm(sig, frm):
    raise _CompileTimeout()


def try_inductor(m, x, budget_s=240):
    """torch.compile the SOURCE module (the realistic comparison),
    timeout-guarded.  Returns (callable|None, status)."""
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    t0 = time.perf_counter()
    try:
        cm = torch.compile(m)
        with torch.no_grad():
            cm(x)
            cm(x)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        return cm, f"compiled module in {time.perf_counter() - t0:.0f}s"
    except _CompileTimeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile TIMEOUT >{budget_s:.0f}s"
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile failed: {type(e).__name__}: {e}"


def make_omd_direct_mqa(m):
    """Hand-rolled omd floor for the MQA (dense-fiber) form.

    cumprod/cumsum prefix maps (O(T) kernels) → per-key dense
    coefficients A = P⊙Wv / b = B@Wvᵀ → the omd_elem/omd_applym
    bindings.  ~7 kernels total; the x/P division can overflow, so
    this is a timing estimate, not a certified form.
    """

    def fn(x):
        a, h0 = m.a, m.h0
        T = x.shape[0]
        P = torch.cumprod(a, dim=0)  # (T,D) prefixes
        B = P * torch.cumsum(x / P, dim=0)
        Wv = m.wv.weight  # (dv,D)
        A = P.unsqueeze(-2) * Wv  # (T,dv,D)
        b = B @ Wv.T  # (T,dv)
        q = m.wq(x).view(T, m.nh, m.hd).transpose(0, 1)
        k = m.wk(x)
        s = q @ k.transpose(-1, -2) * m.sq + m.cm
        f = _IR_TO_TORCH["omd_elem"](s, A, b)
        return _IR_TO_TORCH["omd_applym"](f, h0)

    return fn


def fmt_err(e):
    if e is None:
        return "—"
    if e != e:
        return "NaN"
    return f"{e:.1e}"


# ---------------------------------------------------------------------------
#  Structural gap analysis (non-firing variants)
# ---------------------------------------------------------------------------


def gap_report(m, x, tag, note=""):
    print(f"\n----- structural: {tag} -----", flush=True)
    if note:
        print(f"  ({note})", flush=True)
    t0 = time.perf_counter()
    eg, root, ir, src, stats = bounded_build_xc(m, x)
    print(
        f"  build {time.perf_counter() - t0:.1f}s enodes={eg.n_enodes} "
        f"lifts={stats.get('nonlocal_lifts')} "
        f"trace_crash={stats.get('trace_lift_crashed')}",
        flush=True,
    )
    info = carrier_analysis(eg, root)
    print(f"  root-class ops: {info['root_ops']}", flush=True)
    print(
        f"  om_apply enodes anywhere: {info['om_apply_any']}   "
        f"omd enodes anywhere: {info['omd_any']}   "
        f"omd at root: {info['omd_at_root']}",
        flush=True,
    )
    for e in info["elems"][:8]:
        print(
            f"    om_elem@{e['class']}: value-class ops={e['v_ops']} "
            f"affine-opts(h_eids)={e['aff_h_eids']} "
            f"map={e['map_shape']}",
            flush=True,
        )
    if len(info["elems"]) > 8:
        print(
            f"    … {len(info['elems']) - 8} more om_elem leaves",
            flush=True,
        )
    return eg, root, ir, src, info


# ---------------------------------------------------------------------------
#  Main benchmark — the firing variant
# ---------------------------------------------------------------------------


def run_mqa(args, devices):
    sizes = [int(s) for s in args.sizes.split(",")]
    all_rows = {}
    meta = {}
    for T in sizes:
        D, nh, hd, dv = 64, args.heads, args.hd, args.dv
        print(
            f"\n===== ScanAttnMQA T={T} D={D} nh={nh} hd={hd} "
            f"dv={dv} (fp64) =====",
            flush=True,
        )
        torch.manual_seed(0)
        m = ScanAttnMQA(T, D, nh, hd, dv).eval().double()
        x = torch.randn(T, D, dtype=torch.float64)

        t0 = time.perf_counter()
        eg, root, ir, src, stats = bounded_build_xc(
            m, x, max_nodes=args.max_nodes, trace=args.trace
        )
        build_s = time.perf_counter() - t0
        print(
            f"  build: {build_s:.1f}s enodes={eg.n_enodes} "
            f"lifts={stats.get('nonlocal_lifts')} "
            f"trace_crash={stats.get('trace_lift_crashed')}",
            flush=True,
        )

        t0 = time.perf_counter()
        omd_term, n_omd = extract_omd_term(eg, root)
        print(
            f"  omd enodes at root: {n_omd} "
            f"(extract {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        # Node-cap truncation makes the lift's arrival ORDER-DEPENDENT
        # at T>=128 (hash-seed dependent frontier — observed omd@root
        # flip between runs at the same cap).  Retry with a larger
        # saturation budget on the SAME graph before declaring miss.
        retries = 0
        while omd_term is None and retries < 2:
            retries += 1
            bigger = args.max_nodes * (2**retries)
            print(
                f"  retry {retries}: saturating further "
                f"(max_nodes={bigger})",
                flush=True,
            )
            eg.run(
                default_rules(),
                root,
                max_iterations=2,
                max_nodes=bigger,
            )
            eg.run(XC_LAWS, root, max_iterations=2, max_nodes=bigger)
            more = (
                gather_applyd_stack(eg)
                + gather_apply_stack(eg)
                + omd_tree_lift(eg)
            )
            if more:
                eg.rebuild()
            omd_term, n_omd = extract_omd_term(eg, root)
            print(f"    -> omd enodes at root: {n_omd}", flush=True)
        if omd_term is None:
            print(
                "  !! no omd member after retries — skipping sizes row",
                flush=True,
            )
            meta[T] = {
                "build_s": build_s,
                "enodes": eg.n_enodes,
                "n_omd": 0,
            }
            continue

        t0 = time.perf_counter()
        if eg.n_enodes <= args.extract_cap:
            best_term = eg.extract_best(eg.find(root), flops_cost)
            broot = best_term.op if isinstance(best_term, Op) else "?"
            print(
                f"  best-flops extract: {time.perf_counter() - t0:.1f}s"
                f" root={broot}",
                flush=True,
            )
        else:
            best_term = None
            print(
                f"  best-flops extract: SKIPPED "
                f"(enodes={eg.n_enodes} > {args.extract_cap} — "
                f"extract_best on the full graph was OOM-killed "
                f"at T=256/300k)",
                flush=True,
            )

        terms = {"eager": ir.root, "omd": omd_term, "best": best_term}
        meta[T] = {
            "build_s": build_s,
            "enodes": eg.n_enodes,
            "n_omd": n_omd,
            "ops": {k: distinct_ops(t) for k, t in terms.items() if t},
            "flops": {k: flops_cost(t) for k, t in terms.items() if t},
            "census_omd": census(omd_term),
            "stats": stats,
        }
        print(f"  distinct ops: {meta[T]['ops']}", flush=True)
        print(f"  flops_cost: {meta[T]['flops']}", flush=True)
        print(
            f"  omd census (tree): {meta[T]['census_omd']}", flush=True
        )

        for dev in devices:
            print(f"  -- device={dev} --", flush=True)
            xd = x.to(dev)
            md = m.to(dev)
            with torch.no_grad():
                ref = md(xd)
            rows = {}
            tms, n = bench(md, xd)
            rows["torch-eager (ref)"] = (tms, 0.0, n, None)

            mods = {}
            for name, term in terms.items():
                if term is None:
                    continue
                try:
                    mods[name] = ir_to_torch_module(
                        IR(
                            root=term,
                            inputs=ir.inputs,
                            params=ir.params,
                        ),
                        src,
                    ).to(dev)
                except Exception as e:
                    print(f"    {name}: lower FAILED {e}", flush=True)
            # the batched executor on the SAME omd term
            try:
                bomd = to_batched_omd_module(
                    IR(
                        root=omd_term,
                        inputs=ir.inputs,
                        params=ir.params,
                    ),
                    src,
                ).to(dev)
                print(
                    f"    omd-batched: is_batched={bomd.is_batched} "
                    f"map_mode={bomd.map_mode}",
                    flush=True,
                )
                meta[T]["map_mode"] = bomd.map_mode
                meta[T]["is_batched"] = bomd.is_batched
            except Exception as e:
                bomd = None
                print(f"    omd-batched: lower FAILED {e}", flush=True)

            for name, mod in mods.items():
                try:
                    with torch.no_grad():
                        err = (mod(xd) - ref).abs().max().item()
                    tms, n = bench(mod, xd)
                    rows[f"{name}-IR (generic)"] = (
                        tms,
                        err,
                        n,
                        meta[T]["ops"].get(name),
                    )
                    print(
                        f"    {name}: {tms:.3f}ms err={fmt_err(err)} "
                        f"reps={n}",
                        flush=True,
                    )
                except Exception as e:
                    rows[f"{name}-IR (generic)"] = (None, None, 0, None)
                    print(
                        f"    {name}: FAILED {type(e).__name__}: {e}",
                        flush=True,
                    )

            if bomd is not None:
                try:
                    with torch.no_grad():
                        err = (bomd(xd) - ref).abs().max().item()
                    tms, n = bench(bomd, xd)
                    rows["omd-batched"] = (
                        tms,
                        err,
                        n,
                        meta[T]["ops"].get("omd"),
                    )
                    print(
                        f"    omd-batched: {tms:.3f}ms "
                        f"err={fmt_err(err)} reps={n}",
                        flush=True,
                    )
                    if dev == "cuda":
                        try:
                            with torch.no_grad():
                                bomd.capture_cuda_graph(xd)
                                err = (
                                    (bomd(xd) - ref).abs().max().item()
                                )
                            tms, n = bench(bomd, xd)
                            rows["omd-batched+graph"] = (
                                tms,
                                err,
                                n,
                                None,
                            )
                            print(
                                f"    omd-graph: {tms:.3f}ms "
                                f"err={fmt_err(err)} reps={n}",
                                flush=True,
                            )
                            bomd.drop_cuda_graph()
                        except Exception as e:
                            print(
                                f"    omd-graph FAILED: "
                                f"{type(e).__name__}: {e}",
                                flush=True,
                            )
                except Exception as e:
                    print(
                        f"    omd-batched FAILED: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )

            cm, status = try_inductor(
                md, xd, budget_s=args.compile_timeout
            )
            print(f"    inductor: {status}", flush=True)
            if cm is not None:
                try:
                    with torch.no_grad():
                        err = (cm(xd) - ref).abs().max().item()
                    tms, n = bench(cm, xd)
                    rows["inductor (module)"] = (tms, err, n, None)
                    print(
                        f"    inductor: {tms:.3f}ms "
                        f"err={fmt_err(err)} reps={n}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"    inductor bench FAILED: {e}", flush=True)

            try:
                fn = make_omd_direct_mqa(md)
                with torch.no_grad():
                    err = (fn(xd) - ref).abs().max().item()
                tms, n = bench(fn, xd)
                rows["omd-direct (floor est.)"] = (tms, err, n, None)
                print(
                    f"    omd-direct: {tms:.3f}ms err={fmt_err(err)} "
                    f"reps={n}",
                    flush=True,
                )
            except Exception as e:
                print(f"    omd-direct FAILED: {e}", flush=True)

            all_rows[(T, dev)] = rows
    return all_rows, meta


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def print_tables(sizes, devices, all_rows, meta):
    print(
        "\n\n# Results — ScanAttnMQA (multi-query attention over a "
        "diagonal gated scan; causal, scaled, real Wq/Wk/Wv)\n"
    )
    for dev in devices:
        print(
            f"\n## device = {dev}  (fp64, lower-quartile ms, "
            "warmup 5, ~30 reps)\n"
        )
        forms = []
        for (_T, d), rows in all_rows.items():
            if d == dev:
                for k in rows:
                    if k not in forms:
                        forms.append(k)
        print(
            "| T | form | ms | vs eager-IR | vs torch-eager "
            "| equiv err | ops |"
        )
        print("|" + "---|" * 7)
        for T in sizes:
            rows = all_rows.get((T, dev), {})
            base_ir = rows.get("eager-IR (generic)", (None,))[0]
            base_m = rows.get("torch-eager (ref)", (None,))[0]
            for k in forms:
                if k not in rows:
                    continue
                tms, err, _n, ops = rows[k]
                if tms is None:
                    print(f"| {T} | {k} | FAIL | — | — | — | — |")
                    continue
                su_ir = f"{base_ir / tms:.2f}x" if base_ir else "—"
                su_m = f"{base_m / tms:.2f}x" if base_m else "—"
                print(
                    f"| {T} | {k} | {tms:.3f} | {su_ir} | {su_m} "
                    f"| {fmt_err(err)} "
                    f"| {ops if ops is not None else '—'} |"
                )
        print()

    print("\n## Build stats\n")
    print(
        "| T | build s | enodes | omd@root | map_mode | "
        "omd tree census (top) |"
    )
    print("|---|---|---|---|---|---|")
    for T in sizes:
        mt = meta.get(T, {})
        cen = mt.get("census_omd", {})
        cen_s = ", ".join(
            f"{k}:{v}"
            for k, v in sorted(cen.items(), key=lambda kv: -kv[1])[:6]
        )
        print(
            f"| {T} | {mt.get('build_s', 0):.0f} "
            f"| {mt.get('enodes')} | {mt.get('n_omd')} "
            f"| {mt.get('map_mode', '—')} | {cen_s} |"
        )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hd", type=int, default=16)
    ap.add_argument("--dv", type=int, default=24)
    ap.add_argument(
        "--gap-size",
        type=int,
        default=32,
        help="T for the structural (non-firing) variants",
    )
    ap.add_argument("--max-nodes", type=int, default=300_000)
    ap.add_argument(
        "--extract-cap",
        type=int,
        default=1_000_000,
        help="skip best-flops extraction above this many "
        "enodes (full-graph extract_best is OOM-prone)",
    )
    ap.add_argument("--skip-gap", action="store_true")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="also run lift_scan_to_trace (off: the JSV "
        "carrier never feeds omd and its block-matrix "
        "terms blow the graph to ~3M enodes at T=256)",
    )
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--compile-timeout", type=float, default=240.0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    devices = ["cpu"] + (["cuda"] if use_gpu else [])
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- gap analysis first (cheap — small T, structural only) ----
    if not args.skip_gap:
        T = args.gap_size
        torch.manual_seed(0)
        x = torch.randn(T, 64, dtype=torch.float64)
        gap_report(
            ScanAttnMH(T, 64, 4, 16).eval().double(),
            x,
            f"ScanAttnMH T={T} (natural view+transpose heads)",
            note="value reaches the om leaf through "
            "reshape+transpose — no commute law",
        )
        mc = ScanAttnChunkMH(T, 64, 4, 16).eval().double()
        egc, rootc, irc, srcc, infoc = gap_report(
            mc,
            x,
            f"ScanAttnChunkMH T={T} (chunk+stack head packing)",
            note="per-head applies are reachable — nested omd expected",
        )
        # verify the nested-omd term still evaluates exactly
        if infoc["omd_any"]:
            pins = {}
            for cid in list(egc._classes):
                c = egc.find(cid)
                ec = egc._classes.get(c)
                if ec is None:
                    continue
                cands = [n for n in ec.nodes if n.op in OMD_ROOT_OPS]
                if cands:
                    pins[c] = sorted(cands, key=repr)[0]
            t = egc.extract_best(
                egc.find(rootc), flops_cost, overrides=pins
            )
            if t is not None:
                mod = ir_to_torch_module(
                    IR(root=t, inputs=irc.inputs, params=irc.params),
                    srcc,
                )
                with torch.no_grad():
                    ref = mc(x)
                    out = mod(x)
                print(
                    f"  nested-omd extracted term: root={t.op}, "
                    f"max|Δ| vs module = "
                    f"{(out - ref).abs().max().item():.2e}",
                    flush=True,
                )
        gap_report(
            ScanAttnSDPA(T, 64, 4, 16).eval().double(),
            x,
            f"ScanAttnSDPA T={T} (fused kernel)",
            note="sdpa only decomposes on concat'd operands — "
            "om never lifts",
        )

    all_rows, meta = run_mqa(args, devices)
    print_tables(sizes, devices, all_rows, meta)

    print(
        "\nNotes: eager/omd/best all run through the SAME generic "
        "IRModule evaluator (one Python call + torch dispatch per "
        "distinct op); 'ops' is the DAG size.  omd-batched is "
        "BatchedOmdModule on the same extracted term "
        "(is_batched/map_mode printed above).  omd-direct is a "
        "hand-rolled floor estimate — cumprod/cumsum prefix maps + "
        "the raw omd bindings — not a certified form.  At T>=128 "
        "core saturation hits the node cap; lifts still run after "
        "and the omd member still lands at root."
    )


if __name__ == "__main__":
    main()
