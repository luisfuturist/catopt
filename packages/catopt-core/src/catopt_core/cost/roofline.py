"""Roofline cost model, target-profile constants and measured kernels.

Prices each op as ``max(flops / peak, bytes / bandwidth) + launch``
(:func:`_local_roofline`), reads calibration constants from a
``TargetProfile`` (:func:`_profile_constants`) and floors estimates at
a measured per-op kernel table (:func:`_kernel_lookup`).  Also holds
the critical-path (:func:`depth_cost`) and profile-calibrated
(:func:`roofline_cost_for`, :func:`depth_cost_for`) entry points and
the executor-side per-dispatch / per-graph overhead readers.
"""

from __future__ import annotations

import math
from typing import Any, cast

from catopt_core.ir import Op
from catopt_core.typing import (
    _INVALID,
    _infer_op_shape,
    _numel,
    _shape_of,
)

from .basic import _INVALID_COST, _VIEW_OPS, _CostMarkers, _flops_of
from .params import _FUSION_POINTWISE_OPS

# ---------------------------------------------------------------------------
#  Roofline cost model — item: layout/memory-aware costing
# ---------------------------------------------------------------------------
#
# flops_cost can only see arithmetic.  It cannot express why the fused
# SwiGLU form is a wash on CPU: the single wide GEMM saves a launch, but
# its chunk projections hand *strided* views to the elementwise kernels,
# which are memory-bound and pay for the wasted bandwidth.  The roofline
# model prices each op as
#
#     max(flops / PEAK_FLOPS, bytes / PEAK_BW) + launch_time
#
# which captures both regimes: GEMMs are compute-bound (flops term wins),
# elementwise and copy ops are bandwidth-bound (bytes term wins), and
# non-contiguous chunk views multiply the bytes a consumer must move.

# Calibrated on the dev GPU (RTX 2050 mobile, fp32): measured sustained
# matmul throughput ~2.5 TFLOPS, device copy bandwidth ~89 GB/s, eager
# kernel-launch overhead ~8.7 µs.  Raw strided copies measured ~1.0x
# (no penalty), so _STRIDE_PENALTY is set to 1.0 — the *real* cost of a
# strided view is not slower reads but forced materialisation when a
# layout-strict consumer (e.g. SDPA) needs contiguous input, priced in
# _local_roofline as an extra copy kernel.
_PEAK_FLOPS = 2.5e12  # measured: ~2.5 TFLOPS fp32 GEMM (RTX 2050)
_PEAK_BW = 8.9e10  # measured: ~89 GB/s copy bandwidth
_LAUNCH_S = 8.7e-6  # measured: ~8.7 µs eager launch overhead
_STRIDE_PENALTY = 1.0  # measured: strided copies ~1.0x on this GPU


def _is_strided(term: Any, memo: dict | None = None) -> bool:
    """Return True if *term* is a non-contiguous view.

    chunk on the LAST dim splits each row — consumers read with a row
    stride of 2x the logical row.  chunk on any other dim yields
    contiguous blocks.
    """
    if not (isinstance(term, Op) and term.op in ("chunk", "split")):
        return False
    s = _infer_op_shape(term, memo)
    if not isinstance(s, tuple) or not s:
        return False
    dim = term.attrs.get("dim", -1) % len(s)
    return dim == len(s) - 1


def _bytes_of(term: Op, memo: dict | None = None) -> float:
    """Bytes moved by a single op: inputs read + output written (fp32)."""
    in_bytes = 0.0
    for a in term.args:
        n = _numel(_shape_of(a, memo))
        w = _STRIDE_PENALTY if _is_strided(a, memo) else 1.0
        in_bytes += n * 4.0 * w
    # view ops share storage with their input — no output write
    out_bytes = (
        0.0
        if term.op in _VIEW_OPS
        else _numel(_infer_op_shape(term, memo)) * 4.0
    )
    return in_bytes + out_bytes


def _local_roofline(
    term: Op,
    memo: dict | None = None,
    *,
    peak_flops: float = _PEAK_FLOPS,
    peak_bw: float = _PEAK_BW,
    launch_s: float = _LAUNCH_S,
    kernel_ns=None,
) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch.

    ``kernel_ns`` (a ``_kernel_lookup`` callable) floors the estimate
    at the op's measured kernel time when the profile carries an
    ``op_kernel_ns`` table — see the "measured per-op kernel table"
    section above.
    """
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term, memo)
    if (
        flops >= _INVALID_COST
    ):  # pragma: no cover — INVALID shapes checked above
        return _INVALID_COST
    compute_s = flops / peak_flops
    memory_s = _bytes_of(term, memo) / peak_bw
    launch = 0.0 if term.op in _VIEW_OPS else launch_s
    # True views emit no kernel: no launch AND no memory traffic — the
    # read happens at the consumer, priced there via _STRIDE_PENALTY.
    if term.op in _VIEW_OPS:
        return 0.0
    base = (max(compute_s, memory_s) + launch) * 1e9
    if kernel_ns is not None:
        measured = kernel_ns(term, memo)
        if measured is not None:
            base = max(base, measured)
    return base


def _roofline_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    kernel_ns=None,
) -> float:
    """Shared traversal for roofline_cost and roofline_cost_for.

    The memo key carries the constants so two profiles can share a memo
    dict (e.g. inside dag_cost) without colliding.  When a measured
    kernel table is bound, the callable's identity joins the key —
    keeping the table-less ``("rc",pf,bw,ls,term)`` form intact for
    the extraction fast-path's pre-seeded entries (egraph/extract.py).
    """
    ck = (
        ("rc", peak_flops, peak_bw, launch_s, term)
        if kernel_ns is None
        else ("rc", peak_flops, peak_bw, launch_s, id(kernel_ns), term)
    )
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_roofline(
            term,
            memo,
            peak_flops=peak_flops,
            peak_bw=peak_bw,
            launch_s=launch_s,
            kernel_ns=kernel_ns,
        )
        for arg in term.args:
            base += _roofline_cost(
                arg, memo, peak_flops, peak_bw, launch_s, kernel_ns
            )
        memo[ck] = float(base)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0


def roofline_cost(term: Any, memo: dict | None = None) -> float:
    """Roofline cost in estimated nanoseconds (per-op, additive).

    max(flops/PEAK_FLOPS, bytes/PEAK_BW) + launch per op; view ops are
    free except for the strided-read penalty they impose on consumers.
    This is the honest model for questions like "does the fused GEMM
    pay?" — it answers differently at different batch sizes, which is
    what the measurements show.

    The constants are the RTX 2050 profile hardcoded above; use
    :func:`roofline_cost_for` with a measured ``TargetProfile``
    (``catopt_torch.calibrate.calibrate``) for other targets.
    """
    memo = {} if memo is None else memo
    return _roofline_cost(term, memo, _PEAK_FLOPS, _PEAK_BW, _LAUNCH_S)


def _profile_constants(profile: Any) -> tuple[float, float, float]:
    """(peak_flops, peak_bw, launch_s) from a TargetProfile-like object.

    Accepts anything with ``.tflops`` / ``.gbps`` / ``.launch_us``
    attributes (e.g. ``catopt_torch.calibrate.TargetProfile``) or a dict with
    those keys; ``None`` yields the built-in RTX 2050 constants.
    """
    if profile is None:
        return _PEAK_FLOPS, _PEAK_BW, _LAUNCH_S
    if isinstance(profile, dict):
        get = profile.__getitem__
    else:
        get = lambda k: getattr(profile, k)  # noqa: E731
    return (
        float(get("tflops")) * 1e12,
        float(get("gbps")) * 1e9,
        float(get("launch_us")) * 1e-6,
    )


# ---------------------------------------------------------------------------
#  Measured per-op kernel table — the shape-dependent floor
# ---------------------------------------------------------------------------
#
# The roofline prices an op as max(flops/peak, bytes/bw) + launch —
# blind to how a real kernel's time scales with its shape (BLAS
# efficiency curves, cache-resident bandwidth, gather costs).  A
# calibrated profile carries ``op_kernel_ns``:
# ``{op_class: {shape_key: measured_ns}}`` where each value is the
# median wall time of one eager kernel call at that shape — launch,
# dispatch and kernel work inside (``catopt_torch.calibrate``).
# A term whose op-class and shape signature lands near a measured
# bucket prices at ``max(roofline_ns, measured_ns)``: the measurement
# is a floor on the estimate — it can only raise the model toward the
# observed latency, never undercut it (the fidelity sweep's failure
# mode is under-prediction, so the asymmetric correction is the safe
# direction).

#: Reduction-style ops whose measured table class is ``"reduce"``,
#: bucketed by input element count (the traffic the kernel streams).
_MEASURED_REDUCE_OPS = frozenset(
    {"sum", "mean", "max", "min", "prod", "softmax"}
)

#: Gather ops whose measured table class is ``"index_select"``,
#: bucketed by output element count.
_MEASURED_GATHER_OPS = frozenset({"index_select", "embedding"})


def _mm_signature(
    term: Op, out_shape: tuple, memo: dict | None
) -> tuple[float, float, float] | None:
    """(M, K, N) signature for a matmul/linear op, or ``None``.

    ``M`` is the collapsed row count (``n_out / N`` — batch and row
    dims together), ``N`` the output's last dim, ``K`` the reduction
    dim of the weight argument (``w[-2]`` for matmul, ``w[-1]`` =
    in-features for linear).  Anything unshapeable — a rank-1 result
    (matvec/dot), a missing or dimensionless weight — yields ``None``
    and the op stays on the roofline.
    """
    if len(out_shape) < 2 or len(term.args) < 2:
        return None
    n_dim = out_shape[-1]
    w = _shape_of(term.args[1], memo)
    if (
        not isinstance(w, tuple)
        or len(w) < 2
        or not isinstance(n_dim, int)
        or n_dim <= 0
    ):
        return None
    k_dim = w[-1] if term.op == "linear" else w[-2]
    if not isinstance(k_dim, int) or k_dim <= 0:
        return None
    n_out = float(_numel(out_shape))
    return (n_out / n_dim, float(k_dim), float(n_dim))


def _kernel_signature(
    term: Op, memo: dict | None
) -> tuple[str, tuple[float, ...]] | None:
    """(op-class, shape signature) an ``op_kernel_ns`` bucket matches.

    The mapping mirrors the classes ``calibrate._measure_op_kernels``
    times: ``"matmul"`` (``"MxKxN"``) for matmul/linear, ``"reduce"``
    (input numel) for reductions, ``"concat"`` / ``"stack"`` /
    ``"index_select"`` and ``"pointwise"`` (output numel — for reduce,
    the input numel is what streams).  Ops without a measured class
    return ``None`` and keep the roofline price.
    """
    op = term.op
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID or not isinstance(shape, tuple):
        return None
    n_out = float(_numel(shape))
    if op in ("matmul", "linear"):
        sig = _mm_signature(term, shape, memo)
        return ("matmul", sig) if sig is not None else None
    if op in _MEASURED_REDUCE_OPS:
        # The streamed size is the reduction's INPUT (the output
        # shrinks); _numel tolerates None/() arg shapes (→ 1).
        return (
            "reduce",
            (float(_numel(_shape_of(term.args[0], memo))),),
        )
    if op == "concat":
        return ("concat", (n_out,))
    if op == "stack":
        return ("stack", (n_out,))
    if op in _MEASURED_GATHER_OPS:
        return ("index_select", (n_out,))
    if op in _FUSION_POINTWISE_OPS:
        return ("pointwise", (n_out,))
    return None


def _profile_kernel_table(profile: Any) -> dict | None:
    """Return the raw ``op_kernel_ns`` dict of a profile-like."""
    if profile is None:
        return None
    if isinstance(profile, dict):
        return profile.get("op_kernel_ns") or None
    return getattr(profile, "op_kernel_ns", None) or None


def _kernel_lookup(table: dict | None):
    """Build ``kns(term, memo) -> measured ns | None`` from a table.

    Parses the ``{op_class: {shape_key: ns}}`` profile dict once; the
    returned callable maps a term to the measured wall time of the
    NEAREST bucket in log-space — sum of ``|log2 ratio|`` over the
    signature dims — so shapes between measured points price at their
    closest probe.  ``None`` table (or one with no usable entries)
    yields ``None``, i.e. the pure roofline path.
    """
    if not table:
        return None
    parsed: dict[str, list[tuple[tuple[float, ...], float]]] = {}
    for cls, entries in table.items():
        if not isinstance(entries, dict):
            continue
        pts: list[tuple[tuple[float, ...], float]] = []
        for key, ns in entries.items():
            try:
                sig = tuple(float(v) for v in str(key).split("x"))
                pts.append((sig, float(ns)))
            except (TypeError, ValueError):
                continue
        if pts:
            parsed[cls] = pts
    if not parsed:
        return None

    def kns(term: Op, memo: dict | None) -> float | None:
        sig = _kernel_signature(term, memo)
        if sig is None:
            return None
        pts = parsed.get(sig[0])
        if pts is None:
            return None
        dims = sig[1]
        best_ns = None
        best_d = float("inf")
        for esig, ns in pts:
            if len(esig) != len(dims):
                continue
            d = 0.0
            for s, e in zip(dims, esig, strict=True):
                d += abs(math.log2(max(s, 1.0) / max(e, 1.0)))
            if d < best_d:
                best_d, best_ns = d, ns
        return best_ns

    return kns


def roofline_cost_for(
    profile: Any = None,
    *,
    peak_flops: float | None = None,
    peak_bw: float | None = None,
    launch_s: float | None = None,
):
    """Return a roofline cost fn calibrated to a measured target profile.

    ``profile`` is a ``catopt_torch.calibrate.TargetProfile`` (or any object
    / dict with ``tflops``, ``gbps``, ``launch_us``); ``None`` plus
    keyword overrides gives a one-off calibration.  The returned
    closure has the standard cost-fn signature ``fn(term, memo=None)``
    and can be dropped into ``Regime(cost_fn=...)``,
    ``EGraph.extract_best``, or ``dag_cost``.

    When the profile carries an ``op_kernel_ns`` table (measured
    per-op-class kernel latencies — see
    :func:`catopt_torch.calibrate.calibrate`), each op's estimate
    is floored at the measured time of its nearest shape bucket:
    measurements can only raise the price toward observed latency,
    never undercut the roofline.

    ``roofline_cost_for()`` (no args) is exactly ``roofline_cost``.
    """
    pf, bw, ls = _profile_constants(profile)
    if peak_flops is not None:
        pf = float(peak_flops)
    if peak_bw is not None:
        bw = float(peak_bw)
    if launch_s is not None:
        ls = float(launch_s)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        return _roofline_cost(term, memo, pf, bw, ls, kns)

    cost.__name__ = "roofline_cost_for"
    cast(_CostMarkers, cost).profile = profile
    return cost


def _depth_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    kernel_ns=None,
) -> float:
    """Shared critical-path traversal.

    Used by depth_cost_for and the ``base="depth"`` arm of
    :func:`executor_cost_for`.  The memo key carries the constants so
    two profiles can share a memo dict without colliding (the
    ``_roofline_cost`` convention).
    """
    ck = (
        ("dc", peak_flops, peak_bw, launch_s, term)
        if kernel_ns is None
        else ("dc", peak_flops, peak_bw, launch_s, id(kernel_ns), term)
    )
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        local = _local_roofline(
            term,
            memo,
            peak_flops=peak_flops,
            peak_bw=peak_bw,
            launch_s=launch_s,
            kernel_ns=kernel_ns,
        )
        if local >= _INVALID_COST:
            local = launch_s * 1e9
        child = max(
            (
                _depth_cost(
                    a, memo, peak_flops, peak_bw, launch_s, kernel_ns
                )
                for a in term.args
            ),
            default=0.0,
        )
        out = local + child
        memo[ck] = float(out)
        return out
    memo[ck] = 0.0
    return 0.0


def depth_cost_for(profile: Any = None):
    """Return a critical-path cost fn calibrated to a target profile.

    Same closure convention as :func:`roofline_cost_for`, but the
    objective is depth (local roofline latency + max child depth) like
    :func:`depth_cost` — the axis on which a sequential recurrence and
    its log-depth scan differ.
    """
    pf, bw, ls = _profile_constants(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        return _depth_cost(term, memo, pf, bw, ls, kns)

    cost.__name__ = "depth_cost_for"
    cast(_CostMarkers, cost).profile = profile
    return cost


def depth_cost(term: Any, memo: dict | None = None) -> float:
    """Critical-path cost: the longest dependency chain in seconds.

    Each op's latency is its roofline time (max(flops/peak, bytes/bw)
    + launch); the term's cost is local latency + max child depth.
    Work-preserving reassociations (parallel scans, balanced sums,
    repeated squaring) win here even when total FLOPs are identical —
    this is the axis on which a sequential recurrence and its
    log-depth Blelloch form differ.
    """
    memo = {} if memo is None else memo
    ck = ("dc", term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        local = _local_roofline(term, memo=memo)
        if local >= _INVALID_COST:
            # Unshapeable op: charge a launch, not a veto — depth is a
            # structural metric, not a soundness gate.
            local = _LAUNCH_S * 1e9
        child = max(
            (depth_cost(a, memo) for a in term.args), default=0.0
        )
        out = local + child
        memo[ck] = float(out)
        return out
    memo[ck] = 0.0
    return 0.0


def _profile_dispatch_s(profile: Any) -> float:
    """Per-dispatch executor overhead in seconds, from a profile-like.

    Reads ``dispatch_us`` (attribute or dict key) — per-call
    dispatcher overhead *on top of* the kernel launch the roofline
    model already prices, e.g. Python-side op dispatch in the generic
    evaluator.  Absent a measurement it falls back to the built-in
    launch constant: dispatch ≈ launch.
    """
    if profile is None:
        return _LAUNCH_S
    if isinstance(profile, dict):
        us = profile.get("dispatch_us", _LAUNCH_S * 1e6)
    else:
        us = getattr(profile, "dispatch_us", _LAUNCH_S * 1e6)
    return float(us) * 1e-6


def _profile_graph_overhead_s(profile: Any) -> float:
    """Per-call overhead of a COMPILED graph in seconds.

    ``calibrate`` measures it as ``graph_overhead_us`` — the
    guards+graph-call boundary cost of an Inductor-compiled module,
    minus the launch constants already billed per kernel.  The fused
    price uses ``max(dispatch_s, this)`` for its one per-graph term —
    the measured residual the per-kernel table cannot see.  Fallback
    ``80us`` is conservative (measured ~6us CPU, ~50-150us with
    guards on real graphs).
    """
    fallback = 80.0
    if profile is None:
        return fallback * 1e-6
    if isinstance(profile, dict):
        us = profile.get("graph_overhead_us", fallback)
    else:
        us = getattr(profile, "graph_overhead_us", fallback)
    return float(us) * 1e-6


def _profile_leaf_eval_s(profile: Any) -> float:
    """Per-leaf scan-eval machinery overhead in seconds.

    Reads ``leaf_eval_us`` — one ``apply``/``applyd`` leaf-operand eval
    through the executor's ``eval_term`` machinery (gather/select plus
    elementwise combine) on top of its kernels, as measured by
    ``catopt_torch.calibrate``.  Absent a measurement the fallback is
    conservative — a leaf eval is a handful of dispatches, so it
    prices at ``4 * dispatch_s``.
    """
    fallback = 4.0 * _profile_dispatch_s(profile)
    if profile is None:
        return fallback
    if isinstance(profile, dict):
        us = profile.get("leaf_eval_us", fallback * 1e6)
    else:
        us = getattr(profile, "leaf_eval_us", fallback * 1e6)
    return float(us) * 1e-6
