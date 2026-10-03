"""Calibrate a cost-model target profile on the current machine.

catopt's roofline cost model prices each op as

    max(flops / peak_flops, bytes / peak_bw) + launch_overhead

whose constants are only honest when measured on the deployment
target.  ``calibrate()`` measures them where it runs — a timed matmul
sweep for peak FLOPS, a timed copy/reduction sweep for memory
bandwidth, a tiny-op loop for eager kernel-launch overhead, two
executor-overhead probes (generic-eval dispatch per IR node; per-leaf
eval for batched-scan executors), and a per-op-class kernel sweep
(timed torch kernels at representative shapes for the dominant op
classes — matmul by (M,K,N), pointwise/reduction/concat/stack/
index_select by element count) — and returns a
:class:`~catopt_core.profile.TargetProfile`.

The *data* half — the profile type, persistence, buckets and
correction math — lives in torch-free :mod:`catopt_core.profile`
(plan 0007 split) and is re-exported here so the historical import
paths (``catopt_torch.calibrate.TargetProfile`` and friends) keep working.

Every probe reduces its timed samples through the one timing
contract, :mod:`catopt_core.timing` (median + IQR), so calibration
and the :class:`~catopt_core.ports.Meter` port agree on what a
measurement's summary number is.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from catopt_core.profile import (
    PROFILE_DIR_ENV,
    TargetProfile,
    _bucket_coord,
    _correction_factor,
    _correction_of,
    _corrections_table,
    _learned_corrections,
    _measured_table,
    _safe_name,
    corrected_price_ns,
    list_profiles,
    load_profile,
    measured_price_ns,
    profile_graph_overhead_us,
    profiles_dir,
    record_measured,
    save_profile,
    shape_bucket,
)
from catopt_core.timing import median

__all__ = [
    "PROFILE_DIR_ENV",
    "TargetProfile",
    "_bucket_coord",
    "_correction_factor",
    "_correction_of",
    "_corrections_table",
    "_learned_corrections",
    "_measured_table",
    "_safe_name",
    "calibrate",
    "corrected_price_ns",
    "list_profiles",
    "load_profile",
    "measured_price_ns",
    "profile_graph_overhead_us",
    "profiles_dir",
    "record_measured",
    "save_profile",
    "shape_bucket",
]

#: The logger name is the module's own — caplog tests and downstream
#: filters key on this channel.
logger = logging.getLogger("catopt_torch.calibrate")

#: Conservative fallbacks for the executor-overhead constants, used
#: when a probe cannot run (missing adapter stack, exotic device) or
#: measures non-positive.  Chosen above typical measured values so a
#: partially-calibrated profile errs toward over-pricing executor
#: overhead rather than pretending it is free.
_FALLBACK_DISPATCH_US = 5.0
_FALLBACK_LEAF_EVAL_US = 15.0
#: Compiled-call per-graph overhead when the probe cannot run — one
#: ``torch.compile``d graph invocation pays guard evaluation +
#: Inductor/cudagraph dispatch machinery even for a single kernel.
#: Measured values run ~25 µs (small CPU graph) to ~150 µs; the
#: fallback sits mid-range so a partially-calibrated profile never
#: prices the compiled-call boundary as free.
_FALLBACK_GRAPH_OVERHEAD_US = 80.0

#: Chain length of the dispatch probe: long enough that per-forward
#: fixed cost (``nn.Module.__call__``, env setup) amortises to noise,
#: short enough that the module builds instantly.
_DISPATCH_PROBE_OPS = 100

#: Checkout root above this package — ``packages/catopt-torch/src/
#: catopt_torch/calibrate.py`` is four levels down.  Used only by the
#: best-effort provenance probe; absent for a non-editable install.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD``; ``None`` when unavailable.

    Provenance only — never raises.  A non-editable install (no
    ``.git`` above the package) or a missing/failing ``git`` yields
    ``None`` so a provenance gap never breaks calibration.
    """
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPO_ROOT),
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def _provenance(dtype: torch.dtype) -> dict:
    """Best-effort provenance for a profile's ``meta``.

    The dtype, torch version and platform the measurement ran on plus
    the reproducibility/identity fields plan 0016 stage 3 asks for:
    the CUDA runtime version (``"none"`` on a CPU build), the git HEAD
    sha (absent when git metadata is unavailable) and the torch RNG
    seed in force.  Every field is gathered without raising, so an
    exotic environment degrades the record rather than the run.
    """
    meta: dict = {
        "dtype": str(dtype).replace("torch.", ""),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda or "none",
        "seed": torch.initial_seed(),
    }
    sha = _git_sha()
    if sha is not None:
        meta["git_sha"] = sha
    return meta


@contextlib.contextmanager
def _verbose_ctx(log: logging.Logger, verbose: bool):
    """Drop ``log``'s level to DEBUG for the block when ``verbose``.

    Emission stays level-based — ``verbose=True`` only widens what the
    module logger lets through, surfacing INFO/DEBUG records on whatever
    handlers are configured (pytest's ``caplog`` included) without
    permanently mutating logger state.
    """
    if not verbose:
        yield
        return
    prev = log.level
    log.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        log.setLevel(prev)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _measure_flops(
    dev: torch.device,
    dtype: torch.dtype,
    sizes: tuple,
    iters: int,
    warmup: int,
) -> float:
    """Best sustained matmul throughput over the size sweep, in FLOP/s."""
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        try:
            for _ in range(warmup):
                a @ b
            _sync(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                a @ b
            _sync(dev)
            dt = time.perf_counter() - t0
        finally:
            del a, b
        if dt > 0:
            best = max(best, 2.0 * n * n * n * iters / dt)
    return best


def _measure_bandwidth(
    dev: torch.device,
    dtype: torch.dtype,
    sizes_mb: tuple,
    iters: int,
    warmup: int,
) -> float:
    """Best copy/reduction bandwidth over the size sweep, in bytes/s."""
    esize = torch.empty(0, dtype=dtype).element_size()
    best = 0.0
    for mb in sizes_mb:
        n = max(int(mb * 1e6) // esize, 1)
        src = torch.randn(n, device=dev, dtype=dtype)
        dst = torch.empty(n, device=dev, dtype=dtype)
        try:
            # copy: reads n elements, writes n elements
            for _ in range(warmup):
                dst.copy_(src)
            _sync(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                dst.copy_(src)
            _sync(dev)
            dt = time.perf_counter() - t0
            if dt > 0:
                best = max(best, 2.0 * n * esize * iters / dt)
            # reduction: read-only traffic
            for _ in range(warmup):
                src.sum()
            _sync(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                src.sum()
            _sync(dev)
            dt = time.perf_counter() - t0
            if dt > 0:
                best = max(best, float(n) * esize * iters / dt)
        finally:
            del src, dst
    return best


def _measure_launch(
    dev: torch.device, dtype: torch.dtype, iters: int, warmup: int
) -> float:
    """Per-op wall time of a back-to-back tiny kernel loop, in seconds.

    A 1-element ``add_`` is launch-bound: the measured per-op time is
    the eager dispatch + kernel-launch overhead the cost model prices
    per non-view op.
    """
    x = torch.zeros(1, device=dev, dtype=dtype)
    for _ in range(warmup):
        x.add_(1.0)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        x.add_(1.0)
    _sync(dev)
    return (time.perf_counter() - t0) / iters


def _timed_median(
    fn, dev: torch.device, reps: int, iters: int
) -> float:
    """Median wall-clock seconds of ``iters`` calls to ``fn``.

    Median-of-reps instead of a single timed block: Python-side eval
    overhead is noise-dominated at this scale, so a GC pause or
    scheduler blip must not set the constant.  The reduction is
    :func:`catopt_core.timing.median` — the one timing contract.
    """
    times = []
    for _ in range(reps):
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(dev)
        times.append(time.perf_counter() - t0)
    return median(times)


def _measure_dispatch(
    dev: torch.device,
    dtype: torch.dtype,
    reps: int,
    iters: int,
    warmup: int,
    n_ops: int,
) -> float:
    """Per-node ``eval_term`` dispatch overhead, in seconds.

    An ``n_ops``-long alternating add/mul chain over a tiny ``Var``
    input is lowered with ``ir_to_torch_module``; its median forward
    time minus the median of the equivalent inline-torch chain,
    amortised over ``n_ops * iters``, isolates the per-node cost of generic
    IR eval — recursion, env/memo dict lookups, binding dispatch,
    ``fn(*args, **attrs)`` — on top of the kernels the roofline model
    already prices.  catopt pieces are imported lazily so
    ``calibrate()`` stays importable where the adapter stack is not.
    """
    from catopt_core.ir import IR, Op, TensorType, Var

    from catopt_torch.torch_bridge import ir_to_torch_module

    x = Var("x", TensorType((16,)))
    term: Var | Op = x
    for i in range(n_ops):
        term = Op.make("add" if i % 2 == 0 else "mul", term, x)
    mod = ir_to_torch_module(IR(root=term, inputs=[x]))
    mod.to(dev).eval()
    xt = torch.full((16,), 0.5, device=dev, dtype=dtype)

    def ref_chain() -> torch.Tensor:
        y = xt
        for i in range(n_ops):
            y = y + xt if i % 2 == 0 else y * xt
        return y

    for _ in range(warmup):
        mod(xt)
        ref_chain()
    t_mod = _timed_median(lambda: mod(xt), dev, reps, iters)
    t_ref = _timed_median(ref_chain, dev, reps, iters)
    return max(t_mod - t_ref, 0.0) / (n_ops * iters)


def _measure_leaf_eval(
    dev: torch.device,
    dtype: torch.dtype,
    reps: int,
    iters: int,
    warmup: int,
) -> float:
    """Per-leaf scan-leaf eval overhead, in seconds.

    An extracted ``apply``/``applyd`` leaf reads its per-step
    operands — a gather (``select``) feeding elementwise combines —
    through the executor's ``eval_term`` machinery:
    ``BatchedScanModule`` routes leaf/h evaluation through its
    embedded ``IRModule`` (``BatchedExecutorBase.ev_factory``).  So a
    leaf-shaped term ``add(mul(select(x,0,i), a), b)`` lowered with
    ``ir_to_torch_module`` exercises the same path; subtracting the
    inline-torch equivalent isolates the per-leaf machinery cost on
    top of the kernels, matching ``dispatch_us``'s overhead-only
    convention.  The batched level's stack/index_select work is priced
    by the executor model, not folded in here.
    """
    from catopt_core.ir import IR, Op, TensorType, Var

    from catopt_torch.torch_bridge import ir_to_torch_module

    xs = Var("xs", TensorType((8, 16)))
    a = Var("a", TensorType((16,)))
    b = Var("b", TensorType((16,)))
    leaf = Op.make(
        "add",
        Op.make("mul", Op.make("select", xs, dim=0, index=3), a),
        b,
    )
    mod = ir_to_torch_module(IR(root=leaf, inputs=[xs, a, b]))
    mod.to(dev).eval()
    xt = torch.randn(8, 16, device=dev, dtype=dtype)
    at = torch.randn(16, device=dev, dtype=dtype)
    bt = torch.randn(16, device=dev, dtype=dtype)

    def ref_leaf() -> torch.Tensor:
        return xt.select(0, 3) * at + bt

    for _ in range(warmup):
        mod(xt, at, bt)
        ref_leaf()
    t_mod = _timed_median(lambda: mod(xt, at, bt), dev, reps, iters)
    t_ref = _timed_median(ref_leaf, dev, reps, iters)
    return max(t_mod - t_ref, 0.0) / iters


def _measure_graph_overhead(
    dev: torch.device,
    dtype: torch.dtype,
    reps: int,
    iters: int,
    warmup: int,
    launch_s: float,
) -> float:
    """Per-invocation overhead of one compiled-graph call, seconds.

    ``torch.compile`` of a trivial pointwise chain emits ONE fused
    kernel whose work at this size is launch-bound, so the per-call
    wall minus ``launch_s`` isolates what ``fused_cost_for``'s
    per-graph term must carry but ``dispatch_us`` understates: dynamo
    guard evaluation, compiled-module dispatch and cudagraph-tree
    bookkeeping.  Best-effort like the other executor probes — a
    toolchain without inductor raises into the caller's fallback.
    """
    x = torch.zeros(4096, device=dev, dtype=dtype)

    def fwd(t: torch.Tensor) -> torch.Tensor:
        return torch.relu(t) * 2.0 + 1.0

    compiled = torch.compile(fwd)
    for _ in range(warmup):  # first call compiles
        compiled(x)
    _sync(dev)
    t_c = _timed_median(lambda: compiled(x), dev, reps, iters)
    return max(t_c / iters - launch_s, 0.0)


# ---------------------------------------------------------------------------
# Per-op-class kernel probes — the shape-dependent kernel table
# ---------------------------------------------------------------------------
#
# The five constants above price an op as max(flops/peak, bytes/bw) +
# launch — blind to how a real kernel's time actually scales with its
# shape (BLAS efficiency curves, cache-resident bandwidth, gather
# costs).  The probes below time the torch kernels themselves at a
# handful of representative shapes, producing the ``op_kernel_ns``
# table the cost model consults as a measured floor.

#: Matmul probes as (M, K, N): square GEMMs for the compute-bound
#: regime plus the skinny-K / wide-M rectangles lowered graphs emit
#: (row-batched weight products, T-by-d attention shapes).
_MM_PROBE_FULL = (
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (128, 64, 64),
    (512, 64, 64),
    (2048, 128, 128),
    (4096, 64, 64),
    (16384, 64, 64),
)
_MM_PROBE_QUICK = (
    (128, 128, 128),
    (512, 512, 512),
    (128, 64, 64),
    (512, 64, 64),
    (2048, 128, 128),
)

#: Element-count probes for the traffic-priced classes.  The range
#: spans L1-resident to DRAM-resident sizes so the nearest-bucket
#: lookup sees both the launch floor and the bandwidth regime.
_EW_PROBE_FULL = (4096, 65536, 262144, 1048576, 4194304, 16777216)
_EW_PROBE_QUICK = (4096, 1048576, 4194304)
_REDUCE_PROBE_FULL = (16384, 262144, 1048576, 4194304)
_REDUCE_PROBE_QUICK = (16384, 1048576)
_CAT_PROBE_FULL = (16384, 262144, 1048576, 4194304)
_CAT_PROBE_QUICK = (16384, 1048576)
_IX_PROBE_FULL = (4096, 16384, 65536, 262144, 1048576)
_IX_PROBE_QUICK = (4096, 65536)

#: Width of the index_select probe's gather: n indices over a (2n, W)
#: base read n·W elements — the (T,d)-row gathers scan executors do.
_IX_PROBE_WIDTH = 64


def _kernel_probe(fn, dev: torch.device, quick: bool) -> float:
    """Median wall-clock NANOSECONDS of one ``fn()`` kernel call.

    Timed like :func:`_timed_median` — ``reps`` median-ed blocks of
    ``iters`` back-to-back calls — but with the block length adapted
    to the kernel: a ~ms-timescale op gets a handful of calls per
    rep, a launch-bound tiny op hundreds, so every block lands in the
    same few-ms timing window.
    """
    reps = 3 if quick else 5
    for _ in range(2):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    fn()
    _sync(dev)
    est = time.perf_counter() - t0
    iters = max(3, min(200, int(3e-3 / max(est, 1e-7))))
    return _timed_median(fn, dev, reps, iters) * 1e9 / iters


def _measure_op_kernels(
    dev: torch.device, dtype: torch.dtype, quick: bool
) -> dict:
    """Measured kernel table: ``{op_class: {shape_key: ns}}``.

    Each probe times the eager torch kernel the IR op lowers to —
    ``a @ b``, ``x + y``, ``x.sum()``, ``cat``/``stack``,
    ``index_select`` — so the values carry launch, dispatch and
    kernel work together, matching what a lowered module actually
    pays per op.  A failed probe (exotic device, OOM on the largest
    point) drops its own entry; whatever measures survives — the
    cost model falls back to the roofline for missing classes.

    The sweep runs with ``torch.set_num_threads(1)``: per-call
    latency through a serialized executor is what the table prices —
    the standard timing harness (``torch.utils.benchmark.Timer``,
    the fidelity bench's measurement) defaults to ``num_threads=1``,
    and a multithreaded probe under-reports per-op latency by the
    BLAS thread-scaling factor (~4x on a 12-core box).  The coarse
    ``tflops``/``gbps`` sweeps keep their multithreaded peaks — those
    are hardware-headline constants, not per-op latencies.  (The
    thread pin is a no-op for the CUDA kernel work itself; it only
    shapes the CPU-side dispatch the wall time includes.)
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _op_kernel_sweep(dev, dtype, quick)
    finally:
        torch.set_num_threads(prev_threads)


def _op_kernel_sweep(
    dev: torch.device, dtype: torch.dtype, quick: bool
) -> dict:
    table: dict[str, dict[str, float]] = {}

    def rec(cls: str, key: str, fn) -> None:
        try:
            table.setdefault(cls, {})[key] = _kernel_probe(
                fn, dev, quick
            )
        except Exception:
            logger.debug("op-kernel probe failed: %s[%s]", cls, key)

    mm_shapes = _MM_PROBE_QUICK if quick else _MM_PROBE_FULL
    for m, k, n in mm_shapes:
        a = torch.randn(m, k, device=dev, dtype=dtype)
        b = torch.randn(k, n, device=dev, dtype=dtype)
        rec("matmul", f"{m}x{k}x{n}", lambda a=a, b=b: a @ b)
        del a, b

    ew_numels = _EW_PROBE_QUICK if quick else _EW_PROBE_FULL
    for n in ew_numels:
        x = torch.randn(n, device=dev, dtype=dtype)
        y = torch.randn(n, device=dev, dtype=dtype)
        rec("pointwise", str(n), lambda x=x, y=y: x + y)
        del x, y

    rd_numels = _REDUCE_PROBE_QUICK if quick else _REDUCE_PROBE_FULL
    for n in rd_numels:
        x = torch.randn(n, device=dev, dtype=dtype)
        rec("reduce", str(n), lambda x=x: x.sum())
        del x

    cat_numels = _CAT_PROBE_QUICK if quick else _CAT_PROBE_FULL
    for n in cat_numels:
        a = torch.randn(n // 2, device=dev, dtype=dtype)
        b = torch.randn(n - n // 2, device=dev, dtype=dtype)
        rec("concat", str(n), lambda a=a, b=b: torch.cat([a, b]))
        rec("stack", str(n), lambda a=a, b=b: torch.stack([a, b]))
        del a, b

    ix_numels = _IX_PROBE_QUICK if quick else _IX_PROBE_FULL
    w = _IX_PROBE_WIDTH
    for n in ix_numels:
        n_idx = max(n // w, 1)
        base = torch.randn(
            max(2 * n_idx, w), w, device=dev, dtype=dtype
        )
        idx = torch.randperm(base.shape[0], device=dev)[:n_idx]
        rec(
            "index_select",
            str(n_idx * w),
            lambda base=base, idx=idx: base.index_select(0, idx),
        )
        del base, idx

    return {cls: entries for cls, entries in table.items() if entries}


def _default_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        try:
            return torch.cuda.get_device_name(dev)
        except Exception:
            return f"cuda:{dev.index or 0}"
    proc = platform.processor() or platform.machine() or "cpu"
    return f"{proc} (cpu, {torch.get_num_threads()}t)"


def calibrate(
    device: str | torch.device | None = None,
    *,
    name: str | None = None,
    dtype: torch.dtype = torch.float32,
    quick: bool = False,
    verbose: bool = False,
    save: bool = False,
) -> TargetProfile:
    """Measure the roofline constants of ``device``.

    ``device`` defaults to cuda if available, else cpu.  Returns a
    :class:`~catopt_core.profile.TargetProfile` whose ``meta`` records
    best-effort provenance — dtype, torch/CUDA version, platform, the
    git HEAD sha and the RNG seed — so a measured constant is never
    quoted without the context that produced it (a provenance gap
    never breaks the run).

    Six micro-benchmarks, sized so the whole run takes a few seconds:

    * peak FLOPS   — square-matmul sweep, best sustained rate;
    * bandwidth    — copy + reduction sweep, best bytes/s;
    * launch cost  — mean wall time of a back-to-back tiny-op loop;
    * dispatch     — median-of-reps difference between an N-op IR chain
      and its inline-torch equivalent, per node;
    * leaf eval    — same protocol on a scan-leaf-shaped term, per leaf;
    * graph overhead — per-call wall of a tiny ``torch.compile``d
      pointwise graph minus one launch: the guards+dispatch boundary
      the fused model's per-graph term charges;
    * op kernels   — timed torch kernels for the dominant op classes
      (matmul by (M,K,N); pointwise / reduce / concat / stack /
      index_select by element count), stored as ``op_kernel_ns``.

    ``quick=True`` shrinks the sweeps (for tests / smoke checks) at some
    accuracy cost.  ``save=True`` additionally persists the profile under
    :func:`profiles_dir`.  The executor-overhead probes are best-effort:
    a failure or non-positive reading falls back to
    ``_FALLBACK_DISPATCH_US`` / ``_FALLBACK_LEAF_EVAL_US``.
    """
    dev = (
        torch.device(device)
        if device is not None
        else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    is_cuda = dev.type == "cuda"

    if is_cuda:
        flops_sizes = (512, 1024) if quick else (1024, 2048, 4096)
        flops_iters, flops_warmup = (10, 3) if quick else (20, 5)
        bw_mb = (64,) if quick else (64, 256)
        bw_iters, bw_warmup = (15, 3) if quick else (40, 5)
        launch_iters, launch_warmup = (
            (2000, 200) if quick else (5000, 500)
        )
    else:
        flops_sizes = (256, 512) if quick else (512, 1024, 2048)
        flops_iters, flops_warmup = (3, 1) if quick else (5, 2)
        bw_mb = (32,) if quick else (64, 128)
        bw_iters, bw_warmup = (10, 3) if quick else (20, 5)
        launch_iters, launch_warmup = (
            (2000, 200) if quick else (5000, 500)
        )

    # Executor-overhead probes size the same on either device: they
    # time Python-side eval machinery, not kernel throughput — and are
    # cheap enough that even the full sizes add well under 0.5 s.
    disp_reps, disp_iters, disp_warmup = (
        (5, 15, 5) if quick else (7, 30, 10)
    )
    leaf_reps, leaf_iters, leaf_warmup = (
        (5, 150, 50) if quick else (7, 400, 100)
    )
    # Compiled-overhead probe: warmup calls trigger the compile itself
    # (seconds on a cold cache) — the timed block then measures
    # steady-state per-call dispatch.
    go_reps, go_iters, go_warmup = (3, 60, 4) if quick else (5, 150, 6)

    # Measure honest fp32: TF32 tensor cores would inflate the matmul
    # number past what fp32 elementwise consumers actually get.
    prev_tf32 = None
    if is_cuda and dtype == torch.float32:
        try:
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            prev_tf32 = None
    try:
        flops = _measure_flops(
            dev, dtype, flops_sizes, flops_iters, flops_warmup
        )
        bw = _measure_bandwidth(dev, dtype, bw_mb, bw_iters, bw_warmup)
        launch = _measure_launch(
            dev, dtype, launch_iters, launch_warmup
        )
        try:
            op_kernels = _measure_op_kernels(dev, dtype, quick)
        except Exception:
            op_kernels = {}
    finally:
        if prev_tf32 is not None:
            with contextlib.suppress(Exception):
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # Executor-overhead probes need the adapter stack (catopt_core IR +
    # torch_bridge lowering); any failure — or a non-positive reading
    # on a noisy clock — keeps the conservative fallback so the
    # profile never prices executor overhead as free.
    try:
        dispatch_us = (
            _measure_dispatch(
                dev,
                dtype,
                disp_reps,
                disp_iters,
                disp_warmup,
                _DISPATCH_PROBE_OPS,
            )
            * 1e6
        )
    except Exception:
        dispatch_us = _FALLBACK_DISPATCH_US
    if dispatch_us <= 0:
        dispatch_us = _FALLBACK_DISPATCH_US
    try:
        leaf_eval_us = (
            _measure_leaf_eval(
                dev, dtype, leaf_reps, leaf_iters, leaf_warmup
            )
            * 1e6
        )
    except Exception:
        leaf_eval_us = _FALLBACK_LEAF_EVAL_US
    if leaf_eval_us <= 0:
        leaf_eval_us = _FALLBACK_LEAF_EVAL_US
    # The compiled-call probe needs a working torch.compile backend;
    # same fallback convention as the other executor overheads.
    try:
        graph_overhead_us = (
            _measure_graph_overhead(
                dev, dtype, go_reps, go_iters, go_warmup, launch
            )
            * 1e6
        )
    except Exception:
        graph_overhead_us = _FALLBACK_GRAPH_OVERHEAD_US
    if graph_overhead_us <= 0:
        graph_overhead_us = _FALLBACK_GRAPH_OVERHEAD_US

    profile = TargetProfile(
        name=name or _default_name(dev),
        tflops=flops / 1e12,
        gbps=bw / 1e9,
        launch_us=launch * 1e6,
        device=str(dev),
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        meta=_provenance(dtype),
        dispatch_us=dispatch_us,
        leaf_eval_us=leaf_eval_us,
        op_kernel_ns=op_kernels,
        graph_overhead_us=graph_overhead_us,
    )
    with _verbose_ctx(logger, verbose):
        logger.info(
            "calibrated %s on %s: %.3f TFLOPS, %.1f GB/s, "
            "%.2f µs launch, %.2f µs dispatch, %.2f µs leaf-eval, "
            "%.2f µs graph-overhead",
            profile.name,
            profile.device,
            profile.tflops,
            profile.gbps,
            profile.launch_us,
            profile.dispatch_us,
            profile.leaf_eval_us,
            profile.graph_overhead_us,
        )
        logger.debug(
            "calibration raw: flops=%g flop/s, bw=%g B/s, launch=%g s, "
            "dispatch=%g s, leaf_eval=%g s, graph_overhead=%g s, "
            "op-kernel classes=%s",
            flops,
            bw,
            launch,
            dispatch_us * 1e-6,
            leaf_eval_us * 1e-6,
            graph_overhead_us * 1e-6,
            sorted(profile.op_kernel_ns),
        )
    if save:
        save_profile(profile)
    return profile
