"""Calibrate a cost-model target profile on the current machine.

catopt's roofline cost model prices each op as

    max(flops / peak_flops, bytes / peak_bw) + launch_overhead

whose constants are only honest when measured on the deployment
target.  ``calibrate()`` measures them where it runs — a timed matmul
sweep for peak FLOPS, a timed copy/reduction sweep for memory
bandwidth, a tiny-op loop for eager kernel-launch overhead, and two
executor-overhead probes (generic-eval dispatch per IR node; per-leaf
eval for batched-scan executors) — and returns a
:class:`TargetProfile`.

Profiles round-trip through JSON and persist under
``~/.cache/catopt/profiles`` (override with ``$CATOPT_PROFILE_DIR`` or
``$XDG_CACHE_HOME``), so "discover once, optimize per target" becomes::

    profile = calibrate()                       # once per machine
    save_profile(profile)
    cost_fn = roofline_cost_for(load_profile(profile.name))
    frontier = regime_frontier(eg, root, {"prefill": (cost_fn, "om_batched")})
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

__all__ = [
    "PROFILE_DIR_ENV",
    "TargetProfile",
    "calibrate",
    "list_profiles",
    "load_profile",
    "profiles_dir",
    "save_profile",
]

#: Environment variable overriding the profile-store directory.
PROFILE_DIR_ENV = "CATOPT_PROFILE_DIR"

logger = logging.getLogger("catopt_optimize.calibrate")

#: Conservative fallbacks for the executor-overhead constants, used
#: when a probe cannot run (missing adapter stack, exotic device) or
#: measures non-positive.  Chosen above typical measured values so a
#: partially-calibrated profile errs toward over-pricing executor
#: overhead rather than pretending it is free.
_FALLBACK_DISPATCH_US = 5.0
_FALLBACK_LEAF_EVAL_US = 15.0

#: Chain length of the dispatch probe: long enough that per-forward
#: fixed cost (``nn.Module.__call__``, env setup) amortises to noise,
#: short enough that the module builds instantly.
_DISPATCH_PROBE_OPS = 100


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
# Profile object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetProfile:
    """Measured constants for one execution target.

    * ``tflops``    — peak sustained fp32 matmul throughput (TFLOP/s)
    * ``gbps``      — device memory bandwidth (GB/s, decimal)
    * ``launch_us`` — eager kernel-launch overhead (µs)
    * ``device``    — the torch device measured (e.g. ``"cuda:0"``)
    * ``measured_at`` — ISO-8601 timestamp of the measurement
    * ``meta``      — free-form extras (dtype, torch version, …)
    * ``dispatch_us``   — generic-eval dispatch overhead per IR op
      node (µs): the ``eval_term`` per-node cost (env/memo lookups,
      binding dispatch, ``fn(*args, **attrs)``) on top of the kernels
      the roofline already prices.
    * ``leaf_eval_us``  — per-leaf batched-scan leaf-evaluation
      overhead (µs): one ``apply``/``applyd`` leaf operand eval
      (gather/select + elementwise combine) through the executor's
      eval machinery, beyond its kernel time.

    ``dispatch_us`` / ``leaf_eval_us`` default to conservative
    fallbacks (``_FALLBACK_DISPATCH_US`` / ``_FALLBACK_LEAF_EVAL_US``),
    so profiles saved before the probes existed — or built by hand —
    still price executor overhead honestly.

    Feed it to ``catopt_core.cost.roofline_cost_for`` /
    ``depth_cost_for`` — both accept any object with ``tflops`` /
    ``gbps`` / ``launch_us``, so a regime can later carry per-target
    cost fns without this module being imported by the cost model.
    """

    name: str
    tflops: float
    gbps: float
    launch_us: float
    device: str
    measured_at: str
    meta: dict = field(default_factory=dict)
    dispatch_us: float = _FALLBACK_DISPATCH_US
    leaf_eval_us: float = _FALLBACK_LEAF_EVAL_US

    # -- serialisation ------------------------------------------------
    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(
        cls, data: str | bytes | dict
    ) -> TargetProfile:
        """Rebuild from a JSON string/bytes or an already-parsed dict."""
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise TypeError(
                f"cannot parse TargetProfile from {type(data)}"
            )
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # -- persistence --------------------------------------------------
    def save(self, dir: str | Path | None = None) -> Path:
        """Write this profile under the profiles dir; returns the path."""
        return save_profile(self, dir)

    @classmethod
    def load(
        cls,
        name_or_path: str | Path,
        dir: str | Path | None = None,
    ) -> TargetProfile:
        """Load by profile name or explicit file path."""
        return load_profile(name_or_path, dir)

    def cost_fn(self):
        """The additive roofline cost fn for this target."""
        from catopt_core.cost import roofline_cost_for

        return roofline_cost_for(self)


def profiles_dir() -> Path:
    """Directory where profiles persist.

    ``$CATOPT_PROFILE_DIR`` wins; else ``$XDG_CACHE_HOME/catopt/profiles``;
    else ``~/.cache/catopt/profiles``.
    """
    env = os.environ.get(PROFILE_DIR_ENV)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "catopt" / "profiles"


def _safe_name(name: str) -> str:
    return (
        "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in name
        ).strip("_")
        or "profile"
    )


def save_profile(
    profile: TargetProfile, dir: str | Path | None = None
) -> Path:
    """Persist ``profile`` as ``<dir>/<safe-name>.json``; returns path."""
    d = Path(dir) if dir is not None else profiles_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_safe_name(profile.name)}.json"
    path.write_text(profile.to_json())
    return path


def load_profile(
    name_or_path: str | Path,
    dir: str | Path | None = None,
) -> TargetProfile:
    """Load a profile by name (in the profiles dir) or by file path."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return TargetProfile.from_json(p.read_text())
    d = Path(dir) if dir is not None else profiles_dir()
    # exact filename first, then a scan matching the stored name field
    cand = d / f"{_safe_name(str(name_or_path))}.json"
    if cand.exists():
        return TargetProfile.from_json(cand.read_text())
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                prof = TargetProfile.from_json(f.read_text())
            except (ValueError, TypeError):
                continue
            if prof.name == name_or_path:
                return prof
    raise FileNotFoundError(
        f"no profile named {name_or_path!r} under {d}"
    )


def list_profiles(dir: str | Path | None = None) -> list[str]:
    """Names of all persisted profiles."""
    d = Path(dir) if dir is not None else profiles_dir()
    names = []
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                names.append(
                    TargetProfile.from_json(f.read_text()).name
                )
            except (ValueError, TypeError):
                continue
    return names


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
    scheduler blip must not set the constant.
    """
    times = []
    for _ in range(reps):
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(dev)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


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
    """Measure the roofline constants of ``device`` (default: cuda if
    available, else cpu) and return a :class:`TargetProfile`.

    Five micro-benchmarks, sized so the whole run takes a few seconds:

    * peak FLOPS   — square-matmul sweep, best sustained rate;
    * bandwidth    — copy + reduction sweep, best bytes/s;
    * launch cost  — mean wall time of a back-to-back tiny-op loop;
    * dispatch     — median-of-reps difference between an N-op IR chain
      and its inline-torch equivalent, per node;
    * leaf eval    — same protocol on a scan-leaf-shaped term, per leaf.

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

    profile = TargetProfile(
        name=name or _default_name(dev),
        tflops=flops / 1e12,
        gbps=bw / 1e9,
        launch_us=launch * 1e6,
        device=str(dev),
        measured_at=datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        meta={
            "dtype": str(dtype).replace("torch.", ""),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        dispatch_us=dispatch_us,
        leaf_eval_us=leaf_eval_us,
    )
    with _verbose_ctx(logger, verbose):
        logger.info(
            "calibrated %s on %s: %.3f TFLOPS, %.1f GB/s, "
            "%.2f µs launch, %.2f µs dispatch, %.2f µs leaf-eval",
            profile.name,
            profile.device,
            profile.tflops,
            profile.gbps,
            profile.launch_us,
            profile.dispatch_us,
            profile.leaf_eval_us,
        )
        logger.debug(
            "calibration raw: flops=%g flop/s, bw=%g B/s, launch=%g s, "
            "dispatch=%g s, leaf_eval=%g s",
            flops,
            bw,
            launch,
            dispatch_us * 1e-6,
            leaf_eval_us * 1e-6,
        )
    if save:
        save_profile(profile)
    return profile
