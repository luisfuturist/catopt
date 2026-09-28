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
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "PROFILE_DIR_ENV",
    "TargetProfile",
    "calibrate",
    "list_profiles",
    "load_profile",
    "measured_price_ns",
    "profile_graph_overhead_us",
    "profiles_dir",
    "record_measured",
    "save_profile",
    "shape_bucket",
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

    * ``op_kernel_ns`` — measured per-op-class kernel latencies (ns):
      ``{op_class: {shape_key: measured_ns}}`` where ``op_class`` is
      one of ``"matmul"`` (``"MxKxN"`` keys), ``"pointwise"``,
      ``"reduce"``, ``"concat"``, ``"stack"``, ``"index_select"``
      (element-count keys, decimal strings).  Each value is the
      median wall time of one eager kernel call at that shape —
      launch, dispatch and kernel work all inside — which the cost
      model uses as a floor on its roofline estimate for ops whose
      shape signature lands near a measured bucket.
    * ``graph_overhead_us`` — per-invocation overhead of one compiled
      (``torch.compile``) graph call, in µs: guard evaluation plus
      Inductor/cudagraph dispatch machinery, measured as the compiled
      call's wall time minus a single launch.  The fused cost model's
      per-graph term is ``max(dispatch_s, graph_overhead_s)`` — it
      replaces the bare dispatch charge, it does not add to it.
    * ``measured_ns`` — the measured-feedback map written by
      ``optimize_model_autotuned`` (opt-in via its ``profile=``
      argument): ``{candidate: {bucket: {"median_ns": float,
      "model_ns": float}}}`` where ``candidate`` is a lowering-path
      name (``"generic"``/``"batched"``/``"compiled"``/… or a custom
      candidate name) and ``bucket`` a :func:`shape_bucket` key —
      corrections are per-bucket, never global.  ``model_ns`` is the
      cost model's price of the measured graph at recording time, so
      consumers apply the *residual* (``median - model``) rather than
      the absolute time; entries without ``model_ns`` (candidates the
      model cannot price, e.g. ``"eager"``) substitute the measured
      median directly.  See :func:`measured_price_ns` for the
      consumption contract.

    ``dispatch_us`` / ``leaf_eval_us`` / ``graph_overhead_us`` default
    to conservative fallbacks (``_FALLBACK_DISPATCH_US`` /
    ``_FALLBACK_LEAF_EVAL_US`` / ``_FALLBACK_GRAPH_OVERHEAD_US``), and
    ``op_kernel_ns`` / ``measured_ns`` default to ``{}``, so profiles
    saved before the probes existed — or built by hand — still price
    executor overhead honestly and fall back to the pure roofline
    formula.

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
    op_kernel_ns: dict = field(default_factory=dict)
    graph_overhead_us: float = _FALLBACK_GRAPH_OVERHEAD_US
    measured_ns: dict = field(default_factory=dict)

    # -- serialisation ------------------------------------------------
    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, data: str | bytes | dict) -> TargetProfile:
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
        """Return the additive roofline cost fn for this target."""
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
# Measured-feedback channel + compiled-graph overhead — profile consumers
# ---------------------------------------------------------------------------
#
# ``optimize_model_autotuned`` measures real wall-clock latency per
# lowering candidate and then throws the numbers away; these helpers
# persist them on the profile and define how a consumer applies them.
# The contract is deliberately small and duck-typed (dict profile or
# any object with the field):
#
# * ``graph_overhead_us`` — a scalar constant: per-call overhead of one
#   compiled graph invocation.  The fused cost model's per-graph term
#   is ``max(dispatch_s, graph_overhead_s)``.
# * ``measured_ns`` — ``{candidate: {bucket: record}}``: measured
#   corrections keyed by (lowering-candidate, shape-bucket) — never
#   global.  ``measured_price_ns`` applies them.


def shape_bucket(example_input: Any) -> str:
    """The shape bucket measured corrections key on: ``"<device>:2^e"``
    where ``e`` is ``ceil(log2(total input numel))``.

    Deliberately coarse — a correction measured on one graph transfers
    to another graph only inside the same bucket, so an order-of-
    magnitude bucket is the honest granularity.  The device prefix
    keeps a CUDA measurement from correcting CPU prices (and vice
    versa).  Tuple inputs sum their numels; a non-tensor input lands
    in ``"cpu:2^0"``.
    """
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    numel = 0
    device = None
    for a in args:
        if isinstance(a, torch.Tensor):
            numel += a.numel()
            if device is None:
                device = str(a.device)
    e = math.ceil(math.log2(max(numel, 1)))
    return f"{device or 'cpu'}:2^{e}"


def profile_graph_overhead_us(profile: Any) -> float:
    """Compiled per-graph call overhead (µs) carried by *profile*.

    Reads ``graph_overhead_us`` (dict key or attribute); absent,
    unreadable or non-positive yields ``_FALLBACK_GRAPH_OVERHEAD_US``
    so the fused per-graph term errs toward over-pricing the compiled-
    call boundary rather than pretending it is free.  Consumers charge
    ``max(dispatch_s, graph_overhead_s)`` — the field REPLACES the bare
    dispatch term when larger.
    """
    if profile is None:
        return _FALLBACK_GRAPH_OVERHEAD_US
    if isinstance(profile, dict):
        us = profile.get(
            "graph_overhead_us", _FALLBACK_GRAPH_OVERHEAD_US
        )
    else:
        us = getattr(
            profile, "graph_overhead_us", _FALLBACK_GRAPH_OVERHEAD_US
        )
    try:
        us = float(us)
    except (TypeError, ValueError):
        return _FALLBACK_GRAPH_OVERHEAD_US
    return us if us > 0 else _FALLBACK_GRAPH_OVERHEAD_US


def _measured_table(profile: Any) -> dict | None:
    """The ``measured_ns`` map off a dict or attribute profile."""
    if profile is None:
        return None
    tab = (
        profile.get("measured_ns")
        if isinstance(profile, dict)
        else getattr(profile, "measured_ns", None)
    )
    return tab if isinstance(tab, dict) else None


def record_measured(
    profile: Any,
    candidate: str,
    bucket: str,
    median_ns: float,
    model_ns: float | None = None,
) -> Any:
    """Write one measured-feedback entry into *profile*'s
    ``measured_ns`` map; returns the updated profile.

    ``candidate`` is a lowering-path name (``"generic"`` /
    ``"batched"`` / ``"compiled"`` / a custom candidate name);
    ``bucket`` a :func:`shape_bucket` key — entries are per
    (candidate, bucket), never global.  ``median_ns`` is the measured
    median wall time; ``model_ns`` the cost model's price of the
    measured graph at recording time (omit when the model cannot price
    the candidate — e.g. ``"eager"`` — and the entry substitutes the
    measured median directly; see :func:`measured_price_ns`).

    *dict* profiles are updated in place (and returned); a frozen
    :class:`TargetProfile` — or any dataclass — yields a NEW instance
    via ``dataclasses.replace``; any other object gets ``measured_ns``
    set on it (objects that reject the attribute propagate the usual
    error, e.g. ``AttributeError``/``TypeError``).
    """
    entry: dict[str, float] = {"median_ns": float(median_ns)}
    if model_ns is not None:
        entry["model_ns"] = float(model_ns)
    if isinstance(profile, dict):
        tab = profile.get("measured_ns")
        if not isinstance(tab, dict):
            tab = {}
            profile["measured_ns"] = tab
        tab.setdefault(candidate, {})[bucket] = entry
        return profile
    cur = _measured_table(profile)
    tab = {c: dict(b) for c, b in cur.items()} if cur else {}
    tab.setdefault(candidate, {})[bucket] = entry
    if is_dataclass(profile) and not isinstance(profile, type):
        return replace(profile, measured_ns=tab)
    target: Any = profile
    target.measured_ns = tab
    return target


def measured_price_ns(
    profile: Any,
    candidate: str,
    bucket: str,
    model_ns: float | None,
) -> float | None:
    """Delivered price (ns) of a ``(candidate, bucket)`` pair whose
    uncorrected model estimate is *model_ns* — the measured-feedback
    consumption contract.

    * no ``measured_ns`` entry → *model_ns* unchanged (pure model);
    * entry with ``model_ns`` recorded → ``model_ns + (median_ns -
      recorded model_ns)``: the additive residual transfers the
      systematic gap (e.g. unmodelled per-graph overhead) to the term
      being priced — for the same graph the price IS the measurement;
    * entry without a recorded ``model_ns``, or no model price
      available now → the measured ``median_ns`` itself (the measured
      latency is the best honest price);
    * nothing anywhere → ``None`` (unpriced).
    """
    tab = _measured_table(profile)
    cand = tab.get(candidate) if tab is not None else None
    rec = cand.get(bucket) if isinstance(cand, dict) else None
    if rec is None:
        return model_ns
    if isinstance(rec, dict):
        med, ref = rec.get("median_ns"), rec.get("model_ns")
    else:  # bare-number entries are allowed: absolute substitution
        med, ref = rec, None
    if med is None:
        return model_ns
    if ref is None or model_ns is None:
        return float(med)
    return float(model_ns) + (float(med) - float(ref))


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
    :class:`TargetProfile`.

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
        meta={
            "dtype": str(dtype).replace("torch.", ""),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
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
