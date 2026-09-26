"""Calibrate a cost-model target profile on the current machine.

catopt's roofline cost model prices each op as

    max(flops / peak_flops, bytes / peak_bw) + launch_overhead

whose constants are only honest when measured on the deployment
target.  ``calibrate()`` measures them where it runs — a timed matmul
sweep for peak FLOPS, a timed copy/reduction sweep for memory
bandwidth, and a tiny-op loop for eager kernel-launch overhead — and
returns a :class:`TargetProfile`.

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

    Three micro-benchmarks, sized so the whole run takes a few seconds:

    * peak FLOPS   — square-matmul sweep, best sustained rate;
    * bandwidth    — copy + reduction sweep, best bytes/s;
    * launch cost  — mean wall time of a back-to-back tiny-op loop.

    ``quick=True`` shrinks the sweeps (for tests / smoke checks) at some
    accuracy cost.  ``save=True`` additionally persists the profile under
    :func:`profiles_dir`.
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
    )
    with _verbose_ctx(logger, verbose):
        logger.info(
            "calibrated %s on %s: %.3f TFLOPS, %.1f GB/s, %.2f µs launch",
            profile.name,
            profile.device,
            profile.tflops,
            profile.gbps,
            profile.launch_us,
        )
        logger.debug(
            "calibration raw: flops=%g flop/s, bw=%g B/s, launch=%g s",
            flops,
            bw,
            launch,
        )
    if save:
        save_profile(profile)
    return profile
