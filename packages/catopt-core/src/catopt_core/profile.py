"""Target profiles — measured cost-model constants (torch-free).

catopt's roofline cost model prices each op as

    max(flops / peak_flops, bytes / peak_bw) + launch_overhead

whose constants are only honest when measured on the deployment
target.  This module holds the *data* half of calibration —
:class:`TargetProfile`, profile persistence, the shape buckets the
measured-feedback channel keys on, and the correction math — the
parts that name no tensor library, so the cost model and the
orchestrator can consume a profile in any backend's process.  The
*measurement* half (``calibrate()`` and the torch micro-benchmarks
that fill these constants) lives in ``catopt_torch.calibrate``.

Profiles round-trip through JSON and persist under
``~/.cache/catopt/profiles`` (override with ``$CATOPT_PROFILE_DIR`` or
``$XDG_CACHE_HOME``), so "discover once, optimize per target"
becomes::

    profile = calibrate()                       # once per machine
    save_profile(profile)
    cost_fn = roofline_cost_for(load_profile(profile.name))
    frontier = regime_frontier(eg, root, {"prefill": (cost_fn, "om_batched")})
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

__all__ = [
    "PROFILE_DIR_ENV",
    "TargetProfile",
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

#: Environment variable overriding the profile-store directory.
PROFILE_DIR_ENV = "CATOPT_PROFILE_DIR"

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


# ---------------------------------------------------------------------------
# Profile object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetProfile:
    """Measured constants for one execution target.

    * ``tflops``    — peak sustained fp32 matmul throughput (TFLOP/s)
    * ``gbps``      — device memory bandwidth (GB/s, decimal)
    * ``launch_us`` — eager kernel-launch overhead (µs)
    * ``device``    — the device measured (e.g. ``"cuda:0"``)
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
    * ``measured_ns`` — the measured-feedback map written by the
      autotuned strategy (opt-in via its ``profile=`` argument):
      ``{candidate: {bucket: {"median_ns": float,
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
    * ``corrections`` — the learned correction table
      ``record_measured`` maintains alongside ``measured_ns``:
      ``{candidate: {bucket: {"factor": float, "n": int}}}`` where
      ``factor`` is the running geometric mean of the observed
      ``median_ns / model_ns`` ratios (clamped to
      ``[_CORRECTION_LO, _CORRECTION_HI]``) and ``n`` the observation
      count.  Once a (candidate, bucket) pair reaches
      ``_CORRECTION_MIN_SAMPLES`` observations,
      :func:`corrected_price_ns` multiplies model prices by the
      learned factor — the closed-loop correction of the cost MODEL,
      versus the per-entry residuals of ``measured_ns``.

    ``dispatch_us`` / ``leaf_eval_us`` / ``graph_overhead_us`` default
    to conservative fallbacks (``_FALLBACK_DISPATCH_US`` /
    ``_FALLBACK_LEAF_EVAL_US`` / ``_FALLBACK_GRAPH_OVERHEAD_US``), and
    ``op_kernel_ns`` / ``measured_ns`` / ``corrections`` default to
    ``{}``, so profiles
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
    corrections: dict = field(default_factory=dict)

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
# The autotuned strategy measures real wall-clock latency per
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
# * ``corrections`` — ``{candidate: {bucket: {"factor", "n"}}}``: the
#   learned correction table ``record_measured`` folds every priced
#   measurement into.  ``corrected_price_ns`` multiplies model prices
#   by the factor once enough observations landed — closing the loop
#   from "remember the last measurement" to "learn the model's bias".


def shape_bucket(example_input: Any) -> str:
    """Return the shape bucket measured corrections key on.

    ``"<device>:2^e"`` where ``e`` is
    ``ceil(log2(total input numel))``.

    Deliberately coarse — a correction measured on one graph transfers
    to another graph only inside the same bucket, so an order-of-
    magnitude bucket is the honest granularity.  The device prefix
    keeps a CUDA measurement from correcting CPU prices (and vice
    versa).  Tuple inputs sum their numels; a non-tensor input lands
    in ``"cpu:2^0"``.  Element counting is duck-typed — any value
    exposing ``numel()`` and ``device`` counts, so the bucket is
    backend-neutral.
    """
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    numel = 0
    device = None
    for a in args:
        numel_of = getattr(a, "numel", None)
        if callable(numel_of):
            numel += numel_of()
            if device is None:
                device = str(getattr(a, "device", "cpu"))
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
    """Return the ``measured_ns`` map off a dict or attribute profile."""
    if profile is None:
        return None
    tab = (
        profile.get("measured_ns")
        if isinstance(profile, dict)
        else getattr(profile, "measured_ns", None)
    )
    return tab if isinstance(tab, dict) else None


def _corrections_table(profile: Any) -> dict | None:
    """Return the ``corrections`` map off a dict or attribute profile."""
    if profile is None:
        return None
    tab = (
        profile.get("corrections")
        if isinstance(profile, dict)
        else getattr(profile, "corrections", None)
    )
    return tab if isinstance(tab, dict) else None


#: Minimum ratio observations before a learned correction applies —
#: a single measurement still transfers through ``measured_ns``'s
#: residual path.
_CORRECTION_MIN_SAMPLES = 2
#: Learned multiplicative corrections live inside this range: a stale
#: or outlier-driven factor can misprice a candidate by at most 10x.
_CORRECTION_LO = 0.1
_CORRECTION_HI = 10.0
#: Effective memory of the running geometric mean — once ``n`` exceeds
#: this, a new ratio moves the factor by ~``1/(W+1)`` in log space, so
#: the table still adapts when a device/driver change shifts the true
#: ratio instead of freezing at its full history.
_CORRECTION_WINDOW = 32


def _correction_of(rec: Any) -> tuple[float, int] | None:
    """Validate one ``{"factor", "n"}`` corrections entry.

    Returns ``(factor, n)`` with the factor clamped to
    ``[_CORRECTION_LO, _CORRECTION_HI]`` — hand-edited profiles and
    stale tables get sane bounds at consumption — or ``None`` for
    malformed entries.
    """
    if not isinstance(rec, dict):
        return None
    try:
        f = float(rec["factor"])
        n = int(rec["n"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0.0:
        return None
    return min(max(f, _CORRECTION_LO), _CORRECTION_HI), max(n, 0)


def _learned_corrections(
    corr: dict | None,
    candidate: str,
    bucket: str,
    median_ns: float,
    model_ns: float | None,
) -> dict:
    """Return the ``corrections`` table updated by one observation.

    Learning rule: ``factor`` is a running GEOMETRIC MEAN of the
    observed ``median_ns / model_ns`` ratios — in log space each new
    measurement moves the estimate by at most ``1/(min(n, W)+1)``:
    bounded, monotone-ish (a single wild ratio shifts the factor by a
    bounded step, never replaces it), and still adaptive after the
    window saturates.  Ratios are clamped to
    ``[_CORRECTION_LO, _CORRECTION_HI]`` before folding in, so the
    stored factor can never leave the sane range; ``n`` counts the
    observations.

    Observations without a usable ratio leave the table unchanged: no
    ``model_ns`` (the model cannot price the candidate — its
    ``measured_ns`` entry substitutes the median outright), a zero
    ``model_ns``, or a non-positive/non-finite ratio.
    """
    tab: dict = (
        {c: dict(b) for c, b in corr.items() if isinstance(b, dict)}
        if isinstance(corr, dict)
        else {}
    )
    if model_ns is None:
        return tab
    try:
        ratio = float(median_ns) / float(model_ns)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return tab
    if not math.isfinite(ratio) or ratio <= 0.0:
        return tab
    r = min(max(ratio, _CORRECTION_LO), _CORRECTION_HI)
    cand = tab.setdefault(candidate, {})
    cur = _correction_of(cand.get(bucket))
    if cur is None or cur[1] <= 0:
        cand[bucket] = {"factor": r, "n": 1}
    else:
        f_old, n_old = cur
        w = min(n_old, _CORRECTION_WINDOW)
        cand[bucket] = {
            "factor": math.exp(
                (w * math.log(f_old) + math.log(r)) / (w + 1.0)
            ),
            "n": n_old + 1,
        }
    return tab


def record_measured(
    profile: Any,
    candidate: str,
    bucket: str,
    median_ns: float,
    model_ns: float | None = None,
) -> Any:
    """Write one measured-feedback entry into *profile*.

    Updates ``measured_ns`` — and folds the observation into the
    learned ``corrections`` table — and returns the updated profile.

    ``candidate`` is a lowering-path name (``"generic"`` /
    ``"batched"`` / ``"compiled"`` / a custom candidate name);
    ``bucket`` a :func:`shape_bucket` key — entries are per
    (candidate, bucket), never global.  ``median_ns`` is the measured
    median wall time; ``model_ns`` the cost model's price of the
    measured graph at recording time (omit when the model cannot price
    the candidate — e.g. ``"eager"`` — and the entry substitutes the
    measured median directly; see :func:`measured_price_ns`).  When
    both are usable the ratio ``median_ns / model_ns`` feeds the
    running geometric mean in ``corrections[candidate][bucket]``
    (:func:`_learned_corrections`), which :func:`corrected_price_ns`
    consumes.

    *dict* profiles are updated in place (and returned); a frozen
    :class:`TargetProfile` — or any dataclass — yields a NEW instance
    via ``dataclasses.replace``; any other object gets ``measured_ns``
    (and, when settable, ``corrections``) set on it — objects that
    reject ``measured_ns`` propagate the usual error, while a
    ``corrections`` attribute that cannot be set is skipped silently.
    """
    entry: dict[str, float] = {"median_ns": float(median_ns)}
    if model_ns is not None:
        entry["model_ns"] = float(model_ns)
    if isinstance(profile, dict):
        tab = profile.get("measured_ns")
        if not isinstance(tab, dict):
            tab = {}
            profile["measured_ns"] = tab
        cand = tab.get(candidate)
        if not isinstance(cand, dict):
            cand = tab[candidate] = {}
        cand[bucket] = entry
        profile["corrections"] = _learned_corrections(
            profile.get("corrections"),
            candidate,
            bucket,
            median_ns,
            model_ns,
        )
        return profile
    cur = _measured_table(profile)
    tab = (
        {c: dict(b) for c, b in cur.items() if isinstance(b, dict)}
        if cur
        else {}
    )
    cand = tab.get(candidate)
    if not isinstance(cand, dict):
        cand = tab[candidate] = {}
    cand[bucket] = entry
    ctab = _learned_corrections(
        _corrections_table(profile),
        candidate,
        bucket,
        median_ns,
        model_ns,
    )
    if is_dataclass(profile) and not isinstance(profile, type):
        if "corrections" in profile.__dataclass_fields__:
            return replace(profile, measured_ns=tab, corrections=ctab)
        # dataclass predating the field: measured_ns still lands
        return replace(profile, measured_ns=tab)
    target: Any = profile
    target.measured_ns = tab
    with contextlib.suppress(Exception):
        target.corrections = ctab
    return target


def measured_price_ns(
    profile: Any,
    candidate: str,
    bucket: str,
    model_ns: float | None,
) -> float | None:
    """Delivered price (ns) of a ``(candidate, bucket)`` pair.

    The uncorrected model estimate is *model_ns* — the
    measured-feedback consumption contract.

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


def _bucket_coord(bucket: Any) -> tuple[str, int] | None:
    """Parse a :func:`shape_bucket` key into ``(device, exponent)``.

    Keys are ``"<device>:2^e"``; device strings may themselves carry
    colons (``"cuda:0:2^10"``), so the split is from the right.
    ``None`` for keys that do not parse — they can still exact-match,
    never interpolate.
    """
    if not isinstance(bucket, str):
        return None
    dev, sep, e = bucket.rpartition(":2^")
    if not sep or not dev:
        return None
    try:
        return dev, int(e)
    except ValueError:
        return None


def _correction_factor(
    profile: Any,
    candidate: str,
    bucket: str,
    min_samples: int,
) -> float | None:
    """Look up the learned multiplicative correction for ``(candidate, bucket)``.

    Exact-bucket hit → its (clamped) factor.  Else the NEAREST
    same-device bucket's factor, dampened toward 1.0 by
    ``0.5 ** |Δexponent|`` — a correction measured one bucket away
    applies at half its log-space strength, two away at a quarter, so
    a neighbouring estimate nudges the price without pretending to be
    a local measurement.  Entries with ``n < min_samples`` and
    malformed entries are skipped; cross-device buckets never
    interpolate.  ``None`` when nothing applies.
    """
    tab = _corrections_table(profile)
    cand = tab.get(candidate) if tab is not None else None
    if not isinstance(cand, dict):
        return None
    key = _bucket_coord(bucket)
    best: tuple[int, float] | None = None
    for b, rec in cand.items():
        fn = _correction_of(rec)
        if fn is None or fn[1] < min_samples:
            continue
        if b == bucket:
            return fn[0]
        if key is None:
            continue
        k2 = _bucket_coord(b)
        if k2 is None or k2[0] != key[0]:
            continue
        dist = abs(k2[1] - key[1])
        if best is None or dist < best[0]:
            best = (dist, fn[0])
    if best is None:
        return None
    return best[1] ** (0.5 ** best[0])


def corrected_price_ns(
    profile: Any,
    candidate: str,
    bucket: str,
    model_ns: float | None,
    *,
    min_samples: int = _CORRECTION_MIN_SAMPLES,
) -> float | None:
    """Delivered price (ns) with the LEARNED correction applied.

    The closed-loop counterpart of :func:`measured_price_ns`:

    * a learned ``corrections`` factor for ``(candidate, bucket)`` —
      the exact bucket's own factor, else the nearest same-device
      bucket's dampened factor — with ``n >= min_samples``
      observations returns ``model_ns * factor``: the aggregated
      history's multiplicative correction of the model itself;
    * else the :func:`measured_price_ns` contract — residual transfer
      on a direct ``measured_ns`` entry, the measured median when no
      model price exists now or was recorded, ``model_ns`` unchanged
      when nothing is recorded;
    * ``model_ns=None`` cannot take a multiplicative factor — the
      measured path handles it (median substitution or ``None``).
    """
    if model_ns is not None:
        f = _correction_factor(profile, candidate, bucket, min_samples)
        if f is not None:
            return float(model_ns) * f
    return measured_price_ns(profile, candidate, bucket, model_ns)
