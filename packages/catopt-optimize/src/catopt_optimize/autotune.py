"""Measured autotuning over catopt's lowering paths.

``optimize_model`` commits to a single lowering up front: the
extracted term routes to the serial ``IRModule`` evaluator or — for
carrier-apply roots — a level-batched executor, optionally wrapped
in ``torch.compile``.  Which of those is actually fastest on a given
device is an empirical question the static cost model can only guess
at (per-node eval dispatch vs. batched compose levels vs. inductor
fusion reorder differently across CPU and GPU).

:func:`optimize_model_autotuned` runs the e-graph search ONCE, then
re-lowers the same extracted term through each requested lowering
path, verifies every candidate against the original model (the same
``sink.verify`` gate the pipeline uses — an unverified candidate is
never timed, let alone returned), measures each survivor on the real
input, and returns the measured winner.

Honesty contract:

* every timed candidate passed ``sink.verify`` on ``example_input``;
* every candidate's outcome — build failure, verify failure, timing,
  budget skip — is recorded in ``stats["autotune"]["candidates"]``;
* if every candidate fails, the unmodified ``optimize_model`` output
  is returned and ``stats["autotune"]["fallback"]`` is True.

Built-in candidate names (see :data:`CANDIDATE_BUILDERS`):

``generic``
    Serial ``IRModule`` evaluation — ``sink.lower`` directly, no
    carrier routing (the executor ``lowering="generic"`` prices).
``batched``
    The pipeline's executor routing — ``_lower_extracted``: the
    level-batched carrier executor when the extracted root plans
    one, a plain ``IRModule`` otherwise.  This is exactly what
    ``optimize_model()`` returns.
``compiled``
    ``torch.compile`` over the routed executor — what
    ``optimize_model(runner=CompiledRunner())`` returns.
``compiled_generic``
    ``torch.compile`` over the serial ``IRModule``.
``cuda_graph``
    A fresh routed executor captured into a CUDA graph — what
    ``optimize_model(runner=CudaGraphRunner())`` delivers.  Needs a
    CUDA example input and a capture-capable executor; otherwise the
    candidate records ``status="unavailable"``.
``eager``
    The original model, unchanged.  Always verifies (it is the
    reference) — when nothing beats it, it wins honestly.

Custom candidates: a ``candidates`` entry may be a
``(name, builder)`` tuple where ``builder`` is a
:data:`CandidateBuilder` callable receiving the
:class:`AutotuneContext` and returning a runnable module.

Measured feedback (opt-in)
--------------------------

Every timed candidate's median latency would otherwise be thrown away
after picking the winner.  Pass a profile to ``profile=`` — a
:class:`~catopt_optimize.calibrate.TargetProfile` or a plain dict —
and two things happen:

* candidates are *attempted* cheapest-predicted-first, where the
  prediction is the cost model's delivered price corrected by the
  profile's learned ``corrections`` factors (once a
  (candidate, bucket) pair has enough observations) and
  ``measured_ns`` residuals for this input's
  :func:`~catopt_optimize.calibrate.shape_bucket`
  (see :func:`~catopt_optimize.calibrate.corrected_price_ns`); and
* this run's measurements are written back into the profile's
  ``measured_ns`` map AND folded into its ``corrections`` table
  (a running geometric mean of ``median/model`` ratios per
  (candidate, bucket)), so the NEXT call — or any other consumer of
  the profile — prices those lowerings closer to measured, and the
  model's per-bucket bias is learned across runs.

Corrections are keyed per (candidate, shape-bucket) — never global:
a measurement on one input size only ever corrects prices in its own
bucket.  ``stats["autotune"]`` records what the model predicted
(``predicted_ns`` per candidate, ``predicted_winner``), what was
written (``measured_ns``), and the updated profile object
(``profile``).
"""

from __future__ import annotations

import copy
import logging
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, cast

import torch
from catopt_core.cost import (
    _profile_dispatch_s,
    executor_cost_for,
    fused_cost_for,
)
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.ports import Sink, Source
from catopt_torch.adapters import TorchSink

from catopt_optimize.calibrate import (
    corrected_price_ns,
    profile_graph_overhead_us,
    record_measured,
    shape_bucket,
)
from catopt_optimize.optimize import _lower_extracted, optimize_model

logger = logging.getLogger("catopt_optimize.autotune")

__all__ = [
    "CANDIDATE_BUILDERS",
    "AutotuneContext",
    "CandidateBuilder",
    "CandidateUnavailableError",
    "optimize_model_autotuned",
]


class CandidateUnavailableError(RuntimeError):
    """A candidate legitimately cannot run here.

    E.g. ``cuda_graph`` on a CPU input — recorded as
    ``status="unavailable"``, which is not a failure.
    """


@dataclass
class AutotuneContext:
    """What a candidate builder sees.

    The extracted program, its materialised parameters, and the
    pipeline's own output.

    ``ir`` is the *delivered* IR — ``term`` (``ir.root``) is the
    extracted term after ``IRModule``'s weight-chain folding, so
    ``fused_*`` param leaves already have entries in
    ``param_values`` and re-lowering needs no re-fold.
    ``delivered`` is the module ``optimize_model`` itself produced
    (``lowering`` is its ``stats["lowering"]``) — a builder may
    reuse it directly when it is already the right executor.

    ``ir`` is ``None`` only when the delivered module does not
    expose the IRModule internals (a custom ``sink``'s executor) —
    lowerers then report ``unavailable``; ``delivered``-reuse and
    ``eager`` candidates still work.
    """

    model: torch.nn.Module
    example_input: Any  # tensor or args tuple
    ir: IR | None
    term: Any
    param_values: dict[str, torch.Tensor]
    sink: Sink
    delivered: torch.nn.Module
    lowering: str


#: ``builder(ctx) -> runnable module``; raising
#: :class:`CandidateUnavailableError` marks the candidate
#: ``"unavailable"``, any other exception ``"build_failed"``.
CandidateBuilder = Callable[[AutotuneContext], Any]


def _require_ir(ctx: AutotuneContext) -> IR:
    if ctx.ir is None:
        raise CandidateUnavailableError(
            "delivered module did not expose its extracted IR"
        )
    return ctx.ir


def _build_generic(ctx: AutotuneContext) -> Any:
    """Build the serial ``IRModule`` via ``sink.lower``.

    Reuse ``delivered`` when the pipeline itself routed to the
    generic executor.
    """
    if ctx.lowering == "generic":
        return ctx.delivered
    return ctx.sink.lower(_require_ir(ctx), ctx.param_values)


def _build_batched(ctx: AutotuneContext) -> Any:
    """Build ``_lower_extracted`` carrier routing.

    Reuse ``delivered`` when it already is the routed executor.
    """
    if ctx.lowering == "batched":
        return ctx.delivered
    return _lower_extracted(
        ctx.term, _require_ir(ctx), ctx.param_values, ctx.sink
    )


def _build_compiled(ctx: AutotuneContext) -> Any:
    """``torch.compile`` over the routed executor.

    The ``optimize_model(runner=CompiledRunner())`` delivery.

    Always a FRESH module: ``torch.compile`` rewrites the module's
    ``forward`` attribute (dynamo dispatch), so compiling
    ``ctx.delivered`` would contaminate the ``batched`` candidate —
    they are the same object when the pipeline routed there.
    """
    return torch.compile(
        _lower_extracted(
            ctx.term, _require_ir(ctx), ctx.param_values, ctx.sink
        )
    )


def _build_compiled_generic(ctx: AutotuneContext) -> Any:
    """``torch.compile`` over a FRESH serial ``IRModule``.

    Fresh for the same ``forward``-mutation reason as ``compiled``.
    """
    return torch.compile(
        cast(
            Any,
            ctx.sink.lower(_require_ir(ctx), ctx.param_values),
        )
    )


def _capture_routed(  # pragma: no cover — CUDA-only body
    ctx: AutotuneContext,
) -> Any:
    """Fresh routed executor captured into a CUDA graph.

    Fresh because ``capture_cuda_graph`` mutates the module —
    capturing ``ctx.delivered`` would silently upgrade the
    ``batched`` candidate too.
    """
    mod = _lower_extracted(
        ctx.term, _require_ir(ctx), ctx.param_values, ctx.sink
    )
    capture = getattr(mod, "capture_cuda_graph", None)
    if capture is None:
        raise CandidateUnavailableError(
            "routed executor has no capture_cuda_graph"
        )
    args = (
        ctx.example_input
        if isinstance(ctx.example_input, tuple)
        else (ctx.example_input,)
    )
    capture(*args)
    return mod


def _build_cuda_graph(ctx: AutotuneContext) -> Any:
    """Build the ``cuda_graph`` candidate.

    ``optimize_model(runner=CudaGraphRunner())`` semantics on a
    fresh module (see :func:`_capture_routed`).
    """
    if not _input_is_cuda(ctx.example_input):
        raise CandidateUnavailableError(
            "cuda_graph needs a CUDA example input"
        )
    return _capture_routed(ctx)  # pragma: no cover — CUDA-only


def _build_eager(ctx: AutotuneContext) -> Any:
    """Return the original model — the reference, always verified."""
    return ctx.model


#: Built-in lowering candidates; users may add entries or pass
#: ``(name, builder)`` tuples in ``candidates`` instead.
CANDIDATE_BUILDERS: dict[str, CandidateBuilder] = {
    "generic": _build_generic,
    "batched": _build_batched,
    "compiled": _build_compiled,
    "compiled_generic": _build_compiled_generic,
    "cuda_graph": _build_cuda_graph,
    "eager": _build_eager,
}


def _input_is_cuda(example_input: Any) -> bool:
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    return any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)


def _term_params(term: Any) -> dict[str, Param]:
    """Collect the ``Param`` leaves of a term.

    Interned terms are content-hashed, so a set dedupes shared
    subtrees.
    """
    out: dict[str, Param] = {}
    seen: set[Any] = set()

    def rec(t: Any) -> None:
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Param):
            out.setdefault(t.name, t)
        elif isinstance(t, Op):
            for a in t.args:
                rec(a)

    rec(term)
    return out


def _recover_ir(
    delivered: torch.nn.Module,
) -> tuple[IR, dict[str, torch.Tensor]]:
    """Reconstruct ``(IR, param_values)`` from a delivered module.

    Recover from the module ``optimize_model`` delivered, so the
    SAME extracted term can be re-lowered through another executor
    without re-running the search.

    Every lowering path exposes ``_root`` (the folded extracted
    term), ``_inputs`` and ``_param_map`` — a plain ``IRModule``
    natively, the batched executors through delegated properties —
    so one read covers every route ``_lower_extracted`` takes.
    """
    mod = cast(Any, delivered)
    root = mod._root
    inputs = list(mod._inputs)
    ir = IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=_term_params(root),
    )
    pvals = {
        name: p.detach().clone() for name, p in mod._param_map.items()
    }
    return ir, pvals


def _time_forward(
    mod: Any,
    example_input: Any,
    *,
    n_calls: int,
    warmup: int,
) -> tuple[float, float]:
    """Median and IQR of one forward's wall seconds.

    ``warmup`` untimed calls first (inductor autotune, cache fill);
    every timed call ends in ``cuda.synchronize`` on CUDA inputs so
    the measured time includes the GPU tail.
    """
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    is_cuda = _input_is_cuda(example_input)

    def call() -> None:
        with torch.no_grad():
            mod(*args)

    for _ in range(max(warmup, 0)):
        call()
    if is_cuda:
        torch.cuda.synchronize()  # pragma: no cover — CUDA-only
    times: list[float] = []
    for _ in range(max(n_calls, 1)):
        t0 = time.perf_counter()
        call()
        if is_cuda:
            torch.cuda.synchronize()  # pragma: no cover — CUDA-only
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    if len(times) >= 4:
        q1, _, q3 = statistics.quantiles(times, n=4)
        iqr = q3 - q1
    else:
        iqr = 0.0
    return med, iqr


# ---------------------------------------------------------------------------
# Measured feedback — predicting and recording candidate latency
# ---------------------------------------------------------------------------
#
# The cost model predicts candidate latency so a ``profile=`` run can
# attempt candidates cheapest-first and record residuals for the next
# call.  Model prices are whole-graph delivered estimates in
# nanoseconds, composed from ``catopt_core.cost``'s executor models.

#: Candidate name → the executor-cost lowering that prices it.
#: ``"batched"`` resolves per model to the pipeline's own route
#: (``stats["lowering"]``); ``"eager"`` and custom names get no model
#: price — their ``measured_ns`` entries substitute the measured
#: median outright (``measured_price_ns``).
_CANDIDATE_LOWERING: dict[str, str | None] = {
    "generic": "generic",
    "batched": None,
    "compiled": "compiled",
    "compiled_generic": "compiled",
    "cuda_graph": "compiled",
    "eager": None,
}

#: Probe overhead used by :func:`_fused_charges_graph_overhead` — far
#: above any real value, so a cost model that consumes the field can
#: never price the probe identically.
_PROBE_OVERHEAD_US = 1e6


def _with_graph_overhead(profile: Any, us: float) -> Any | None:
    """Return *profile* with ``graph_overhead_us`` set to *us*.

    ``None`` for ``None`` or profile types that cannot carry the
    field.
    """
    if profile is None:
        return None
    if isinstance(profile, dict):
        return {**profile, "graph_overhead_us": us}
    if is_dataclass(profile) and not isinstance(profile, type):
        try:
            return replace(profile, graph_overhead_us=us)
        except TypeError:  # dataclass without the field
            return None
    try:
        clone: Any = copy.copy(profile)
        clone.graph_overhead_us = us
        return clone
    except Exception:
        return None


def _fused_charges_graph_overhead(profile: Any) -> bool:
    """Check whether :func:`fused_cost_for` consumes the field.

    True when the installed :func:`fused_cost_for` already consumes
    ``graph_overhead_us`` in its per-graph term.

    Behavioural probe — price a one-op term with the profile's
    overhead bumped to :data:`_PROBE_OVERHEAD_US`: a consuming model
    changes the price.  Keeps :func:`_compiled_model_ns` correct on
    both sides of the cost-model adoption boundary (no double-charge
    once ``_fused_cost`` reads the field natively).
    """
    bumped = _with_graph_overhead(profile, _PROBE_OVERHEAD_US)
    if bumped is None:
        return False
    x = Var("_go_x", TensorType((8,)))
    y = Var("_go_y", TensorType((8,)))
    probe = Op.make("add", x, y)
    try:
        return fused_cost_for(bumped)(probe) != fused_cost_for(profile)(
            probe
        )
    except Exception:
        return False


def _compiled_model_ns(term: Any, profile: Any) -> float:
    """Fusion-region price of *term* in ns plus call overhead.

    The compiled per-graph call overhead.

    ``fused_cost_for`` charges ``dispatch_us`` once per graph; the
    profile's ``graph_overhead_us`` (measured guards + inductor
    dispatch) widens that term to ``max(dispatch_s,
    graph_overhead_s)`` — the contract documented on
    :func:`~catopt_optimize.calibrate.profile_graph_overhead_us`.  If
    the installed cost model already consumes the field
    (``_fused_charges_graph_overhead``) the base price carries it and
    nothing is added here.
    """
    base = float(fused_cost_for(profile)(term))
    if _fused_charges_graph_overhead(profile):
        return base
    surplus_s = max(
        profile_graph_overhead_us(profile) * 1e-6
        - _profile_dispatch_s(profile),
        0.0,
    )
    return base + surplus_s * 1e9


def _candidate_model_ns(
    name: str, ctx: AutotuneContext, profile: Any
) -> float | None:
    """Return the cost model's UNCORRECTED delivered price (ns).

    Prices one candidate; ``None`` when the term was not recovered
    or the candidate has no priced lowering (``"eager"``, custom
    names).

    The learned ``corrections`` factors are deliberately applied one
    layer up, where the attempt-order price map is built
    (:func:`~catopt_optimize.calibrate.corrected_price_ns`) — the
    ``model_ns`` recorded in stats and paired with each write-back
    must stay the honest raw model estimate, since it is the
    denominator of the ratios the correction table learns.

    ``"batched"`` prices under the executor the pipeline actually
    routed to; ``"cuda_graph"`` shares the fused model — its real
    per-call overhead differs, which is exactly what the measured
    residual absorbs on write-back.
    """
    if ctx.term is None:
        return None
    if name == "batched":
        lowering: str | None = (
            "batched_scan" if ctx.lowering == "batched" else "generic"
        )
    else:
        lowering = _CANDIDATE_LOWERING.get(name)
    if lowering is None:
        return None
    try:
        if lowering == "compiled":
            return _compiled_model_ns(ctx.term, profile)
        return float(
            executor_cost_for(profile, lowering=lowering)(ctx.term)
        )
    except Exception:  # pricing must never break a measurement run
        return None


def optimize_model_autotuned(
    model: torch.nn.Module,
    example_input: Any,  # tensor or positional-args tuple
    *,
    candidates: Iterable[str | tuple[str, CandidateBuilder]] = (
        "generic",
        "batched",
        "compiled",
    ),
    budget_s: float | None = None,
    n_calls: int = 30,
    warmup: int = 5,
    rtol: float = 1e-4,
    atol: float | None = None,
    source: Source | None = None,
    sink: Sink | None = None,
    profile: Any = None,
    verbose: bool = False,
    **optimize_kwargs: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Optimize ``model``, then autotune over lowering paths.

    Steps:

    1. Run :func:`optimize_model` once (uncompiled, no CUDA graph —
       those are candidates, not presets) to get the extracted term
       and the pipeline's own lowered module.
    2. Re-lower the same extracted term through each requested
       candidate (see :data:`CANDIDATE_BUILDERS`).
    3. ``sink.verify`` each candidate against ``model`` on
       ``example_input`` — failures are recorded and excluded;
       a candidate is never timed unverified.
    4. Time each survivor: ``warmup`` calls + ``n_calls`` timed
       forwards (median + IQR, CUDA-synchronised).
    5. Return the measured-fastest candidate.

    Parameters
    ----------
    model
        The model to optimize, then autotune the lowerings of.
    example_input
        A representative input — a tensor or a positional-args
        tuple.
    candidates
        Names into :data:`CANDIDATE_BUILDERS` and/or
        ``(name, builder)`` tuples.
    budget_s
        Wall-clock budget for the WHOLE call (the search counts
        against it).  Candidates left unstarted when it expires are
        recorded ``status="skipped"``.  ``None`` — no bound.
    n_calls, warmup
        Timing shape: ``warmup`` untimed forwards, then ``n_calls``
        timed ones; the median decides.
    rtol, atol
        The ``sink.verify`` equivalence gate.
    source, sink
        The port adapters — defaults :class:`TorchSource` /
        :class:`TorchSink`.  The same ``sink`` both lowers and
        verifies candidates.
    profile
        Measured-feedback channel (opt-in): a
        :class:`~catopt_optimize.calibrate.TargetProfile`, a dict, or
        any object with a ``measured_ns`` mapping.  When given,
        candidates are attempted cheapest-predicted-first — the model
        price corrected by the profile's learned ``corrections``
        factors and ``measured_ns`` residuals for this input's
        :func:`~catopt_optimize.calibrate.shape_bucket` — and this
        run's timings are written back into the profile
        (:func:`~catopt_optimize.calibrate.record_measured`), so the
        next ``optimize_model_autotuned`` call on this shape prices
        candidates closer to measured.  The updated object is
        returned as ``stats["autotune"]["profile"]`` — dicts update in
        place, a frozen ``TargetProfile`` comes back replaced.  The
        same ``measured_ns``/``graph_overhead_us`` keys are the
        contract a profile-aware selection path (e.g. a
        ``profile=``-accepting ``optimize_model`` /
        ``_delivered_cost``) consumes.
    verbose
        Print progress.
    **optimize_kwargs
        Forwarded to :func:`optimize_model` (``ruleset``,
        ``max_iterations``, ``ops``, ``runner``, …).  Compilation
        and graph capture are candidates here, not presets — pass
        ``runner`` only to decorate the pipeline's own delivered
        module, not as a substitute candidate.

    Returns
    -------
    (module, stats)
        ``module`` is the measured-fastest VERIFIED candidate (the
        plain ``optimize_model`` output when every candidate fails
        — ``stats["autotune"]["fallback"]``).  ``stats`` is the
        ``optimize_model`` stats dict plus ``stats["autotune"]``:
        ``winner``, ``winner_median_s``, per-candidate records
        (``status``/``median_s``/``iqr_s``/``verified``/``max_rel``/
        ``model_ns``/``predicted_ns``/``error``), ``fallback``,
        ``shape_bucket``, ``predicted_ns``/``predicted_winner``,
        ``measured_ns``/``profile`` (only with ``profile=``),
        ``search_s``, ``elapsed_s``.

    """
    t_start = time.monotonic()
    if sink is None:
        sink = cast(Sink, TorchSink(ops=optimize_kwargs.get("ops")))

    # -- (a) the one search -----------------------------------------
    delivered, stats = optimize_model(
        model,
        example_input,
        source=source,
        sink=sink,
        verbose=verbose,
        **optimize_kwargs,
    )
    search_s = time.monotonic() - t_start
    lowering = str(stats.get("lowering", "generic"))
    if verbose:
        logger.info(
            "[Autotune] search done in %.2fs "
            "(lowering=%s); re-lowering candidates...",
            search_s,
            lowering,
        )

    # -- (b) re-lower the same extracted term ------------------------
    try:
        ir, pvals = _recover_ir(delivered)
    except Exception:
        # A custom sink may deliver a non-IRModule executor — lowerer
        # candidates then report unavailable; delivered-reuse and
        # eager candidates still work.
        ir, pvals = None, {}
    ctx = AutotuneContext(
        model=model,
        example_input=example_input,
        ir=ir,
        term=ir.root if ir is not None else None,
        param_values=pvals,
        sink=sink,
        delivered=delivered,
        lowering=lowering,
    )

    records: dict[str, dict[str, Any]] = {}
    built: dict[str, Any] = {}

    def over_budget() -> bool:
        return (
            budget_s is not None
            and time.monotonic() - t_start >= budget_s
        )

    # Normalise entries up front — measured-feedback pricing and the
    # predicted-order sort below need every candidate's name.
    entries: list[tuple[str, CandidateBuilder | None]] = []
    for entry in candidates:
        if isinstance(entry, str):
            entries.append((entry, CANDIDATE_BUILDERS.get(entry)))
        else:
            entries.append(entry)

    # -- measured feedback: price every candidate ----------------------
    # model_ns  — the cost model's uncorrected delivered price;
    # price_map — corrected by the profile's learned corrections
    # factors and measured_ns residuals for THIS input's shape bucket
    # (never global).  Either may be absent for unpriceable candidates
    # (eager, custom names, unrecovered IR).
    bucket = shape_bucket(example_input)
    model_ns: dict[str, float] = {}
    for name, _builder in entries:
        m = _candidate_model_ns(name, ctx, profile)
        if m is not None:
            model_ns[name] = m
    price_map: dict[str, float] = {}
    for name, _builder in entries:
        p = corrected_price_ns(
            profile, name, bucket, model_ns.get(name)
        )
        if p is not None:
            price_map[name] = p
    # A profile opt-in sorts the attempt order cheapest-predicted-first:
    # under a budget_s the likeliest winners get timed before the
    # budget expires.  Unpriced candidates keep their declared order
    # after the priced ones.  Without profile= the declared order is
    # untouched.
    if profile is not None and price_map:
        entries.sort(
            key=lambda e: (
                0 if e[0] in price_map else 1,
                price_map.get(e[0], 0.0),
            )
        )

    for name, builder in entries:
        if builder is None:
            records[name] = {
                "status": "unknown",
                "error": f"no candidate {name!r}; "
                f"known: {sorted(CANDIDATE_BUILDERS)}",
            }
            continue
        rec: dict[str, Any] = {}
        records[name] = rec
        if name in model_ns:
            rec["model_ns"] = model_ns[name]
        if name in price_map:
            rec["predicted_ns"] = price_map[name]
        if over_budget():
            rec["status"] = "skipped"
            rec["reason"] = "budget_s exhausted"
            continue
        try:
            mod = builder(ctx)
        except CandidateUnavailableError as e:
            rec["status"] = "unavailable"
            rec["reason"] = str(e)
            continue
        except Exception as e:
            rec["status"] = "build_failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            continue

        # -- (c) verify BEFORE timing: an unverified candidate is
        #        never timed, let alone returned. -----------------
        try:
            vr = sink.verify(
                model, mod, example_input, rtol=rtol, atol=atol
            )
        except Exception as e:
            rec["status"] = "verify_error"
            rec["error"] = f"{type(e).__name__}: {e}"
            continue
        rec["verified"] = vr.passed
        rec["max_abs"] = vr.max_abs
        rec["max_rel"] = vr.max_rel
        if not vr.passed:
            rec["status"] = "verify_failed"
            continue

        # -- (d) time ----------------------------------------------
        try:
            med, iqr = _time_forward(
                mod, example_input, n_calls=n_calls, warmup=warmup
            )
        except Exception as e:
            rec["status"] = "time_failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            continue
        rec["status"] = "timed"
        rec["median_s"] = med
        rec["iqr_s"] = iqr
        rec["n_calls"] = n_calls
        built[name] = mod
        if verbose:
            logger.info(
                "[Autotune] %s: %.3f ms (verified, rel=%.2e)",
                name,
                med * 1e3,
                vr.max_rel,
            )

    # -- (e) pick the measured winner ------------------------------
    timed = {
        n: r for n, r in records.items() if r.get("status") == "timed"
    }
    if timed:
        winner = min(timed, key=lambda n: timed[n]["median_s"])
        module = cast(torch.nn.Module, built[winner])
        fallback = False
    else:
        # "falls back to generic": the unmodified optimize_model
        # output.  Verify it too — the fallback record must not
        # claim an unchecked win either.
        winner = None
        module = delivered
        fallback = True
        try:
            vr = sink.verify(
                model,
                delivered,
                example_input,
                rtol=rtol,
                atol=atol,
            )
            fallback_ok = bool(vr.passed)
        except Exception:
            fallback_ok = False
        records["_pipeline_fallback"] = {
            "status": "fallback",
            "verified": fallback_ok,
        }

    # -- measured feedback: persist this run's timings -----------------
    # Opt-in via profile=: every timed candidate's median joins the
    # profile's measured_ns map under (candidate, shape_bucket), paired
    # with the model price that failed to predict it — the residual is
    # what a corrected price transfers (measured_price_ns).
    written: dict[str, dict[str, Any]] = {}
    if profile is not None:
        for name, r in timed.items():
            med_ns = float(r["median_s"]) * 1e9
            mns = model_ns.get(name)
            profile = record_measured(
                profile, name, bucket, med_ns, mns
            )
            w: dict[str, Any] = {"bucket": bucket, "median_ns": med_ns}
            if mns is not None:
                w["model_ns"] = mns
                w["residual_ns"] = med_ns - mns
            written[name] = w

    predicted_winner = (
        min(price_map, key=lambda n: price_map[n])
        if price_map
        else None
    )

    if verbose:
        logger.info(
            "[Autotune] winner: %s (%d/%d candidates timed)",
            winner or "pipeline fallback",
            len(timed),
            len(records),
        )

    stats["autotune"] = {
        "winner": winner,
        "winner_median_s": (
            timed[winner]["median_s"] if winner is not None else None
        ),
        "predicted_winner": predicted_winner,
        "predicted_ns": price_map or None,
        "shape_bucket": bucket,
        "candidates": records,
        "fallback": fallback,
        "ir_recovered": ir is not None,
        "n_calls": n_calls,
        "warmup": warmup,
        "rtol": rtol,
        "budget_s": budget_s,
        "search_s": search_s,
        "elapsed_s": time.monotonic() - t_start,
    }
    if profile is not None:
        stats["autotune"]["measured_ns"] = written
        stats["autotune"]["profile"] = profile
    return module, stats
