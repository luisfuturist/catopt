"""Measured autotuning over the backend's lowering paths (plan 0007).

The pipeline commits to a single lowering up front: the extracted
term routes through the sink's executor table — a level-batched
carrier executor when the root plans one, the generic evaluator
otherwise — and a delivery runner may wrap it.  Which route is
actually fastest on a given device is an empirical question the
static cost model can only guess at (per-node eval dispatch vs.
batched compose levels vs. compiler fusion reorder differently
across backends and devices).

The :class:`~catopt_orchestrator.optimize.Autotuned` strategy runs the
e-graph search ONCE, then re-lowers the same extracted term through
each requested lowering path, verifies every candidate against the
model through ``sink.verify`` (an unverified candidate is never
timed, let alone returned), measures each survivor through the
:class:`~catopt_core.ports.Meter` port, and returns the measured
winner.

Honesty contract:

* every timed candidate passed ``sink.verify`` on ``example_input``;
* every candidate's outcome — build failure, verify failure, timing,
  budget skip — is recorded in ``stats["autotune"]["candidates"]``;
* if every candidate fails, the pipeline's own delivered module is
  returned and ``stats["autotune"]["fallback"]`` is True.

Built-in candidate names (see :data:`CANDIDATE_BUILDERS`):

``generic``
    The sink's plain lowering — ``sink.lower`` directly, no carrier
    routing (the executor ``lowering="generic"`` prices).
``batched``
    The pipeline's executor routing — ``_lower_extracted`` against
    ``sink.executors``: the first carrier entry whose ``accepts``
    probe holds, the generic lowering otherwise.  This is exactly
    what the pipeline delivers.
``eager``
    The original model, unchanged.  Always verifies (it is the
    reference) — when nothing beats it, it wins honestly.
``<executor name>``
    Every name in ``sink.executors`` resolves to that
    :class:`~catopt_core.ports.ExecutorSpec` — ``spec.lower`` on a
    fresh lowering of the extracted term (``"scan"`` /
    ``"om_batched"`` / ``"omd_batched"`` / ``"om_streaming"`` /
    ``"trace"`` for the torch backend).

Backend-provided builders — ``torch.compile`` / CUDA-graph paths —
arrive through the ``builders`` map (the torch wrapper supplies
``catopt_torch.autotune.TORCH_BUILDERS``: ``"torch_compile"``,
``"torch_compile_generic"``, ``"cuda_graph"``).  Custom candidates:
a ``candidates`` entry may be a ``(name, builder)`` tuple where
``builder`` is a :data:`CandidateBuilder` callable receiving the
:class:`AutotuneContext` and returning a runnable.

Measured feedback (opt-in)
--------------------------

Every timed candidate's median latency would otherwise be thrown away
after picking the winner.  Pass a profile to ``profile=`` — a
:class:`~catopt_core.profile.TargetProfile` or a plain dict —
and two things happen:

* candidates are *attempted* cheapest-predicted-first, where the
  prediction is the cost model's delivered price corrected by the
  profile's learned ``corrections`` factors (once a
  (candidate, bucket) pair has enough observations) and
  ``measured_ns`` residuals for this input's
  :func:`~catopt_core.profile.shape_bucket`
  (see :func:`~catopt_core.profile.corrected_price_ns`); and
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
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, cast

from catopt_core.cost import (
    _profile_dispatch_s,
    executor_cost_for,
    fused_cost_for,
)
from catopt_core.failures import FailureClass, classify
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.ports import (
    ExecutorSpec,
    Sink,
    Source,
)
from catopt_core.profile import (
    corrected_price_ns,
    profile_graph_overhead_us,
    record_measured,
    shape_bucket,
)

from catopt_orchestrator.optimize import Optimizer, _lower_extracted

logger = logging.getLogger("catopt_orchestrator.autotune")

__all__ = [
    "CANDIDATE_BUILDERS",
    "AutotuneContext",
    "CandidateBuilder",
    "CandidateUnavailableError",
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
    ``delivered`` is the module the pipeline itself produced
    (``lowering`` is its ``stats["lowering"]``) — a builder may
    reuse it directly when it is already the right executor.

    ``ir`` is ``None`` only when the delivered module does not
    expose the IRModule internals (a custom ``sink``'s executor) —
    lowerers then report ``unavailable``; ``delivered``-reuse and
    ``eager`` candidates still work.
    """

    model: Any
    example_input: Any  # tensor or args tuple
    ir: IR | None
    term: Any
    param_values: dict[str, Any]
    sink: Sink
    delivered: Any
    lowering: str


#: ``builder(ctx) -> runnable``; raising
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
    """Build the serial executor via ``sink.lower``.

    Reuse ``delivered`` when the pipeline itself routed to the
    generic executor.
    """
    if ctx.lowering == "generic":
        return ctx.delivered
    return ctx.sink.lower(_require_ir(ctx), ctx.param_values)


def _build_batched(ctx: AutotuneContext) -> Any:
    """Build the sink's executor-routed lowering.

    Reuse ``delivered`` when it already is the routed executor.
    """
    if ctx.lowering == "batched":
        return ctx.delivered
    return _lower_extracted(
        ctx.term, _require_ir(ctx), ctx.param_values, ctx.sink
    )


def _build_eager(ctx: AutotuneContext) -> Any:
    """Return the original model — the reference, always verified."""
    return ctx.model


#: Built-in backend-neutral lowering candidates; backend-provided
#: names arrive through the ``builders`` map, and ``<executor name>``
#: resolves against ``sink.executors``.  Users may add entries or
#: pass ``(name, builder)`` tuples in ``candidates`` instead.
CANDIDATE_BUILDERS: dict[str, CandidateBuilder] = {
    "generic": _build_generic,
    "batched": _build_batched,
    "eager": _build_eager,
}


def _executor_builder(spec: ExecutorSpec) -> CandidateBuilder:
    """Turn a sink executor entry into a candidate builder.

    ``spec.lower`` runs on a FRESH lowering of the extracted term
    (``ctx.ir``) — never ``ctx.delivered``, which the ``batched``
    candidate may already own.
    """

    def build(ctx: AutotuneContext) -> Any:
        return spec.lower(_require_ir(ctx), ctx.param_values)

    return build


def _resolve_builder(
    name: str,
    sink: Sink,
    builders: dict[str, CandidateBuilder] | None,
) -> CandidateBuilder | None:
    """Resolve a candidate name — built-in, backend, executor-table.

    Order: the backend-provided ``builders`` map first (backend
    candidates shadow neutral names deliberately — e.g. a backend
    override of ``"batched"``), then :data:`CANDIDATE_BUILDERS`,
    then ``sink.executors`` entries.
    """
    if builders is not None and name in builders:
        return builders[name]
    b = CANDIDATE_BUILDERS.get(name)
    if b is not None:
        return b
    table = getattr(sink, "executors", None) or {}
    spec = table.get(name)
    if spec is not None:
        return _executor_builder(spec)
    return None


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
    delivered: Any,
) -> tuple[IR, dict[str, Any]]:
    """Reconstruct ``(IR, param_values)`` from a delivered module.

    Recover from the module the pipeline delivered, so the
    SAME extracted term can be re-lowered through another executor
    without re-running the search.

    Every lowering path exposes ``_root`` (the folded extracted
    term), ``_inputs`` and ``_param_map`` — a plain ``IRModule``
    natively, the batched executors through delegated properties —
    so one read covers every route ``_lower_extracted`` takes.
    """

    def clone(p: Any) -> Any:
        det = getattr(p, "detach", None)
        if callable(det):
            p = det()
        cl = getattr(p, "clone", None)
        return cl() if callable(cl) else copy.copy(p)

    root = delivered._root
    inputs = list(delivered._inputs)
    ir = IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=_term_params(root),
    )
    pvals = {name: clone(p) for name, p in delivered._param_map.items()}
    return ir, pvals


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
#: (``stats["lowering"]``); ``"eager"``, executor-table names and
#: custom names get no model price — their ``measured_ns`` entries
#: substitute the measured median outright (``measured_price_ns``).
_CANDIDATE_LOWERING: dict[str, str | None] = {
    "generic": "generic",
    "batched": None,
    "torch_compile": "compiled",
    "torch_compile_generic": "compiled",
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
    :func:`~catopt_core.profile.profile_graph_overhead_us`.  If
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
    names, executor-table entries — the model prices delivered
    routes, not names).

    The learned ``corrections`` factors are deliberately applied one
    layer up, where the attempt-order price map is built
    (:func:`~catopt_core.profile.corrected_price_ns`) — the
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


def _autotuned_impl(
    model: Any,
    example_input: Any,  # tensor or positional-args tuple
    *,
    candidates: Iterable[str | tuple[str, CandidateBuilder]] = (
        "generic",
        "batched",
        "torch_compile",
    ),
    builders: dict[str, CandidateBuilder] | None = None,
    budget_s: float | None = None,
    n_calls: int = 30,
    warmup: int = 5,
    rtol: float = 1e-4,
    atol: float | None = None,
    optimizer: Optimizer,
    profile: Any = None,
    verbose: bool = False,
    **optimize_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Optimize ``model``, then autotune over lowering paths.

    The engine behind the :class:`~catopt_orchestrator.optimize.Autotuned`
    strategy (plan 0006/0007).  Steps:

    1. Run the monolithic pipeline once (uncompiled, no graph
       capture — those are candidates, not presets) to get the
       extracted term and the pipeline's own lowered module.
    2. Re-lower the same extracted term through each requested
       candidate (see :data:`CANDIDATE_BUILDERS`, ``builders``, and
       ``sink.executors``).
    3. ``sink.verify`` each candidate against ``model`` on
       ``example_input`` — failures are recorded and excluded;
       a candidate is never timed unverified.
    4. Time each survivor through the ``meter`` port: ``warmup``
       calls + ``n_calls`` timed forwards (median + IQR, device
       synchronisation is the meter's business).
    5. Return the measured-fastest candidate.

    Parameters
    ----------
    model
        The model to optimize, then autotune the lowerings of.
    example_input
        A representative input — a tensor or a positional-args
        tuple.
    candidates
        Names resolvable by :func:`_resolve_builder` and/or
        ``(name, builder)`` tuples.
    builders
        Backend-provided candidate builders — names the neutral
        table does not know (the torch wrapper supplies
        ``TORCH_BUILDERS``).  Merged over :data:`CANDIDATE_BUILDERS`.
    budget_s
        Wall-clock budget for the WHOLE call (the search counts
        against it).  Candidates left unstarted when it expires are
        recorded ``status="skipped"``.  ``None`` — no bound.
    n_calls, warmup
        Timing shape: ``warmup`` untimed forwards, then ``n_calls``
        timed ones; the median decides.
    rtol, atol
        The ``sink.verify`` equivalence gate.
    optimizer
        The configured orchestrator — its ``source``/``sink`` run the
        one search, its ``meter`` times the survivors, its
        ``criteria``/``runner`` defaults apply to the search and the
        pipeline delivery.  Required — there is no assumed backend.
    profile
        Measured-feedback channel (opt-in): a
        :class:`~catopt_core.profile.TargetProfile`, a dict, or
        any object with a ``measured_ns`` mapping.  When given,
        candidates are attempted cheapest-predicted-first — the model
        price corrected by the profile's learned ``corrections``
        factors and ``measured_ns`` residuals for this input's
        :func:`~catopt_core.profile.shape_bucket` — and this
        run's timings are written back into the profile
        (:func:`~catopt_core.profile.record_measured`), so the
        next autotuned call on this shape prices
        candidates closer to measured.  The updated object is
        returned as ``stats["autotune"]["profile"]`` — dicts update in
        place, a frozen ``TargetProfile`` comes back replaced.  The
        same ``measured_ns``/``graph_overhead_us`` keys are the
        contract a profile-aware selection path (e.g. a
        ``profile=``-accepting delivered-cost path) consumes.
    verbose
        Print progress.
    **optimize_kwargs
        Forwarded to the underlying optimize call (``rules``,
        ``max_iterations``, …).  Compilation and graph capture are
        candidates here, not presets — ``runner`` only decorates the
        pipeline's own delivered module, not a substitute candidate.

    Returns
    -------
    (module, stats)
        ``module`` is the measured-fastest VERIFIED candidate (the
        pipeline's own delivery when every candidate fails —
        ``stats["autotune"]["fallback"]``).  ``stats`` is the
        pipeline stats dict plus ``stats["autotune"]``:
        ``winner``, ``winner_median_s``, per-candidate records
        (``status``/``median_s``/``iqr_s``/``verified``/``max_rel``/
        ``model_ns``/``predicted_ns``/``error``/``failure``),
        ``fallback``, ``shape_bucket``,
        ``predicted_ns``/``predicted_winner``,
        ``measured_ns``/``profile`` (only with ``profile=``),
        ``search_s``, ``elapsed_s``.

        A failed candidate records its :class:`FailureClass` under
        ``failure`` (plan 0016 stage 3) alongside the stage-only
        ``status`` — a ``build_failed``/``verify_error``/``time_failed``
        record names *why* (OOM / timeout / kernel / unavailable / …)
        rather than only *where* it stopped.

    """
    t_start = time.monotonic()

    sink = optimizer.sink
    meter = optimizer.meter
    if meter is None:
        raise TypeError(
            "Autotuned strategy needs a Meter port — pass meter= "
            "(or backend=) to Optimizer"
        )
    # ``Optimizer.__post_init__`` already rejects a missing source/sink,
    # so by construction both ports are present here.
    source = optimizer.source
    sink = cast(Sink, optimizer.sink)
    # Runner-style kwargs belong to the optimizer, not the phase
    # verbs — pull them out like optimize_model did.
    opt_kw = dict(optimize_kwargs)
    runner = opt_kw.pop("runner", None)
    criteria = opt_kw.pop("criteria", None)
    opt_kw.pop("ops", None)  # sink already resolved — ops is moot

    # -- (a) the one search -----------------------------------------
    lr = Optimizer(
        source=cast(Source, source),
        sink=sink,
        composer=optimizer.composer,
        meter=meter,
        criteria=criteria,
        runner=(runner if runner is not None else optimizer.runner),
    ).optimize(
        model, example_input, verify=verbose, verbose=verbose, **opt_kw
    )
    delivered = lr.module
    stats = lr.stats
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
    known = (
        sorted(set(CANDIDATE_BUILDERS) | set(builders or {}))
        if builders
        else sorted(CANDIDATE_BUILDERS)
    )
    for entry in candidates:
        if isinstance(entry, str):
            entries.append(
                (entry, _resolve_builder(entry, sink, builders))
            )
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
                "error": f"no candidate {name!r}; known: {known}",
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
            rec["failure"] = classify(e)
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
            rec["failure"] = classify(e)
            rec["error"] = f"{type(e).__name__}: {e}"
            continue
        rec["verified"] = vr.passed
        rec["max_abs"] = vr.max_abs
        rec["max_rel"] = vr.max_rel
        if not vr.passed:
            rec["status"] = "verify_failed"
            continue

        # -- (d) time through the meter port -------------------------
        # A stage-only status hides WHY a run failed: the meter's own
        # classified ``failure`` (OOM / timeout / kernel / …) and any
        # exception the meter itself raises both land in ``failure``,
        # so OOM vs timeout vs device-absent stay distinguishable.
        try:
            timing = meter.time(
                mod, example_input, n_calls=n_calls, warmup=warmup
            )
        except Exception as e:
            rec["status"] = "time_failed"
            rec["failure"] = classify(e)
            rec["error"] = f"{type(e).__name__}: {e}"
            continue
        if timing.failure != FailureClass.OK:
            rec["status"] = "time_failed"
            rec["failure"] = timing.failure
            continue
        med, iqr = timing.median_s, timing.iqr_s
        rec["status"] = "timed"
        rec["median_s"] = med
        rec["iqr_s"] = iqr
        rec["n_calls"] = timing.n_calls
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
        module = built[winner]
        fallback = False
    else:
        # "falls back to generic": the pipeline's own delivered
        # module.  Verify it too — the fallback record must not
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
