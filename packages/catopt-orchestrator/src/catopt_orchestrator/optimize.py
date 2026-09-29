"""Top-level optimization pipeline — backend-neutral (plan 0007).

This module implements the four-phase killer experiment:

Phase 1 — Equivalence:   model → IR (through the ``source`` port)
Phase 2 — Search:        IR → e-graph → equality saturation → best term
Phase 3 — Lower:          best term → runnable (through the ``sink``
                        port's executor routing)
Phase 4 — Compare:        benchmark vs. vanilla TorchInductor

The public surface is the *verb* pair plus the configured entry
object (plan 0006), and the orchestrator is backend-neutral
(plan 0007): it orchestrates ANY backend through the
:class:`catopt_core.ports` protocols and imports no torch.

* :func:`search` — phases 1+2: ``model -> SearchResult`` (the IR,
  saturated e-graph, extracted term, leaf values and search-record
  stats, all inspectable).
* :func:`lower` — phase 3 (+verify): ``SearchResult -> LowerResult``;
  re-lowering the same result under different runners delivers
  different executables from ONE search.
* :class:`Optimizer` — the configured entry point: an explicit
  :class:`~catopt_core.pipeline.Backend` (or explicit
  ``source``/``sink``/``composer``/``meter`` ports — no default
  backend) plus ``criteria``/``runner`` delivery defaults;
  ``.search`` / ``.lower`` / ``.optimize`` / ``.discover``.
* :class:`Monolithic` / :class:`Compositional` / :class:`Autotuned` —
  the :class:`~catopt_core.ports.Strategy` seam behind
  ``Optimizer.optimize(..., strategy=...)``.

Backend specifics live on the adapter side: carrier executors arrive
through ``sink.executors``, causal-mask specialization through the
sink's optional ``specialize_causal`` hook, per-block structural
machinery through the ``composer`` port, and wall-clock timing
through the ``meter`` port.  The deprecated ``optimize_*`` wrappers
with their torch defaults moved to ``catopt_torch.api`` (resolving
lazily here for compatibility); ``Optimizer(backend=TorchBackend())``
is the supported torch spelling — no default is assumed anywhere in
this package.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from catopt_core.cost import (
    backend_cost,
    dag_cost,
    executor_cost_for,
    flops_cost,
)
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, op_repr
from catopt_core.laws import (
    ALL_RULES_WITH_LAYOUT,
    CATEGORICAL_RULES,
    SIMPLIFICATION_RULES,
    all_rules,
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from catopt_core.pipeline import Backend, LowerResult, SearchResult
from catopt_core.ports import (
    Capabilities,
    Composer,
    CostFn,
    ExecutorSpec,
    Meter,
    Runner,
    Sink,
    Source,
    Strategy,
)

from catopt_optimize.criteria import (
    Criteria,
    Criterion,
    criteria_cost,
)
from catopt_optimize.runners import IdentityRunner

#: Rules whose saturation closure is combinatorially explosive on
#: stacked blocks: the pure-symmetry monoid laws enumerate every
#: bracketing/ordering of a summation (Catalan-scale on the residual
#: accumulator), the scale-hoist laws pair every scale member with
#: every linear, and the distribute/factor/naturality/assoc algebra
#: generates cross-product closures (distribute splits a sum into two
#: matmuls that factor rules then re-pair against *every other*
#: summand — enodes grew 337 → 40k in four iterations on a
#: DeepParallel stack).  The pipeline runs these under a
#: per-rule enode budget — *bounded saturation* — which truncates the
#: reordering closure but leaves every content-bearing rewrite at the
#: exact fixed point.  Structural fusions (qkv/swiglu/sdpa folds,
#: gqa_absorb) and the simplification singletons stay unbudgeted:
#: their matches are pattern-specific, not closure-generating.
#: Measured on stacked ParallelBlocks (the model that motivated
#: ``optimize_compositional``): identical extracted cost at every
#: budget ≥ 512 while saturation drops from minutes to ~1s.
logger = logging.getLogger("catopt_optimize.optimize")

_EXPANSIVE_RULES = frozenset(
    {
        # monoid symmetries
        "comm_add",
        "comm_mul",
        "assoc_add",
        "assoc_mul",
        # diagonal-scale naturality (norm folding)
        "linear_row_scale",
        "linear_row_scale_rev",
        "linear_channel_scale",
        "linear_channel_scale_rev",
        # bilinearity: distribute / factor pairs (both directions)
        "distribute_matmul_over_add",
        "factor_matmul",
        "right_distribute_matmul",
        "right_factor_matmul",
        "weight_factor_matmul",
        "weight_distribute_matmul",
        "weight_factor_linear",
        "weight_distribute_linear",
        "right_factor_linear",
        # composition chains / scalar naturality
        "assoc_linear",
        "assoc_linear_bias",
        "assoc_linear_bias_rev",
        "naturality_scalar",
        "naturality_scalar_rev",
        "assoc_matmul",
        "assoc_matmul_rev",
    }
)


class OptimizationResourceError(RuntimeError):
    """Raised when the optimizer crosses a resource bound.

    Sources: the e-graph reached ``max_enodes`` (checked once per
    saturation iteration inside ``EGraph.run`` and again at phase
    boundaries here), the process/device memory footprint crossed
    ``max_memory_mb``, or an allocator-failure exception — host
    ``MemoryError``, the backend's ``OutOfMemoryError`` (torch's
    ``torch.cuda.OutOfMemoryError`` is one), or the RuntimeError
    variants allocator failures surface as — anywhere in the
    export → saturation → lowering pipeline.

    ``optimize_compositional`` records these as ordinary per-block
    failures with ``reason == "resource_limit"``; a standalone
    ``optimize_model`` caller gets this dedicated type instead of a
    raw OOM.
    """


def _looks_like_oom(exc: BaseException) -> bool:
    """Return True for allocator-failure exceptions — backend-agnostic.

    Host ``MemoryError``, a backend-named ``*OutOfMemoryError*`` class
    (``torch.cuda.OutOfMemoryError`` and any future backend's
    equivalent), and the ``RuntimeError`` variants allocator failures
    surface as ("CUDA out of memory", DefaultCPUAllocator's
    "can't allocate memory").
    """
    if isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError)
        and "outofmemoryerror" in type(exc).__name__.lower()
    ):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "can't allocate memory" in msg
            or "cannot allocate memory" in msg
        )
    return False


def _oom_to_resource_error(fn):
    """Wrap an optimizer entry point against allocator failures.

    They surface as :class:`OptimizationResourceError` instead of a raw
    OOM.  ``functools.wraps`` keeps the public signature and docstring.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OptimizationResourceError:
            raise
        except Exception as e:
            if _looks_like_oom(e):
                raise OptimizationResourceError(
                    f"{type(e).__name__}: {e}"
                ) from e
            raise

    return wrapper


def _current_memory_mb(meter: Any = None) -> float:
    """Return the current process memory footprint in MiB.

    Host RSS (read from ``/proc/self/status``, backend-agnostic) plus
    backend-device bytes the adapter reports — the meter's optional
    ``device_memory_mb()`` hook (torch's implementation reads
    ``torch.cuda.memory_allocated``; device memory lives outside RSS).
    A meter without the hook contributes zero.
    """
    rss = 0.0
    try:
        with open("/proc/self/status") as fh:
            for line in (
                fh
            ):  # pragma: no branch — VmRSS always present on Linux
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
                    break
    except OSError:  # pragma: no cover — non-Linux only
        import resource

        rss = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
    dev_fn = getattr(meter, "device_memory_mb", None)
    dev = float(dev_fn()) if callable(dev_fn) else 0.0
    return rss + dev


def _check_resources(
    eg, max_enodes, max_memory_mb, meter: Any = None
) -> None:
    """Cheap watermark check, called at phase boundaries.

    ``max_enodes`` is already enforced inside ``EGraph.run`` once per
    saturation iteration — and mid-iteration for budgeted rules via the
    match-loop ``enode_budget`` — so a crossing observed here means the
    run truncated against the cap: fail fast rather than spending
    extraction/lowering effort on an over-budget graph.  The memory
    check catches tensor pressure an e-node count cannot see
    (materialised weight folds, lowered parameters).
    """
    if (
        max_enodes is not None
        and eg is not None
        and eg.n_enodes >= max_enodes
    ):
        raise OptimizationResourceError(
            f"e-graph reached {eg.n_enodes} e-nodes "
            f"(max_enodes={max_enodes})"
        )
    if max_memory_mb is not None:
        used = _current_memory_mb(meter)
        if used > max_memory_mb:
            raise OptimizationResourceError(
                f"memory footprint {used:.0f} MiB exceeds "
                f"max_memory_mb={max_memory_mb}"
            )


def _default_cost_fn() -> CostFn:
    """Return the default extraction model.

    Roofline pricing plus the executor overhead of the lowering this
    pipeline delivers — ``"generic"`` per-node eval or
    Roofline + the additive generic-dispatch overhead (solver ops
    surcharged — the fidelity sweep caught a ``trace`` resolvent priced
    ~free hiding a 14.5 s ``linalg.solve``).  The level-batched carrier
    executors are *not* priced into selection: their cost is a
    whole-spine property, which ``extract_best``'s additive local-cost
    decomposition can't express — instead the lowering routes apply
    roots to them after extraction (``stats["lowering"]``), so a
    selected carrier member gets its fast executor for free.
    """
    return executor_cost_for(lowering="generic")


def _carrier_plans() -> dict[str, Callable]:
    """Carrier-apply root ops → (batched plan builder) — deferred.

    A term rooted at one of these lowers through the level-batched
    executor.  The builders are carrier-package machinery (torch
    executors); they resolve at call time so the orchestrator never
    imports a backend, and a partial install simply yields an empty
    map (no carrier upgrades).
    """
    try:
        from catopt_carriers.om_lower import build_om_plan
        from catopt_carriers.omd_lower import build_omd_plan
        from catopt_carriers.scan_lower import build_scan_plan
    except ModuleNotFoundError:
        return {}
    return {
        "apply": build_scan_plan,
        "applyd": build_scan_plan,
        "om_apply": build_om_plan,
        "omd_apply": build_omd_plan,
        "omd_applym": build_omd_plan,
    }


def _delivered_cost(
    term: Any, profile: Any = None, compiled: bool = False
) -> float:
    """Price a term under the lowering it would actually get.

    The batched carrier executor for plannable apply roots, generic
    eval otherwise (solver ops surcharged).  With ``compiled=True``
    the delivered module is torch.compile-wrapped whichever route
    ran, so the fusion-region model prices every term.
    """
    if compiled:
        return executor_cost_for(profile, lowering="compiled")(term)
    if isinstance(term, Op) and term.op in _carrier_plans():
        plan = _carrier_plans()[term.op](term)
        if plan is not None:
            return executor_cost_for(profile, lowering="batched_scan")(
                term
            )
    return executor_cost_for(profile, lowering="generic")(term)


def _carrier_upgrade(
    eg: Any,
    root_eid: int,
    best_term: Any,
    cost_fn: CostFn,
    profile: Any = None,
    compiled: bool = False,
) -> Any:
    """Coordinated carrier selection.

    ``extract_best``'s additive local-cost decomposition can't price
    the batched-scan win (a whole-spine property: log(T) batched
    levels vs T serial leaves), so apply-rooted members lose to terms
    that merely count cheaper — even though they run ~6x faster when
    lowered.  Re-examine the root eclass: force-extract each
    carrier-apply enode and compare *delivered* prices — each term
    billed under the executor it would route to.  Swap only when the
    carrier member is cheaper AND its batched plan exists (else the
    module degrades to serial eval and the price lied).
    """
    plans = _carrier_plans()
    cid = eg.find(root_eid)
    carriers = [n for n in eg._classes[cid].nodes if n.op in plans]
    if not carriers:
        return best_term
    best_price = _delivered_cost(best_term, profile, compiled)
    for node in carriers:
        cand = eg.extract_best(root_eid, cost_fn, overrides={cid: node})
        if cand is None:
            continue
        price = _delivered_cost(cand, profile, compiled)
        if price < best_price:
            best_term, best_price = cand, price
    return best_term


def _lower_extracted(
    best_term: Any,
    optimized_ir: Any,
    source_tensors: dict | None,
    sink: Sink,
) -> Any:
    """Route the extracted term through the sink's executor table.

    Executor routing: term-level cost is blind to the lowering — a
    carrier-apply term evaluated by the generic evaluator runs its
    leaves one-by-one (~6-30x slower than the level-batched schedule
    it was priced for).  Route the term to the first *carrier*
    :class:`~catopt_core.ports.ExecutorSpec` of
    :attr:`~catopt_core.ports.Sink.executors` whose ``accepts`` probe
    holds (mapping order is routing order); carrier-agnostic sinks
    and non-carrier terms fall through to ``sink.lower`` — the
    backend-neutral default.  A declined carrier plan degrades to the
    generic path inside each builder rather than failing.
    """
    spec = _route_spec(best_term, sink)
    if spec is not None:
        return spec.lower(optimized_ir, source_tensors)
    return sink.lower(optimized_ir, source_tensors)


def _route_spec(best_term: Any, sink: Sink) -> ExecutorSpec | None:
    """First carrier executor spec accepting *best_term*, else None."""
    table = getattr(sink, "executors", None) or {}
    return next(
        (
            s
            for s in table.values()
            if s.carrier is not None and s.accepts(best_term)
        ),
        None,
    )


# ---------------------------------------------------------------------------
#  The verbs — search (phases 1+2) and lower (phase 3 + verify)
# ---------------------------------------------------------------------------


def _resolve_cost_fn(
    cost_fn: CostFn | None,
    criteria: Any,
    capabilities: Capabilities | None,
) -> tuple[CostFn, Any]:
    """Selection-model precedence: ``cost_fn`` > ``criteria`` > default.

    Returns ``(cost_fn, criteria_used)`` — the normalised-blend marker
    ``criteria_cost`` sets is read BEFORE the ``backend_cost`` wrap,
    which propagates only the billing markers.  With ``capabilities``
    given, members using an op the backend cannot lower price at
    ``+inf``, so extraction never commits to one; omitted, pricing is
    backend-agnostic.
    """
    if cost_fn is None:
        cost_fn = (
            criteria_cost(criteria)
            if criteria is not None
            else _default_cost_fn()
        )
    criteria_used = getattr(cost_fn, "criteria", None)
    if capabilities is not None:
        cost_fn = backend_cost(cost_fn, capabilities.supported_ops)
    return cost_fn, criteria_used


#: Term-local fusion rules the pipeline SUBSUMES with the non-local
#: ``pair_shared_input_*`` pass (it needs no consumer pattern, and
#: keeping them would let extraction pick consumer-level chunk
#: alternatives that bypass the globally-coordinated split choice).
_SUBSUMED = frozenset(
    {"swiglu_fuse", "parallel_mul_fuse", "qkv_fuse", "qkv_fuse_asym"}
)


def _ruleset_rules(ruleset: str) -> list:
    """Map a ruleset name to its rewrite list.

    ``"all"`` / ``"all+layout"`` / ``"simpl"`` / ``"categorical"`` —
    every set but ``"simpl"`` drops :data:`_SUBSUMED`.
    """
    if ruleset == "all":
        return [r for r in all_rules() if r.name not in _SUBSUMED]
    if ruleset == "all+layout":
        return [
            r for r in ALL_RULES_WITH_LAYOUT if r.name not in _SUBSUMED
        ]
    if ruleset == "simpl":
        return SIMPLIFICATION_RULES
    if ruleset == "categorical":
        return [r for r in CATEGORICAL_RULES if r.name not in _SUBSUMED]
    raise ValueError(f"Unknown ruleset: {ruleset}")


def _pairing_and_lifts(
    eg: EGraph,
    rules: list,
    root_eid: int,
    stats: dict[str, Any],
    run_cap: int,
    rule_budgets: dict[str, int] | None,
    max_enodes: int | None,
    max_memory_mb: float | None,
    source_tensors: dict,
    meter: Any = None,
) -> list:
    """Non-local passes with a brief re-saturation between them.

    The diagram-level product law pairs every linear sharing an input
    into one GEMM + split views (no consumer pattern needed); then the
    non-local lifts — unrolled recurrences -> ``trace(F)``, stacks of
    same-state carrier applications -> one application, whole om trees
    over scanned values -> the deferred omd carrier, and exact weight
    tying (duplicate Param leaves share one class).  All witnessed so
    certificates stay replayable.

    Returns the pairing-groups list — the coordinated (paired)
    extraction in :func:`_select_best_term` needs it.
    """
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    if groups:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb, meter)
        stats["pairing_groups"] = len(groups)
        # brief second saturation so other rules see the new enodes
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb, meter)

    lifts = _carrier_lifts(eg, source_tensors)
    if lifts:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb, meter)
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb, meter)
    return groups


def _carrier_lifts(eg: EGraph, source_tensors: dict) -> list:
    """Run the non-local carrier/tying lifts, carriers lazily resolved.

    ``catopt_carriers`` machinery (the carrier lifts) resolves at call
    time; the weight-tying lifts are core.  A partial install without
    carriers contributes only the tying passes.
    """
    try:
        from catopt_carriers.trace_lift import (
            lift_scan_to_applyd,
            lift_scan_to_trace,
        )
        from catopt_carriers.xcarrier import (
            gather_apply_stack,
            gather_applyd_stack,
            omd_tree_lift,
        )
    except ModuleNotFoundError:
        carrier: list = []
    else:
        carrier = (
            lift_scan_to_applyd(eg)
            + lift_scan_to_trace(eg)
            + gather_applyd_stack(eg)
            + gather_apply_stack(eg)
            + omd_tree_lift(eg)
        )
    return (
        carrier
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
    )


def _select_best_term(
    eg: EGraph,
    root_eid: int,
    cost_fn: CostFn,
    stats: dict[str, Any],
    *,
    groups: list,
    fusion_epsilon: float,
    delivers_compiled: bool,
    specialize_causal: bool,
    capabilities: Capabilities | None,
    source_tensors: dict,
) -> Any:
    """Extract the search's term: greedy -> paired -> carrier -> causal.

    * ``extract_best`` — the additive DAG-cost minimum under
      ``cost_fn`` (``fusion_epsilon`` arms the near-tie fusion band);
    * paired extraction — with pairing groups present, force every
      paired member to its split enode and compare true DAG costs
      (one shared memo prices the baseline once);
    * :func:`_carrier_upgrade` — the whole-spine batched-executor win
      the additive decomposition can't price, billed under the
      intended delivery (``delivers_compiled``);
    * the causal-mask const fold — the opt-out specialization, run
      through the capabilities object's optional
      ``specialize_causal(term, params, memo) -> term`` hook (a
      ``Sink`` carries it for the torch backend): no hook, no fold.
    """
    best_term = eg.extract_best(
        root_eid, cost_fn, fusion_epsilon=fusion_epsilon
    )
    if groups:
        # Coordinated extraction: force every paired member to its
        # split enode AND steer consumers through the shared GEMM.
        # Compare true DAG costs — forcing loses if a group is only
        # partially reachable or a bypassing alternative was already
        # cheaper.  One shared memo across both calls: a forced term
        # shares most subterms with the best term, and the baseline
        # price is computed once, not per candidate.
        _dc_memo: dict = {}
        _best_dag = dag_cost(best_term, cost_fn, memo=_dc_memo)
        forced = eg.extract_paired(root_eid, cost_fn, groups)
        # Honest-decline bookkeeping: an un-extractable forced term
        # prices at +inf, so the same comparison decides and the stats
        # record the verdict with the cost delta.
        _forced_dag = (
            dag_cost(forced, cost_fn, memo=_dc_memo)
            if forced is not None
            else float("inf")
        )
        if _forced_dag <= _best_dag:
            best_term = forced
            stats["paired_extract"] = True
        else:
            stats["paired_extract"] = False
            stats["paired_delta"] = _forced_dag - _best_dag
    # Coordinated carrier selection: a batched-executor win is a
    # whole-spine property the additive extraction can't price.
    best_term = _carrier_upgrade(
        eg, root_eid, best_term, cost_fn, compiled=delivers_compiled
    )
    # Causal specialization: a param-only attn_mask that evaluates to a
    # lower-triangular keep-mask is is_causal=True — no mask op at all.
    # The fold is backend machinery (it evaluates parameter subtrees in
    # the backend's own runtime) — the capabilities port surfaces it
    # via the optional ``specialize_causal`` hook; a backend without
    # one simply skips the pass.
    if specialize_causal and capabilities is not None:
        fold = getattr(capabilities, "specialize_causal", None)
        if fold is not None:
            _cm: dict = {}
            best_term = fold(best_term, source_tensors, _cm)
            if _cm.get("_hit"):
                stats["causal_specialized"] = True
    return best_term


@_oom_to_resource_error
def search(
    model: Any,
    x: Any,
    *,
    source: Source,
    capabilities: Capabilities | None = None,
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None,
    cost_fn: CostFn | None = None,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    symmetry_budget: int | None = 2048,
    max_memory_mb: float | None = None,
    specialize_causal: bool = True,
    fusion_epsilon: float = 0.0,
    delivers_compiled: bool = False,
    meter: Meter | None = None,
    verbose: bool = False,
) -> SearchResult:
    """Run the search phase: ``model -> SearchResult``.

    Export → e-graph → bounded equality saturation → non-local passes
    → extraction.  The result carries everything the lower phase and
    an interactive caller need: the exported ``IR`` and leaf values,
    the saturated ``EGraph`` and root e-class, the extracted ``term``,
    the pricing actually used, and the search-record ``stats``.

    Parameters
    ----------
    model
        The model to optimize — the type only ``source`` interprets.
    x
        An example input for tracing (tensor or positional-args tuple).
    source : Source
        The graph-source port (``model -> (IR, leaves)``) — REQUIRED.
        There is no assumed frontend; choosing torch means importing
        ``catopt_torch`` and passing ``TorchSource``.
    capabilities : Capabilities, optional
        The backend's op surface — ``supported_ops`` bounds extraction
        to forms the backend can lower (``backend_cost`` pricing), and
        its optional ``specialize_causal`` hook supplies the causal
        const fold's evaluator.  Omitted: pricing is backend-agnostic
        and the causal fold is skipped.  A :class:`Sink` satisfies it
        (every sink is a ``Capabilities``).
    cost_fn : CostFn, optional
        Term-extraction pricing; ``None`` falls to ``criteria`` then
        the executor-aware default.
    criteria : dict, Criterion, Criteria, or sequence, optional
        Selection axes blended into the extraction model — see
        ``optimize_model``.
    ruleset : str
        ``"all"`` / ``"all+layout"`` / ``"simpl"`` / ``"categorical"``.
    max_iterations : int
        Maximum equality-saturation iterations.
    max_enodes : int, optional
        E-node bound; crossing it raises
        :class:`OptimizationResourceError`.  ``None`` disables.
    max_memory_mb : float, optional
        Process memory bound in MiB.  ``None`` disables.
    symmetry_budget : int, optional
        Per-rule enode budget for the expansive rules; ``None`` is
        unbounded saturation.
    fusion_epsilon : float, default 0.0
        Near-tie fusion-preferred extraction band — see
        ``optimize_model``.
    specialize_causal : bool, default True
        The causal-mask const fold is an opt-out search pass: a
        param-only ``sdpa`` mask that evaluates to the lower-triangular
        keep-mask becomes ``is_causal=True``.
    delivers_compiled : bool, default False
        Delivery hint for carrier selection only: when the eventual
        delivery will be ``torch.compile``-wrapped, the carrier upgrade
        bills terms under the fusion-region (``lowering="compiled"``)
        price.  The runner itself is a ``lower`` concern.
    meter : Meter, optional
        The backend's timing port — consulted only for the optional
        device-memory read the ``max_memory_mb`` bound needs (host
        RSS is read portably).  ``None`` counts host bytes only.
    verbose : bool
        Print progress.

    Returns
    -------
    SearchResult
        ``ir`` / ``eg`` / ``root_eid`` / ``term`` / ``param_values`` /
        ``stats`` (the search record) / ``cost_fn`` / ``source`` /
        ``model`` (the verify reference ``lower`` uses).

    """
    cost_fn, criteria_used = _resolve_cost_fn(
        cost_fn, criteria, capabilities
    )

    # Recursive walks (extraction, member resolution) descend the
    # e-class DAG, whose depth grows with the saturation closure —
    # thousands of levels on deep stacks.
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    # -- Phase 1: Export to IR -------------------------------------------
    if verbose:
        print(
            f"[Phase 1] Exporting {model.__class__.__name__} to IR..."
        )
    ir, source_tensors = source.to_ir(model, x)
    if verbose:
        print(f"  IR root: {op_repr(ir.root)}")
        print(f"  Inputs:  {[str(v) for v in ir.inputs]}")
        print(f"  Params:  {list(ir.params.keys())}")

    # -- Phase 2: Build e-graph and saturate -----------------------------
    if verbose:
        print(
            "[Phase 2] Building e-graph and running equality "
            "saturation..."
        )
    eg = EGraph()
    root_eid = eg.add_term(ir.root)
    rules = _ruleset_rules(ruleset)
    if verbose:
        print(f"  Rules: {[r.name for r in rules]}")

    # Bounded-saturation budget for the expansive rules (see
    # ``_EXPANSIVE_RULES``); enforced inside the matcher so a giant
    # e-class cannot spend the whole budget in one enumeration.
    rule_budgets = (
        {n: symmetry_budget for n in _EXPANSIVE_RULES}
        if symmetry_budget is not None
        else None
    )
    # ``None`` = unbounded: the run loop wants a concrete watermark.
    run_cap = max_enodes if max_enodes is not None else sys.maxsize

    stats: dict[str, Any] = eg.run(
        rules,
        root_eid,
        max_iterations=max_iterations,
        max_nodes=run_cap,
        rule_budgets=rule_budgets,
    )
    _check_resources(eg, max_enodes, max_memory_mb, meter)

    groups = _pairing_and_lifts(
        eg,
        rules,
        root_eid,
        stats,
        run_cap,
        rule_budgets,
        max_enodes,
        max_memory_mb,
        source_tensors,
        meter,
    )

    stats["rule_fires"] = dict(eg.rule_fires)
    stats["criteria"] = criteria_used
    if fusion_epsilon:
        stats["fusion_epsilon"] = fusion_epsilon
    if verbose:
        print(f"  E-graph: {stats}")

    # -- Extract best term -----------------------------------------------
    best_term = _select_best_term(
        eg,
        root_eid,
        cost_fn,
        stats,
        groups=groups,
        fusion_epsilon=fusion_epsilon,
        delivers_compiled=delivers_compiled,
        specialize_causal=specialize_causal,
        capabilities=capabilities,
        source_tensors=source_tensors,
    )

    if verbose:
        print(f"  Best term: {op_repr(best_term)}")
        print(f"  Cost: {cost_fn(best_term):.2f} FLOPs (est.)")

    # Final watermark before the lower phase materialises parameters —
    # the phase that turns graph choices into real tensor bytes.
    _check_resources(eg, max_enodes, max_memory_mb, meter)

    return SearchResult(
        ir=ir,
        eg=eg,
        root_eid=root_eid,
        term=best_term,
        param_values=source_tensors,
        stats=stats,
        cost_fn=cost_fn,
        source=source,
        model=model,
    )


@_oom_to_resource_error
def lower(
    result: SearchResult,
    x: Any,
    *,
    sink: Sink,
    runner: Runner | None = None,
    verify: bool = True,
    rtol: float = 1e-4,
    atol: float | None = None,
    verbose: bool = False,
) -> LowerResult:
    """Run the lower phase: ``SearchResult -> LowerResult``.

    Executor routing (``_lower_extracted``): a term the sink's
    executor table claims — the first carrier ``ExecutorSpec`` whose
    ``accepts`` probe holds — goes to that level-batched executor;
    everything else to ``sink.lower``.  The delivery runner then
    decides HOW the routed executor ships — identity
    (``IdentityRunner`` — the default) or a backend-provided
    transform like ``catopt_torch.runners.TorchCompileRunner`` /
    ``catopt_cuda.CudaGraphRunner`` / a ``ChainedRunner``
    composition.
    ``stats`` is a FRESH dict — the search record plus the lowering
    keys; :attr:`SearchResult.stats` is never mutated — so one search
    can feed many deliveries.

    Parameters
    ----------
    result : SearchResult
        What :func:`search` (or ``Optimizer.search``) produced.
    x
        The example input — the runner needs it (compile probes,
        graph capture) and ``sink.verify`` runs on it.
    sink : Sink
        The graph-sink port — REQUIRED.  There is no assumed backend.
    runner : Runner, optional
        The delivery transform; ``None`` ships as lowered.
    verify : bool, default True
        Run the sink's equivalence gate.  The reference is a fresh
        lowering of ``result.ir`` — the un-optimized program the
        search exported — via ``sink.lower``: the verify contract is
        "the delivery preserves the IR the search certified", stated
        in the sink's own runtime so a non-torch backend works
        unchanged.  The report lands in ``LowerResult.verified``.
    rtol, atol
        The equivalence tolerances, forwarded to ``sink.verify``.
    verbose : bool
        Print progress.

    """
    if runner is None:
        runner = IdentityRunner()
    stats: dict[str, Any] = dict(result.stats)
    if verbose:
        print("[Phase 3] Lowering optimized IR to torch module...")
    params = dict(result.param_values)
    optimized_ir = IR(
        root=result.term,
        inputs=result.ir.inputs,
        input_names=result.ir.input_names,
        params=result.ir.params,
    )
    optimized_module = _lower_extracted(
        result.term, optimized_ir, params, sink
    )
    stats["lowering"] = (
        "batched"
        if getattr(optimized_module, "is_batched", False)
        else "generic"
    )
    # Delivery: the runner decides how the routed executor ships.
    # stats["runner"] records the name (member-name list for a chain);
    # runners write their own outcome keys (stats["compiled"],
    # stats["cuda_graph"]).
    runner_names = getattr(runner, "names", None)
    stats["runner"] = (
        list(runner_names)
        if runner_names is not None
        else getattr(runner, "name", type(runner).__name__)
    )
    optimized_module = runner.apply(optimized_module, x, stats)

    # Verify semantic equivalence — the delivered module against the
    # un-optimized IR lowered by the same sink (backend-neutral: the
    # reference is a runnable in the sink's runtime, which the
    # source-side model need not be).
    verified = None
    if verify:
        if verbose:
            print("[Verify] Checking output equivalence...")
        ref = sink.lower(result.ir, params)
        verified = sink.verify(
            ref, optimized_module, x, rtol=rtol, atol=atol
        )
        if verbose:
            print(f"  Max abs diff:  {verified.max_abs:.6e}")
            print(f"  Max rel diff:  {verified.max_rel:.6e}")
            if verified.passed:
                print("  ✓ Semantically equivalent (within tolerance)")
            else:
                print("  ✗ WARNING: large difference detected!")

    return LowerResult(
        module=optimized_module, stats=stats, verified=verified
    )


# ---------------------------------------------------------------------------
#  The strategy seam — Optimizer + Monolithic / Compositional / Autotuned
# ---------------------------------------------------------------------------

#: Keyword names each phase verb accepts — ``Monolithic`` partitions
#: ``optimize(..., **kw)`` between them (``verbose`` reaches both).
_SEARCH_KW = frozenset(inspect.signature(search).parameters) - {
    "model",
    "x",
    "source",
}
_LOWER_KW = frozenset(inspect.signature(lower).parameters) - {
    "result",
    "x",
    "sink",
}


class Monolithic:
    """The default strategy: ``lower ∘ search`` in one pass.

    One whole-model search, one delivery — what ``optimize_model``
    always did.  ``optimize(..., **kw)`` keywords partition by phase:
    search knobs (``ruleset``, ``max_iterations``, ``max_enodes``,
    ``cost_fn``, ``symmetry_budget``, ``specialize_causal``,
    ``fusion_epsilon``, ``delivers_compiled``, ``capabilities``,
    ``criteria``, ``meter``) reach the optimizer's ``.search``; lower
    knobs (``runner``, ``verify``, ``rtol``, ``atol``) reach
    ``.lower``; ``verbose`` reaches both.  Unknown names raise
    ``TypeError``.
    """

    name = "monolithic"

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run search, then lower — the single-shot pipeline."""
        unknown = sorted(set(kw) - _SEARCH_KW - _LOWER_KW)
        if unknown:
            raise TypeError(
                f"optimize() got unexpected keywords: {unknown}"
            )
        s_kw = {k: v for k, v in kw.items() if k in _SEARCH_KW}
        l_kw = {k: v for k, v in kw.items() if k in _LOWER_KW}
        # A runner chosen at call time also hints the search: the
        # carrier upgrade prices delivered cost under the
        # fusion-region model when the delivery will be compiled.
        runner = l_kw.get("runner")
        if runner is not None:
            s_kw.setdefault(
                "delivers_compiled",
                bool(getattr(runner, "delivers_compiled", False)),
            )
        result = optimizer.search(model, x, **s_kw)
        return optimizer.lower(result, x, **l_kw)


class Compositional:
    """Per-block search+lower, then recompose — composer-driven.

    The strategy view of ``optimize_compositional``: selects blocks
    through the optimizer's :class:`~catopt_core.ports.Composer`
    port, optimizes each on its captured input, runs the pairwise
    cross-block pass and grafts the results into a parameter-sharing
    clone — every backend-native step delegated to the composer, so
    the strategy itself is backend-neutral.

    ``block_pred`` / ``verify_tol`` / ``max_cross_pairs`` are strategy
    configuration; the per-block search knobs (``ruleset``,
    ``max_iterations``, ``cost_fn``, ``max_enodes``, ``max_memory_mb``,
    ``verbose``) ride in ``**kw``.
    """

    name = "compositional"

    def __init__(
        self,
        *,
        block_pred: Callable | None = None,
        verify_tol: float = 1e-4,
        max_cross_pairs: int = 8,
    ) -> None:
        """Store the strategy configuration."""
        self.block_pred = block_pred
        self.verify_tol = verify_tol
        self.max_cross_pairs = max_cross_pairs

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run the per-block pipeline through the optimizer's ports."""
        mod, stats = _optimize_compositional(
            model,
            x,
            optimizer=optimizer,
            block_pred=self.block_pred,
            verify_tol=self.verify_tol,
            max_cross_pairs=self.max_cross_pairs,
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


class Autotuned:
    """Measured autotune: one search, N verified timed deliveries.

    The strategy view of ``optimize_model_autotuned`` — re-lowers
    the extracted term through each candidate lowering path, verifies
    every candidate against the model, times the survivors through
    the optimizer's :class:`~catopt_core.ports.Meter` port and ships
    the measured winner.

    ``candidates`` are names resolved against the backend's
    :attr:`~catopt_core.ports.Sink.executors` table (``"generic"``
    plus every executor name — ``"batched"`` is the routing
    pseudo-name for the pipeline's own delivery) and
    :data:`catopt_optimize.autotune.CANDIDATE_BUILDERS`
    (``"eager"``), or ``(name, builder)`` tuples; backend-provided
    builders arrive through ``builders=`` (the torch wrapper maps
    ``"torch_compile"`` / ``"torch_compile_generic"`` /
    ``"cuda_graph"`` via ``catopt_torch.api.TORCH_BUILDERS``).  The
    remaining fields are the timing / budget / verify knobs.
    ``optimize`` kwargs (``ruleset``, ``max_iterations``, ``runner``,
    …) forward to the underlying pipeline call; an explicit
    ``verbose=`` keyword overrides the strategy field.
    """

    name = "autotuned"

    def __init__(
        self,
        candidates: Any = ("generic", "batched", "torch_compile"),
        *,
        budget_s: float | None = None,
        n_calls: int = 30,
        warmup: int = 5,
        rtol: float = 1e-4,
        atol: float | None = None,
        profile: Any = None,
        verbose: bool = False,
        builders: dict[str, Any] | None = None,
    ) -> None:
        """Store the autotune configuration."""
        self.candidates = candidates
        self.budget_s = budget_s
        self.n_calls = n_calls
        self.warmup = warmup
        self.rtol = rtol
        self.atol = atol
        self.profile = profile
        self.verbose = verbose
        #: Backend-provided candidate builders — names the
        #: orchestrator-side ``CANDIDATE_BUILDERS`` map does not know
        #: (e.g. the torch wrapper's ``TORCH_BUILDERS``).  Merged over
        #: the neutral table; ``None`` is backend-agnostic autotuning.
        self.builders = builders

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Search once via the ports, then time each candidate."""
        # Local import: autotune imports this module at top level, so
        # the reverse edge must defer to call time (the same pattern
        # runners.runner_candidate documents).
        from catopt_optimize.autotune import _autotuned_impl

        mod, stats = _autotuned_impl(
            model,
            x,
            candidates=self.candidates,
            builders=self.builders,
            budget_s=self.budget_s,
            n_calls=self.n_calls,
            warmup=self.warmup,
            rtol=self.rtol,
            atol=self.atol,
            optimizer=optimizer,
            profile=self.profile,
            verbose=kw.pop("verbose", self.verbose),
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


@dataclass
class Optimizer:
    """The configured entry point — explicit ports, no assumed backend.

    The pipeline runs through four ports — ``source`` (``model ->
    (IR, leaves)``), ``sink`` (``IR -> runnable`` + verify +
    executors), ``composer`` (per-block structural machinery) and
    ``meter`` (timing) — bundled as an immutable
    :class:`~catopt_core.pipeline.Backend` or given individually.
    Resolution is ``backend`` first, then the explicit arguments
    override each port it carries; every port is required — there is
    NO default backend (choosing torch means importing
    ``catopt_torch.backend.TorchBackend``).  ``criteria`` is the
    default selection blend; ``runner`` the default delivery
    (``IdentityRunner`` — the no-op, backend-neutral shipping step).

    The phase verbs mirror the module-level functions with the
    optimizer's ports wired in; ``optimize`` composes them through a
    :class:`~catopt_core.ports.Strategy`::

        opt = Optimizer(backend=TorchBackend())
        mod, stats = opt.optimize(model, x)
        res = opt.search(model, x)          # the inspectable mid-state
        fast = opt.lower(res, x, runner=TorchCompileRunner())

    """

    source: Source | None = None
    sink: Sink | None = None
    composer: Composer | None = None
    meter: Meter | None = None
    backend: Backend | None = None
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None
    runner: Runner = field(default_factory=IdentityRunner)

    def __post_init__(self) -> None:
        """Resolve ports — explicit args override the backend's.

        ``source`` and ``sink`` are REQUIRED — from the backend or
        explicitly.  ``composer``/``meter`` are the optional ports the
        :class:`Compositional` / :class:`Autotuned` strategies need;
        a strategy raises a clear error when its port is missing, so
        a bare ``Optimizer(source=..., sink=...)`` still runs
        :class:`Monolithic` — the historical minimal form.
        """
        if self.backend is not None:
            if self.source is None:
                object.__setattr__(self, "source", self.backend.source)
            if self.sink is None:
                object.__setattr__(self, "sink", self.backend.sink)
            if self.composer is None:
                object.__setattr__(
                    self, "composer", self.backend.composer
                )
            if self.meter is None:
                object.__setattr__(self, "meter", self.backend.meter)
        missing = [
            n for n in ("source", "sink") if getattr(self, n) is None
        ]
        if missing:
            raise TypeError(
                "Optimizer requires source/sink — pass "
                "backend=Backend(...) or explicit "
                "source=/sink= ports; composer=/meter= are needed "
                f"only by the per-block/measured strategies "
                f"(missing: {', '.join(missing)})"
            )

    def search(self, model: Any, x: Any, **kw: Any) -> SearchResult:
        """Run :func:`search` through this optimizer's ports.

        ``capabilities`` defaults to the sink (a ``Sink`` is a
        ``Capabilities``); ``criteria`` to the configured blend;
        ``delivers_compiled`` to the runner's marker; ``meter`` to the
        configured meter.  Keyword arguments override each default
        outright.
        """
        kw.setdefault("source", self.source)
        kw.setdefault("capabilities", self.sink)
        kw.setdefault("criteria", self.criteria)
        kw.setdefault(
            "delivers_compiled",
            bool(getattr(self.runner, "delivers_compiled", False)),
        )
        kw.setdefault("meter", self.meter)
        return _search(model, x, **kw)

    def lower(
        self, result: SearchResult, x: Any, **kw: Any
    ) -> LowerResult:
        """Run :func:`lower` through this optimizer's sink.

        ``runner=None`` means the configured default, not the
        identity — pass ``runner=IdentityRunner()`` explicitly to
        override a configured delivery.
        """
        kw.setdefault("sink", self.sink)
        if kw.get("runner") is None:
            kw["runner"] = self.runner
        return _lower(result, x, **kw)

    def optimize(
        self,
        model: Any,
        x: Any,
        *,
        strategy: Strategy | None = None,
        **kw: Any,
    ) -> LowerResult:
        """Run *strategy* (default :class:`Monolithic`) end to end.

        The result unpacks as ``(module, stats)`` — ``mod, stats =
        opt.optimize(model, x)`` mirrors the historical tuple.
        """
        strat = strategy if strategy is not None else Monolithic()
        return strat.run(model, x, optimizer=self, **kw)

    def discover(self, model: Any, x: Any, **kw: Any) -> SearchResult:
        """Run the discovery view — :func:`search` with frontier defaults.

        ``ruleset="categorical"`` / ``max_iterations=6`` unless the
        caller says otherwise; the :class:`SearchResult` carries the
        whole inspectable mid-state (``.alternatives()`` /
        ``.certificate()`` / ``.eg`` / ``.stats``).
        """
        kw.setdefault("ruleset", "categorical")
        kw.setdefault("max_iterations", 6)
        return self.search(model, x, **kw)


# ---------------------------------------------------------------------------
#  The discover verb and the compositional driver
# ---------------------------------------------------------------------------


#: The verbs bound to module level — ``Optimizer``'s methods alias the
#: module functions so the method bodies can't be shadowed by the
#: same-named methods.
_search = search
_lower = lower


@_oom_to_resource_error
def discover_alternatives(
    model: Any,
    x: Any,
    *,
    source: Source,
    capabilities: Capabilities | None = None,
    ruleset: str = "categorical",
    max_iterations: int = 6,
    cost_fn: CostFn | None = None,
) -> SearchResult:
    """Enumerate the frontier — return the :class:`SearchResult`.

    The discovery-engine view: the same export → e-graph → saturation
    → pairing pipeline as :func:`search` with frontier-oriented
    defaults (``ruleset="categorical"``, ``max_iterations=6``) — and
    the result is the inspectable mid-state, not a fixed report:

    * ``res.alternatives(top_k)`` — the top-k cheapest distinct
      members of the root class under the result's pricing
      (``cost_fn`` — or its default — is backend-relative when
      ``capabilities`` is given, so the frontier only lists forms the
      backend can lower);
    * ``res.eg.diverse_classes()`` — the e-classes holding
      structurally distinct equivalent terms, where emergent
      compositions hide;
    * ``res.stats["rule_fires"]`` — the provenance: which generic
      laws actually fired;
    * ``res.term`` / ``res.eg`` / ``res.stats`` — the extraction and
      its record.

    Human inspection of this frontier is how level-3 candidates —
    emergent compositions of known laws — are found.  ``source`` is
    required (no assumed frontend); ``capabilities`` supplies
    backend-relative pricing and the const-fold registry.

    Ruleset names follow the pipeline's selection
    (:func:`_ruleset_rules`) — ``"all"`` drops the pairing-subsumed
    fusion rules here too.
    """
    return search(
        model,
        x,
        source=source,
        capabilities=capabilities,
        cost_fn=cost_fn,
        ruleset=ruleset,
        max_iterations=max_iterations,
    )


def ir_to_string(term: Any) -> str:
    """Pretty-print an IR term as an S-expression."""
    return op_repr(term)


def term_cost(term: Any, cost_fn: CostFn | None = None) -> float:
    """Compute the cost of a term using the given cost function."""
    if cost_fn is None:
        cost_fn = flops_cost
    return cost_fn(term)


def _optimize_compositional(
    model: Any,
    example_input: Any,
    *,
    optimizer: Optimizer,
    block_pred: Callable | None = None,
    cost_fn: CostFn | None = None,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verify_tol: float = 1e-4,
    max_cross_pairs: int = 8,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Optimize a stacked/multi-block model one block at a time.

    Whole-model equality saturation is monolithic: the e-graph grows with
    the product of block structures, so deep stacks saturate slowly.
    This backend-neutral driver delegates every backend-native step to
    the optimizer's ports and

    1. selects *blocks* via the composer port — leaf sub-objects, plus
       any child where ``block_pred(parent, name, child)`` holds,
    2. runs the ORIGINAL model once (``composer.capture_inputs``) to
       capture each block's real input (a block's input is not the
       model input),
    3. runs the optimizer's ``search``/``lower`` phases on each block
       with its captured input, verifying the lowered block against
       the original on that input — a block that fails to export,
       saturate, lower, or verify keeps its original implementation;
       a block that crosses a resource bound (``max_enodes``,
       ``max_memory_mb``, or a caught OOM) fails with
       ``reason == "resource_limit"``,
    4. runs the pairwise cross-block pass (the composer's optional
       ``cross_pairs`` hook): for each *adjacent* pair whose boundary
       is a simple value flow (``composer.boundary``), build the
       joint micro-model, optimize it, verify it against the eager
       pair, and graft it in place of both blocks when it is verified
       AND cheaper than the separate results — capped at
       ``max_cross_pairs`` joint runs (``0`` disables),
    5. clones the model — structure deep-copied, parameter/buffer
       values *shared* with the original
       (``composer.clone_sharing``; no second copy of the weights, so
       recompose does not double device memory) — and grafts the
       optimized executors back in place
       (``composer.graft``), preserving the original forward
       structure, then verifies end-to-end equivalence on
       ``example_input`` through ``sink.verify``.

    Returns ``(recomposed_model, stats)`` where ``stats["blocks"]`` maps
    each block's dotted name to ``{"status", "stats", "param_report",
    "time_s", ...}`` and ``stats["param_report"]`` aggregates the
    per-block parameter diffs (eliminated/derived names are prefixed by
    block name for auditability) — the composer's ``param_report`` hook
    supplies the per-block diff when it exists.  ``stats["cross_pairs"]``
    maps each attempted pair ``"a+b"`` to ``{"status", "boundary", ...}``
    — ``"grafted"`` / ``"declined"`` / ``"skipped"``.  ``stats["shared_
    params"]`` is True when the recomposed model shares the original's
    storage (the normal path); ``stats["in_place"]`` True means cloning
    failed and the input was returned unmodified.
    """
    t_start = time.time()
    composer = optimizer.composer
    if composer is None:
        raise TypeError(
            "Compositional strategy needs a Composer port — "
            "pass composer= (or backend=) to Optimizer"
        )
    # ``Optimizer.__post_init__`` already rejects a missing source/sink,
    # so by construction both ports are present here.
    source = cast(Source, optimizer.source)
    sink = cast(Sink, optimizer.sink)
    if cost_fn is None:
        cost_fn = _default_cost_fn()

    blocks = composer.blocks(model, predicate=block_pred)
    if verbose:
        names = [n for n, _ in blocks]
        print(
            f"[Compositional] {len(blocks)} candidate blocks: {names}"
        )

    captured, io = composer.capture_inputs(model, blocks, example_input)
    # A second capture on a perturbed probe input arms the residual
    # boundary check against a coincidence of values (it only runs when
    # the pair pass is enabled; a probe failure fails closed — residual
    # pairs decline, identity-proven chain pairs still work).
    captured2: dict[str, Any] = {}
    io2: dict[str, Any] = {}
    if max_cross_pairs:
        import contextlib

        with contextlib.suppress(Exception):
            captured2, io2 = composer.capture_inputs(
                model, blocks, composer.perturbed(example_input)
            )

    replacements: dict[str, Any] = {}
    block_reports: dict[str, dict[str, Any]] = {}
    agg = {
        "original_params": 0,
        "optimized_params": 0,
        "original_bytes": 0,
        "optimized_bytes": 0,
        "eliminated": [],
        "derived": [],
    }
    # The weight-file diff is an optional composer hook — a backend
    # without one reports no parameter audit.
    param_diff = getattr(composer, "param_report", None)

    for name, block in blocks:
        rep: dict[str, Any] = {"status": "not_executed"}
        block_reports[name] = rep
        cap = captured.get(name)
        if cap is None:
            continue
        args, kwargs = cap
        if kwargs:
            rep["status"] = "skipped"
            rep["reason"] = f"non-positional kwargs {sorted(kwargs)}"
            continue
        ex = args[0] if len(args) == 1 else args
        t0 = time.time()
        try:
            # Per-block search+lower — the same phases the monolithic
            # pipeline runs, through the optimizer's ports.
            res = optimizer.search(
                block,
                ex,
                ruleset=ruleset,
                max_iterations=max_iterations,
                max_enodes=max_enodes,
                max_memory_mb=max_memory_mb,
                cost_fn=cost_fn,
                delivers_compiled=False,
                verbose=verbose,
            )
            lr = optimizer.lower(
                res,
                ex,
                runner=IdentityRunner(),
                verify=verbose,
                verbose=verbose,
            )
            opt_mod, st = lr.module, lr.stats
            # Per-block verification on the captured input — soundness
            # gate independent of the pipeline's own (verbose-gated)
            # check.  Any mismatch or eval failure falls back.
            vr = sink.verify(block, opt_mod, args, rtol=verify_tol)
            rep["rel_diff"] = vr.max_rel
            if not vr.passed:
                raise RuntimeError(
                    f"block verification failed: "
                    f"rel diff {vr.max_rel:.3e}"
                )
            replacements[name] = opt_mod
            rep["status"] = "optimized"
            rep["stats"] = st
            if param_diff is not None:
                pr = param_diff(block, opt_mod)
                rep["param_report"] = pr
                agg["original_params"] += pr["original_params"]
                agg["optimized_params"] += pr["optimized_params"]
                agg["original_bytes"] += pr["original_bytes"]
                agg["optimized_bytes"] += pr["optimized_bytes"]
                agg["eliminated"] += [
                    f"{name}:{n}" for n in pr["eliminated"]
                ]
                agg["derived"] += [f"{name}:{n}" for n in pr["derived"]]
            if verbose:
                print(
                    f"[Compositional] {name}: optimized "
                    f"({rep['rel_diff']:.2e})"
                )
        except Exception as e:
            rep["status"] = "failed"
            rep["error"] = f"{type(e).__name__}: {e}"
            if isinstance(
                e, OptimizationResourceError
            ) or _looks_like_oom(e):
                rep["reason"] = "resource_limit"
            if verbose:
                print(f"[Compositional] {name}: keeping original ({e})")
        rep["time_s"] = time.time() - t0

    # -- Pairwise cross-block pass --------------------------------------
    # Per-block optimization is blind across the boundary: adjacent
    # blocks can share transforms a per-block search cannot see (block
    # i's output projection composing with block i+1's input
    # projections; a residual add absorbing a shared affine).  The
    # pass is adapter machinery — verified and cost-gated; every
    # decline keeps the separate results.
    cross_pairs: dict[str, dict[str, Any]] = {}
    cross = getattr(composer, "cross_pairs", None)
    if max_cross_pairs and callable(cross):
        cross_pairs = cross(
            blocks,
            captured,
            io,
            captured2,
            io2,
            replacements,
            block_reports,
            agg,
            ruleset=ruleset,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            max_memory_mb=max_memory_mb,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            source=source,
            sink=sink,
            max_cross_pairs=max_cross_pairs,
            verbose=verbose,
        )

    # -- Recompose -------------------------------------------------------
    # The clone grafts structure, never values, so it shares the
    # original's parameter/buffer storage — recomposing costs no extra
    # weight bytes (the old plain deepcopy doubled the footprint and
    # OOMed at ~0.5B fp16 on small GPUs).
    in_place = False
    try:
        new_model = composer.clone_sharing(model)
    except Exception:
        # Never graft into the caller's live model: the replacements
        # carry shape-specialized state baked by the export for the
        # example input, and a mutated caller fails at the NEXT input
        # shape (and the e2e check degenerates to self-comparison).
        new_model = model
        in_place = True
        replacements = {}
    composer.graft(new_model, replacements)

    # -- End-to-end verification ----------------------------------------
    agg["bytes_saved"] = agg["original_bytes"] - agg["optimized_bytes"]
    agg["ratio"] = (
        agg["optimized_bytes"] / agg["original_bytes"]
        if agg["original_bytes"]
        else 1.0
    )
    stats: dict[str, Any] = {
        "compositional": True,
        "n_blocks": len(blocks),
        "n_optimized": len(replacements),
        "n_failed": sum(
            1 for r in block_reports.values() if r["status"] == "failed"
        ),
        "n_skipped": sum(
            1
            for r in block_reports.values()
            if r["status"] in ("skipped", "not_executed")
        ),
        "blocks": block_reports,
        "in_place": in_place,
        "shared_params": not in_place,
        "param_report": agg,
        "cross_pairs": cross_pairs,
    }

    if in_place:
        # new_model IS the input model — a verify would be a
        # self-comparison that always reports 0.0.  Record the
        # degenerate case honestly instead of a false pass.
        stats["end_to_end"] = {
            "skipped": "in_place",
            "reason": "param-sharing clone failed — returned model "
            "is the input module, unmodified",
        }
        logger.warning(
            "optimize_compositional: clone failed — returning the "
            "input module unmodified (no blocks grafted; "
            "report.in_place=True)"
        )
    else:
        args2 = (
            example_input
            if isinstance(example_input, tuple)
            else (example_input,)
        )
        try:
            vr = sink.verify(model, new_model, args2)
            stats["end_to_end"] = {
                "max_abs_diff": vr.max_abs,
                "max_rel_diff": vr.max_rel,
            }
            if verbose:
                print(
                    f"[Compositional] end-to-end rel diff: "
                    f"{stats['end_to_end']['max_rel_diff']:.3e}"
                )
        except Exception as e:
            stats["end_to_end"] = {"error": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"[Compositional] end-to-end check failed: {e}")

    stats["wall_time_s"] = time.time() - t_start
    return new_model, stats


# ---------------------------------------------------------------------------
# Compatibility delegation — moved torch-facing names (lazy)
# ---------------------------------------------------------------------------
#
# The deprecated ``optimize_*`` wrappers (torch-defaulted entry
# points) moved to ``catopt_torch.api`` (plan 0007); the torch-native
# structural internals (hook capture, shared-param clone, pair
# boundary) moved to ``catopt_torch.composer``; the causal-fold
# internals to ``catopt_torch.folds``; the typed reports stay in
# ``catopt_torch.report``.  All resolve lazily through this module's
# ``__getattr__`` so the historical private/compat paths
# (``catopt.optimize.optimize_model``,
# ``catopt.optimize._pair_boundary``, …) keep working on a torch
# install — while ``import catopt_optimize.optimize`` itself loads no
# backend.

_API_NAMES = frozenset(
    {
        "optimize_model",
        "optimize_compositional",
        "save_optimized_weights",
    }
)
_COMPOSER_NAMES = frozenset(
    {
        "param_report",
        "_default_block_pred",
        "_select_blocks",
        "_capture_block_inputs",
        "_MODEL_KEY",
        "_replace_submodule",
        "_shared_param_clone",
        "_perturbed_input",
        "_residual_probe",
        "_executor_flops",
        "_JointPair",
        "_FusedPair",
        "_Zero",
        "_plain_consumers",
        "_io_has_value",
        "_pair_boundary",
        "_cross_pair_pass",
        "_CROSS_PAIR_SYMMETRY_BUDGET",
    }
)
_FOLD_NAMES = frozenset(
    {
        "_eval_const",
        "_is_causal_keep_mask",
        "_specialize_causal",
    }
)
_REPORT_NAMES = frozenset(
    {
        "OptReport",
        "BlockReport",
        "CompositionalReport",
        "verify_module",
    }
)

#: Composer cross_pairs is a TorchComposer method — its historical
#: free-function form is reproduced by a bound call.
_CARRIER_PLANS_LAZY: dict[str, Callable] | None = None


def __getattr__(name: str) -> Any:
    """Resolve the moved torch-facing names lazily."""
    global _CARRIER_PLANS_LAZY
    if name == "_CARRIER_PLANS":
        if _CARRIER_PLANS_LAZY is None:
            _CARRIER_PLANS_LAZY = _carrier_plans()
        return _CARRIER_PLANS_LAZY
    if name in _API_NAMES:
        mod = importlib.import_module("catopt_torch.api")
        return getattr(mod, name)
    if name in _COMPOSER_NAMES:
        mod = importlib.import_module("catopt_torch.composer")
        if name == "_cross_pair_pass":
            # Historical shape: the free function — now the composer
            # method (same signature, self dropped).
            return mod.TorchComposer().cross_pairs
        return getattr(mod, name)
    if name in _FOLD_NAMES:
        mod = importlib.import_module("catopt_torch.folds")
        return getattr(mod, name)
    if name in _REPORT_NAMES:
        mod = importlib.import_module("catopt_torch.report")
        return getattr(mod, name)
    if name == "torch":
        # Historical patch point: ``monkeypatch.setattr(O.torch,
        # "compile", ...)`` mutated the module attribute — the same
        # module object the delivery runners look up at call time.
        return importlib.import_module("torch")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
