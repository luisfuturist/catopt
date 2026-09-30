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
through the ``meter`` port.  ``Optimizer(backend=TorchBackend())``
is the supported torch spelling — no default is assumed anywhere in
this package.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from catopt_core.cost import (
    backend_cost,
    dag_cost,
    executor_cost_for,
    flops_cost,
)
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, Param, Var, op_repr
from catopt_core.laws import (
    RuleSet,
    pair_shared_input_convs,
    pair_shared_input_linears,
    preset,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from catopt_core.laws import (
    tags as _law_tags,
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

from catopt_orchestrator.criteria import (
    Criteria,
    Criterion,
    criteria_cost,
)
from catopt_orchestrator.runners import IdentityRunner

#: Bounded saturation: the rules tagged
#: :data:`catopt_core.laws.tags.EXPANSIVE` — the pure-symmetry monoid
#: laws enumerating every bracketing/ordering of a summation
#: (Catalan-scale on the residual accumulator), the scale-hoist laws
#: pairing every scale member with every linear, and the
#: distribute/factor/naturality/assoc algebra generating cross-product
#: closures (enodes grew 337 → 40k in four iterations on a
#: DeepParallel stack) — run under a per-rule enode budget, which
#: truncates the reordering closure but leaves every content-bearing
#: rewrite at the exact fixed point.  Structural fusions
#: (qkv/swiglu/sdpa folds, gqa_absorb) and the simplification
#: singletons stay unbudgeted: their matches are pattern-specific,
#: not closure-generating.  Measured on stacked ParallelBlocks (the
#: model that motivated ``optimize_compositional``): identical
#: extracted cost at every budget ≥ 512 while saturation drops from
#: minutes to ~1s.
logger = logging.getLogger("catopt_orchestrator.optimize")


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

    The :class:`Compositional` strategy records these as ordinary
    per-block failures with ``reason == "resource_limit"``; a
    standalone ``Optimizer.optimize`` caller gets this dedicated type
    instead of a raw OOM.
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

    An engine without a materialised ``_classes`` view (the
    :class:`~catopt_core.ports.Engine` port does not require one)
    skips the pass — greedy extraction already ran.
    """
    eclasses = getattr(eg, "_classes", None)
    if eclasses is None:
        return best_term
    plans = _carrier_plans()
    cid = eg.find(root_eid)
    carriers = [n for n in eclasses[cid].nodes if n.op in plans]
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


#: The composed default rule set — ``catopt_core.laws.DEFAULT`` plus
#: the carrier-package ``CARRIERS`` preset.  Composed **once,
#: lazily**: ``catopt_carriers`` is a different, torch-coupled
#: package, so reaching into it here at import time would break the
#: orchestrator's backend-neutral contract; a partial install simply
#: contributes the core default.  Subsumed- and symmetry-tagged
#: rules are excluded by the *preset*, not by a filter in this module
#: — the ``_SUBSUMED`` hidden list and the ``ruleset: str`` switch
#: are gone (plan 0009).
_DEFAULT_RULES: RuleSet | None = None


def default_rules() -> RuleSet:
    """Return the pipeline's composed default rule set (``DEFAULT_RULES``)."""
    global _DEFAULT_RULES
    if _DEFAULT_RULES is None:
        from catopt_core import laws

        rs = laws.DEFAULT
        with contextlib.suppress(ModuleNotFoundError):
            from catopt_carriers import CARRIERS

            rs = rs + CARRIERS
        _DEFAULT_RULES = replace(
            rs,
            name="default_rules",
            description=(
                "the pipeline default — core DEFAULT + the "
                "carrier-package CARRIERS preset"
            ),
        )
    return _DEFAULT_RULES


def __getattr__(name: str) -> Any:
    """Lazily materialise ``DEFAULT_RULES`` (carrier composition)."""
    if name == "DEFAULT_RULES":
        return default_rules()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _resolve_rules(rules: Any) -> RuleSet:
    """Normalise a ``rules`` argument to a :class:`RuleSet`.

    ``None`` → :data:`DEFAULT_RULES` (the composed default above).
    A string resolves to a named preset — the orchestrator-level
    names (``default`` / ``carrier_search`` / ``xc``) first, then
    :func:`catopt_core.laws.preset`; anything else iterable becomes
    an anonymous set.  The old ``ruleset: str`` switch and the hidden
    ``_SUBSUMED`` exclusion are gone — the preset itself says what
    is in.
    """
    if rules is None:
        return default_rules()
    if isinstance(rules, RuleSet):
        return rules
    if isinstance(rules, str):
        if rules == "carrier_search":
            from catopt_orchestrator.regime import _carrier_search

            return _carrier_search()
        if rules == "xc":
            from catopt_orchestrator.regime import _xc_rules

            return _xc_rules()
        if rules in ("default", "default_rules"):
            return default_rules()
        return preset(rules)
    return RuleSet("custom", tuple(rules))


def _resolve_engine(engine: Any) -> Any:
    """Materialise the saturation engine for :func:`search`.

    ``None`` → the pure-Python reference :class:`EGraph` (the
    default).  A class or zero-arg factory is called to produce the
    engine; an engine instance — anything carrying the
    :class:`~catopt_core.ports.Engine` surface, duck-typed on
    ``add_term`` — is used directly.  Engines are **never
    auto-detected**: installing ``catopt_native`` changes nothing
    until ``engine=`` is passed.
    """
    if engine is None:
        return EGraph()
    if isinstance(engine, type) or not hasattr(engine, "add_term"):
        engine = engine()
    return engine


def _pairing_and_lifts(
    eg: EGraph,
    rules: Iterable,
    root_eid: int,
    stats: dict[str, Any],
    run_cap: int,
    rule_budgets: dict[str, int] | None,
    max_enodes: int | None,
    max_memory_mb: float | None,
    source_tensors: dict,
    meter: Any = None,
    detect_factors: bool = False,
) -> list:
    """Non-local passes with a brief re-saturation between them.

    The diagram-level product law pairs every linear sharing an input
    into one GEMM + split views (no consumer pattern needed); then the
    non-local lifts — unrolled recurrences -> ``trace(F)``, stacks of
    same-state carrier applications -> one application, whole om trees
    over scanned values -> the deferred omd carrier, exact weight
    tying (duplicate Param leaves share one class), and — opt-in via
    ``detect_factors`` — the low-rank factored-parameter offers of
    :func:`catopt_core.laws.factored.offer_low_rank_factors`.  All
    witnessed so certificates stay replayable.

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

    lifts = _carrier_lifts(eg, source_tensors, stats, detect_factors)
    if lifts:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb, meter)
        stats["nonlocal_lifts"] = len(lifts)
        # Provenance the compositional structural cache replays
        # against: exact-tie clusters (``share_duplicate_params``
        # returns ``list[list[str]]`` of param names) and slice-dedup
        # derivations (``share_duplicate_param_slices`` returns dicts
        # carrying ``dedup_param``).  A replayed term that dropped a
        # tied name or references a derived ``__heads`` stack is only
        # valid for a block whose own values reproduce the same
        # sharing structure — the records below are what a cache hit
        # re-checks / re-derives.
        ties = [
            r
            for r in lifts
            if isinstance(r, list)
            and all(isinstance(n, str) for n in r)
        ]
        derived = {
            r["dedup_param"]: {
                "base": r["param"],
                "heads": r["heads"],
                "imap": tuple(r["index_map"]),
            }
            for r in lifts
            if isinstance(r, dict) and "dedup_param" in r
        }
        stats["param_sharing"] = {"ties": ties, "derived": derived}
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb, meter)
    return groups


def _carrier_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_factors: bool,
) -> list:
    """Run the non-local carrier/tying lifts, carriers lazily resolved.

    ``catopt_carriers`` machinery (the carrier lifts) resolves at call
    time; the weight-tying lifts are core.  A partial install without
    carriers contributes only the tying passes.  ``detect_factors``
    arms the opt-in low-rank detection offers of
    :func:`catopt_core.laws.factored.offer_low_rank_factors`.
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
        + _factor_lifts(eg, source_tensors, stats, detect_factors)
    )


def _factor_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_factors: bool,
) -> list:
    """Opt-in low-rank factored-parameter offers (``detect_factors``).

    The detection pass itself is the branch — when off this returns
    ``[]`` without touching the graph; when on, each certified
    offer lands in ``stats["low_rank_factors"]`` (minus the e-class
    id, which means nothing outside this run).
    """
    if not detect_factors:
        return []
    from catopt_core.laws.factored import offer_low_rank_factors

    offers = offer_low_rank_factors(eg, source_tensors)
    if offers:
        stats["low_rank_factors"] = [
            {k: v for k, v in r.items() if k != "eid"} for r in offers
        ]
    return offers


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
            pre_fold = best_term
            best_term = fold(best_term, source_tensors, _cm)
            if _cm.get("_hit"):
                stats["causal_specialized"] = True
                # The compositional cache stores the pre-fold term and
                # re-runs the fold on each hit block's own values —
                # the fold's verdict is value-dependent (the mask must
                # evaluate to the causal triangle), so the cached form
                # is the one BEFORE this rewrite.
                stats["pre_causal_term"] = pre_fold
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
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    symmetry_budget: int | None = 2048,
    max_memory_mb: float | None = None,
    specialize_causal: bool = True,
    fusion_epsilon: float = 0.0,
    delivers_compiled: bool = False,
    meter: Meter | None = None,
    verbose: bool = False,
    stop: str = "fixed_point",
    patience: int = 3,
    engine: Any = None,
    detect_factors: bool = False,
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
    rules : RuleSet, str, or iterable, optional
        The saturation rule set — a first-class
        :class:`~catopt_core.laws.RuleSet`, a preset *name* (resolved
        by :func:`catopt_core.laws.preset` plus the orchestrator-level
        ``"default"`` / ``"carrier_search"`` / ``"xc"``), or a plain
        iterable of rewrites.  ``None`` resolves to
        :data:`DEFAULT_RULES` — ``core.DEFAULT + carriers.CARRIERS``.
        Subsumed/symmetry exclusion is the preset's business: pass
        ``laws.DEFAULT + laws.SYMMETRY`` (or ``laws.FULL``) to opt
        into the symmetry generators.
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
    stop : {"fixed_point", "improving"}, default "fixed_point"
        Saturation stop policy (plan 0010, lever 1d).
        ``"fixed_point"`` runs to the exact fixed point — the
        historical behaviour and the default.  ``"improving"`` is an
        **opt-in heuristic**: the best term is re-extracted after
        every iteration and saturation stops once its cost has not
        strictly improved for ``patience`` consecutive iterations —
        most wins land in the first few iterations, but it can stop
        before the true optimum.  The outcome is recorded in
        ``stats["stop"]`` and ``stats["improved"]``.
    patience : int, default 3
        Consecutive non-improving iterations tolerated under
        ``stop="improving"``.
    engine : Engine, optional
        The saturation engine — an instance, class, or zero-arg
        factory conforming to :class:`catopt_core.ports.Engine`.
        ``None`` uses the pure-Python reference ``EGraph``.  The
        optional native accelerator is
        ``catopt_native.NativeEngine`` — engines are never
        auto-detected, and ``stats["engine"]`` records which ran.
        The engine port covers the *search* only: the non-local
        pairing/lift passes (which mutate through the proof-carrying
        ``union(witness=...)`` surface) run only on the Python
        engine and are skipped otherwise; certificates likewise stay
        a Python-engine concern.
    detect_factors : bool, default False
        Opt-in low-rank detection pass
        (:func:`catopt_core.laws.factored.offer_low_rank_factors`):
        weight parameters whose *stored values* factor at rank ``r``
        below the ``r·(i+o) < i·o`` break-even get a witnessed
        factored member (``x@A@B`` / ``linear(linear(x,A),B)``) in
        their consumer's e-class — a dense-but-low-rank ``Linear``
        then extracts as the two-GEMM chain when the cost model
        prefers it.  Spelled factorisations (LoRA ``lora_A`` /
        ``lora_B`` pairs, ``x@(A@B)`` terms) need no detection — the
        ``assoc_*`` laws already derive both directions and the cost
        model picks; this flag covers weights whose low rank is only
        visible in the values.
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
    eg = _resolve_engine(engine)
    root_eid = eg.add_term(ir.root)
    rules = _resolve_rules(rules)
    if verbose:
        print(f"  Rules: {[r.name for r in rules]}")

    # Bounded-saturation budget for the EXPANSIVE-tagged rules (the
    # tag replaced the ``_EXPANSIVE_RULES`` name list); enforced
    # inside the matcher so a giant e-class cannot spend the whole
    # budget in one enumeration.
    rule_budgets = (
        {
            r.name: symmetry_budget
            for r in rules.tagged(_law_tags.EXPANSIVE)
        }
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
        stop=stop,
        patience=patience,
        cost_fn=cost_fn,
    )
    # Which saturation core ran — the reference engine needs no marker,
    # engines declare ``engine_name`` ("native", ...).
    stats["engine"] = getattr(eg, "engine_name", "python")
    _check_resources(eg, max_enodes, max_memory_mb, meter)

    if isinstance(eg, EGraph):
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
            detect_factors,
        )
    else:
        # Non-local passes (pairing + carrier lifts) offer members
        # through the proof-carrying ``union(witness=...)`` surface —
        # a Python-engine capability the ``Engine`` port does not
        # require.  A run needing them (or certificates) uses the
        # Python engine; see the ``engine`` docstring.
        groups = []
        stats["nonlocal_passes"] = "skipped (engine is not an EGraph)"

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
    search knobs (``rules``, ``max_iterations``, ``max_enodes``,
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

    ``block_pred`` / ``verify_tol`` / ``max_cross_pairs`` /
    ``cache`` are strategy configuration; the per-block search knobs
    (``rules``, ``max_iterations``, ``cost_fn``, ``max_enodes``,
    ``max_memory_mb``, ``verbose``) ride in ``**kw``.

    ``cache`` controls the structural-signature search cache:
    transformer-style stacks repeat ONE block structure N times with
    different parameters, and the second..Nth searches are redundant.
    ``None``/``True`` (the default) enables a fresh per-run dict;
    ``False`` disables; an explicit dict is used as-is — pass a
    caller-owned mapping to reuse search results across models.
    Replays re-leaf the cached term under the hit block's own names
    and values, re-run the value-dependent folds, and verify
    numerically before grafting; a replay that cannot honour the
    template's assumptions declines to the ordinary per-block search.
    Hits/misses land in ``stats["cache"]``.
    """

    name = "compositional"

    def __init__(
        self,
        *,
        block_pred: Callable | None = None,
        verify_tol: float = 1e-4,
        max_cross_pairs: int = 8,
        cache: dict | bool | None = None,
    ) -> None:
        """Store the strategy configuration."""
        self.block_pred = block_pred
        self.verify_tol = verify_tol
        self.max_cross_pairs = max_cross_pairs
        self.cache = cache

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run the per-block pipeline through the optimizer's ports."""
        if self.cache is False:
            cache = None
        elif self.cache is None or self.cache is True:
            cache = {}
        else:
            cache = self.cache
        mod, stats = _optimize_compositional(
            model,
            x,
            optimizer=optimizer,
            block_pred=self.block_pred,
            verify_tol=self.verify_tol,
            max_cross_pairs=self.max_cross_pairs,
            cache=cache,
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
    :data:`catopt_orchestrator.autotune.CANDIDATE_BUILDERS`
    (``"eager"``), or ``(name, builder)`` tuples; backend-provided
    builders arrive through ``builders=`` (the torch wrapper maps
    ``"torch_compile"`` / ``"torch_compile_generic"`` /
    ``"cuda_graph"`` via ``catopt_torch.autotune.TORCH_BUILDERS``).  The
    remaining fields are the timing / budget / verify knobs.
    ``optimize`` kwargs (``rules``, ``max_iterations``, ``runner``,
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
        from catopt_orchestrator.autotune import _autotuned_impl

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
    #: The optimizer's default saturation rule set — ``None`` resolves
    #: to the composed :data:`DEFAULT_RULES` at search time.
    rules: RuleSet | str | Iterable | None = None
    #: The default saturation engine — ``None`` uses the pure-Python
    #: reference ``EGraph``; an instance, class, or zero-arg factory
    #: conforming to :class:`catopt_core.ports.Engine` (e.g.
    #: ``catopt_native.NativeEngine``) selects explicitly.  Never
    #: auto-detected.
    engine: Any = None

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
        kw.setdefault("rules", self.rules)
        kw.setdefault(
            "delivers_compiled",
            bool(getattr(self.runner, "delivers_compiled", False)),
        )
        kw.setdefault("meter", self.meter)
        kw.setdefault("engine", self.engine)
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

        ``max_iterations=6`` unless the caller says otherwise (the
        ``rules`` default is the composed :data:`DEFAULT_RULES`);
        the :class:`SearchResult` carries the
        whole inspectable mid-state (``.alternatives()`` /
        ``.certificate()`` / ``.eg`` / ``.stats``).
        """
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
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 6,
    cost_fn: CostFn | None = None,
) -> SearchResult:
    """Enumerate the frontier — return the :class:`SearchResult`.

    The discovery-engine view: the same export → e-graph → saturation
    → pairing pipeline as :func:`search` with frontier-oriented
    defaults (``max_iterations=6``; ``rules=None`` resolves to the
    composed :data:`DEFAULT_RULES`) — and
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

    ``rules`` follows :func:`search`'s contract — a ``RuleSet``, a
    preset name, a plain iterable, or ``None`` for the composed
    default.  The pairing-subsumed fusion rules are excluded by the
    presets, never by a filter inside the pipeline.
    """
    return search(
        model,
        x,
        source=source,
        capabilities=capabilities,
        cost_fn=cost_fn,
        rules=rules,
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


# ---------------------------------------------------------------------------
#  Structural-signature cache — repeated-block search reuse
# ---------------------------------------------------------------------------
#
# Stacked transformer-style models repeat one block structure N times
# with different parameters: ``Compositional`` would run N identical
# equality-saturation searches.  The cache canonicalises each block's
# exported IR modulo leaf *names* (``structural_key``) and stores the
# extracted term under it; a hit re-leafs that term with the current
# block's own parameter names, re-derives the value-dependent pieces
# (tied-weight assumptions, ``__heads`` dedup stacks, the causal-mask
# fold), lowers it and verifies numerically before grafting — a replay
# that cannot honour the template's assumptions declines to the
# ordinary per-block search.


def _canon_attr(v: Any, ctx: dict) -> Any:
    """Canonical key material for one op attr value.

    Term-likes in attr position canonicalise through the same walk
    (attrs may embed subterms); containers recurse; other unhashable
    values fall back to ``repr``.
    """
    if isinstance(v, (Op, Param, Var, Const, list, tuple, dict)):
        return _canon_term(v, ctx)
    try:
        hash(v)
    except TypeError:
        return repr(v)
    return v


def _canon_leaf(t: Any, ctx: dict) -> Any:
    """Canonical key for a leaf — Param/Var/Const or a non-term node.

    No memo needed: Param positions are idempotent (``setdefault`` on
    the first-occurrence index) and a leaf key is O(1) to recompute.
    A Var absent from ``inputs`` falls back to its name — a
    conservative miss, never a collision.
    """
    if isinstance(t, Param):
        pos = ctx["params"].setdefault(t.name, len(ctx["params"]))
        return (
            "param",
            pos,
            tuple(t.typ.shape),
            ctx["meta"].get(t.name),
        )
    if isinstance(t, Var):
        pos = ctx["vars"].get(t.name)
        return (
            "var",
            pos if pos is not None else t.name,
            tuple(t.typ.shape),
            ctx["meta"].get(t.name),
        )
    if isinstance(t, Const):
        return ("const", type(t.value).__name__, t.value)
    return ("leaf", type(t).__name__, repr(t))


def _canon_term(t: Any, ctx: dict) -> Any:
    """Canonical form of one term under the leaf-renaming ``ctx``.

    ``ctx`` carries ``vars`` (Var name -> input position), ``params``
    (Param name -> first-occurrence position, built by the walk),
    ``meta`` (leaf name -> extra key material, e.g. dtype) and the
    DAG ``memo`` — interned ``Op`` objects only (always hashable), so
    shared subterms canonicalise once and hash-consed duplicates are
    free.
    """
    if isinstance(t, (list, tuple)):
        return ("seq", tuple(_canon_term(a, ctx) for a in t))
    if isinstance(t, dict):
        return (
            "map",
            tuple(
                sorted((k, _canon_term(v, ctx)) for k, v in t.items())
            ),
        )
    if not isinstance(t, Op):
        return _canon_leaf(t, ctx)
    hit = ctx["memo"].get(t)
    if hit is not None:
        return hit
    key = (
        "op",
        t.op,
        tuple(_canon_term(a, ctx) for a in t.args),
        tuple(
            sorted((n, _canon_attr(v, ctx)) for n, v in t.attrs.items())
        ),
    )
    ctx["memo"][t] = key
    return key


def _canonical(
    root: Any,
    inputs: Iterable,
    leaf_meta: Mapping | None,
) -> tuple[tuple, tuple[str, ...]]:
    """Canonical key plus the Param first-occurrence name order.

    The key is ``(body, inputs_sig)``: the body canonicalises the term
    DAG modulo leaf names; ``inputs_sig`` pins the input arity and
    every input's declared shape/meta — including inputs the root
    never references.  The second return lists Param names in the same
    first-occurrence order the key's positional indices encode.
    """
    meta = leaf_meta if leaf_meta is not None else {}
    inputs = tuple(inputs)  # may be any iterable — consumed twice
    ctx = {
        "meta": meta,
        "vars": {v.name: i for i, v in enumerate(inputs)},
        "params": {},
        "memo": {},
    }
    inputs_sig = tuple(
        (tuple(v.typ.shape), meta.get(v.name)) for v in inputs
    )
    return (_canon_term(root, ctx), inputs_sig), tuple(ctx["params"])


def structural_key(
    root: Any,
    inputs: Iterable = (),
    leaf_meta: Mapping | None = None,
) -> tuple:
    """Canonical hashable signature of a term, modulo leaf names.

    Two exported blocks are structurally identical iff their roots
    have equal keys: the same op-DAG (op names, argument order, attr
    values), the same ``Const`` leaves (type-tagged, so ``Const(2)``
    and ``Const(2.0)`` differ), and ``Var``/``Param`` leaves that vary
    at most by name.  A ``Var`` canonicalises to its position in
    ``inputs`` (so ``sub(x, y)`` and ``sub(y, x)`` differ), a ``Param``
    to its first-occurrence position in the deterministic traversal.

    ``leaf_meta`` maps a leaf name to extra key material — the
    compositional pass supplies each leaf's dtype (``TensorType``
    carries shape only), so same-shape different-dtype blocks never
    share an entry.
    """
    return _canonical(root, inputs, leaf_meta)[0]


def _param_order(root: Any) -> tuple[str, ...]:
    """Param leaf names in ``structural_key``'s first-occurrence order."""
    return _canonical(root, (), None)[1]


def _param_leaves(term: Any) -> Any:
    """Yield every ``Param`` leaf reachable in *term* (DAG-memoized)."""
    seen: set = set()

    def walk(t: Any) -> Any:
        try:
            if t in seen:
                return
            seen.add(t)
        except TypeError:
            pass  # unhashable container arg — walk through it anyway
        if isinstance(t, Param):
            yield t
        elif isinstance(t, Op):
            for a in t.args:
                yield from walk(a)
        elif isinstance(t, (list, tuple)):
            for a in t:
                yield from walk(a)

    yield from walk(term)


def _term_param_names(term: Any) -> set[str]:
    """Set of ``Param`` leaf names reachable in *term*."""
    return {p.name for p in _param_leaves(term)}


def _param_shapes_ok(term: Any, params: Mapping) -> bool:
    """Check remapped Param leaves' declared shapes against values.

    A leaf whose declared shape is fully concrete must match the
    supplied tensor's shape — a cheap structural guard that catches a
    term cached under the wrong key before any lowering runs.  Leaves
    with unknown dims (``None``) or names absent from ``params`` are
    skipped; coverage is enforced separately.
    """
    for p in _param_leaves(term):
        declared = tuple(p.typ.shape)
        if None in declared:
            continue
        t = params.get(p.name)
        if t is not None and tuple(getattr(t, "shape", ())) != declared:
            return False
    return True


def _leaf_equal(a: Any, b: Any) -> bool:
    """Exact value equality of two tensor-like leaves (torch-free).

    Mirrors ``catopt_core.laws.pairing._exact_equal`` — same shape and
    elementwise equal via the objects' own ``==``/``.all()``.
    """
    try:
        if a is None or b is None or tuple(a.shape) != tuple(b.shape):
            return False
        r = a == b
        return bool(r.all() if hasattr(r, "all") else r)
    except Exception:
        return False


def _dedup_blocks(
    t: Any, h: int
) -> tuple[list, tuple[int, ...], int, int] | None:
    """First-occurrence unique head-blocks of ``t`` plus the index map.

    Mirrors ``share_duplicate_param_slices``: split the (o, i) tensor
    into ``h`` row-blocks, keep first occurrences by raw bytes, return
    ``(unique_blocks, index_map, d, i)`` — ``None`` when the split is
    degenerate.  Tensor ops are duck-typed (``new_zeros`` / slicing /
    ``numpy().tobytes()``) — the orchestrator stays backend-neutral.
    """
    o, i = int(t.shape[0]), int(t.shape[1])
    if h < 2 or o % h or o // h <= 0:
        return None
    d = o // h
    blocks = [t[j * d : (j + 1) * d] for j in range(h)]
    sigs = [
        b.detach().cpu().contiguous().numpy().tobytes() for b in blocks
    ]
    uniq_sigs: list[bytes] = []
    uniq_blocks: list = []
    imap: list[int] = []
    for j, s in enumerate(sigs):
        try:
            imap.append(uniq_sigs.index(s))
        except ValueError:
            uniq_sigs.append(s)
            uniq_blocks.append(blocks[j])
            imap.append(len(uniq_sigs) - 1)
    return uniq_blocks, tuple(imap), d, i


def _derive_dedup(
    rec: Mapping, pmap: Mapping, source_tensors: Mapping
) -> tuple[Any, str] | None:
    """Recompute a ``{base}__heads{h}`` dedup stack for new values.

    The replay is servable only when the new block's redundancy
    pattern reproduces the recorded index map exactly — otherwise the
    template term's ``index_select`` would gather the wrong rows.
    Returns ``(value, current_derived_name)`` or ``None``.
    """
    base = pmap.get(rec["base"])
    t = source_tensors.get(base) if base is not None else None
    h, imap = rec["heads"], tuple(rec["imap"])
    if (
        t is None
        or getattr(t, "dim", lambda: -1)() != 2
        or len(imap) != h
    ):
        return None
    try:
        out = _dedup_blocks(t, h)
    except Exception:
        return None
    if out is None or out[1] != imap:
        return None
    uniq_blocks, _cur_imap, d, i = out
    dedup = uniq_blocks[0].new_zeros((len(uniq_blocks), d, i))
    for j, u in enumerate(uniq_blocks):
        dedup[j] = u
    return dedup.detach(), f"{base}__heads{h}"


def _remap_term(
    term: Any, pmap: Mapping, vmap: Mapping, dmap: Mapping
) -> Any | None:
    """Re-leaf *term* under the name maps; None if a leaf is unmappable.

    ``pmap``: template Param name -> current name; ``vmap``: template
    Var name -> the current ``Var`` object; ``dmap``: derived Param
    name -> current derived name.  An unmapped leaf aborts — an
    unrenamed template name would mis-bind at eval time.
    """
    bad: list[str] = []
    memo: dict = {}

    def go(t: Any) -> Any:
        hit = memo.get(t)
        if hit is not None:
            return hit
        if isinstance(t, Op):
            out = Op.make(
                t.op, *(go(a) for a in t.args), **dict(t.attrs)
            )
        elif isinstance(t, Param):
            new = pmap.get(t.name, dmap.get(t.name))
            if new is None:
                bad.append(t.name)
                out = t
            else:
                out = Param(new, t.typ)
        elif isinstance(t, Var):
            out = vmap.get(t.name)
            if out is None:
                bad.append(t.name)
                out = t
        else:
            out = t
        memo[t] = out
        return out

    result = go(term)
    return None if bad else result


def _cache_entry(res: SearchResult) -> dict[str, Any]:
    """Build the replayable template from a completed search.

    The entry stores the extracted term — the PRE-causal-fold form
    when the fold fired, so a replay re-specializes against the new
    block's own mask values — plus the leaf correspondence tables and
    the value-dependent assumptions recorded in
    ``stats["param_sharing"]``: exact-tie clusters to re-check and
    ``__heads`` derivations to recompute.  An unrecorded derived name
    gets a ``None`` recipe, making hits on this entry decline.
    """
    term = res.stats.get("pre_causal_term", res.term)
    term_params = _term_param_names(term)
    sharing = res.stats.get("param_sharing") or {}
    order = _param_order(res.ir.root)
    return {
        "term": term,
        "params": tuple(order),
        "inputs": tuple(v.name for v in res.ir.inputs),
        "term_params": term_params,
        "ties": sharing.get("ties", ()),
        "derived": {
            n: sharing.get("derived", {}).get(n)
            for n in term_params - set(order)
        },
        "stats": res.stats,
    }


def _block_key(ir: IR, source_tensors: Mapping, args: tuple) -> tuple:
    """Structural signature of one exported block.

    ``leaf_meta`` carries each leaf's dtype — the IR's ``TensorType``
    is shape-only, so a float64 block and a float32 twin must not
    share a template.  Param dtypes come from ``source_tensors``;
    input dtypes from the captured positional ``args``.
    """
    meta = {
        n: str(getattr(t, "dtype", type(t).__name__))
        for n, t in source_tensors.items()
    }
    for i, v in enumerate(ir.inputs):
        a = args[i] if i < len(args) else None
        meta[v.name] = str(getattr(a, "dtype", type(a).__name__))
    return structural_key(ir.root, inputs=ir.inputs, leaf_meta=meta)


def _replay_derived(
    entry: dict, pmap: Mapping, source_tensors: Mapping, params: dict
) -> dict[str, str] | None:
    """Materialise derived-param values for the hit block.

    Each ``__heads``-style recipe recomputes the deduplicated stack
    from the CURRENT block's base tensor and lands it under the
    current derived name — the name remap alone cannot supply it since
    the value was computed, at search time, from the template's own
    weights.  Returns the derived-name remap, or ``None`` when any
    recipe fails (unservable entry → caller falls back to search).
    """
    dmap: dict[str, str] = {}
    for dname, rec in entry["derived"].items():
        if rec is None:
            return None
        pair = _derive_dedup(rec, pmap, source_tensors)
        if pair is None:
            return None
        value, cur_name = pair
        dmap[dname] = cur_name
        params[cur_name] = value
    return dmap


def _ties_hold(
    ties: Iterable, term_params: set, pmap: Mapping, params: Mapping
) -> bool:
    """Re-check the template's exact-tie assumptions on new values.

    A param dropped in favour of a tied representative must hold an
    equal tensor in this block too — otherwise the canonical leaf
    substitutes a different weight.  Clusters the term kept wholesale
    (or dropped wholesale) commit to nothing and are skipped; names
    unmappable in this block are likewise inert.  All names are
    template-side, mapped to current names through ``pmap``.
    """
    for cluster in ties:
        kept = [
            pmap[n] for n in cluster if n in term_params and n in pmap
        ]
        dropped = [
            pmap[n]
            for n in cluster
            if n not in term_params and n in pmap
        ]
        if not kept or not dropped:
            continue
        anchor = params.get(kept[0])
        if any(not _leaf_equal(params.get(n), anchor) for n in dropped):
            return False
    return True


def _cache_replay(
    entry: dict | None,
    ir: IR,
    source_tensors: Mapping,
    *,
    sink: Sink,
    source: Source,
    model: Any,
) -> SearchResult | None:
    """Instantiate a cached template for a structurally-equal block.

    Returns a lowering-ready :class:`SearchResult` — the template term
    re-leafed under this block's leaf names, billed against this
    block's own ``param_values`` — or ``None`` when the entry cannot
    serve this block (leaf-arity mismatch, an unrecorded derived
    parameter, a violated tie/dedup assumption, an unmappable leaf, or
    a shape/param coverage failure).  ``None`` sends the caller down
    the ordinary per-block search path.
    """
    if entry is None:
        return None
    order = _param_order(ir.root)
    if len(order) != len(entry["params"]) or len(
        entry["inputs"]
    ) != len(ir.inputs):
        return None
    pmap = dict(zip(entry["params"], order, strict=True))
    vmap = dict(zip(entry["inputs"], ir.inputs, strict=True))

    params = dict(source_tensors)
    dmap = _replay_derived(entry, pmap, source_tensors, params)
    if dmap is None or not _ties_hold(
        entry["ties"], entry["term_params"], pmap, params
    ):
        return None

    term = _remap_term(entry["term"], pmap, vmap, dmap)
    if term is None:
        return None
    # Re-run the causal-mask const fold on this block's own values:
    # a mask still evaluating to the causal triangle re-folds; one
    # that does not stays a materialised argument — either way a
    # correct term.
    fold = getattr(sink, "specialize_causal", None)
    if fold is not None:
        term = fold(term, params, {})
    if not _term_param_names(term) <= set(
        params
    ) or not _param_shapes_ok(term, params):
        return None

    stats = dict(entry["stats"])
    stats["cache_replay"] = True
    return SearchResult(
        ir=ir,
        # The replayed result carries no saturated e-graph — the search
        # was skipped.  ``lower`` never touches the engine fields.
        eg=EGraph(),
        root_eid=-1,
        term=term,
        param_values=params,
        stats=stats,
        source=source,
        model=model,
    )


def _optimize_compositional(
    model: Any,
    example_input: Any,
    *,
    optimizer: Optimizer,
    block_pred: Callable | None = None,
    cost_fn: CostFn | None = None,
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verify_tol: float = 1e-4,
    max_cross_pairs: int = 8,
    cache: dict | None = None,
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

    ``cache`` (the :class:`Compositional` strategy resolves its
    ``cache=`` flag into this mapping): when a dict is given, each
    block's exported IR is canonicalised by :func:`structural_key` —
    same op-DAG, same constants, Params/Vars differing at most by
    name (positions, shapes and dtypes are pinned) — and a block whose
    key already has a template replays the cached extracted term under
    its own leaf names/values instead of re-running the search.  The
    replay re-derives the template's value-dependent assumptions
    (exact-tie clusters and ``__heads`` dedup stacks recorded in
    ``stats["param_sharing"]``; the causal-mask fold re-evaluated on
    the hit block's tensors) and the delivered module is verified
    numerically like any other — an unservable or failing replay
    falls back to the ordinary search.

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
    failed and the input was returned unmodified.  With caching enabled,
    ``stats["cache"]`` reports ``{"hits", "misses", "fallbacks"}`` and
    each block's report carries ``"cache"``: ``"hit"`` / ``"miss"`` /
    ``"replay_failed"``.
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
    cache_hits = 0
    cache_misses = 0
    cache_fallbacks = 0

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
            opt_mod = st = None
            key = None
            entry = None
            if cache is not None:
                # Export once up-front for the structural signature —
                # a cache hit skips the search phase entirely.
                kir, ktensors = source.to_ir(block, ex)
                key = _block_key(kir, ktensors, args)
                entry = cache.get(key)
                res2 = _cache_replay(
                    entry,
                    kir,
                    ktensors,
                    sink=sink,
                    source=source,
                    model=block,
                )
                if res2 is not None:
                    try:
                        lr2 = optimizer.lower(
                            res2,
                            ex,
                            runner=IdentityRunner(),
                            verify=verbose,
                            verbose=verbose,
                        )
                        vh = sink.verify(
                            block, lr2.module, args, rtol=verify_tol
                        )
                        if vh.passed:
                            opt_mod, st = lr2.module, lr2.stats
                            cache_hits += 1
                            rep["cache"] = "hit"
                            rep["rel_diff"] = vh.max_rel
                            if verbose:
                                print(
                                    f"[Compositional] {name}: "
                                    f"cache hit ({vh.max_rel:.2e})"
                                )
                    except Exception as exc:
                        # A failed replay simply reverts to search.
                        logger.debug(
                            "[Compositional] %s: cache replay failed: %s",
                            name,
                            exc,
                        )
            if opt_mod is None:
                if entry is not None:
                    # An entry existed but could not serve this block
                    # (unmappable leaf, violated value assumption, or a
                    # failed replay verify) — fall back to the search.
                    cache_fallbacks += 1
                    rep["cache"] = "replay_failed"
                elif cache is not None:
                    rep["cache"] = "miss"
                # Per-block search+lower — the same phases the monolithic
                # pipeline runs, through the optimizer's ports.
                res = optimizer.search(
                    block,
                    ex,
                    rules=rules,
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
                if cache is not None:
                    cache_misses += 1
                    # ``key`` is always set here: a cache-enabled block
                    # that failed its export exited through the outer
                    # ``except`` before reaching the miss path.
                    # ``setdefault``: a fallback block's fresh template
                    # must not evict the existing one — an entry that
                    # declined THIS block may still serve later blocks
                    # whose values honour its assumptions.
                    cache.setdefault(key, _cache_entry(res))
                # Per-block verification on the captured input —
                # soundness gate independent of the pipeline's own
                # (verbose-gated) check.  Any mismatch or eval failure
                # falls back.
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
            rules=rules,
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
    if cache is not None:
        stats["cache"] = {
            "hits": cache_hits,
            "misses": cache_misses,
            "fallbacks": cache_fallbacks,
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
