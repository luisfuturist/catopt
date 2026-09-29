"""Top-level optimization pipeline.

This module implements the four-phase killer experiment:

Phase 1 — Equivalence:   PyTorch → IR
Phase 2 — Search:        IR → e-graph → equality saturation → best term
Phase 3 — Lower:          best term → torch.nn.Module
Phase 4 — Compare:        benchmark vs. vanilla TorchInductor

The main entry point is :func:`optimize_model`.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from catopt_carriers.om_lower import (
    build_om_plan,
    is_om_apply_term,
    to_batched_om_module,
)
from catopt_carriers.omd_lower import (
    build_omd_plan,
    is_omd_apply_term,
    to_batched_omd_module,
)
from catopt_carriers.scan_lower import (
    build_scan_plan,
    is_scan_apply_term,
    to_batched_scan_module,
)
from catopt_carriers.trace_lift import (
    lift_scan_to_applyd,
    lift_scan_to_trace,
)
from catopt_carriers.xcarrier import (
    gather_apply_stack,
    gather_applyd_stack,
    omd_tree_lift,
)
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
from catopt_core.ops import OpTable
from catopt_core.ports import CostFn, Executor, OpRegistry, Sink, Source
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.report import (
    BlockReport,
    CompositionalReport,
    OptReport,
    verify_module,
)

from catopt_optimize.criteria import (
    Criteria,
    Criterion,
    criteria_cost,
)
from catopt_optimize.runners import GenericRunner, Runner

#: Rules whose saturation closure is combinatorially explosive on
#: stacked blocks: the pure-symmetry monoid laws enumerate every
#: bracketing/ordering of a summation (Catalan-scale on the residual
#: accumulator), the scale-hoist laws pair every scale member with
#: every linear, and the distribute/factor/naturality/assoc algebra
#: generates cross-product closures (distribute splits a sum into two
#: matmuls that factor rules then re-pair against *every other*
#: summand — enodes grew 337 → 40k in four iterations on a
#: DeepParallel stack).  ``optimize_model`` runs these under a
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
    ``max_memory_mb``, or a ``torch.cuda.OutOfMemoryError`` /
    ``MemoryError`` surfaced anywhere in the export → saturation →
    lowering pipeline.

    ``optimize_compositional`` records these as ordinary per-block
    failures with ``reason == "resource_limit"``; a standalone
    :func:`optimize_model` caller gets this dedicated type instead of a
    raw OOM.
    """


def _looks_like_oom(exc: BaseException) -> bool:
    """Return True for allocator-failure exceptions.

    Host ``MemoryError``, ``torch.cuda.OutOfMemoryError`` and the
    ``RuntimeError`` variants allocator failures surface as on older
    torch / host-side paths ("CUDA out of memory", DefaultCPUAllocator's
    "can't allocate memory").
    """
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
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


def _current_memory_mb() -> float:
    """Return the current process memory footprint in MiB.

    Host RSS plus CUDA-allocated bytes (device memory lives outside
    RSS).
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
    dev = (
        torch.cuda.memory_allocated() / float(1 << 20)
        if torch.cuda.is_available()
        else 0.0
    )
    return rss + dev


def _check_resources(eg, max_enodes, max_memory_mb) -> None:
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
        used = _current_memory_mb()
        if used > max_memory_mb:
            raise OptimizationResourceError(
                f"memory footprint {used:.0f} MiB exceeds "
                f"max_memory_mb={max_memory_mb}"
            )


def _eval_const(
    term: Any, params: dict, ops: OpRegistry | None = None
) -> torch.Tensor | None:
    """Evaluate a parameter-only subtree to a concrete tensor.

    Permissive compile-time fold — any un-evaluatable piece (Var,
    missing Param, missing binding, raising binding, non-tensor
    result) yields ``None``.  Delegates to
    :func:`catopt_torch.torch_bridge.eval_term` (plan 0002 phase D);
    ``tensor_only`` reproduces the per-level isinstance check.
    """
    from catopt_torch.torch_bridge import _IR_TO_TORCH, eval_term

    bindings = _IR_TO_TORCH if ops is None else ops.torch_bindings
    return eval_term(
        term,
        param_env=params,
        bindings=bindings,
        tensor_only=True,
    )


def _is_causal_keep_mask(mask_val: torch.Tensor, q_shape) -> bool:
    """Return True when a mask is exactly the causal lower triangle.

    ``(…, T, T)`` keeps the lower triangle and T matches q's sequence
    dim — i.e. the mask IS is_causal.
    """
    if not isinstance(q_shape, tuple) or len(q_shape) < 2:
        return False
    if (
        mask_val.ndim < 2
        or mask_val.shape[-1] != mask_val.shape[-2]
        or mask_val.shape[-1] != q_shape[-2]
    ):
        return False
    keep = (
        mask_val.bool()
        if mask_val.dtype == torch.bool
        else mask_val > -1e30
    )
    tril = torch.tril(
        torch.ones(
            mask_val.shape[-2],
            mask_val.shape[-1],
            dtype=torch.bool,
            device=mask_val.device,
        )
    )
    return bool((keep == tril).all())


def _specialize_causal(
    term: Any,
    params: dict,
    memo: dict | None = None,
    ops: OpRegistry | None = None,
) -> Any:
    """sdpa(q,k,v, mask) → sdpa(q,k,v, is_causal=True).

    Applies when mask is parameter-only and evaluates to a causal
    keep-mask.  Dropping the materialised mask unlocks the fused
    flash/mem-efficient kernels.
    """
    from catopt_core.typing import _shape_of as _so

    if memo is None:
        memo = {}
    if not isinstance(term, Op):
        return term
    key = term  # content-keyed: interned terms hash by structure
    if key in memo:
        return memo[key]
    args = tuple(
        _specialize_causal(a, params, memo, ops) for a in term.args
    )
    attrs = dict(term.attrs)
    if term.op == "sdpa" and len(args) >= 4 and not attrs.get("arg5"):
        mv = _eval_const(args[3], params, ops)
        if mv is not None and _is_causal_keep_mask(mv, _so(args[0])):
            args = args[:3]
            attrs["arg5"] = True
            memo["_hit"] = True
    out = Op.make(term.op, *args, **attrs)
    memo[key] = out
    return out


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


#: Carrier-apply root enode ops → (batched plan builder).  A term
#: rooted at one of these lowers through the level-batched executor.
_CARRIER_PLANS = {
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
    if isinstance(term, Op) and term.op in _CARRIER_PLANS:
        plan = _CARRIER_PLANS[term.op](term)
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
    cid = eg.find(root_eid)
    carriers = [
        n for n in eg._classes[cid].nodes if n.op in _CARRIER_PLANS
    ]
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
    sink: Any,
) -> Any:
    """Route the extracted term to the executor that runs it best.

    Executor routing: term-level cost is blind to the lowering — a
    carrier-apply term evaluated by the generic IRModule runs its
    leaves one-by-one (~6-30x slower than the level-batched schedule
    it was priced for).  Route apply roots to their batched executor;
    non-matching roots delegate to IRModule inside each builder, so a
    declined plan degrades to the generic path rather than failing.
    """
    if is_scan_apply_term(best_term):
        return to_batched_scan_module(
            optimized_ir, param_values=source_tensors
        )
    if is_om_apply_term(best_term):
        return to_batched_om_module(
            optimized_ir, param_values=source_tensors
        )
    if is_omd_apply_term(best_term):
        return to_batched_omd_module(
            optimized_ir, param_values=source_tensors
        )
    return sink.lower(optimized_ir, source_tensors)


def discover_alternatives(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    ruleset: str = "categorical",
    max_iterations: int = 6,
    cost_fn: CostFn | None = None,
    top_k: int = 8,
    source: Source | None = None,
    sink: Sink | None = None,
) -> dict:
    """Enumerate the cheapest distinct members of class [G].

    The discovery-engine view.  Runs the same export → e-graph →
    saturation → pairing pipeline as optimize_model, but instead of
    committing to the single best term it returns the top-k alternatives
    under the cost model, plus the rule-fire provenance (which generic
    laws actually fired).  Human inspection of this frontier is how
    level-3 candidates — emergent compositions of known laws — are
    found.

    ``source`` / ``sink`` select the graph source and the (backend-
    relative) sink, defaulting to :class:`TorchSource` /
    :class:`TorchSink`; alternatives are priced against
    ``sink.supported_ops`` so the frontier only ever lists forms the
    sink can lower.
    """
    if source is None:
        source = TorchSource()
    if sink is None:
        sink = TorchSink()
    if cost_fn is None:
        cost_fn = _default_cost_fn()
    cost_fn = backend_cost(cost_fn, sink.supported_ops)
    ir, source_tensors = source.to_ir(model, example_input)
    eg = EGraph()
    root_eid = eg.add_term(ir.root)
    rules = {
        "all": all_rules(),
        "all+layout": ALL_RULES_WITH_LAYOUT,
        "simpl": SIMPLIFICATION_RULES,
        "categorical": CATEGORICAL_RULES,
    }[ruleset]
    if ruleset == "categorical":
        _SUBSUMED = {
            "swiglu_fuse",
            "qkv_fuse",
            "qkv_fuse_asym",
            "parallel_mul_fuse",
        }
        rules = [r for r in rules if r.name not in _SUBSUMED]
    # Same bounded-saturation policy as optimize_model — the frontier
    # stays representative but the call returns in bounded time.
    rule_budgets = {n: 2048 for n in _EXPANSIVE_RULES}
    stats = eg.run(
        rules,
        root_eid,
        max_iterations=max_iterations,
        rule_budgets=rule_budgets,
    )
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    if groups:
        eg.rebuild()
        stats["pairing_groups"] = len(groups)
        eg.run(
            rules, root_eid, max_iterations=5, rule_budgets=rule_budgets
        )

    # Non-local lifts: unrolled recurrences -> trace(F), stacks of
    # same-state carrier applications -> one application, whole om
    # trees over scanned values -> the deferred omd carrier, and exact
    # weight tying (duplicate Param leaves share one class).
    # All witnessed so certificates stay replayable.
    lifts = (
        lift_scan_to_applyd(eg)
        + lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
    )
    if lifts:
        eg.rebuild()
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(
            rules, root_eid, max_iterations=5, rule_budgets=rule_budgets
        )
    alts = eg.extract_alternatives(root_eid, cost_fn, top_k=top_k)
    return {
        "alternatives": alts,
        "diverse_classes": eg.diverse_classes(),
        "rule_fires": dict(
            sorted(eg.rule_fires.items(), key=lambda kv: -kv[1])
        ),
        "stats": stats,
        "ir": ir,
        "eg": eg,
        "root_eid": root_eid,
        "source_tensors": source_tensors,
    }


@_oom_to_resource_error
def optimize_model(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    cost_fn: CostFn | None = None,
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None,
    symmetry_budget: int | None = 2048,
    ops: OpTable | None = None,
    source: Source | None = None,
    sink: Sink | None = None,
    runner: Runner | None = None,
    verbose: bool = True,
) -> tuple[Executor, dict[str, Any]]:
    """End-to-end categorical optimization of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize.
    example_input : torch.Tensor
        An example input for tracing.
    ruleset : str
        Which rewrite rules to use: ``"all"``, ``"simpl"``, or ``"categorical"``.
    max_iterations : int
        Maximum equality-saturation iterations.
    max_enodes : int, optional
        E-node bound on the e-graph.  Enforced once per saturation
        iteration inside ``EGraph.run`` (plus per-match for budgeted
        rules) and re-checked at each phase boundary; reaching it
        raises :class:`OptimizationResourceError` rather than
        extracting from a truncated graph.  ``None`` disables the
        bound (and the run-loop watermark).
    max_memory_mb : float, optional
        Process memory bound in MiB — host RSS plus CUDA-allocated
        bytes — checked at phase boundaries; crossing it raises
        :class:`OptimizationResourceError`.  ``None`` (default)
        disables the check.
    cost_fn : CostFn | None
        Cost function for term extraction — the
        :class:`catopt_core.ports.CostFn` port.  Defaults to
        :func:`executor_cost_for` with ``lowering="generic"`` —
        roofline plus per-node dispatch overhead (solver ops
        surcharged).  Carrier-apply selections are routed to their
        level-batched executors at lowering time rather than priced
        in, because batched cost is non-additive over the spine and
        ``extract_best``'s local-cost decomposition can't see it.
    criteria : dict, Criterion, Criteria, or sequence, optional
        Selection axes blended into the extraction model — a
        ``{axis: weight}`` dict over the named axes
        (:data:`~catopt_optimize.criteria.AXES`), a single
        :class:`~catopt_optimize.criteria.Criterion`, a
        ``Criteria``/``Blend`` composition (e.g.
        ``LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3``),
        or a list of criteria / ``(criterion, weight)`` pairs; see
        :func:`catopt_optimize.criteria.criteria_cost`.  Consulted only
        when ``cost_fn`` is ``None`` — precedence is explicit
        ``cost_fn`` > ``criteria`` > the default model.
        ``stats["criteria"]`` records the normalised axes priced.
    symmetry_budget : int, optional
        Per-rule enode budget for the expansive rules in
        ``_EXPANSIVE_RULES`` (monoid symmetries and scale hoists) —
        bounded saturation.  The reordering closure these rules
        generate grows Catalan-fast on stacked blocks (the residual
        accumulator's bracketings), which is what pushed monolithic
        eqsat past ~2 blocks.  ``None`` restores unbounded
        saturation.  The bound can only *miss* optimizations, never
        introduce wrong ones — every recorded merge is still a real
        equality.
    ops : OpTable, optional
        The op table the optimized term is lowered through (plan 0001
        phase 2c) — used to build the default :class:`TorchSink` and
        for the causal-mask constant evaluation.  ``None`` (default)
        resolves to ``OpTable.full()`` — the ambient ``_IR_TO_TORCH``
        table.  Ignored when ``sink`` is given.
    source : Source, optional
        The graph-source port (``model -> (IR, leaves)``), defaulting
        to :class:`TorchSource` (``torch.export``).  Pass a different
        source to optimize a non-torch frontend — the core search never
        imports torch itself.
    sink : Sink, optional
        The graph-sink port (``IR -> runnable``, plus its op set and
        equivalence gate), defaulting to ``TorchSink(ops=ops)``.
        Extraction is priced against ``sink.supported_ops``
        (:func:`catopt_core.cost.backend_cost`), so the optimizer only
        commits to forms the sink can lower.  Takes precedence over
        ``ops``.
    runner : Runner, optional
        Delivery-stage object deciding HOW the lowered executor is
        executed — see :mod:`catopt_optimize.runners`
        (:class:`GenericRunner` identity, :class:`CompiledRunner`
        ``torch.compile``, :class:`CudaGraphRunner` CUDA-graph
        capture, :class:`ChainedRunner` left-to-right composition —
        e.g. ``ChainedRunner([CompiledRunner(), CudaGraphRunner()])``
        compiles first, then defers capture to the compile's
        outcome).  Applied once to the routed executor;
        ``stats["runner"]`` records its name (a list of member names
        for a chain) and the runner writes its own outcome keys
        (``stats["compiled"]``, ``stats["cuda_graph"]``).  ``None``
        delivers the module as lowered (:class:`GenericRunner`).
        Duck-typed — any object with ``name`` and
        ``apply(module, example_input, stats)`` conforms.
    verbose : bool
        Print progress.

    Returns
    -------
    (optimized_module, stats)
        The optimized ``torch.nn.Module`` and a dictionary of e-graph stats.

    Raises
    ------
    OptimizationResourceError
        When a resource bound is crossed (``max_enodes``,
        ``max_memory_mb``) or an allocator failure —
        ``torch.cuda.OutOfMemoryError``, ``MemoryError``, or the
        equivalent ``RuntimeError`` — is raised anywhere in the
        pipeline.  ``optimize_compositional`` treats this as a normal
        per-block fallback (status ``"failed"``,
        ``reason == "resource_limit"``).

    """
    if source is None:
        source = TorchSource()
    if sink is None:
        sink = TorchSink(ops=ops)
    if cost_fn is None:
        cost_fn = (
            criteria_cost(criteria)
            if criteria is not None
            else _default_cost_fn()
        )
    # Criteria-based selection reports the normalised blend actually
    # priced (the marker criteria_cost sets) — read before the
    # backend_cost wrap, which propagates only the billing markers.
    criteria_used = getattr(cost_fn, "criteria", None)
    # Backend-relative pricing: members using an op the sink cannot
    # lower price at +inf, so extraction never commits to one.
    cost_fn = backend_cost(cost_fn, sink.supported_ops)

    # Delivery runner — the only execution control: None ships the
    # routed executor as lowered (GenericRunner).
    if runner is None:
        runner = GenericRunner()

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
    ir, source_tensors = source.to_ir(model, example_input)
    if verbose:
        print(f"  IR root: {op_repr(ir.root)}")
        print(f"  Inputs:  {[str(v) for v in ir.inputs]}")
        print(f"  Params:  {list(ir.params.keys())}")

    # -- Phase 2: Build e-graph and saturate -----------------------------
    if verbose:
        print(
            "[Phase 2] Building e-graph and running equality saturation..."
        )
    eg = EGraph()
    root_eid = eg.add_term(ir.root)

    # Choose rules.  The term-local fusion rules (swiglu_fuse, qkv_fuse,
    # parallel_mul_fuse, qkv_fuse_asym) are special cases of the product
    # law; in the pipeline they are SUBSUMED by the non-local
    # pair_shared_input_linears pass, which needs no consumer pattern.
    # Keeping them would let extraction pick consumer-level chunk
    # alternatives that bypass the globally-coordinated split choice.
    _SUBSUMED = {
        "swiglu_fuse",
        "parallel_mul_fuse",
        "qkv_fuse",
        "qkv_fuse_asym",
    }
    if ruleset == "all":
        rules = [r for r in all_rules() if r.name not in _SUBSUMED]
    elif ruleset == "all+layout":
        rules = [
            r
            for r in ALL_RULES_WITH_LAYOUT
            if r.name not in _SUBSUMED
        ]
    elif ruleset == "simpl":
        rules = SIMPLIFICATION_RULES
    elif ruleset == "categorical":
        rules = [
            r for r in CATEGORICAL_RULES if r.name not in _SUBSUMED
        ]
    else:
        raise ValueError(f"Unknown ruleset: {ruleset}")

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
    _check_resources(eg, max_enodes, max_memory_mb)

    # Diagram-level product law: pair every linear sharing an input into
    # one GEMM + split views.  Non-local — no consumer pattern needed.
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    if groups:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb)
        stats["pairing_groups"] = len(groups)
        # brief second saturation so other rules see the new enodes
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb)

    # Non-local lifts: unrolled recurrences -> trace(F), stacks of
    # same-state carrier applications -> one application, whole om
    # trees over scanned values -> the deferred omd carrier, and exact
    # weight tying (duplicate Param leaves share one class).
    # All witnessed so certificates stay replayable.
    lifts = (
        lift_scan_to_applyd(eg)
        + lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
    )
    if lifts:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb)
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb)

    stats["rule_fires"] = dict(eg.rule_fires)
    stats["criteria"] = criteria_used
    if verbose:
        print(f"  E-graph: {stats}")

    # -- Extract best term -----------------------------------------------
    best_term = eg.extract_best(root_eid, cost_fn)
    if groups:
        # Coordinated extraction: force every paired member to its split
        # enode AND steer consumers through the shared GEMM.  Compare
        # true DAG costs — forcing loses if a group is only partially
        # reachable or a bypassing alternative was already cheaper.
        # One shared memo across both calls: a forced term shares most
        # subterms with the best term, and the baseline price is
        # computed once, not per candidate.
        _dc_memo: dict = {}
        _best_dag = dag_cost(best_term, cost_fn, memo=_dc_memo)
        forced = eg.extract_paired(root_eid, cost_fn, groups)
        # Honest-decline bookkeeping: an un-extractable forced term
        # prices at +inf, so the same comparison decides and the
        # stats record the verdict with the cost delta.
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
        eg,
        root_eid,
        best_term,
        cost_fn,
        compiled=bool(getattr(runner, "delivers_compiled", False)),
    )
    # Causal specialization: a param-only attn_mask that evaluates to a
    # lower-triangular keep-mask is is_causal=True — no mask op at all.
    _cm: dict = {}
    best_term = _specialize_causal(
        best_term, source_tensors, _cm, ops=sink.ops
    )
    if _cm.get("_hit"):
        stats["causal_specialized"] = True

    if verbose:
        print(f"  Best term: {op_repr(best_term)}")
        print(f"  Cost: {cost_fn(best_term):.2f} FLOPs (est.)")

    # -- Phase 3: Lower back to torch -----------------------------------
    # Final watermark before materialising the lowered parameters —
    # the phase that turns graph choices into real tensor bytes.
    _check_resources(eg, max_enodes, max_memory_mb)
    if verbose:
        print("[Phase 3] Lowering optimized IR to torch module...")
    optimized_ir = IR(
        root=best_term,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    optimized_module = _lower_extracted(
        best_term, optimized_ir, source_tensors, sink
    )
    stats["lowering"] = (
        "batched"
        if getattr(optimized_module, "is_batched", False)
        else "generic"
    )
    # Delivery: the runner decides how the routed executor ships —
    # identity (generic), torch.compile, CUDA-graph capture, or a
    # left-to-right composition.  stats["runner"] records the name
    # (member-name list for a chain); runners write their own outcome
    # keys (stats["compiled"], stats["cuda_graph"]).
    runner_names = getattr(runner, "names", None)
    stats["runner"] = (
        list(runner_names)
        if runner_names is not None
        else getattr(runner, "name", type(runner).__name__)
    )
    optimized_module = runner.apply(
        optimized_module, example_input, stats
    )

    # Verify semantic equivalence
    if verbose:
        print("[Verify] Checking output equivalence...")
        vr = sink.verify(
            model, optimized_module, example_input, rtol=1e-4
        )
        print(f"  Max abs diff:  {vr.max_abs:.6e}")
        print(f"  Max rel diff:  {vr.max_rel:.6e}")
        if vr.passed:
            print("  ✓ Semantically equivalent (within tolerance)")
        else:
            print("  ✗ WARNING: large difference detected!")

    return optimized_module, stats


def param_report(
    model: torch.nn.Module, optimized_module: torch.nn.Module
) -> dict:
    """Joint graph+parameter view of the optimized weights file.

    Which original parameters survive in the optimized realization,
    which were eliminated, and which were derived (folded) — the
    'optimized weights file' diff.

    The optimized module's state_dict IS the smaller weights file:
    ``_fold_weight_chains`` materialises derived tensors (``fused_*``)
    and ``_build_params`` registers only parameters the extracted term
    actually references, so eliminated subgraphs drop their weights
    automatically.  This function makes that auditable.
    """
    orig = {n: p for n, p in model.state_dict().items()}
    opt = {n: p for n, p in optimized_module.state_dict().items()}
    orig_names = {f"p_{n.replace('.', '_')}" for n in orig}
    opt_names = set(opt)
    eliminated = sorted(orig_names - opt_names)
    derived = sorted(n for n in opt_names if n not in orig_names)
    orig_bytes = sum(
        p.numel() * p.element_size() for p in orig.values()
    )
    opt_bytes = sum(p.numel() * p.element_size() for p in opt.values())
    return {
        "original_params": len(orig),
        "optimized_params": len(opt),
        "original_bytes": orig_bytes,
        "optimized_bytes": opt_bytes,
        "eliminated": eliminated,
        "derived": derived,
        "bytes_saved": orig_bytes - opt_bytes,
        "ratio": opt_bytes / orig_bytes if orig_bytes else 1.0,
    }


def save_optimized_weights(
    optimized_module: torch.nn.Module, path: str
) -> None:
    """Emit the optimized weights file.

    Only the parameters the certified form actually needs (folded
    derived tensors included).
    """
    torch.save(optimized_module.state_dict(), path)


def ir_to_string(term: Any) -> str:
    """Pretty-print an IR term as an S-expression."""
    return op_repr(term)


def term_cost(term: Any, cost_fn: CostFn | None = None) -> float:
    """Compute the cost of a term using the given cost function."""
    if cost_fn is None:
        cost_fn = flops_cost
    return cost_fn(term)


# ---------------------------------------------------------------------------
#  Compositional optimization — per-block eqsat, then recompose
# ---------------------------------------------------------------------------


def _default_block_pred(
    parent: torch.nn.Module, name: str, module: torch.nn.Module
) -> bool:
    """Select direct children of ``nn.ModuleList`` / ``nn.Sequential``.

    The default block selector for the standard 'stacked blocks'
    structure.
    """
    return isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    )


def _select_blocks(
    model: torch.nn.Module, block_pred: Callable | None
) -> list[tuple[str, torch.nn.Module]]:
    """Pick the top-most submodules to optimize independently.

    Walks the module tree; the top-most matching blocks are chosen.

    A child is selected when it is a leaf (no children of its own) or when
    ``block_pred(parent, child_name, child)`` is true.  Selected blocks are
    opaque: we never descend into them, so e.g. the ``nn.Linear`` leaves
    inside a matched ``ParallelBlock`` are not optimized separately.
    """
    pred = block_pred or _default_block_pred
    blocks: list[tuple[str, torch.nn.Module]] = []

    def visit(module: torch.nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            full = f"{prefix}.{child_name}" if prefix else child_name
            is_leaf = next(child.children(), None) is None
            if is_leaf or pred(module, child_name, child):
                blocks.append((full, child))
            else:
                visit(child, full)

    visit(model, "")
    return blocks


#: ``io`` key under which the capture pass stores the model's own
#: return value — ``<`` is not a legal module-attribute character, so
#: it can never collide with a real block name.
_MODEL_KEY = "<model>"


def _capture_block_inputs(
    model: torch.nn.Module,
    blocks: list[tuple[str, torch.nn.Module]],
    example_input: torch.Tensor | tuple,
) -> tuple[dict[str, tuple[tuple, dict]], dict[str, dict[str, Any]]]:
    """Record each selected block's first forward inputs via hooks.

    Runs the ORIGINAL model once.  Returns ``(captured, io)``:

    * ``captured`` maps ``{name: (args, kwargs)}`` — detached clones of
      the first call's arguments;
    * ``io`` carries the cross-block dataflow evidence the pairwise
      pass reads: per block ``{"calls", "in_objs", "out_obj", "out"}``
      — the call count, the live arg/output OBJECTS of the first call
      (kept referenced so ``is``-identity stays valid: a freed object's
      id could be reused by a later allocation), and a detached clone
      of the first output — plus ``io["<model>"]`` with the model's
      own return.

      The object identity answers "did B literally consume A's
      output?"; the clones answer "was it modified in between?".
    """
    captured: dict[str, tuple[tuple, dict]] = {}
    io: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(mod, args, kwargs, out):
            entry = io.setdefault(
                name,
                {
                    "calls": 0,
                    "in_objs": (),
                    "out_obj": None,
                    "out": None,
                },
            )
            entry["calls"] += 1
            if name not in captured:
                captured[name] = (
                    tuple(
                        a.detach().clone()
                        if isinstance(a, torch.Tensor)
                        else a
                        for a in args
                    ),
                    {
                        k: (
                            v.detach().clone()
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in kwargs.items()
                    },
                )
                entry["in_objs"] = args
                entry["out_obj"] = out
                entry["out"] = (
                    out.detach().clone()
                    if isinstance(out, torch.Tensor)
                    else out
                )

        return hook

    for name, mod in blocks:
        handles.append(
            mod.register_forward_hook(make_hook(name), with_kwargs=True)
        )
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    try:
        model.eval()
        with torch.no_grad():
            out = model(*args)
        io[_MODEL_KEY] = {
            "out_obj": out,
            "out": (
                out.detach().clone()
                if isinstance(out, torch.Tensor)
                else out
            ),
        }
    finally:
        for h in handles:
            h.remove()
    return captured, io


def _replace_submodule(
    model: torch.nn.Module, dotted: str, new_mod: torch.nn.Module
) -> None:
    """Set ``model.<dotted>`` to ``new_mod``.

    Handles ModuleList / Sequential integer children.
    """
    parent_name, _, child_name = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    ):
        parent[int(child_name)] = new_mod
    else:
        setattr(parent, child_name, new_mod)


def _shared_param_clone(model: torch.nn.Module) -> torch.nn.Module:
    """Deepcopy ``model``'s module structure without copying tensors.

    A plain ``copy.deepcopy`` of an ``nn.Module`` clones every parameter
    and buffer — at ~0.5B fp16 params that doubles device memory before
    a single optimized block is grafted, which is what OOMed
    compositional recompose on small GPUs.  Recomposing only ever
    *replaces* submodules (``_replace_submodule`` rebinds entries in the
    clone's ``_modules`` dicts); it never mutates a tensor in place, so
    the clone can share the original's tensor storage safely.

    The mechanism is deepcopy's own memo: ``copy.deepcopy`` checks
    ``memo[id(obj)]`` before dispatching to ``__deepcopy__``, so
    pre-seeding every reachable tensor's id makes the clone reuse those
    objects while the module ``__dict__``s, ``_modules`` /
    ``_parameters`` / ``_buffers`` dicts, and plain attributes still
    copy normally — the result is a real clone (``training`` flag,
    hooks, structure) whose weights alias the original's.  Grafting
    into it cannot touch the caller's model.

    Seeding covers registered ``parameters()``/``buffers()`` plus
    *unregistered* tensor attributes — ``mod.foo = tensor`` lands in
    ``__dict__`` (only Parameters go to ``_parameters`` and only
    registered buffers to ``_buffers``), which the default deepcopy
    walk traverses by id, so the memo shares them too.  Tensors nested
    inside non-tensor container/attribute objects (e.g.
    ``self.cache = {"k": t}``) are not memo-hit and still clone; if
    even that fails, the caller's in-place fallback applies.
    """
    memo: dict[int, Any] = {}
    for t in list(model.parameters()) + list(model.buffers()):
        memo[id(t)] = t
    for mod in model.modules():
        for v in vars(mod).values():
            if isinstance(v, torch.Tensor):
                memo[id(v)] = v
    return copy.deepcopy(model, memo)


# ---------------------------------------------------------------------------
#  Pairwise cross-block pass — jointly optimize adjacent block pairs
# ---------------------------------------------------------------------------
#
# Per-block optimization is blind across the boundary: block i's output
# projection can compose with block i+1's input projections (a weight-only
# chain that folds to one stored matrix), and a residual ``+`` between
# them can absorb a shared affine.  For each *adjacent* pair the pass
# classifies the boundary, builds a joint micro-model wrapping
# ``B(A(x))``, runs the ordinary :func:`optimize_model` on it, verifies
# it against the eager pair, and grafts the joint module into the clone —
# only when it is verified AND cheaper than the two separately-optimized
# results.

#: Symmetry budget for joint runs.  The joint is a two-block
#: micro-model — the reordering closure that motivated bounded
#: saturation dominates its cost, and the documented break-even is
#: identical extracted cost at every budget ≥ 512.  Truncating the
#: reordering closure can only *miss* rewrites (the pair then simply
#: declines on cost), never produce a wrong one.
_CROSS_PAIR_SYMMETRY_BUDGET = 512


def _perturbed_input(example_input: Any) -> Any:
    """Return a second probe input — different values, same structure.

    The residual-boundary check requires ``b_in == a_in + a_out``; on
    ONE example a coincidental value match could promote a false
    boundary, so the relation must also hold on a perturbed probe.
    Only floating tensors are perturbed — a perturbation would corrupt
    non-float (index) inputs.
    """

    def _perturb(t: Any) -> Any:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            return t * 1.5 + 0.01
        return t

    if isinstance(example_input, tuple):
        return tuple(_perturb(a) for a in example_input)
    return _perturb(example_input)


def _residual_probe(
    name_a: str,
    name_b: str,
    captured2: dict[str, tuple[tuple, dict]],
    io2: dict[str, dict[str, Any]],
) -> bool:
    """Second-probe confirmation of a residual boundary.

    On the perturbed-input capture, ``b_in == a_in + a_out`` must hold
    again — a coincidence of values on one example can't promote the
    pair.  Any absence (block not executed on the probe path, extra
    call, non-tensor piece) fails closed.
    """
    ca = captured2.get(name_a)
    cb = captured2.get(name_b)
    ia = io2.get(name_a)
    if ca is None or cb is None or ia is None or ia["calls"] != 1:
        return False
    args_a, _ = ca
    args_b, _ = cb
    a_out = ia["out"]
    if len(args_a) != 1 or len(args_b) != 1:
        return False
    a_in, b_in = args_a[0], args_b[0]
    if not (
        isinstance(a_in, torch.Tensor)
        and isinstance(b_in, torch.Tensor)
        and isinstance(a_out, torch.Tensor)
    ):
        return False
    return bool(
        a_in.shape == a_out.shape and torch.equal(b_in, a_in + a_out)
    )


def _executor_flops(mod: Any) -> float:
    """Delivered FLOPs of a lowered executor.

    Prices the module's post-fold root term — the computation that
    actually runs (weight chains are already materialised to single
    params).  Carrier-batched executors expose the serial root through
    ``eval_mod``; anything unpriceable returns ``inf`` so the pair
    comparison simply keeps the separate modules.
    """
    root = getattr(mod, "_root", None)
    if root is None:
        root = getattr(getattr(mod, "eval_mod", None), "_root", None)
    if root is None:
        return float("inf")
    return float(flops_cost(root))


class _JointPair(torch.nn.Module):
    """Joint micro-model for one adjacent pair, in the boundary's mode.

    ``mode`` is the :func:`_pair_boundary` verdict — the joint function
    of A's input ``x`` the optimizer sees:

    * ``"chain"`` — ``x |-> B(A(x))``
    * ``"chain_wrapped"`` — ``x |-> A(x) + B(A(x))`` (the parent's own
      ``y + B(y)`` around B is part of the segment)
    * ``"residual"`` — ``x |-> B(x + A(x))``
    * ``"residual_wrapped"`` — ``x |-> (x + A(x)) + B(x + A(x))``
    """

    def __init__(
        self,
        a: torch.nn.Module,
        b: torch.nn.Module,
        mode: str,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a(x)
        if self.mode.startswith("residual"):
            y = x + y
        out = self.b(y)
        if self.mode.endswith("_wrapped"):
            out = y + out
        return out


class _FusedPair(torch.nn.Module):
    """Delivery wrapper grafted at block A's slot for a fused pair.

    ``inner`` is the jointly-optimized executor computing the whole
    segment as a function of A's input; B's slot becomes
    ``nn.Identity`` (B consumed plainly) or :class:`_Zero` (the parent
    residual-wraps B, so its slot must contribute a zero addend).

    For a residual A-boundary the parent's own ``x + ·`` still runs, so
    the wrapper returns the *delta* ``inner(x) - x`` and the outer add
    reconstructs ``inner(x)`` (to within one rounding step).
    """

    def __init__(self, inner: torch.nn.Module, delta: bool) -> None:
        super().__init__()
        self.inner = inner
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        return out - x if self.delta else out


class _Zero(torch.nn.Module):
    """Exact-zero placeholder for a consumed, residual-wrapped slot.

    The fused pair delivers the whole segment upstream; the parent's
    ``y + ·`` around B still executes, so this slot contributes an
    exact zero addend — adding literal zeros is lossless, unlike a
    computed ``y - y`` on non-finite values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _plain_consumers(
    io: dict[str, dict[str, Any]],
    captured: dict[str, tuple[tuple, dict]],
    out_obj: Any,
    out_val: torch.Tensor,
) -> list[str]:
    """Block names whose captured args hold ``out`` (identity + value).

    Object identity proves the same tensor flowed in; comparing the
    captured clone rules out an in-place rewrite between producer and
    consumer.
    """
    hits = []
    for n, m in io.items():
        c_args = captured.get(n, ((), {}))[0]
        for arg, real in zip(
            c_args, m.get("in_objs", ()), strict=False
        ):
            if (
                real is out_obj
                and isinstance(arg, torch.Tensor)
                and torch.equal(arg, out_val)
            ):
                hits.append(n)
                break
    return hits


def _io_has_value(
    captured: dict[str, tuple[tuple, dict]],
    model_out: Any,
    val: torch.Tensor,
) -> bool:
    """Return True when ``val`` appears verbatim in the captured flow.

    Checks the model's return and every block's captured positional
    args — the evidence that a *computed* sum (like ``b_in + b_out``)
    is what actually flows on.
    """
    if isinstance(model_out, torch.Tensor) and torch.equal(
        model_out, val
    ):
        return True
    return any(
        isinstance(a, torch.Tensor) and torch.equal(a, val)
        for args, _ in captured.values()
        for a in args
    )


def _pair_boundary(
    name_a: str,
    name_b: str,
    captured: dict[str, tuple[tuple, dict]],
    io: dict[str, dict[str, Any]],
    captured2: dict[str, tuple[tuple, dict]],
    io2: dict[str, dict[str, Any]],
) -> str | None:
    """Classify the A→B dataflow of an adjacent block pair.

    Returns the joint-micro-model mode — ``"chain"`` /
    ``"chain_wrapped"`` / ``"residual"`` / ``"residual_wrapped"`` — or
    ``None`` when the boundary is not a simple value flow.

    A-side (what flows INTO B):

    * ``chain`` — B's input IS A's output: the same live tensor object,
      unmodified between the two calls (the captured values still
      compare equal), and consumed by B alone;
    * ``residual`` — B's input is ``a_in + a_out`` (the ``x + A(x)``
      residual pattern), confirmed on the perturbed second-probe
      capture too, and A's output feeds nothing else.

    B-side (how B's OUTPUT is consumed — it decides the graft, because
    the parent's ``b_in + B(b_in)`` wrap makes B's slot an addend, not
    a value):

    * plain — B's output object reaches another block or the model
      return unmodified → B's slot becomes ``nn.Identity``;
    * ``_wrapped`` — the sum ``b_in + b_out`` is what flows on → B's
      slot becomes :class:`_Zero` and the joint absorbs B's residual;
    * both or neither → ``None`` (ambiguous or unknown downstream).

    Identity checks run on the retained live objects (``is``); equality
    on detached clones — a coincidence of values alone never promotes a
    boundary.
    """
    ca = captured.get(name_a)
    cb = captured.get(name_b)
    ia = io.get(name_a)
    ib = io.get(name_b)
    if ca is None or cb is None or ia is None or ib is None:
        return None
    if ia["calls"] != 1 or ib["calls"] != 1:
        # A re-entered block is called again outside the pair window —
        # a fused graft would rewrite that later call too.
        return None
    args_a, kw_a = ca
    args_b, kw_b = cb
    if kw_a or kw_b or len(args_a) != 1 or len(args_b) != 1:
        return None
    a_in, b_in, a_out = args_a[0], args_b[0], ia["out"]
    if not (
        isinstance(a_in, torch.Tensor)
        and isinstance(b_in, torch.Tensor)
        and isinstance(a_out, torch.Tensor)
    ):
        return None
    model = io.get(_MODEL_KEY, {})
    model_out, model_out_obj = model.get("out"), model.get("out_obj")
    a_out_obj = ia["out_obj"]
    if a_out_obj is model_out_obj:
        return None  # A's output escapes the pair entirely
    a_fans = _plain_consumers(io, captured, a_out_obj, a_out)
    if ib["in_objs"][0] is a_out_obj:
        # B literally consumed A's output object — and nothing else did.
        if a_fans != [name_b]:
            return None
        a_mode = "chain"
    elif a_fans:
        # A's output feeds another block as well — not a simple edge.
        return None
    elif not (
        a_in.shape == a_out.shape
        and torch.equal(b_in, a_in + a_out)
        and _residual_probe(name_a, name_b, captured2, io2)
    ):
        return None
    else:
        a_mode = "residual"
    # B-side: how is B's own output consumed?
    b_out, b_out_obj = ib["out"], ib["out_obj"]
    if not isinstance(b_out, torch.Tensor):
        return None
    plain_ev = bool(
        _plain_consumers(io, captured, b_out_obj, b_out)
    ) or (
        b_out_obj is model_out_obj
        and isinstance(model_out, torch.Tensor)
        and torch.equal(model_out, b_out)
    )
    wrapped_ev = b_in.shape == b_out.shape and _io_has_value(
        captured, model_out, b_in + b_out
    )
    if plain_ev == wrapped_ev:
        # Neither evidence, or both (ambiguous downstream) — decline.
        return None
    return a_mode + ("_wrapped" if wrapped_ev else "")


def _cross_pair_pass(
    blocks: list[tuple[str, torch.nn.Module]],
    captured: dict[str, tuple[tuple, dict]],
    io: dict[str, dict[str, Any]],
    captured2: dict[str, tuple[tuple, dict]],
    io2: dict[str, dict[str, Any]],
    replacements: dict[str, torch.nn.Module],
    block_reports: dict[str, BlockReport],
    agg: dict[str, Any],
    *,
    ruleset: str,
    max_iterations: int,
    max_enodes: int | None,
    max_memory_mb: float | None,
    cost_fn: CostFn,
    verify_tol: float,
    ops: OpTable | None,
    max_cross_pairs: int,
    verbose: bool,
) -> dict[str, dict[str, Any]]:
    """Jointly optimize adjacent block pairs across their boundary.

    For each consecutive pair ``(blocks[i], blocks[i+1])`` in execution
    order whose boundary is a simple value flow (:func:`_pair_boundary`),
    build the :class:`_JointPair` micro-model, run the ordinary
    :func:`optimize_model` on it (pairing, residual folds and scale
    hoists apply across the two-block composition), verify the lowered
    joint against the eager pair at ``verify_tol``, and graft it — a
    :class:`_FusedPair` at A's slot, ``nn.Identity`` / :class:`_Zero`
    at B's — only when verified AND its delivered FLOPs beat the sum
    of the two separately-optimized results.

    Combinatorics are capped: adjacent pairs only, no overlap (a block
    consumed by a graft cannot re-pair), and at most
    ``max_cross_pairs`` joint optimization runs.  Every failure is a
    silent decline recorded as ``{pair: {"status", ...}}`` —
    ``"grafted"``, ``"declined"`` (with ``reason``), or ``"skipped"``
    (with ``reason``).  ``replacements``/``block_reports``/``agg`` are
    updated in place for grafted pairs so the aggregate param report
    keeps describing what is actually delivered.
    """
    reports: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    attempts = 0
    for i in range(len(blocks) - 1):
        name_a, mod_a = blocks[i]
        name_b, mod_b = blocks[i + 1]
        pair = f"{name_a}+{name_b}"
        if name_a in consumed or name_b in consumed:
            reports[pair] = {
                "status": "skipped",
                "reason": "member already fused",
            }
            continue
        if name_a not in replacements or name_b not in replacements:
            reports[pair] = {
                "status": "skipped",
                "reason": "block not optimized",
            }
            continue
        mode = _pair_boundary(
            name_a, name_b, captured, io, captured2, io2
        )
        if mode is None:
            reports[pair] = {
                "status": "skipped",
                "reason": "no simple boundary",
            }
            continue
        if attempts >= max_cross_pairs:
            reports[pair] = {
                "status": "skipped",
                "reason": f"max_cross_pairs={max_cross_pairs}",
            }
            continue
        attempts += 1
        t0 = time.time()
        entry: dict[str, Any] = {"boundary": mode}
        try:
            joint = _JointPair(mod_a, mod_b, mode)
            (x,) = captured[name_a][0]
            opt_j, st_j = optimize_model(
                joint,
                x,
                ruleset="all+layout" if ruleset == "all" else ruleset,
                max_iterations=max_iterations,
                max_enodes=max_enodes,
                max_memory_mb=max_memory_mb,
                cost_fn=cost_fn,
                ops=ops,
                symmetry_budget=_CROSS_PAIR_SYMMETRY_BUDGET,
                verbose=verbose,
            )
            entry["stats"] = st_j
            # Soundness gate, same tolerance convention as the
            # per-block verify — the eager pair vs its lowering.
            vr = verify_module(joint, opt_j, (x,), rtol=verify_tol)
            entry["rel_diff"] = vr.max_rel
            if not vr.passed:
                entry["status"] = "declined"
                entry["reason"] = (
                    f"joint verify failed: {vr.max_rel:.3e}"
                )
            else:
                j_cost = _executor_flops(opt_j)
                sep = _executor_flops(
                    replacements[name_a]
                ) + _executor_flops(replacements[name_b])
                entry["joint_cost"] = j_cost
                entry["separate_cost"] = sep
                if j_cost >= sep:
                    entry["status"] = "declined"
                    entry["reason"] = "no cost improvement"
                else:
                    fused = _FusedPair(
                        opt_j, delta=mode.startswith("residual")
                    )
                    pr_j = param_report(joint, fused)
                    pa = block_reports[name_a].param_report or {}
                    pb = block_reports[name_b].param_report or {}
                    delta = {
                        k: pr_j[k] - pa.get(k, 0) - pb.get(k, 0)
                        for k in (
                            "original_params",
                            "optimized_params",
                            "original_bytes",
                            "optimized_bytes",
                        )
                    }
                    drop = (f"{name_a}:", f"{name_b}:")
                    for k, v in delta.items():
                        agg[k] += v
                    agg["eliminated"] = [
                        e
                        for e in agg["eliminated"]
                        if not e.startswith(drop)
                    ]
                    agg["derived"] = [
                        e
                        for e in agg["derived"]
                        if not e.startswith(drop)
                    ]
                    agg["eliminated"] += [
                        f"{pair}:{n}" for n in pr_j["eliminated"]
                    ]
                    agg["derived"] += [
                        f"{pair}:{n}" for n in pr_j["derived"]
                    ]
                    replacements[name_a] = fused
                    replacements[name_b] = (
                        _Zero()
                        if mode.endswith("_wrapped")
                        else torch.nn.Identity()
                    )
                    block_reports[name_a].extra["cross_pair"] = pair
                    block_reports[name_b].extra["cross_pair"] = pair
                    consumed.update((name_a, name_b))
                    entry["status"] = "grafted"
        except Exception as e:
            entry["status"] = "declined"
            entry["reason"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["time_s"] = time.time() - t0
        reports[pair] = entry
        if verbose:
            print(
                f"[Compositional] pair {pair}: "
                f"{entry['status']} ({mode})"
            )
    return reports


def optimize_compositional(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple,
    *,
    block_pred: Callable[[torch.nn.Module, str, torch.nn.Module], bool]
    | None = None,
    cost_fn: CostFn | None = None,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verify_tol: float = 1e-4,
    ops: OpTable | None = None,
    max_cross_pairs: int = 8,
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Optimize a stacked/multi-block model one block at a time.

    Whole-model equality saturation is monolithic: the e-graph grows with
    the product of block structures, so deep stacks saturate slowly.
    This driver instead

    1. walks the module tree and selects *blocks* — leaf submodules, plus
       any child where ``block_pred(parent, name, child)`` holds
       (default: children of ``nn.ModuleList``/``nn.Sequential``),
    2. runs the ORIGINAL model once with forward hooks to capture each
       block's real input (a block's input is not the model input),
    3. runs :func:`optimize_model` on each block with its captured input,
       verifying the lowered block against the original on that input —
       a block that fails to export, saturate, lower, or verify keeps its
       original implementation; a block that crosses a resource bound
       (``max_enodes``, ``max_memory_mb``, or a caught OOM) fails with
       ``reason == "resource_limit"``,
    4. runs the pairwise cross-block pass (:func:`_cross_pair_pass`):
       for each *adjacent* pair whose boundary is a simple value flow
       (B's input IS A's output, or the ``x + A(x)`` residual), build
       the joint micro-model, optimize it with :func:`optimize_model`,
       verify it against the eager pair, and graft it in place of both
       blocks when it is verified AND cheaper than the separate results
       — capped at ``max_cross_pairs`` joint runs (``0`` disables),
    5. clones the model — structure deep-copied, parameter/buffer
       tensors *shared* with the original (``_shared_param_clone``; no
       second copy of the weights, so recompose does not double device
       memory) — and grafts the optimized ``IRModule`` back in place,
       preserving the original forward structure, then verifies
       end-to-end equivalence on ``example_input``.

    Returns ``(recomposed_module, stats)`` where ``stats["blocks"]`` maps
    each block's dotted name to ``{"status", "stats", "param_report",
    "time_s", ...}`` and ``stats["param_report"]`` aggregates the
    per-block parameter diffs (eliminated/derived names are prefixed by
    block name for auditability).  ``stats["cross_pairs"]`` maps each
    attempted pair ``"a+b"`` to ``{"status", "boundary", ...}`` —
    ``"grafted"`` / ``"declined"`` / ``"skipped"``.  ``stats["shared_
    params"]`` is True when the recomposed module shares the original's
    tensor storage (the normal path); ``stats["in_place"]`` True means
    cloning failed and the input module was returned unmodified.
    """
    t_start = time.time()
    if cost_fn is None:
        cost_fn = _default_cost_fn()

    blocks = _select_blocks(model, block_pred)
    if verbose:
        print(
            f"[Compositional] {len(blocks)} candidate blocks: "
            f"{[n for n, _ in blocks]}"
        )

    captured, io = _capture_block_inputs(model, blocks, example_input)
    # A second capture on a perturbed probe input arms the residual
    # boundary check against a coincidence of values (it only runs when
    # the pair pass is enabled; a probe failure fails closed — residual
    # pairs decline, identity-proven chain pairs still work).
    captured2: dict[str, tuple[tuple, dict]] = {}
    io2: dict[str, dict[str, Any]] = {}
    if max_cross_pairs:
        with contextlib.suppress(Exception):
            captured2, io2 = _capture_block_inputs(
                model, blocks, _perturbed_input(example_input)
            )

    replacements: dict[str, torch.nn.Module] = {}
    block_reports: dict[str, BlockReport] = {}
    agg = {
        "original_params": 0,
        "optimized_params": 0,
        "original_bytes": 0,
        "optimized_bytes": 0,
        "eliminated": [],
        "derived": [],
    }

    for name, block in blocks:
        rep = BlockReport(name=name, status="not_executed")
        block_reports[name] = rep
        cap = captured.get(name)
        if cap is None:
            continue
        args, kwargs = cap
        if kwargs:
            rep.status = "skipped"
            rep.reason = f"non-positional kwargs {sorted(kwargs)}"
            continue
        ex = args[0] if len(args) == 1 else args
        t0 = time.time()
        try:
            opt_mod, st = optimize_model(
                block,
                ex,
                ruleset=ruleset,
                max_iterations=max_iterations,
                max_enodes=max_enodes,
                max_memory_mb=max_memory_mb,
                cost_fn=cost_fn,
                ops=ops,
                verbose=verbose,
            )
            # Per-block verification on the captured input — soundness
            # gate independent of optimize_model's own (verbose-gated)
            # check.  Any mismatch or eval failure falls back.
            vr = verify_module(block, opt_mod, args, rtol=verify_tol)
            rep.rel_diff = vr.max_rel
            if not vr.passed:
                raise RuntimeError(
                    f"block verification failed: "
                    f"rel diff {vr.max_rel:.3e}"
                )
            replacements[name] = opt_mod
            rep.status = "optimized"
            rep.stats = OptReport.from_stats(st)
            pr = param_report(block, opt_mod)
            rep.param_report = pr
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
                    f"({rep.rel_diff:.2e})"
                )
        except Exception as e:
            rep.status = "failed"
            rep.error = f"{type(e).__name__}: {e}"
            if isinstance(
                e, OptimizationResourceError
            ) or _looks_like_oom(e):
                rep.reason = "resource_limit"
            if verbose:
                print(f"[Compositional] {name}: keeping original ({e})")
        rep.time_s = time.time() - t0

    # -- Pairwise cross-block pass --------------------------------------
    # Per-block optimization is blind across the boundary: adjacent
    # blocks can share transforms a per-block search cannot see (block
    # i's output projection composing with block i+1's input
    # projections; a residual add absorbing a shared affine).  Verified
    # and cost-gated; every decline keeps the separate results.
    cross_pairs: dict[str, dict[str, Any]] = {}
    if max_cross_pairs:
        cross_pairs = _cross_pair_pass(
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
            ops=ops,
            max_cross_pairs=max_cross_pairs,
            verbose=verbose,
        )

    # -- Recompose -------------------------------------------------------
    # The clone grafts submodules, never tensors, so it shares the
    # original's parameter/buffer storage — recomposing costs no extra
    # weight bytes (the old plain deepcopy doubled the footprint and
    # OOMed at ~0.5B fp16 on small GPUs).
    in_place = False
    try:
        new_model = _shared_param_clone(model)
    except Exception:
        # Never graft into the caller's live model: the replacements
        # carry shape-specialized attrs baked by torch.export for the
        # example input, and a mutated caller fails at the NEXT input
        # shape (and the e2e check degenerates to self-comparison).
        new_model = model
        in_place = True
        replacements = {}
    for name, opt_mod in replacements.items():
        _replace_submodule(new_model, name, opt_mod)

    # -- End-to-end verification ----------------------------------------
    report = CompositionalReport(
        compositional=True,
        n_blocks=len(blocks),
        n_optimized=len(replacements),
        n_failed=sum(
            1 for r in block_reports.values() if r.status == "failed"
        ),
        n_skipped=sum(
            1
            for r in block_reports.values()
            if r.status in ("skipped", "not_executed")
        ),
        blocks=block_reports,
        in_place=in_place,
        shared_params=not in_place,
        extra={"cross_pairs": cross_pairs},
    )
    agg["bytes_saved"] = agg["original_bytes"] - agg["optimized_bytes"]
    agg["ratio"] = (
        agg["optimized_bytes"] / agg["original_bytes"]
        if agg["original_bytes"]
        else 1.0
    )
    report.param_report = agg

    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    if in_place:
        # new_model IS the input model — a verify would be a
        # self-comparison that always reports 0.0.  Record the
        # degenerate case honestly instead of a false pass.
        report.end_to_end = {
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
        try:
            vr = verify_module(model, new_model, args)
            report.end_to_end = {
                "max_abs_diff": vr.max_abs,
                "max_rel_diff": vr.max_rel,
            }
            if verbose:
                print(
                    f"[Compositional] end-to-end rel diff: "
                    f"{report.end_to_end['max_rel_diff']:.3e}"
                )
        except Exception as e:
            report.end_to_end = {"error": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"[Compositional] end-to-end check failed: {e}")

    report.wall_time_s = time.time() - t_start
    return new_model, report.to_dict()
