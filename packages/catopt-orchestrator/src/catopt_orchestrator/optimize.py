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
    headshare_keys_hold,
    pair_shared_input_convs,
    pair_shared_input_linears,
    preset,
    share_duplicate_attention_heads,
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
    TaskMetric,
)

from catopt_orchestrator.carriers import get_carriers
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
    """Carrier-apply root ops → (batched plan builder).

    A term rooted at one of these lowers through the level-batched
    executor.  The builders arrive through
    :mod:`catopt_orchestrator.carriers` (the carrier package registers
    them), so the orchestrator never imports a backend; a process
    without carriers simply yields an empty map (no carrier upgrades).
    """
    m = get_carriers()
    return {} if m is None else m.plans()


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
#  Certified bounded-error offers (plan 0012) — the budget plumbing
# ---------------------------------------------------------------------------


def _check_budget_engine(eg: Any, error_budget: float | None) -> None:
    """Bounded extraction needs proof witnesses — the reference EGraph.

    No-op when ``error_budget`` is ``None``.  A non-``EGraph`` engine
    lacks ``certificate`` / ``extract_best_bounded`` (the
    :class:`~catopt_core.ports.Engine` port does not require them),
    and a level-1 e-graph records no merge witnesses — in either case
    the bound gate would silently pass every bounded member.  Fail
    loudly instead.
    """
    if error_budget is None:
        return
    if not isinstance(eg, EGraph):
        raise TypeError(
            "error_budget requires the reference EGraph engine — "
            "bounded offers and their certificates are Python-engine "
            f"machinery (got {type(eg).__name__})"
        )
    if eg.truncation_level < 2:
        raise TypeError(
            "error_budget requires proof witnesses — the engine was "
            "built at truncation_level < 2"
        )


def _subterms(term: Any) -> Iterable:
    """Yield *term* and every subterm, pre-order."""
    stack = [term]
    while stack:
        t = stack.pop()
        yield t
        if isinstance(t, Op):
            stack.extend(t.args)


def _term_params(term: Any) -> set[str]:
    """Names of every ``Param`` leaf *term* carries."""
    return {t.name for t in _subterms(term) if isinstance(t, Param)}


def _bound_rules(eg: EGraph) -> dict[str, Any]:
    """Return the bound-carrying rewrites this run registered."""
    return {n: r for n, r in eg._rule_objs.items() if r.error_bound}


def _delivered_bound(
    eg: EGraph, term: Any, bound: dict[str, Any]
) -> set[str]:
    """Bound-carrying rule names whose offered member *term* delivers.

    A bound offer survives extraction under two fingerprints, both
    robust to saturation rewriting the offered member after it was
    registered:

    * *verbatim delivery* — the member's root enode is registered to
      the bound witness's synthetic application (``_enode_app``), so
      a term locating to it used the offer directly;
    * *derived params* — bounded members introduce ``Param`` leaves
      the exact side lacks (``__bl`` / ``__lr`` derived weights);
      leaves survive rewriting, so a term still carrying one is a
      rewritten delivery of the member.
    """
    used_params = _term_params(term)
    delivered = {
        nm
        for nm, r in bound.items()
        if (_term_params(r.rhs) - _term_params(r.lhs)) & used_params
    }
    # Enodes registered to bound-carrying applications: a verbatim
    # delivered member locates to one of these.
    tainted: dict[Any, str] = {
        en: eg._applications[ai]["rule"]
        for en, ai in eg._enode_app.items()
        if eg._applications[ai]["rule"] in bound
    }
    for sub in _subterms(term):
        _ce, en = eg._locate(sub)
        if rn := tainted.get(en):
            delivered.add(rn)
    return delivered


def _bound_ledger(
    eg: EGraph, cert: Any, term: Any
) -> tuple[list[dict], float]:
    """Compute the bound ledger of a delivered term — entries + total.

    The certificate is the primary source: each replayed bound step
    contributes its rule's declared ``error_bound``.  But a derivation
    that exhausts its budget records ``egraph_dependent`` stubs and
    any bound merge inside such a gap contributes *nothing* to
    ``cert.error_bound`` — a silent undercount.  :func:`_delivered_bound`
    therefore re-scans the term itself for bound-member fingerprints;
    a detected-but-uncertified rule is appended and its bound added,
    so the ledger is never silently empty on an approximate term.
    """
    entries = _error_bound_entries(cert)
    bound = _bound_rules(eg)
    if not bound:
        return entries, cert.error_bound
    covered = {e["rule"] for e in entries}
    extra = sorted(_delivered_bound(eg, term, bound) - covered)
    for nm in extra:
        r = bound[nm]
        entries.append(
            {
                "rule": nm,
                "law": r.law,
                "bound": r.error_bound,
                "norm": r.bound_norm,
                "measured_max_rel": None,
            }
        )
    return entries, cert.error_bound + sum(
        bound[nm].error_bound for nm in extra
    )


def _ban_bound_members(
    eg: EGraph, term: Any, bound: dict[str, Any], bans: dict[int, set]
) -> bool:
    """Exclude every bound member *term* delivers; True on new bans.

    For each delivered bound rule this bans (a) the offered member's
    root enode, (b) its derived ``Param`` leaf enodes — rewritten
    spellings keep the leaf, so banning it kills every variant — and
    (c) the enode the term actually picked inside the offered class,
    which covers bound members carrying no derived param at all
    (``zero_bounded``).  Each new ban strictly shrinks the candidate
    space, so a caller looping on this either fits the budget or
    stalls — reported by the ``False`` return.
    """
    delivered = _delivered_bound(eg, term, bound)
    if not delivered:
        return False
    want: set = set()
    derived: list = []
    for nm in delivered:
        r = bound[nm]
        ce, _en = eg._locate(r.rhs)
        want.add(ce)
        keep = _term_params(r.lhs)
        derived += [
            sub
            for sub in _subterms(r.rhs)
            if isinstance(sub, Param) and sub.name not in keep
        ]
    progress = False

    def _ban(ce: Any, en: Any) -> None:
        nonlocal progress
        slot = bans.setdefault(ce, set())
        if en not in slot:
            slot.add(en)
            progress = True

    for sub in _subterms(term):
        ce, en = eg._locate(sub)
        if ce in want:
            _ban(ce, en)
    for p in derived:
        _ban(*eg._locate(p))
    return progress


def _bounded_term(
    eg: EGraph,
    root_eid: int,
    src: Any,
    cost_fn: CostFn,
    error_budget: float,
) -> tuple[Any, list[dict], float]:
    """Cheapest member whose delivered bound fits ``error_budget``.

    Iterates ``extract_best`` under accumulating enode bans: each
    round's ledger — the certificate plus the delivered-member
    fingerprints — either fits the budget (done) or names the bound
    members to exclude for the next round.  The original member is
    never bound-carried, so a real search root always satisfies the
    budget; ``None``/stall are reachable only for a degenerate
    fully-cyclic class or a bound member no fingerprint can name.
    """
    bans: dict[int, set] = {}
    bound = _bound_rules(eg)
    while True:
        term = eg.extract_best(root_eid, cost_fn, bans=bans)
        if term is None:
            raise OptimizationResourceError(
                f"error_budget={error_budget}: no member of the root "
                "e-class certifies within the budget"
            )
        cert = eg.certificate(src, term, root_eid=root_eid)
        entries, total = _bound_ledger(eg, cert, term)
        if total <= error_budget:
            return term, entries, total
        if not _ban_bound_members(eg, term, bound, bans):
            raise OptimizationResourceError(
                f"error_budget={error_budget}: the cheapest member "
                "relies on a bound offer that cannot be excluded"
            )


def _bound_gate(
    eg: EGraph,
    root_eid: int,
    src: Any,
    cost_fn: CostFn,
    stats: dict[str, Any],
    best_term: Any,
    error_budget: float | None,
) -> Any:
    """Apply the error-budget ledger gate to the extracted term.

    Pass-through when ``error_budget`` is ``None``.  Otherwise ledger
    the winner — the certificate plus the delivered-member
    fingerprints (a derivation that exhausted its budget undercounts)
    — and re-extract through :func:`_bounded_term`'s ban loop when
    the total exceeds the budget.  ``stats`` always records the
    request and the accepted bound — never silent.
    """
    if error_budget is None:
        return best_term
    cert = eg.certificate(src, best_term, root_eid=root_eid)
    entries, bound_total = _bound_ledger(eg, cert, best_term)
    if bound_total > error_budget:
        best_term, entries, bound_total = _bounded_term(
            eg, root_eid, src, cost_fn, error_budget
        )
    stats["error_budget"] = error_budget
    stats["error_bound_total"] = bound_total
    stats["error_bounds"] = entries
    return best_term


def _error_bound_entries(cert: Any) -> list[dict]:
    """One record per bound-carrying step of the accepted derivation.

    ``measured_max_rel`` stays ``None`` until ``lower``'s verify fills
    it with the delivered-vs-original measurement — the ledger is
    surfaced either way, never silent.
    """
    return [
        {
            "rule": s.rule,
            "law": r.law,
            "bound": r.error_bound,
            "norm": r.bound_norm,
            "measured_max_rel": None,
        }
        for s in cert.steps
        if (r := cert.rules.get(s.rule)) is not None and r.error_bound
    ]


# ---------------------------------------------------------------------------
#  Bound propagation — bridging the weight-space / output-space units
# ---------------------------------------------------------------------------
#
# The ledger's ``bound`` is certified in *weight* space (``max_abs`` =
# ``max|ΔW|`` for the specials' bounded members, ``frobenius`` for the
# low-rank residual).  The verify report measures *output* space —
# ``max|Δy|`` — and a site ``y = x·W`` amplifies the weight bound by
# the input's contraction norm.  Comparing the two directly is a unit
# bug (the bounded_e2e sweep measured ~2-8x output amplification on a
# real head, declining every delivery).  ``_propagate_bounds`` bridges
# the units honestly: per output element
# ``|Δy_j| = |Σ_k x_k·ΔW_jk| ≤ bound · max_i‖x_i‖_p``, so each entry's
# output bound is ``bound · amp`` with ``amp`` the *site* input's max
# row norm — evaluated, not guessed (the witness LHS is the original
# site expression; its data subterm is lowered through the sink and
# run on the verify input).


def _bound_p(bound_norm: str) -> float | None:
    """Row-norm exponent a bound norm propagates through, or ``None``.

    A per-element (``max_abs``) bound gives ``|Δy_j| ≤
    Σ_k|x_k|·max|ΔW|`` — the row-L1 factor.  Frobenius and spectral
    bounds give ``|Δy_j| ≤ ‖x_row‖_2·‖ΔW‖`` — the row-L2 factor.  Any
    other norm has no stated propagation rule here: ``None`` makes the
    gate fail closed rather than guess a bridge.
    """
    return {"max_abs": 1.0, "frobenius": 2.0, "spectral": 2.0}.get(
        bound_norm
    )


def _site_data_term(rule: Any) -> Any | None:
    """Return the data-side subterm of a bound witness's LHS.

    Bounded offers witness ``site -> offered`` where ``site`` is the
    original ``linear``/``matmul`` expression — its first argument is
    the site input.  A non-projection LHS (e.g. a morphism-level joint
    term) carries no isolable site.
    """
    lhs = getattr(rule, "lhs", None)
    if (
        isinstance(lhs, Op)
        and lhs.op in ("linear", "matmul")
        and lhs.args
    ):
        return lhs.args[0]
    return None


def _site_eval(
    sink: Sink, ir: IR, params: dict, term: Any, args: tuple
) -> Any:
    """Evaluate one subterm: lower it through the sink, run on ``args``.

    The subterm's ``Var`` leaves bind positionally to the module's own
    inputs, so running it on the verify input yields the real site
    activation — the operand the bounded weight actually contracts.
    """
    sub = IR(
        root=term,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    mod = sink.lower(sub, params)
    return mod.forward(*args)


def _row_norm(t: Any, p: float) -> float | None:
    """``max_i ‖t_i‖_p`` over the last axis — the contraction factor.

    Duck-typed tensor ops (``abs`` / ``sum`` / ``max``), the same
    convention as the pairing/factored passes — the orchestrator names
    no tensor library.  ``None`` when the value cannot supply them.
    """
    abs_ = getattr(t, "abs", None)
    if not callable(abs_):
        return None
    try:
        a = abs_()
        v = a.sum(-1) if p == 1.0 else (a * a).sum(-1) ** 0.5
        return float(v.max())
    except Exception:
        return None


def _entry_amp(
    rule: Any,
    p: float,
    sink: Sink,
    ir: IR,
    params: dict,
    args: tuple,
) -> tuple[float | None, str]:
    """One ledger entry's site-input amplification, with provenance.

    Primary: evaluate the bound rule's witness-LHS data subterm on the
    verify args — the real site input — and take its row norm.
    Fallback: the module input's own row norm when the site input
    cannot be isolated or evaluated (``"block_input"`` — exact for a
    block that IS its projection site, a documented estimate deeper
    inside one).  ``(None, "none")`` when neither measures.
    """
    site = _site_data_term(rule) if rule is not None else None
    if site is not None:
        with contextlib.suppress(Exception):
            amp = _row_norm(_site_eval(sink, ir, params, site, args), p)
            if amp is not None:
                return amp, "site"
    amps = [n for n in (_row_norm(a, p) for a in args) if n is not None]
    return (max(amps), "block_input") if amps else (None, "none")


def _propagate_bounds(
    stats: dict[str, Any],
    eg: Any,
    ir: IR,
    params: dict,
    sink: Sink,
    args: tuple,
) -> float | None:
    """Propagate the ledger's weight-space bounds into output units.

    Each ``stats["error_bounds"]`` entry certifies ``‖ΔW‖ ≤ bound``
    in its ``norm``; at a site ``y = x_s·W`` the delivered output
    moves by at most ``bound · max_i‖(x_s)_i‖_p`` (``p`` from
    :func:`_bound_p`) — so the total is in the same ``max|Δy|`` units
    the verify report measures, not the weight-space units of the
    certificate.  ``_entry_amp`` prices each entry's site input;
    every entry records ``site_input_norm`` / ``site_input``
    (``"site"`` / ``"block_input"`` / ``"none"``) and the propagated
    ``output_bound`` — the ledger stays honest about which input
    priced it.

    Returns the summed output bound (absolute ``max|Δy|`` units),
    stored in ``stats["error_bound_output"]``, or ``None`` when some
    entry's norm has no propagation rule or no input norm could be
    measured — the gate then fails closed.
    """
    entries = stats.get("error_bounds")
    if not entries:
        return 0.0
    rules = _bound_rules(eg) if isinstance(eg, EGraph) else {}
    out: list[dict] = []
    total = 0.0
    unpropagated = False
    for e in entries:
        e2 = dict(e)
        amp, src = (
            _entry_amp(rules.get(e["rule"]), p, sink, ir, params, args)
            if (p := _bound_p(str(e.get("norm") or ""))) is not None
            else (None, "none")
        )
        e2["site_input_norm"] = amp
        e2["site_input"] = src
        e2["output_bound"] = (
            float(e["bound"]) * amp if amp is not None else None
        )
        if e2["output_bound"] is None:
            unpropagated = True
        else:
            total += e2["output_bound"]
        out.append(e2)
    stats["error_bounds"] = out
    stats["error_bound_output"] = None if unpropagated else total
    return stats["error_bound_output"]


def _bound_record(
    stats: dict[str, Any], report: Any, output_bound: float | None
) -> bool:
    """Fill the ledger's measured/propagated fields; return honored.

    The verify report measures ``max|Δy|`` — so the honored check is
    ``report.max_abs ≤ output_bound``, output bound against output
    measurement in the same units.  ``output_bound_rel`` restates each
    entry's bound in the report's relative units for the record
    (``max_rel = max_abs / (max|ref| + floor)``, so the denominator is
    ``max_abs / max_rel``; ``None`` when the measurement was exact).
    """
    denom = (
        report.max_abs / report.max_rel if report.max_rel > 0 else None
    )
    stats["error_bounds"] = [
        {
            **e,
            "measured_max_rel": report.max_rel,
            "output_bound_rel": (
                e["output_bound"] / denom
                if e.get("output_bound") is not None
                and denom is not None
                else None
            ),
        }
        for e in stats.get("error_bounds", [])
    ]
    honored = bool(
        output_bound is not None and report.max_abs <= output_bound
    )
    stats["error_bounds_honored"] = honored
    return honored


def _rel_bound(output_bound: float | None, report: Any) -> float:
    """Restate an absolute output bound in the report's rel units.

    For messages and records only — the gate itself compares in
    absolute units.  ``nan`` when the bound could not be propagated
    (fail-closed), ``inf`` when the measurement was exactly zero.
    """
    if output_bound is None:
        return float("nan")
    if report.max_rel <= 0 or report.max_abs <= 0:
        return float("inf")
    return output_bound * report.max_rel / report.max_abs


def _bounded_gate(report: Any, bound: float | None) -> bool:
    """Return the verify predicate under an accepted output bound.

    ``bound`` is the propagated OUTPUT bound (absolute ``max|Δy|``
    units, from :func:`_propagate_bounds`) — never the weight-space
    certificate, which is in different units and must not gate the
    measured error directly.  ``report.passed`` is the sink's
    tolerance gate (run with ``rtol=inf`` on bounded deliveries — the
    bound replaces the relative tolerance); the honest bound gate
    additionally requires the measured absolute error to stay within
    the propagated bound.  ``bound=None`` means the bound could not be
    propagated — the gate fails closed; ``bound=0`` means no bound
    members were delivered — the gate is the sink's verdict alone.
    """
    return bool(report.passed) and (
        bound == 0.0 or (bound is not None and report.max_abs <= bound)
    )


def _block_bound_gate(
    sink: Sink,
    block: Any,
    mod: Any,
    inputs: Any,
    verify_tol: float,
    st: dict[str, Any],
    res: Any,
    atol: float | None = None,
) -> tuple[Any, float | None]:
    """Verify one delivered module, propagating any bound first.

    Returns ``(report, output_bound)`` for :func:`_bounded_gate`.
    With no delivered bound members the plain ``verify_tol`` verify
    runs and ``output_bound`` is 0.  With them, the ledger's
    weight-space bound propagates to output units on the block's
    captured inputs (:func:`_propagate_bounds`), the sink verify runs
    at ``rtol=inf`` (the propagated bound replaces the relative
    tolerance), and the ledger gains its measured/propagated fields
    via :func:`_bound_record` — ``output_bound`` is ``None`` when the
    bound could not be propagated, which fails the gate closed.
    ``inputs`` is a tensor or positional-args tuple.
    """
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    bound = float(st.get("error_bound_total", 0.0))
    if bound <= 0.0:
        return (
            sink.verify(block, mod, args, rtol=verify_tol, atol=atol),
            0.0,
        )
    out_b = _propagate_bounds(
        st, res.eg, res.ir, dict(res.param_values), sink, args
    )
    report = sink.verify(block, mod, args, rtol=float("inf"), atol=atol)
    _bound_record(st, report, out_b)
    return report, out_b


def _bound_verify(
    stats: dict[str, Any],
    report: Any,
    bound_total: float,
    output_bound: float | None,
) -> Any:
    """Apply the honest bound gate to a verify report.

    No-op when ``bound_total`` is 0 (the accepted term delivers no
    bound members).  Otherwise each ``stats["error_bounds"]`` entry
    records the measured value and its propagated bound — the ledger
    is never silent — and ``stats["error_bounds_honored"]`` flags the
    verdict: a measured output error exceeding the propagated bound,
    a bound that could not be propagated (``output_bound=None``), or
    a failing sink report substitutes a ``passed=False`` report, even
    when the (unbounded) tolerance alone passed.
    """
    if bound_total <= 0.0:
        return report
    honored = _bound_record(stats, report, output_bound)
    if not honored or not report.passed:
        return _BoundedVerify(
            max_abs=report.max_abs, max_rel=report.max_rel
        )
    return report


@dataclass(frozen=True)
class _BoundedVerify:
    """A ``VerifyResult``-shaped report forced to ``passed=False``.

    ``lower`` substitutes this when a delivered module's measured
    error exceeds the certified bound its accepted members claimed —
    the bound gate is stricter than the verify tolerance.
    """

    max_abs: float
    max_rel: float
    passed: bool = False


# ---------------------------------------------------------------------------
#  Task-metric verify gate (plan 0015) — the certificate's task contract
# ---------------------------------------------------------------------------


def _task_contract(
    task: Any, task_tol: float | None
) -> tuple[str, float] | None:
    """Normalise a ``task=`` / ``task_tol=`` pair to ``(name, tol)``.

    ``None`` when no task metric is configured — the pointwise/bound
    gate is then unchanged.  A ``task_tol`` with no metric, a metric
    without a callable ``distance``, and a metric with no
    ``tolerance`` member and no explicit ``task_tol`` are all loud
    errors — a gate's parameters are never guessed.
    """
    if task is None:
        if task_tol is not None:
            raise TypeError("task_tol= requires a task= metric")
        return None
    if not callable(getattr(task, "distance", None)):
        raise TypeError(
            "task metric must be a TaskMetric — an object with "
            "distance(ref, opt), name and tolerance; "
            f"got {type(task).__name__} (no callable distance)"
        )
    name = str(getattr(task, "name", type(task).__name__))
    tol = (
        float(task_tol)
        if task_tol is not None
        else getattr(task, "tolerance", None)
    )
    if tol is None:
        raise TypeError(
            f"task metric {name!r} declares no tolerance — "
            "pass task_tol= explicitly"
        )
    return name, float(tol)


def _run_exec(mod: Any, args: tuple) -> Any:
    """Run a lowered executor on positional ``args`` (duck-typed).

    The :class:`~catopt_core.ports.Executor` contract is ``forward``;
    a runner-delivered wrapper that only defines ``__call__`` still
    works.  Tensor-ish args are cloned — the same guard
    ``verify_module`` uses against in-place forwards.
    """
    fn = getattr(mod, "forward", None)
    if not callable(fn):
        fn = mod
    return fn(*[a.clone() if hasattr(a, "clone") else a for a in args])


def _task_verify(
    stats: dict[str, Any],
    report: Any,
    task: Any,
    contract: tuple[str, float],
    ref: Any,
    mod: Any,
    args: tuple,
) -> _TaskVerify:
    """Evaluate the task metric on the verify args; gate on it.

    Under a task contract the task distance is the verdict — that is
    what the caller opted into.  The pointwise measurements still ride
    the returned report (``max_abs`` / ``max_rel``) and the bound
    ledger keeps its ``error_bounds_honored`` verdict, so a
    task-accepted delivery records BOTH numbers: certified under the
    task tolerance, drift measured.  ``stats["task"]`` gains the
    measured ``distance`` and the verdict — never silent.

    The metric evaluates on the *verify* input — the certificate is
    calibration-conditioned on exactly this input distribution.
    """
    name, tol = contract
    dist = float(
        task.distance(_run_exec(ref, args), _run_exec(mod, args))
    )
    passed = bool(dist <= tol)
    stats["task"] = {
        "name": name,
        "tolerance": tol,
        "distance": dist,
        "passed": passed,
        "evaluated_on": "verify_input",
    }
    return _TaskVerify(
        max_abs=report.max_abs,
        max_rel=report.max_rel,
        passed=passed,
        task_metric=name,
        task_distance=dist,
        task_tolerance=tol,
    )


def _verify_metric(
    contract: tuple[str, float] | None, bound_total: float
) -> str:
    """Name the metric that produced the verify verdict.

    ``"max_rel"`` — the plain pointwise gate; ``"bound"`` — the
    propagated output bound (a bounded member was delivered); the
    task metric's name when a ``task=`` contract gated.
    """
    if contract is not None:
        return contract[0]
    return "bound" if bound_total > 0.0 else "max_rel"


def _accepted_by(stats: dict[str, Any], verified: Any) -> str:
    """Which contract accepted the delivery — or ``"declined"``.

    ``"task"`` when a task metric gated acceptance, ``"bound"`` when
    the propagated output bound did, ``"pointwise"`` when the plain
    tolerance gate did.  The same label lands on each
    ``stats["error_bounds"]`` entry — a task-accepted bounded member
    records ``accepted_by="task"`` (and its pointwise bound is still
    reported).
    """
    if not verified.passed:
        return "declined"
    if stats.get("task", {}).get("passed"):
        return "task"
    return (
        "bound"
        if float(stats.get("error_bound_total", 0.0)) > 0.0
        else "pointwise"
    )


@dataclass(frozen=True)
class _TaskVerify:
    """A ``VerifyResult``-shaped report carrying the task verdict.

    ``max_abs`` / ``max_rel`` keep the pointwise measurements —
    reported either way — while ``passed`` is the task gate's verdict.
    The extra fields expose which metric gated, its tolerance and the
    measured distance.
    """

    max_abs: float
    max_rel: float
    passed: bool
    task_metric: str = ""
    task_distance: float = 0.0
    task_tolerance: float = 0.0


def _resolve_task(
    task: Any, task_tol: float | None, result: Any
) -> tuple[Any, tuple[str, float] | None]:
    """Resolve the effective ``(task, (name, tol))`` contract pair.

    ``lower`` precedence: an explicit ``task=`` wins outright; absent
    one, the contract recorded on the :class:`SearchResult` by
    :func:`search` governs (metric AND resolved tolerance).  An
    explicit ``task=`` without ``task_tol`` falls back to the
    metric's own ``tolerance`` — never to a tolerance recorded for a
    different metric.  ``_task_contract`` loudly rejects the bad
    combinations (tolerance without a metric, a metric without
    ``distance``, a tolerance-less metric without an override).
    """
    if task is None:
        task = getattr(result, "task", None)
        if task is not None and task_tol is None:
            task_tol = getattr(result, "task_tol", None)
    return task, _task_contract(task, task_tol)


def _contract_tol(contract: tuple[str, float] | None) -> float | None:
    """Return the resolved tolerance of a task contract, else None."""
    return contract[1] if contract is not None else None


def _task_declared(
    stats: dict[str, Any], contract: tuple[str, float] | None
) -> None:
    """Record the declared task contract in the search stats.

    The contract is declared at search and *evaluated* at lower's
    verify — ``distance``/``passed`` stay absent until then, so a
    promised-but-unevaluated contract is visible as such.
    """
    if contract is not None:
        stats["task"] = {"name": contract[0], "tolerance": contract[1]}


def _gate_delivery(
    stats: dict[str, Any],
    report: Any,
    bound_total: float,
    output_bound: float | None,
    task: Any,
    contract: tuple[str, float] | None,
    ref: Any,
    mod: Any,
    x: Any,
) -> Any:
    """Apply the operative verify gate and record the verdict.

    The pointwise/bound machinery always runs first (``stats``
    carries ``error_bounds_honored`` and the measured fields either
    way); under a task contract the task distance then replaces the
    verdict — the bound ledger stays on the record, never silent.
    ``verify_metric`` names the gating metric and ``accepted_by``
    which contract accepted — the same label lands on each
    ``error_bounds`` entry.
    """
    verified = _bound_verify(stats, report, bound_total, output_bound)
    if contract is not None:
        verified = _task_verify(
            stats,
            report,
            task,
            contract,
            ref,
            mod,
            x if isinstance(x, tuple) else (x,),
        )
    stats["verify_metric"] = _verify_metric(contract, bound_total)
    stats["accepted_by"] = _accepted_by(stats, verified)
    for e in stats.get("error_bounds", []):
        e["accepted_by"] = stats["accepted_by"]
    return verified


def _block_verify_task(
    task: Any,
    contract: tuple[str, float] | None,
    block: Any,
    mod: Any,
    args: tuple,
) -> dict[str, Any] | None:
    """Task-gate record for one compositional block verify.

    ``None`` when no task contract is configured (the block's
    pointwise/bound gate is then unchanged).  Otherwise runs both
    sides on the block's *captured* input — the metric is conditioned
    on that calibration input — and returns the serializable record
    (``name`` / ``tolerance`` / ``distance`` / ``passed``) that lands
    in the block report and its stats.
    """
    if contract is None:
        return None
    name, tol = contract
    dist = float(
        task.distance(_run_exec(block, args), _run_exec(mod, args))
    )
    return {
        "name": name,
        "tolerance": tol,
        "distance": dist,
        "passed": bool(dist <= tol),
        "evaluated_on": "captured_input",
    }


def _block_gate(
    sink: Sink,
    block: Any,
    mod: Any,
    args: tuple,
    verify_tol: float,
    st: dict[str, Any],
    res: Any,
    rep: dict[str, Any],
    task: Any,
    contract: tuple[str, float] | None,
) -> tuple[Any, float | None, bool]:
    """Verify one delivered block module; return the gate record.

    Runs ``_block_bound_gate`` first — the pointwise report and the
    bound propagation land in ``st`` either way (the ledger stays
    honest) — then the operative verdict: under a task contract the
    task distance measured on the captured input decides and the
    record lands in ``rep["task"]`` / ``st["task"]`` /
    ``st["verify_metric"]`` / ``st["accepted_by"]``; otherwise the
    propagated-bound gate decides.  Returns ``(report,
    output_bound, ok)``.
    """
    vr, out_b = _block_bound_gate(
        sink, block, mod, args, verify_tol, st, res
    )
    rep["rel_diff"] = vr.max_rel
    trec = _block_verify_task(task, contract, block, mod, args)
    if trec is None:
        return vr, out_b, _bounded_gate(vr, out_b)
    rep["task"] = trec
    st["task"] = trec
    st["verify_metric"] = trec["name"]
    st["accepted_by"] = "task" if trec["passed"] else "declined"
    return vr, out_b, bool(trec["passed"])


def _block_decline(
    rep: dict[str, Any], vr: Any, out_b: float | None
) -> None:
    """Raise the per-block verify failure — task or bound wording."""
    t = rep.get("task")
    if t is not None:
        raise RuntimeError(
            f"block verification failed: task {t['name']} distance "
            f"{t['distance']:.3e} exceeds tolerance "
            f"{t['tolerance']:.3e}"
        )
    raise RuntimeError(
        f"block verification failed: rel diff {vr.max_rel:.3e} "
        f"(accepted bound {_rel_bound(out_b, vr):.3e})"
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
#: lazily**: the carrier preset arrives through
#: :mod:`catopt_orchestrator.carriers` (the carrier package registers
#: it), so the orchestrator imports no backend; a process without
#: carriers simply contributes the core default.  Subsumed- and
#: symmetry-tagged
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
        m = get_carriers()
        if m is not None:
            rs = rs + m.preset()
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
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
) -> list:
    """Non-local passes with a brief re-saturation between them.

    The diagram-level product law pairs every linear sharing an input
    into one GEMM + split views (no consumer pattern needed); then the
    non-local lifts — unrolled recurrences -> ``trace(F)``, stacks of
    same-state carrier applications -> one application, whole om trees
    over scanned values -> the deferred omd carrier, exact weight
    tying (duplicate Param leaves share one class), and — opt-in via
    ``detect_factors`` — the low-rank factored-parameter offers of
    :func:`catopt_core.laws.factored.offer_low_rank_factors`, plus —
    opt-in via ``detect_specials`` — the structurally-special
    weight offers of
    :func:`catopt_core.laws.specials.offer_weight_specials` (exact —
    or, when ``error_budget`` is set, additionally its bounded
    near-dead / near-duplicate elision members).  All witnessed so
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

    lifts = _carrier_lifts(
        eg,
        source_tensors,
        stats,
        detect_factors,
        detect_specials,
        detect_headshare,
        error_budget,
    )
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
        # HeadShare sites carry the same replay burden: a cached term
        # whose sdpa gathers rely on head equality is valid only for
        # blocks whose own weights reproduce it — recorded as recheck
        # recipes for ``_cache_replay``'s ``_headshare_holds`` gate.
        headshare = [
            r["recheck"]
            for r in lifts
            if isinstance(r, dict) and "recheck" in r
        ]
        stats["param_sharing"] = {
            "ties": ties,
            "derived": derived,
            "headshare": headshare,
        }
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
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
) -> list:
    """Run the non-local carrier/tying lifts.

    The carrier lifts arrive through
    :mod:`catopt_orchestrator.carriers`; the weight-tying lifts are
    core.  A process without carriers contributes only the tying
    passes.  ``detect_factors``
    arms the opt-in low-rank detection offers of
    :func:`catopt_core.laws.factored.offer_low_rank_factors`;
    ``detect_specials`` arms the opt-in weight-structure offers of
    :func:`catopt_core.laws.specials.offer_weight_specials` — exact
    by default, additionally bounded when ``error_budget`` is set;
    ``detect_headshare`` arms the opt-in bitwise-equal-head compute
    sharing of
    :func:`catopt_core.laws.headshare.share_duplicate_attention_heads`.
    """
    m = get_carriers()
    carrier: list = [] if m is None else m.lifts(eg)
    return (
        carrier
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
        + _headshare_lifts(eg, source_tensors, detect_headshare)
        + _factor_lifts(eg, source_tensors, stats, detect_factors)
        + _special_lifts(
            eg, source_tensors, stats, detect_specials, error_budget
        )
    )


def _headshare_lifts(
    eg: EGraph, source_tensors: dict, detect_headshare: bool
) -> list:
    """Opt-in bitwise-equal-head sharing offers (``detect_headshare``).

    The detection pass itself is the branch — when off this returns
    ``[]`` and the graph never sees the gather-``sdpa``-gather member.
    When on, each ``sdpa`` site whose (Wq, Wk, Wv) head blocks are
    bitwise-equal gets a witnessed deduplicated alternative; the offer
    is exact by construction and competes on cost like any other.
    """
    if not detect_headshare:
        return []
    return share_duplicate_attention_heads(eg, source_tensors)


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


def _special_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_specials: bool,
    error_budget: float | None = None,
) -> list:
    """Opt-in structurally-special weight offers (``detect_specials``).

    The detection pass itself is the branch — when off this returns
    ``[]`` without touching the graph; when on, each certified offer
    (identity/diagonal/zero/elide/block-diag members, all
    ``error_bound=0``) lands in ``stats["weight_specials"]`` (minus
    the e-class id, which means nothing outside this run).  With
    ``error_budget`` set, the pass additionally offers the bounded
    ``elide_bounded`` / ``zero_bounded`` members — certified
    approximate, each carrying its measured ``error_bound ≤
    error_budget``; the selection gate downstream enforces the
    accumulated bound.
    """
    if not detect_specials:
        return []
    from catopt_core.laws.specials import offer_weight_specials

    offers = offer_weight_specials(
        eg, source_tensors, budget=error_budget
    )
    if offers:
        stats["weight_specials"] = [
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
    error_budget: float | None = None,
    src: Any = None,
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
      ``Sink`` carries it for the torch backend): no hook, no fold;
    * the error-budget gate (``error_budget``, plan 0012) — the
      chosen term's bound ledger (the ``src`` -> ``best_term``
      certificate plus delivered-member fingerprints, since a
      budget-exhausted derivation undercounts) must total at most
      ``error_budget``; a violation re-extracts through
      :func:`_bounded_term`'s ban loop.  Every bound-carrying member
      the accepted term delivers lands in ``stats["error_bounds"]``
      — never silent.
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
    # The bound gate: only members whose delivered bound accumulates
    # to at most ``error_budget`` are acceptable — see ``_bound_gate``.
    best_term = _bound_gate(
        eg, root_eid, src, cost_fn, stats, best_term, error_budget
    )
    return best_term


def _policy_kwargs(policy: Any) -> dict[str, Any]:
    """Return the engine kwargs a policy implies.

    Only the reference engine schedules through a policy today, so it
    is passed conditionally: an engine that does not take ``policy``
    keeps conforming to the ``Engine`` port.
    """
    return {} if policy is None else {"policy": policy}


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
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
    task: TaskMetric | None = None,
    task_tol: float | None = None,
    policy: Any = None,
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
    detect_specials : bool, default False
        Opt-in structurally-special weight pass
        (:func:`catopt_core.laws.specials.offer_weight_specials`):
        weight parameters whose *stored values* are identity,
        diagonal, zero, block-diagonal, or carry bitwise-dead input
        slices / bitwise-duplicate output slices get witnessed
        **exact** members (bound 0) — skip/pointwise-mul, narrowed
        projections plus gathers, per-block splits — in their
        consumer's e-class.  Unlike ``detect_factors`` nothing here
        is approximate: a slice is dead iff every value is exactly
        0.0, and duplicates dedupe bitwise.  See ``error_budget`` for
        the certified-approximate extension.
    detect_headshare : bool, default False
        Opt-in shared-head detection pass
        (:func:`catopt_core.laws.headshare.share_duplicate_attention_heads`):
        ``sdpa`` sites whose per-head (Wq, Wk, Wv) weight blocks are
        *bitwise-equal* under a resolved head structure get a
        witnessed gather-``sdpa``-gather member computing each unique
        head once — exact (bound 0), competing through the normal
        extraction cost.  Applicability is deliberately narrow:
        shared-head architectures, quantization-induced ties, GQA
        kv replication, post-``WeightTie`` merges — ordinary trained
        weights almost never tie bitwise, in which case nothing fires.
    error_budget : float, optional
        Opt-in certified-approximation budget (plan 0012).  ``None``
        (the default) keeps the search exact — no bounded member is
        offered and no bound gate runs.  When a float is given:

        * ``detect_specials``'s pass additionally offers the bounded
          ``elide_bounded`` / ``zero_bounded`` members of
          :func:`catopt_core.laws.specials.offer_weight_specials`
          (each with a measured ``error_bound ≤ error_budget``);
        * extraction accepts a term only when its ledgered bound —
          the certificate's accumulated ``error_bound`` plus any
          bound member the delivered term provably uses — stays
          within the budget (a derivation that exhausts its budget
          records ``egraph_dependent`` stubs which undercount, so
          delivered members are also fingerprinted directly);
        * the ledger is never silent: ``stats["error_budget"]``
          echoes the request, ``stats["error_bound_total"]`` is the
          accepted certificate bound (weight space), and
          ``stats["error_bounds"]`` lists every bound-carrying member
          used (``rule`` / ``law`` / ``bound`` / ``norm``); :func:`lower`
          propagates each ``bound`` to an ``output_bound`` on the
          verify input (:func:`_propagate_bounds`) and fills in
          ``measured_max_rel``.

        Requires the reference ``EGraph`` at ``truncation_level >= 2``
        (the bound ledger reads certificates and rule-application
        witnesses); anything else raises :class:`TypeError`.
    task : TaskMetric, optional
        Opt-in task-level equivalence contract (plan 0015) — a
        :class:`~catopt_core.ports.TaskMetric` value such as
        :class:`~catopt_core.metrics.TopKAgreement` or
        :class:`~catopt_core.metrics.ArgmaxStability`.  The search is
        unchanged — extraction still picks the cheapest member under
        ``cost_fn`` — but the contract is *recorded*: it rides the
        result so :func:`lower` gates the delivered module's verify on
        ``task.distance(ref_out, opt_out) <= tolerance`` evaluated on
        the verify input, and ``stats["task"]`` names the metric and
        tolerance (distance/verdict land at verify time).  The
        certificate is then calibration-conditioned on the verify
        input — distribution shift is the honest caveat.
    task_tol : float, optional
        Override the metric's own ``tolerance``; required when the
        metric declares none.
    policy : Policy, optional
        An in-search action ordering consulted once per saturation
        iteration (``EGraph.run(..., policy=...)``).  It may only
        *reorder* the rules, so the fixed point and the certificate are
        unchanged; the name is recorded in ``stats["policy"]``.
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
    contract = _task_contract(task, task_tol)

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
    _check_budget_engine(eg, error_budget)
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

    run_kw = _policy_kwargs(policy)
    stats: dict[str, Any] = eg.run(
        rules,
        root_eid,
        max_iterations=max_iterations,
        max_nodes=run_cap,
        rule_budgets=rule_budgets,
        stop=stop,
        patience=patience,
        cost_fn=cost_fn,
        **run_kw,
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
            detect_specials,
            detect_headshare,
            error_budget,
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
    _task_declared(stats, contract)
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
        error_budget=error_budget,
        src=ir.root,
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
        task=task,
        task_tol=_contract_tol(contract),
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
    task: TaskMetric | None = None,
    task_tol: float | None = None,
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
        When the search ran under ``error_budget`` and accepted
        bounded members (``stats["error_bound_total"]``), the bound
        replaces the relative tolerance: the certificate's bound is
        in *weight* space, so it is first propagated to output units
        (:func:`_propagate_bounds` — ``bound · site-input norm`` per
        entry, evaluated on ``x``), the sink verify runs at
        ``rtol=inf``, and the *measured* ``max|Δy|`` must stay within
        the propagated ``stats["error_bound_output"]`` — a violation
        declines the delivery (``verified.passed`` False,
        ``stats["error_bounds_honored"]`` False), even when the
        relative difference was small.  Each
        ``stats["error_bounds"]`` entry records its propagated
        ``output_bound`` and the measured ``max_rel`` — the ledger is
        never silent.
    task : TaskMetric, optional
        Opt-in task-level gate (plan 0015) — a
        :class:`~catopt_core.ports.TaskMetric` value.  When set, the
        verify *verdict* is the task distance: the delivered module
        and the fresh reference lowering run on ``x`` and
        ``verified.passed`` becomes
        ``task.distance(ref_out, opt_out) <= tolerance`` — the
        certificate is calibration-conditioned on ``x``.  The
        pointwise measurements are still reported
        (``verified.max_abs`` / ``verified.max_rel``), the bound
        ledger still populates, and a task-accepted bounded member
        records ``accepted_by="task"`` — never silent.  ``None``
        inherits the contract recorded by :func:`search`
        (``result.task`` / ``result.task_tol``).
    task_tol : float, optional
        Tolerance override; defaults to the result's recorded
        tolerance, then the metric's own ``tolerance``.
    verbose : bool
        Print progress.

    """
    if runner is None:
        runner = IdentityRunner()
    stats: dict[str, Any] = dict(result.stats)
    # Task contract precedence — explicit ``task=`` wins; otherwise
    # the search's recorded contract governs (see ``_resolve_task``).
    task, contract = _resolve_task(task, task_tol, result)
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
        bound_total = float(stats.get("error_bound_total", 0.0))
        # The certified bound is in weight space; the gate compares the
        # measured output error against the *propagated* bound (same
        # ``max|Δy|`` units).  Under a bound the sink's relative
        # tolerance opens (rtol=inf): the bound replaces it — honoring
        # it implies ``max_rel ≤ bound`` restated in rel units, so
        # ``max(rtol, bound)`` semantics are preserved.
        report, output_bound = _block_bound_gate(
            sink, ref, optimized_module, x, rtol, stats, result, atol
        )
        verified = _gate_delivery(
            stats,
            report,
            bound_total,
            output_bound,
            task,
            contract,
            ref,
            optimized_module,
            x,
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
        "headshare": sharing.get("headshare", ()),
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
    # HeadShare offers lean on bitwise head equality — re-verify the
    # recorded key slices against this block's own tensors (names
    # remapped through pmap + the derived-name map).
    if not headshare_keys_hold(
        entry.get("headshare", ()),
        lambda n: dmap.get(n, pmap.get(n)),
        params,
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
    error_budget: float | None = None,
    detect_specials: bool = False,
    detect_factors: bool = False,
    detect_headshare: bool = False,
    task: TaskMetric | None = None,
    task_tol: float | None = None,
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

    ``error_budget`` / ``detect_specials`` / ``detect_factors`` /
    ``detect_headshare``
    forward to each per-block :meth:`Optimizer.search` — with a
    budget set, a block's certified weight-space bound propagates to
    an output bound (:func:`_propagate_bounds`, evaluated on the
    captured input), the sink verify runs with its relative
    tolerance open, and the measured error must honor the propagated
    bound (:func:`_bounded_gate`) or the block keeps its original
    implementation.

    ``task`` / ``task_tol`` forward the same way: under a task
    contract each block's verify gates on the task distance evaluated
    on the block's *captured* input (the metric is conditioned on
    that calibration input), and the block report records
    ``task = {name, tolerance, distance, passed}``.

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
    contract = _task_contract(task, task_tol)

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
                        vh, _out2, gate_ok = _block_gate(
                            sink,
                            block,
                            lr2.module,
                            args,
                            verify_tol,
                            lr2.stats,
                            res2,
                            rep,
                            task,
                            contract,
                        )
                        if gate_ok:
                            opt_mod, st = lr2.module, lr2.stats
                            cache_hits += 1
                            rep["cache"] = "hit"
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
                    error_budget=error_budget,
                    detect_specials=detect_specials,
                    detect_factors=detect_factors,
                    detect_headshare=detect_headshare,
                    task=task,
                    task_tol=task_tol,
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
                # falls back.  Under a bounded search the certified
                # bound is in weight space, so it propagates to an
                # output bound on the captured input first
                # (``_propagate_bounds``) and the measured error must
                # honor THAT — same units — while the sink's relative
                # tolerance opens (the bound is the tolerance).
                vr, out_b, gate_ok = _block_gate(
                    sink,
                    block,
                    opt_mod,
                    args,
                    verify_tol,
                    st,
                    res,
                    rep,
                    task,
                    contract,
                )
                if not gate_ok:
                    _block_decline(rep, vr, out_b)
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
