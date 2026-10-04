"""Measured executor-cost probe — price the routing, not the syntax.

``law_wallclock.py`` showed the pipeline's cost model prices the
extracted SSM terms ~32 % cheaper while ``BatchedScanModule`` runs them
*slower* than the raw unrolled module.  The mechanism is a pricing
blind spot, not a law failure — and this tool localises it precisely:

* extraction prices EVERY root-eclass member under
  ``executor_cost_for(lowering="generic")`` — the per-node serial
  evaluator — regardless of which executor the term would be routed
  to at lowering;
* routing is term-level: the first carrier ``ExecutorSpec`` whose
  ``accepts`` probe holds claims the term (``_route_spec``), so an
  ``applyd``-rooted extraction is delivered by ``BatchedScanModule``;
* ``_carrier_upgrade`` can only swap a carrier member IN (when its
  batched price beats the pick's delivered price) — it never demotes
  a carrier pick extraction already made.  On these models the
  ``applyd`` member wins the generic-priced extraction outright.

So the decision that needs an executor-aware price is EXTRACTION
itself.  This tool measures, per case:

* every root-eclass alternative (``SearchResult.alternatives``),
  lowered through BOTH its routed executor and the generic
  ``IRModule``, verified against the raw model and timed on CUDA
  (eager + manual CUDA-graph replay — the ``law_wallclock``
  methodology);
* the modeled prices side-by-side: the generic extraction price and
  the delivered-executor price (``_delivered_cost``);
* then the counterfactual: re-run ``search`` with
  ``cost_fn=`` a delivered-executor-aware pricing (apply-rooted
  plannable terms billed under ``"batched_scan"``, everything else
  under ``"generic"``) — once at the model's own constants, once
  corrected by the measured per-executor factors — and times what the
  corrected extraction actually delivers.

The corrected arm now exercises the SHIPPED selection path: the
measured factors ride a ``TargetProfile.corrections`` table carried
by the cost fn's ``profile`` marker (``backend_cost`` forwards it),
``_select_best_term`` threads it plus the input's
``shape_bucket`` into ``_carrier_upgrade``, and
``_delivered_cost``'s corrected-price consumer applies the factor —
no post-hoc re-implementation of the upgrade loop.  ``--emit-profile``
writes the measured corrections as a ``TargetProfile`` JSON — the
calibration artifact a ``cost_fn``-carrying search consumes.

Honesty contract: the corrected factors are measured on these same
shapes — a same-bucket correction demonstrating the mechanism, not a
deployed calibration.  Where the corrected model still picks wrong,
the report says so.

Run::

    .venv/bin/python tools/executor_cost_probe.py
    .venv/bin/python tools/executor_cost_probe.py --json /tmp/ec.json
    .venv/bin/python tools/executor_cost_probe.py --emit-profile /tmp/p.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from catopt_core.cost import dag_cost, executor_cost_for
from catopt_core.cost.roofline import _LAUNCH_S, _PEAK_BW, _PEAK_FLOPS
from catopt_core.ir import IR, Op
from catopt_core.profile import TargetProfile, shape_bucket
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import (
    Autotuned,
    _carrier_plans,
    _carrier_upgrade,
    _delivered_cost,
    _route_spec,
    default_rules,
)
from catopt_torch.backend import TorchBackend

# Reuse the wall-clock methodology verbatim: same cases, same timing.
from law_wallclock import (
    _DEV_CUDA,
    _ITERS,
    _MAX_ENODES,
    _MAX_ITERS,
    _WARMUP,
    _capture,
    _cases,
    _synced_median,
)

#: Root-eclass frontier size measured per case.
_TOP_K = 6


# ---------------------------------------------------------------------------
#  Delivered-executor-aware extraction pricing
# ---------------------------------------------------------------------------


def _is_plannable_carrier(term: Any) -> bool:
    """Return True when *term* routes to a level-batched carrier executor.

    Mirrors ``_delivered_cost`` / ``_route_spec``: the term is rooted
    at a carrier-apply op whose batched plan builder accepts it — a
    declined plan degrades to serial eval inside the module, priced
    generic.
    """
    plans = _carrier_plans()
    return (
        isinstance(term, Op)
        and term.op in plans
        and plans[term.op](term) is not None
    )


def _delivered_cost_fn(
    factors: dict[str, float] | None = None,
    profile: Any = None,
) -> Any:
    """Cost fn pricing each term under the executor it ROUTES to.

    Carrier-apply-rooted, plannable terms bill at
    ``executor_cost_for(lowering="batched_scan")``; everything else at
    ``"generic"`` — ``_delivered_cost`` lifted into an extraction
    ``cost_fn``.  ``factors`` optionally multiplies each family's
    modeled price by a measured correction (``{"batched": f, ...}``).

    ``profile`` rides the returned fn's ``profile`` marker —
    ``backend_cost`` forwards it and ``_select_best_term`` reads it
    into ``_carrier_upgrade``'s delivered-price comparison (the
    shipped measured-pricing seam this probe demonstrates).

    Non-additive at carrier roots (``_batched_scan_latency`` is a
    whole-spine price): as an ``extract_best`` model it is
    approximate — the applyd member's local cost absorbs the spine's
    batched-vs-generic delta.  That is precisely the mechanism under
    test; the counterfactual reports what it actually extracts.
    """
    gen = executor_cost_for(profile, lowering="generic")
    bat = executor_cost_for(profile, lowering="batched_scan")
    f_gen = (factors or {}).get("generic", 1.0)
    f_bat = (factors or {}).get("batched", 1.0)
    plannable: dict[Any, bool] = {}

    def cost(term: Any, memo: dict | None = None) -> float:
        hit = plannable.get(term)
        if hit is None:
            hit = _is_plannable_carrier(term)
            plannable[term] = hit
        if hit:
            return bat(term, memo) * f_bat
        return gen(term, memo) * f_gen

    cost.__name__ = "delivered_cost_fn"
    cost.profile = profile
    return cost


def _measured_profile(
    factors: dict[str, float], counts: dict[str, int], bucket: str
) -> TargetProfile:
    """Return the measured family factors as a ``TargetProfile``.

    Written under the ``corrections`` contract
    ``catopt_core.profile.record_measured`` maintains —
    ``{candidate: {bucket: {"factor", "n"}}}`` — keyed by the
    candidate names ``_delivered_cost`` bills its routed executor
    under and by this input's ``shape_bucket``, so the shipped
    corrected-price consumer applies them.  The base fields carry
    the model's OWN built-in constants (including the ``dispatch`` /
    ``leaf_eval`` fallbacks a ``None`` profile uses): the only delta
    vs the uncalibrated delivered comparison is the measured
    correction.
    """
    return TargetProfile(
        name="executor-cost-probe",
        tflops=_PEAK_FLOPS / 1e12,
        gbps=_PEAK_BW / 1e9,
        launch_us=_LAUNCH_S * 1e6,
        dispatch_us=_LAUNCH_S * 1e6,
        leaf_eval_us=4.0 * _LAUNCH_S * 1e6,
        device=torch.cuda.get_device_name(0),
        measured_at=datetime.now(UTC).isoformat(),
        corrections={
            cand: {bucket: {"factor": f, "n": counts.get(cand, 2)}}
            for cand, f in factors.items()
        },
        meta={
            "source": "tools/executor_cost_probe.py",
            "note": "pooled same-run factors — mechanism demo",
        },
    )


# ---------------------------------------------------------------------------
#  Per-alternative lowering + timing
# ---------------------------------------------------------------------------


@dataclass
class AltRow:
    """One root-eclass alternative, priced and measured."""

    root_op: str
    sel_cost: float  # generic extraction price (dag_cost)
    delivered_ns: float  # modeled delivered-executor price
    routed: str  # executor the term routes to
    is_batched: bool | None = None
    verified: bool = False
    eager_ns: float | None = None  # routed-lowered, eager
    graph_ns: float | None = None  # routed-lowered, graph replay
    gen_eager_ns: float | None = None  # generic-lowered same term
    gen_graph_ns: float | None = None
    error: str = ""


@dataclass
class CaseRow:
    """One (model, size) case: the alternative frontier + counterfactual."""

    case: str
    picked_root: str = ""
    picked_exec: str = ""
    # The shipped search's term differs from post-hoc greedy
    # re-extraction — the paired/causal/carrier-upgrade stages moved
    # it (the flag cannot tell them apart).
    post_greedy_diff: bool = False
    raw_eager_ns: float = 0.0
    raw_graph_ns: float = 0.0
    alts: list[AltRow] = field(default_factory=list)
    cf: dict[str, dict[str, Any]] = field(default_factory=dict)


def _term_ir(res: Any, term: Any) -> IR:
    """Build the IR the lower phase assembles for an extracted term."""
    return IR(
        root=term,
        inputs=res.ir.inputs,
        input_names=res.ir.input_names,
        params=res.ir.params,
    )


def _lower_term(term: Any, res: Any, sink: Any, via: str) -> Any:
    """Lower *term* through its routed executor or the generic one."""
    ir = _term_ir(res, term)
    params = dict(res.param_values)
    if via == "routed":
        spec = _route_spec(term, sink)
        if spec is not None:
            return spec.lower(ir, params)
    return sink.executors["generic"].lower(ir, params)


def _time(module: Any, xc: torch.Tensor) -> tuple[float, float]:
    """Median eager + graph-replay ns for ``module(xc)`` on CUDA."""
    mod = module.to(_DEV_CUDA).eval()
    eager = _synced_median(lambda m=mod: m(xc))["median"] * 1e6
    try:
        replay = _capture(lambda t, m=mod: m(t), xc)
        graph = _synced_median(replay)["median"] * 1e6
    except Exception:
        graph = float("nan")
    return eager, graph


def _check(module: Any, xc: torch.Tensor, ref: torch.Tensor) -> bool:
    """Numerical sanity: lowered module agrees with the raw model."""
    got = module.to(_DEV_CUDA).eval()(xc)
    return bool(torch.allclose(ref, got, rtol=1e-4, atol=1e-6))


def _probe_case(case: Any, opt: Optimizer, sink: Any) -> CaseRow:
    """Search *case*, then price/lower/time every root alternative."""
    torch.manual_seed(0)
    model = case.build().eval().double()
    x = torch.randn(*case.shape, dtype=torch.float64)
    res = opt.search(
        model,
        x,
        rules=default_rules(),
        max_iterations=_MAX_ITERS,
        max_enodes=_MAX_ENODES,
    )
    row = CaseRow(case=case.name)
    root = res.term.op if isinstance(res.term, Op) else str(res.term)
    row.picked_root = root
    spec = _route_spec(res.term, sink)
    row.picked_exec = spec.name if spec is not None else "generic"
    greedy = res.eg.extract_best(res.root_eid, res.cost_fn)
    row.post_greedy_diff = greedy != res.term

    xc = x.to(_DEV_CUDA)
    with torch.inference_mode():
        ref = model.to(_DEV_CUDA).eval()(xc)
        row.raw_eager_ns, row.raw_graph_ns = _time(model, xc)
        for sel_cost, term in res.alternatives(top_k=_TOP_K):
            r_op = term.op if isinstance(term, Op) else str(term)
            spec = _route_spec(term, sink)
            routed = spec.name if spec is not None else "generic"
            a = AltRow(
                root_op=r_op,
                sel_cost=sel_cost,
                delivered_ns=_delivered_cost(term),
                routed=routed,
            )
            try:
                mod = _lower_term(term, res, sink, "routed")
                a.is_batched = getattr(mod, "is_batched", None)
                a.verified = _check(mod, xc, ref)
                a.eager_ns, a.graph_ns = _time(mod, xc)
            except Exception as e:
                a.error = f"routed: {type(e).__name__}: {e}"
            if routed != "generic" and not a.error:
                # The same term through the generic evaluator — the
                # "unrolled equivalent" executor of the carrier pick.
                try:
                    gmod = _lower_term(term, res, sink, "generic")
                    a.gen_eager_ns, a.gen_graph_ns = _time(gmod, xc)
                except Exception as e:
                    a.error += f" generic: {type(e).__name__}: {e}"
            row.alts.append(a)
    return row


# ---------------------------------------------------------------------------
#  Counterfactual extraction — executor-aware cost_fn through public API
# ---------------------------------------------------------------------------


def _counterfactual(
    case: Any,
    opt: Optimizer,
    sink: Any,
    factors: dict[str, float] | None,
    tag: str,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Re-search *case* under delivered pricing; time what it ships.

    With *factors* given, the measured corrections ride the cost
    fn's ``profile`` marker — a ``TargetProfile`` carrying them as a
    ``corrections`` table under this input's ``shape_bucket`` — so
    the SHIPPED ``_select_best_term`` → ``_carrier_upgrade`` →
    ``_delivered_cost`` chain runs the corrected delivered
    comparison itself (``res.term`` IS the corrected pick).  A
    post-hoc ``_carrier_upgrade`` on the same e-graph re-confirms:
    greedy re-extract under the search's cost_fn, then the shipped
    upgrade priced off the forwarded profile marker — the decision
    the pipeline makes once its delivered pricing is calibrated.
    The paired-extraction stage is skipped (its group objects aren't
    recorded on the result); under this cost_fn it agrees with
    greedy on these cases.
    """
    torch.manual_seed(0)
    model = case.build().eval().double()
    x = torch.randn(*case.shape, dtype=torch.float64)
    bucket = shape_bucket(x) if factors else None
    prof = (
        _measured_profile(factors, counts or {}, bucket)
        if factors
        else None
    )
    res = opt.search(
        model,
        x,
        rules=default_rules(),
        max_iterations=_MAX_ITERS,
        max_enodes=_MAX_ENODES,
        cost_fn=_delivered_cost_fn(factors, profile=prof),
    )
    low = opt.lower(res, x, verify=True)
    root = res.term.op if isinstance(res.term, Op) else str(res.term)
    spec = _route_spec(res.term, sink)
    xc = x.to(_DEV_CUDA)
    out: dict[str, Any] = {
        "model": tag,
        "root": root,
        "exec": spec.name if spec is not None else "generic",
        "module": type(low.module).__name__,
        "verified": bool(low.verified and low.verified.passed),
        "cost": dag_cost(res.term, res.cost_fn),
    }
    with torch.inference_mode():
        out["eager_ns"], out["graph_ns"] = _time(low.module, xc)
        ref = model.to(_DEV_CUDA).eval()(xc)

    greedy = res.eg.extract_best(res.root_eid, res.cost_fn)
    out["greedy_root"] = (
        greedy.op if isinstance(greedy, Op) else str(greedy)
    )
    if factors is None:
        return out

    # The corrected selection chain on the SAME e-graph: greedy under
    # the corrected model, then the SHIPPED carrier-upgrade
    # comparison — fed the profile the ``backend_cost`` wrapper
    # forwarded off the cost fn and this input's bucket, exactly the
    # arguments ``_select_best_term`` supplies in-pipeline.
    pick = _carrier_upgrade(
        res.eg,
        res.root_eid,
        greedy,
        res.cost_fn,
        profile=getattr(res.cost_fn, "profile", None),
        bucket=bucket,
    )
    proot = pick.op if isinstance(pick, Op) else str(pick)
    pspec = _route_spec(pick, sink)
    sel: dict[str, Any] = {
        "root": proot,
        "exec": pspec.name if pspec is not None else "generic",
    }
    if pick == res.term:
        sel["eager_ns"] = out["eager_ns"]
        sel["graph_ns"] = out["graph_ns"]
        sel["module"] = out["module"]
        sel["verified"] = out["verified"]
    else:
        try:
            mod = _lower_term(pick, res, sink, "routed")
            sel["module"] = type(mod).__name__
            sel["verified"] = _check(mod, xc, ref)
            with torch.inference_mode():
                sel["eager_ns"], sel["graph_ns"] = _time(mod, xc)
        except Exception as e:
            sel["error"] = f"{type(e).__name__}: {e}"
    out["corrected_sel"] = sel
    return out


def _autotune_arm(case: Any, opt: Optimizer) -> dict[str, Any]:
    """Run the shipped measured-delivery path: ``Autotuned`` strategy.

    Re-lowers the extracted term through ``eager`` (the raw model),
    ``generic`` (serial IRModule) and ``batched`` (the routed carrier
    executor), verifies, times, ships the measured winner — the
    mechanism the pipeline already has for executor-choice honesty.
    It can only re-deliver the term extraction picked; the member
    choice itself is upstream of it.  Run on CUDA tensors so the
    meter's timings cover the same launch-bound regime the probe
    measures.
    """
    torch.manual_seed(0)
    model = case.build().eval().double().to(_DEV_CUDA)
    x = torch.randn(*case.shape, dtype=torch.float64).to(_DEV_CUDA)
    try:
        mod, stats = opt.optimize(
            model,
            x,
            strategy=Autotuned(
                candidates=("eager", "generic", "batched"), warmup=10
            ),
            max_iterations=_MAX_ITERS,
            max_enodes=_MAX_ENODES,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    at = stats.get("autotune", {})
    xc = x.to(_DEV_CUDA)
    with torch.inference_mode():
        eager, graph = _time(mod, xc)
    return {
        "winner": at.get("winner"),
        "module": type(mod).__name__,
        "eager_ns": eager,
        "graph_ns": graph,
        "candidates": {
            n: {
                "status": r.get("status"),
                "median_s": r.get("median_s"),
            }
            for n, r in at.get("candidates", {}).items()
        },
    }


# ---------------------------------------------------------------------------
#  Measured correction factors
# ---------------------------------------------------------------------------


def _correction_ratios(rows: list[CaseRow]) -> dict[str, list[float]]:
    """Measured/modeled latency ratios per executor family (eager).

    Each measured alternative contributes
    ``measured_routed_ns / modeled_ns`` under its routed family
    (``"batched"`` for the carrier executors, ``"generic"``).
    Same-shape measurements — the mechanism demo, not a deployed
    calibration.
    """
    ratios: dict[str, list[float]] = {"batched": [], "generic": []}
    for row in rows:
        for a in row.alts:
            if a.eager_ns is None or a.delivered_ns <= 0:
                continue
            fam = "generic" if a.routed == "generic" else "batched"
            ratios[fam].append(a.eager_ns / a.delivered_ns)
    return ratios


def _correction_factors(rows: list[CaseRow]) -> dict[str, float]:
    """Geomean measured/modeled ratio per executor family (eager)."""
    ratios = _correction_ratios(rows)
    return {
        fam: math.exp(statistics.fmean(math.log(r) for r in rs))
        for fam, rs in ratios.items()
        if rs
    }


def _emit_profile(
    path: str,
    cases: list[Any],
    factors: dict[str, float],
    counts: dict[str, int],
) -> None:
    """Write the measured corrections as a ``TargetProfile`` JSON.

    The ``corrections`` table ``record_measured`` maintains —
    ``{candidate: {bucket: {"factor", "n"}}}`` — carrying the pooled
    per-family factor under EVERY measured ``shape_bucket`` (one per
    case), the same calibration the counterfactual consumes.  Pooled
    rather than per-case: a single-case family often contributes one
    observation, which sits below the learned-factor
    ``_CORRECTION_MIN_SAMPLES`` gate and would never fire — pooling
    keeps the emitted artifact live, at the documented granularity of
    a same-run mechanism demo.  The base constants are the model's
    built-ins — the correction table is the whole delta.  Feed the
    result to a search as
    ``cost_fn=executor_cost_for(TargetProfile.load(path))``.
    """
    corr: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        bucket = shape_bucket(torch.empty(*case.shape))
        for fam, f in factors.items():
            corr.setdefault(fam, {})[bucket] = {
                "factor": f,
                "n": counts.get(fam, 2),
            }
    prof = TargetProfile(
        name="executor-cost-probe",
        tflops=_PEAK_FLOPS / 1e12,
        gbps=_PEAK_BW / 1e9,
        launch_us=_LAUNCH_S * 1e6,
        dispatch_us=_LAUNCH_S * 1e6,
        leaf_eval_us=4.0 * _LAUNCH_S * 1e6,
        device=torch.cuda.get_device_name(0),
        measured_at=datetime.now(UTC).isoformat(),
        corrections=corr,
        meta={
            "source": "tools/executor_cost_probe.py",
            "note": "same-shape measured factors — mechanism demo",
        },
    )
    Path(path).write_text(prof.to_json() + "\n")


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def _ns(v: float | None) -> str:
    """Render nanoseconds as µs, or a dash."""
    return "-" if v is None else f"{v / 1e3:8.1f}"


def _report(rows: list[CaseRow], factors: dict[str, float]) -> str:
    """Render the measured tables and the counterfactual verdict."""
    out = [
        "== executor_cost_probe — measured executor-family pricing ==",
        f"   device: {torch.cuda.get_device_name(0)}  fp64  "
        f"warmup {_WARMUP}  iters {_ITERS}  (times in µs)",
        "",
    ]
    for row in rows:
        out.append(
            f"-- {row.case}   picked: {row.picked_root} -> "
            f"{row.picked_exec}"
            + (
                "   [post-greedy changed pick]"
                if row.post_greedy_diff
                else ""
            )
        )
        out.append(
            f"   raw: eager {_ns(row.raw_eager_ns)}   "
            f"graph {_ns(row.raw_graph_ns)}"
        )
        out.append(
            f"   {'root':>10} {'routed':>12} {'sel-cost':>10} "
            f"{'model-ns':>10} {'r-eager':>9} {'r-graph':>9} "
            f"{'g-eager':>9} {'g-graph':>9}  ok"
        )
        for a in row.alts:
            out.append(
                f"   {a.root_op:>10} {a.routed:>12} "
                f"{a.sel_cost:>10.3g} {a.delivered_ns:>10.3g} "
                f"{_ns(a.eager_ns):>9} {_ns(a.graph_ns):>9} "
                f"{_ns(a.gen_eager_ns):>9} {_ns(a.gen_graph_ns):>9} "
                f" {'Y' if a.verified else 'n'}"
                + (f"   [{a.error}]" if a.error else "")
            )
        best = min(
            (a for a in row.alts if a.eager_ns is not None),
            key=lambda a: a.eager_ns,
            default=None,
        )
        if best is not None:
            out.append(
                f"   measured-best (eager): {best.root_op} -> "
                f"{best.routed}  {_ns(best.eager_ns)}"
            )
        for tag, cf in row.cf.items():
            if tag == "autotune":
                out.append(
                    f"   autotune: winner={cf.get('winner')}/"
                    f"{cf.get('module')}  "
                    f"eager {_ns(cf.get('eager_ns'))}  "
                    f"graph {_ns(cf.get('graph_ns'))}"
                )
                continue
            out.append(
                f"   cf[{tag}]: {cf['root']} -> {cf['exec']}/"
                f"{cf['module']}  verify={cf['verified']}  "
                f"greedy={cf.get('greedy_root')}  "
                f"eager {_ns(cf['eager_ns'])}  "
                f"graph {_ns(cf['graph_ns'])}"
            )
            sel = cf.get("corrected_sel")
            if sel is not None:
                out.append(
                    f"     corrected-sel: {sel['root']} -> "
                    f"{sel['exec']}/{sel.get('module')}  "
                    f"verify={sel.get('verified')}  "
                    f"eager {_ns(sel.get('eager_ns'))}  "
                    f"graph {_ns(sel.get('graph_ns'))}"
                    + (f"   [{sel['error']}]" if "error" in sel else "")
                )
        out.append("")
    out.append(f"   correction factors (eager): {factors}")
    return "\n".join(out)


def _verdict(rows: list[CaseRow]) -> str:
    """Report whether delivered pricing picked the measured-fastest."""
    out = ["== verdict =="]
    for row in rows:
        ok = [a for a in row.alts if a.eager_ns is not None]
        if not ok:
            out.append(f"  {row.case:<26} no measurable alternatives")
            continue
        measured = min(ok, key=lambda a: a.eager_ns)
        shipped = next(
            (a for a in ok if a.root_op == row.picked_root), None
        )
        modeled = min(ok, key=lambda a: a.delivered_ns)
        cf = row.cf.get("measured", {})
        sel = cf.get("corrected_sel", cf)
        pick = sel.get("root", "?")
        flip = (
            "correct" if pick == measured.root_op else f"still {pick}"
        )
        at = row.cf.get("autotune", {})
        out.append(
            f"  {row.case:<26} shipped {row.picked_root}/"
            f"{row.picked_exec} "
            f"{_ns(shipped.eager_ns if shipped else None)}   "
            f"measured-best {measured.root_op}/{measured.routed} "
            f"{_ns(measured.eager_ns)}   modeled-delivered picks "
            f"{modeled.root_op}   corrected-selection: {flip}   "
            f"autotune winner: {at.get('winner')}"
        )
    return "\n".join(out)


def _dump_json(
    path: str, rows: list[CaseRow], factors: dict[str, float]
) -> None:
    """Write the machine-readable measured cost table."""
    payload = {
        "device": torch.cuda.get_device_name(0),
        "warmup": _WARMUP,
        "iters": _ITERS,
        "correction_factors_eager": factors,
        "cases": [
            {
                "case": r.case,
                "picked_root": r.picked_root,
                "picked_exec": r.picked_exec,
                "post_greedy_diff": r.post_greedy_diff,
                "raw_eager_ns": r.raw_eager_ns,
                "raw_graph_ns": r.raw_graph_ns,
                "alternatives": [
                    {
                        "root_op": a.root_op,
                        "routed": a.routed,
                        "is_batched": a.is_batched,
                        "sel_cost": a.sel_cost,
                        "delivered_ns": a.delivered_ns,
                        "eager_ns": a.eager_ns,
                        "graph_ns": a.graph_ns,
                        "generic_eager_ns": a.gen_eager_ns,
                        "generic_graph_ns": a.gen_graph_ns,
                        "verified": a.verified,
                        "error": a.error,
                    }
                    for a in r.alts
                ],
                "counterfactuals": r.cf,
            }
            for r in rows
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Measure, correct, re-extract, and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument(
        "--emit-profile",
        metavar="PATH",
        help=(
            "write the measured executor corrections as a "
            "TargetProfile JSON (per-case shape buckets)"
        ),
    )
    parser.add_argument(
        "--skip-counterfactual",
        action="store_true",
        help="measure only; skip the corrected re-search",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("executor_cost_probe requires CUDA")

    opt = Optimizer(backend=TorchBackend())
    sink = opt.sink
    _ = sink.executors  # resolve carrier specs (registration effect)

    rows: list[CaseRow] = []
    for case in _cases():
        rows.append(_probe_case(case, opt, sink))
        print(f"  probed {case.name}", flush=True)

    factors = _correction_factors(rows)
    counts = {
        fam: len(rs) for fam, rs in _correction_ratios(rows).items()
    }
    if not args.skip_counterfactual:
        for row, case in zip(rows, _cases(), strict=True):
            row.cf["modeled"] = _counterfactual(
                case, opt, sink, None, "modeled"
            )
            print(f"  cf-modeled {case.name}", flush=True)
            row.cf["measured"] = _counterfactual(
                case, opt, sink, factors, "measured", counts
            )
            print(f"  cf-measured {case.name}", flush=True)
            row.cf["autotune"] = _autotune_arm(case, opt)
            print(f"  autotune {case.name}", flush=True)

    print()
    print(_report(rows, factors))
    print(_verdict(rows))
    if args.json:
        _dump_json(args.json, rows, factors)
        print(f"\nwrote {args.json}")
    if args.emit_profile:
        _emit_profile(
            args.emit_profile, list(_cases()), factors, counts
        )
        print(f"wrote {args.emit_profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
