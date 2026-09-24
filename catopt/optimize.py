"""Top-level optimization pipeline.

This module implements the four-phase killer experiment:

Phase 1 — Equivalence:   PyTorch → IR
Phase 2 — Search:        IR → e-graph → equality saturation → best term
Phase 3 — Lower:          best term → torch.nn.Module
Phase 4 — Compare:        benchmark vs. vanilla TorchInductor

The main entry point is :func:`optimize_model`.
"""

from __future__ import annotations

from typing import Any

import torch

from catopt.ir import IR, Op, Var, Const, Param
from catopt.egraph import EGraph
from catopt.rules import (all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES,
                          pair_shared_input_linears,
                          pair_shared_input_convs,
                          share_duplicate_params)
from catopt.trace_lift import lift_scan_to_trace
from catopt.xcarrier import (gather_applyd_stack, gather_apply_stack,
                             omd_tree_lift)
from catopt.cost import (flops_cost, count_cost, launch_aware_cost,
                         CostModel, dag_cost)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.ir import op_repr


def _eval_const(term: Any, params: dict) -> torch.Tensor | None:
    """Evaluate a parameter-only subtree to a concrete tensor."""
    from catopt.torch_bridge import _IR_TO_TORCH
    if isinstance(term, Param):
        return params.get(term.name)
    if isinstance(term, Const):
        return torch.tensor(term.value)
    if isinstance(term, Var):
        return None
    if isinstance(term, Op):
        vals = [_eval_const(a, params) for a in term.args]
        if any(v is None for v in vals):
            return None
        fn = _IR_TO_TORCH.get(term.op)
        if fn is None:
            return None
        try:
            with torch.no_grad():
                out = fn(*vals, **dict(term.attrs))
            return out if isinstance(out, torch.Tensor) else None
        except Exception:
            return None
    return None


def _is_causal_keep_mask(mask_val: torch.Tensor, q_shape) -> bool:
    """mask (…, T, T) keeps exactly the lower triangle and T matches
    q's sequence dim — i.e. the mask IS is_causal."""
    from catopt.cost import _shape_of as _so
    if not isinstance(q_shape, tuple) or len(q_shape) < 2:
        return False
    if (mask_val.ndim < 2 or mask_val.shape[-1] != mask_val.shape[-2]
            or mask_val.shape[-1] != q_shape[-2]):
        return False
    keep = (mask_val.bool() if mask_val.dtype == torch.bool
            else mask_val > -1e30)
    tril = torch.tril(torch.ones(mask_val.shape[-2], mask_val.shape[-1],
                                 dtype=torch.bool,
                                 device=mask_val.device))
    return bool((keep == tril).all())


def _specialize_causal(term: Any, params: dict,
                       memo: dict | None = None) -> Any:
    """sdpa(q,k,v, mask) where mask is parameter-only and evaluates to
    a causal keep-mask → sdpa(q,k,v, is_causal=True).  Dropping the
    materialised mask unlocks the fused flash/mem-efficient kernels."""
    from catopt.cost import _shape_of as _so
    if memo is None:
        memo = {}
    if not isinstance(term, Op):
        return term
    key = id(term)
    if key in memo:
        return memo[key]
    args = tuple(_specialize_causal(a, params, memo) for a in term.args)
    attrs = dict(term.attrs)
    if (term.op == "sdpa" and len(args) >= 4
            and not attrs.get("arg5")):
        mv = _eval_const(args[3], params)
        if mv is not None and _is_causal_keep_mask(mv, _so(args[0])):
            args = args[:3]
            attrs["arg5"] = True
            memo["_hit"] = True
    out = Op.make(term.op, *args, **attrs)
    memo[key] = out
    return out


def discover_alternatives(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    ruleset: str = "categorical",
    max_iterations: int = 6,
    cost_fn: CostModel = None,
    top_k: int = 8,
) -> dict:
    """Enumerate the cheapest distinct members of the semantic
    equivalence class [G] — the discovery-engine view.

    Runs the same export → e-graph → saturation → pairing pipeline as
    optimize_model, but instead of committing to the single best term
    it returns the top-k alternatives under the cost model, plus the
    rule-fire provenance (which generic laws actually fired).  Human
    inspection of this frontier is how level-3 candidates — emergent
    compositions of known laws — are found.
    """
    if cost_fn is None:
        cost_fn = launch_aware_cost
    ir, source_tensors = export_to_ir(model, example_input)
    eg = EGraph()
    root_eid = eg.add_term(ir.root)
    rules = {"all": all_rules(),
             "simpl": SIMPLIFICATION_RULES,
             "categorical": CATEGORICAL_RULES}[ruleset]
    if ruleset == "categorical":
        _SUBSUMED = {"swiglu_fuse", "qkv_fuse", "qkv_fuse_asym",
                     "parallel_mul_fuse"}
        rules = [r for r in rules if r.name not in _SUBSUMED]
    stats = eg.run(rules, root_eid, max_iterations=max_iterations)
    groups = (pair_shared_input_linears(eg)
              + pair_shared_input_convs(eg))
    if groups:
        eg.rebuild()
        stats["pairing_groups"] = len(groups)
        eg.run(rules, root_eid, max_iterations=5)
    # Non-local lifts: unrolled recurrences -> trace(F), stacks of
    # same-state carrier applications -> one application, whole om
    # trees over scanned values -> the deferred omd carrier, and exact
    # weight tying (duplicate Param leaves share one class).
    # All witnessed so certificates stay replayable.
    lifts = (lift_scan_to_trace(eg)
             + gather_applyd_stack(eg)
             + gather_apply_stack(eg)
             + omd_tree_lift(eg)
             + share_duplicate_params(eg, source_tensors))
    if lifts:
        eg.rebuild()
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(rules, root_eid, max_iterations=5)
    alts = eg.extract_alternatives(root_eid, cost_fn, top_k=top_k)
    return {
        "alternatives": alts,
        "diverse_classes": eg.diverse_classes(),
        "rule_fires": dict(sorted(eg.rule_fires.items(),
                                  key=lambda kv: -kv[1])),
        "stats": stats,
        "ir": ir,
        "eg": eg,
        "root_eid": root_eid,
        "source_tensors": source_tensors,
    }


def optimize_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int = 100_000,
    cost_fn=None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
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
    max_enodes : int
        Stop if the e-graph exceeds this many e-nodes.
    cost_fn : callable
        Cost function for term extraction.  Defaults to
        :func:`launch_aware_cost` (FLOPs + a small per-kernel penalty so
        that forms with identical FLOPs but fewer launches win).
    verbose : bool
        Print progress.

    Returns
    -------
    (optimized_module, stats)
        The optimized ``torch.nn.Module`` and a dictionary of e-graph stats.
    """
    if cost_fn is None:
        cost_fn = launch_aware_cost

    # -- Phase 1: Export to IR -------------------------------------------
    if verbose:
        print(f"[Phase 1] Exporting {model.__class__.__name__} to IR...")
    ir, source_tensors = export_to_ir(model, example_input)
    if verbose:
        print(f"  IR root: {op_repr(ir.root)}")
        print(f"  Inputs:  {[str(v) for v in ir.inputs]}")
        print(f"  Params:  {list(ir.params.keys())}")

    # -- Phase 2: Build e-graph and saturate -----------------------------
    if verbose:
        print("[Phase 2] Building e-graph and running equality saturation...")
    eg = EGraph()
    root_eid = eg.add_term(ir.root)

    # Choose rules.  The term-local fusion rules (swiglu_fuse, qkv_fuse,
    # parallel_mul_fuse, qkv_fuse_asym) are special cases of the product
    # law; in the pipeline they are SUBSUMED by the non-local
    # pair_shared_input_linears pass, which needs no consumer pattern.
    # Keeping them would let extraction pick consumer-level chunk
    # alternatives that bypass the globally-coordinated split choice.
    _SUBSUMED = {"swiglu_fuse", "parallel_mul_fuse",
                 "qkv_fuse", "qkv_fuse_asym"}
    if ruleset == "all":
        rules = [r for r in all_rules() if r.name not in _SUBSUMED]
    elif ruleset == "simpl":
        rules = SIMPLIFICATION_RULES
    elif ruleset == "categorical":
        rules = [r for r in CATEGORICAL_RULES if r.name not in _SUBSUMED]
    else:
        raise ValueError(f"Unknown ruleset: {ruleset}")

    if verbose:
        print(f"  Rules: {[r.name for r in rules]}")

    stats = eg.run(rules, root_eid,
                   max_iterations=max_iterations, max_nodes=max_enodes)

    # Diagram-level product law: pair every linear sharing an input into
    # one GEMM + split views.  Non-local — no consumer pattern needed.
    groups = (pair_shared_input_linears(eg)
              + pair_shared_input_convs(eg))
    if groups:
        eg.rebuild()
        stats["pairing_groups"] = len(groups)
        # brief second saturation so other rules see the new enodes
        eg.run(rules, root_eid, max_iterations=5, max_nodes=max_enodes)

    # Non-local lifts: unrolled recurrences -> trace(F), stacks of
    # same-state carrier applications -> one application, whole om
    # trees over scanned values -> the deferred omd carrier, and exact
    # weight tying (duplicate Param leaves share one class).
    # All witnessed so certificates stay replayable.
    lifts = (lift_scan_to_trace(eg)
             + gather_applyd_stack(eg)
             + gather_apply_stack(eg)
             + omd_tree_lift(eg)
             + share_duplicate_params(eg, source_tensors))
    if lifts:
        eg.rebuild()
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(rules, root_eid, max_iterations=5, max_nodes=max_enodes)

    stats["rule_fires"] = dict(eg.rule_fires)
    if verbose:
        print(f"  E-graph: {stats}")

    # -- Extract best term -----------------------------------------------
    best_term = eg.extract_best(root_eid, cost_fn)
    if groups:
        # Coordinated extraction: force every paired member to its split
        # enode AND steer consumers through the shared GEMM.  Compare
        # true DAG costs — forcing loses if a group is only partially
        # reachable or a bypassing alternative was already cheaper.
        forced = eg.extract_paired(root_eid, cost_fn, groups)
        if (forced is not None
                and dag_cost(forced, cost_fn)
                <= dag_cost(best_term, cost_fn)):
            best_term = forced
            stats["paired_extract"] = True
    # Causal specialization: a param-only attn_mask that evaluates to a
    # lower-triangular keep-mask is is_causal=True — no mask op at all.
    _cm: dict = {}
    best_term = _specialize_causal(best_term, source_tensors, _cm)
    if _cm.get("_hit"):
        stats["causal_specialized"] = True

    if verbose:
        print(f"  Best term: {op_repr(best_term)}")
        print(f"  Cost: {cost_fn(best_term):.2f} FLOPs (est.)")

    # -- Phase 3: Lower back to torch -----------------------------------
    if verbose:
        print("[Phase 3] Lowering optimized IR to torch module...")
    optimized_ir = IR(root=best_term, inputs=ir.inputs,
                      input_names=ir.input_names, params=ir.params)
    optimized_module = ir_to_torch_module(optimized_ir, param_values=source_tensors)

    # Verify semantic equivalence
    if verbose:
        print("[Verify] Checking output equivalence...")
        model.eval()
        optimized_module.eval()
        with torch.no_grad():
            if isinstance(example_input, tuple):
                original_out = model(*[a.clone() for a in example_input])
                opt_out = optimized_module(*[a.clone() for a in example_input])
            else:
                original_out = model(example_input.clone())
                opt_out = optimized_module(example_input.clone())
            max_diff = (original_out - opt_out).abs().max().item()
            rel_diff = max_diff / (original_out.abs().max().item() + 1e-8)
            print(f"  Max abs diff:  {max_diff:.6e}")
            print(f"  Max rel diff:  {rel_diff:.6e}")
            if rel_diff < 1e-4:
                print("  ✓ Semantically equivalent (within tolerance)")
            else:
                print("  ✗ WARNING: large difference detected!")

    return optimized_module, stats


def param_report(model: torch.nn.Module,
                 optimized_module: torch.nn.Module) -> dict:
    """Joint graph+parameter view: which original parameters survive in
    the optimized realization, which were eliminated, and which were
    derived (folded) — the 'optimized weights file' diff.

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
    orig_bytes = sum(p.numel() * p.element_size() for p in orig.values())
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


def save_optimized_weights(optimized_module: torch.nn.Module,
                           path: str) -> None:
    """Emit the optimized weights file — only the parameters the
    certified form actually needs (folded derived tensors included)."""
    torch.save(optimized_module.state_dict(), path)


def ir_to_string(term: Any) -> str:
    """Pretty-print an IR term as an S-expression."""
    return op_repr(term)


def term_cost(term: Any, cost_fn=None) -> float:
    """Compute the cost of a term using the given cost function."""
    if cost_fn is None:
        cost_fn = flops_cost
    return cost_fn(term)
