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
                          pair_shared_input_linears)
from catopt.cost import (flops_cost, count_cost, launch_aware_cost,
                         CostModel, dag_cost)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.ir import op_repr


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
    groups = pair_shared_input_linears(eg)
    if groups:
        eg.rebuild()
        stats["pairing_groups"] = len(groups)
        # brief second saturation so other rules see the new enodes
        eg.run(rules, root_eid, max_iterations=5, max_nodes=max_enodes)

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


def ir_to_string(term: Any) -> str:
    """Pretty-print an IR term as an S-expression."""
    return op_repr(term)


def term_cost(term: Any, cost_fn=None) -> float:
    """Compute the cost of a term using the given cost function."""
    if cost_fn is None:
        cost_fn = flops_cost
    return cost_fn(term)
