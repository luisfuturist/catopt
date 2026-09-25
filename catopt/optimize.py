"""Top-level optimization pipeline.

This module implements the four-phase killer experiment:

Phase 1 — Equivalence:   PyTorch → IR
Phase 2 — Search:        IR → e-graph → equality saturation → best term
Phase 3 — Lower:          best term → torch.nn.Module
Phase 4 — Compare:        benchmark vs. vanilla TorchInductor

The main entry point is :func:`optimize_model`.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable

import torch

from catopt.ir import IR, Op, Var, Const, Param
from catopt.egraph import EGraph
from catopt.rules import (all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES,
                          pair_shared_input_linears,
                          pair_shared_input_convs,
                          share_duplicate_params,
                          share_duplicate_param_slices)
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
             + share_duplicate_params(eg, source_tensors)
             + share_duplicate_param_slices(eg, source_tensors))
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
    eps_rtol: float | None = None,
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
    eps_rtol : float, optional
        Optional certified-approximation toolkit — off by default and
        not part of the core optimizer.  When set, also run the
        certified-approximation passes
        (``eps.low_rank_params`` + ``eps.kron_linear_params``): each
        offer carries an exact Eckart–Young / Frobenius bound and is
        recorded in ``stats["eps_offers"]``.  The offers only *win*
        under a storage-aware cost model (``param_bytes_cost``) or
        explicit selection — the default launch-aware cost keeps the
        exact member, so this never silently trades accuracy.
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
             + share_duplicate_params(eg, source_tensors)
             + share_duplicate_param_slices(eg, source_tensors))
    if eps_rtol is not None:
        from catopt.eps import (low_rank_params, kron_linear_params,
                                low_rank_gather)
        eps_offers = (low_rank_params(eg, source_tensors, rtol=eps_rtol)
                      + low_rank_gather(eg, source_tensors,
                                        rtol=eps_rtol)
                      + kron_linear_params(eg, source_tensors,
                                           rtol=eps_rtol))
        lifts += eps_offers
        stats["eps_offers"] = eps_offers
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


# ---------------------------------------------------------------------------
#  Compositional optimization — per-block eqsat, then recompose
# ---------------------------------------------------------------------------

def _default_block_pred(parent: torch.nn.Module, name: str,
                        module: torch.nn.Module) -> bool:
    """Default block selector: direct children of ``nn.ModuleList`` /
    ``nn.Sequential`` — the standard 'stacked blocks' structure."""
    return isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential))


def _select_blocks(model: torch.nn.Module,
                   block_pred: Callable | None) -> list[tuple[str, torch.nn.Module]]:
    """Walk the module tree and pick the top-most submodules to optimize
    independently.

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


def _capture_block_inputs(
    model: torch.nn.Module,
    blocks: list[tuple[str, torch.nn.Module]],
    example_input: torch.Tensor | tuple,
) -> dict[str, tuple[tuple, dict]]:
    """Run the ORIGINAL model once and record each selected block's first
    forward inputs via hooks.  Returns ``{name: (args, kwargs)}``."""
    captured: dict[str, tuple[tuple, dict]] = {}
    handles = []

    def make_hook(name: str):
        def hook(mod, args, kwargs, out):
            if name not in captured:
                captured[name] = (
                    tuple(a.detach().clone() if isinstance(a, torch.Tensor)
                          else a for a in args),
                    {k: (v.detach().clone() if isinstance(v, torch.Tensor)
                         else v) for k, v in kwargs.items()},
                )
        return hook

    for name, mod in blocks:
        handles.append(mod.register_forward_hook(make_hook(name),
                                                 with_kwargs=True))
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    try:
        model.eval()
        with torch.no_grad():
            model(*args)
    finally:
        for h in handles:
            h.remove()
    return captured


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max().item()
            / (a.abs().max().item() + 1e-8))


def _replace_submodule(model: torch.nn.Module, dotted: str,
                       new_mod: torch.nn.Module) -> None:
    """Set ``model.<dotted>`` to ``new_mod``, handling ModuleList /
    Sequential integer children."""
    parent_name, _, child_name = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(
            parent, (torch.nn.ModuleList, torch.nn.Sequential)):
        parent[int(child_name)] = new_mod
    else:
        setattr(parent, child_name, new_mod)


def optimize_compositional(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple,
    *,
    block_pred: Callable[[torch.nn.Module, str, torch.nn.Module], bool] | None = None,
    cost_fn=None,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int = 100_000,
    verify_tol: float = 1e-4,
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
       original implementation,
    4. clones the model and grafts the optimized ``IRModule`` back in
       place, preserving the original forward structure, then verifies
       end-to-end equivalence on ``example_input``.

    Returns ``(recomposed_module, stats)`` where ``stats["blocks"]`` maps
    each block's dotted name to ``{"status", "stats", "param_report",
    "time_s", ...}`` and ``stats["param_report"]`` aggregates the
    per-block parameter diffs (eliminated/derived names are prefixed by
    block name for auditability).
    """
    t_start = time.time()
    if cost_fn is None:
        cost_fn = launch_aware_cost

    blocks = _select_blocks(model, block_pred)
    if verbose:
        print(f"[Compositional] {len(blocks)} candidate blocks: "
              f"{[n for n, _ in blocks]}")

    captured = _capture_block_inputs(model, blocks, example_input)

    replacements: dict[str, torch.nn.Module] = {}
    block_stats: dict[str, dict[str, Any]] = {}
    agg = {"original_params": 0, "optimized_params": 0,
           "original_bytes": 0, "optimized_bytes": 0,
           "eliminated": [], "derived": []}

    for name, block in blocks:
        entry: dict[str, Any] = {}
        block_stats[name] = entry
        cap = captured.get(name)
        if cap is None:
            entry["status"] = "not_executed"
            continue
        args, kwargs = cap
        if kwargs:
            entry["status"] = "skipped"
            entry["reason"] = f"non-positional kwargs {sorted(kwargs)}"
            continue
        ex = args[0] if len(args) == 1 else args
        t0 = time.time()
        try:
            opt_mod, st = optimize_model(
                block, ex,
                ruleset=ruleset, max_iterations=max_iterations,
                max_enodes=max_enodes, cost_fn=cost_fn,
                verbose=verbose)
            # Per-block verification on the captured input — soundness
            # gate independent of optimize_model's own (verbose-gated)
            # check.  Any mismatch or eval failure falls back.
            with torch.no_grad():
                ref = block(*args)
                got = opt_mod(*args)
            rd = _rel_diff(ref, got)
            entry["rel_diff"] = rd
            if not (rd < verify_tol):
                raise RuntimeError(
                    f"block verification failed: rel diff {rd:.3e}")
            replacements[name] = opt_mod
            entry["status"] = "optimized"
            entry["stats"] = st
            pr = param_report(block, opt_mod)
            entry["param_report"] = pr
            agg["original_params"] += pr["original_params"]
            agg["optimized_params"] += pr["optimized_params"]
            agg["original_bytes"] += pr["original_bytes"]
            agg["optimized_bytes"] += pr["optimized_bytes"]
            agg["eliminated"] += [f"{name}:{n}" for n in pr["eliminated"]]
            agg["derived"] += [f"{name}:{n}" for n in pr["derived"]]
            if verbose:
                print(f"[Compositional] {name}: optimized "
                      f"({entry['rel_diff']:.2e})")
        except Exception as e:  # noqa: BLE001 — fallback keeps original
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[Compositional] {name}: keeping original ({e})")
        entry["time_s"] = time.time() - t0

    # -- Recompose -------------------------------------------------------
    in_place = False
    try:
        new_model = copy.deepcopy(model)
    except Exception:
        new_model = model
        in_place = True
    for name, opt_mod in replacements.items():
        _replace_submodule(new_model, name, opt_mod)

    # -- End-to-end verification ----------------------------------------
    stats: dict[str, Any] = {
        "compositional": True,
        "n_blocks": len(blocks),
        "n_optimized": len(replacements),
        "n_failed": sum(1 for e in block_stats.values()
                        if e.get("status") == "failed"),
        "n_skipped": sum(1 for e in block_stats.values()
                         if e.get("status") in ("skipped", "not_executed")),
        "blocks": block_stats,
        "in_place": in_place,
    }
    agg["bytes_saved"] = agg["original_bytes"] - agg["optimized_bytes"]
    agg["ratio"] = (agg["optimized_bytes"] / agg["original_bytes"]
                    if agg["original_bytes"] else 1.0)
    stats["param_report"] = agg

    args = (example_input if isinstance(example_input, tuple)
            else (example_input,))
    model.eval()
    new_model.eval()
    try:
        with torch.no_grad():
            ref = model(*[a.clone() if isinstance(a, torch.Tensor) else a
                          for a in args])
            out = new_model(*[a.clone() if isinstance(a, torch.Tensor) else a
                              for a in args])
        stats["end_to_end"] = {
            "max_abs_diff": (ref - out).abs().max().item(),
            "max_rel_diff": _rel_diff(ref, out),
        }
        if verbose:
            print(f"[Compositional] end-to-end rel diff: "
                  f"{stats['end_to_end']['max_rel_diff']:.3e}")
    except Exception as e:  # noqa: BLE001
        stats["end_to_end"] = {"error": f"{type(e).__name__}: {e}"}
        if verbose:
            print(f"[Compositional] end-to-end check failed: {e}")

    stats["wall_time_s"] = time.time() - t_start
    return new_model, stats
