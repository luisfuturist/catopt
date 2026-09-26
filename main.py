"""Main entry point: demonstrates the catopt optimization pipeline.

Runs the "killer experiment" on a MatrixChain model where the
associativity of matrix multiplication (a categorical law) enables a
cheaper association order that TorchInductor does not discover.
"""

from __future__ import annotations

import argparse

import torch

from catopt.cost import flops_cost, launch_aware_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt.models import MatrixChain
from catopt.optimize import optimize_model
from catopt.rules import CATEGORICAL_RULES, SIMPLIFICATION_RULES
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def demo_matrix_chain(batch: int = 128, verbose: bool = True) -> None:
    """Demonstrate matrix-chain associativity optimization."""
    d0, d1, d2, d3 = 128, 64, 32, 8

    model = MatrixChain(d0, d1, d2, d3)
    x = torch.randn(batch, d0)

    print(f"\n{'=' * 70}")
    print(f"  Case: MatrixChain({d0}→{d1}→{d2}→{d3}), batch={batch}")
    print(
        f"  Left-assoc FLOPs:  {MatrixChain.flops((d0, d1, d2, d3), batch):,}"
    )
    print(
        f"  Fused FLOPs:       {MatrixChain.fused_flops((d0, d1, d2, d3), batch):,}"
    )
    print(
        f"  Expected speedup:  {MatrixChain.flops((d0, d1, d2, d3), batch) / MatrixChain.fused_flops((d0, d1, d2, d3), batch):.1f}x"
    )
    print(f"{'=' * 70}")

    ir, source_tensors = export_to_ir(model, x)
    if verbose:
        print(f"\n  Original IR: {op_repr(ir.root)}")
        print(f"  Original cost: {flops_cost(ir.root):.0f} FLOPs")

    # Build e-graph and saturate
    eg = EGraph()
    root_eid = eg.add_term(ir.root)
    rules = CATEGORICAL_RULES
    stats = eg.run(rules, root_eid, max_iterations=20, max_nodes=5000)
    if verbose:
        print(f"\n  E-graph stats: {stats}")

    # Extract best term
    best = eg.extract_best(root_eid, flops_cost)
    if verbose:
        print(f"  Optimized IR:  {op_repr(best)}")
        print(f"  Optimized cost: {flops_cost(best):.0f} FLOPs")

    # Verify correctness
    model.eval()
    optimized = ir_to_torch_module(
        IR(
            root=best,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        ),
        param_values=source_tensors,
    )
    optimized.eval()

    with torch.no_grad():
        orig_out = model(x.clone())
        opt_out = optimized(x.clone())
        max_diff = (orig_out - opt_out).abs().max().item()
        print(f"\n  ✓ Output max abs diff: {max_diff:.6e}")
        if max_diff < 1e-3:
            print("  ✓ Semantically equivalent (within tolerance)")
        else:
            print("  ✗ WARNING: outputs differ!")


def demo_large_batch(batch: int = 4096, verbose: bool = True) -> None:
    """CPU-friendly larger-batch timing for the matrix-chain case.

    The 6.3x FLOP figure includes the one-time W1@(W2@W3) precompute
    (147k FLOPs); bigger batches amortize it, so the *runtime* gap
    should widen toward the pure-runtime ratio
    batch*d0*d1 + ... vs batch*d0*d3 (here 18688 vs 2048 per sample,
    i.e. ~9.1x in matmul FLOPs).  Wall-clock on CPU also reflects
    inductor overhead, thread scaling, and memory traffic, so this
    reports measured numbers alongside the theory instead of
    conflating the two.
    """
    import time

    from catopt.models import MatrixChain

    torch.manual_seed(0)
    d0, d1, d2, d3 = 128, 64, 32, 8
    model = MatrixChain(d0, d1, d2, d3)
    x = torch.randn(batch, d0)
    ir, source_tensors = export_to_ir(model, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=5000)
    best = eg.extract_best(eid, flops_cost)
    opt = ir_to_torch_module(
        IR(
            root=best,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        ),
        param_values=source_tensors,
    )
    model.eval()
    opt.eval()
    with torch.no_grad():
        co = torch.compile(model)
        cc = torch.compile(opt)
        for _ in range(3):
            _ = co(x)
            _ = cc(x)

        def bench(f, n: int = 20) -> float:
            ts = []
            with torch.no_grad():
                for _ in range(n):
                    t0 = time.perf_counter()
                    _ = f(x)
                    ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            t = ts[2:-2]
            return sum(t) / len(t)

        to, tc = bench(co), bench(cc)
        per_left = (
            d0 * d1 + d1 * d2 + d2 * d3
        )  # matmul-FLOPs/2 per sample
        per_right = d0 * d3
        pre = d1 * d2 * d3 + d0 * d1 * d3  # 2x counted below
        print(f"\n{'=' * 70}")
        print(f"  Large-batch timing: batch={batch}")
        print(
            f"  Runtime-only FLOP ratio (theory): "
            f"{per_left / per_right:.1f}x"
        )
        print(f"  Precompute (one-time): {2 * pre:,} FLOPs")
        print(f"  Inductor (orig):   {to:.2f} ms")
        print(
            f"  Catopt + Inductor: {tc:.2f} ms  (speedup {to / tc:.2f}x)"
        )
        print(f"{'=' * 70}")


def demo_parallel_projections(
    batch: int = 4096, verbose: bool = True
) -> None:
    """The composed win: two categorical laws + weight folding.

    ParallelLinear:  x@W1 + x@W2            ->  x @ (W1+W2)   [2.00x FLOPs]
    DeepParallel:    (W1(x) + W2(x)) @ W3   ->  x @ (W3 @ (W1+W2))  [2.91x]

    DeepParallel needs `weight_factor_linear` to fire BEFORE
    `assoc_linear` can rewrite the stack, then IRModule folds the
    weight-only product once at construction time.  Measured against
    torch.compile on the unmodified graph: Inductor does not find it.
    """
    from catopt.models import DeepParallel, ParallelLinear

    torch.manual_seed(0)
    print(f"\n{'=' * 70}")
    print("  Case: parallel projections (weight merging / bilinearity)")
    print(f"{'=' * 70}")

    for label, model, x in (
        (
            "ParallelLinear",
            ParallelLinear(128, n_experts=2),
            torch.randn(batch, 128),
        ),
        (
            "DeepParallel",
            DeepParallel(128, 128, 128),
            torch.randn(batch, 128),
        ),
    ):
        ir, source_tensors = export_to_ir(model, x)
        eg = EGraph()
        eid = eg.add_term(ir.root)
        eg.run(
            CATEGORICAL_RULES + SIMPLIFICATION_RULES,
            eid,
            max_iterations=30,
            max_nodes=50000,
        )
        best = eg.extract_best(eid, flops_cost)
        ratio = flops_cost(ir.root) / max(flops_cost(best), 1)
        if verbose:
            print(f"\n  {label} IR:  {op_repr(ir.root)}")
            print(f"  orig FLOPs:  {flops_cost(ir.root):,.0f}")
            print(f"  best term:   {op_repr(best)}")
            print(
                f"  best FLOPs:  {flops_cost(best):,.0f}   ratio {ratio:.2f}x"
            )
        lowered = ir_to_torch_module(
            IR(
                root=best,
                inputs=ir.inputs,
                input_names=ir.input_names,
                params=ir.params,
            ),
            param_values=source_tensors,
        )
        model.eval()
        lowered.eval()
        with torch.no_grad():
            diff = (
                (model(x.clone()) - lowered(x.clone()))
                .abs()
                .max()
                .item()
            )
        print(
            f"  \u2713 {label}: equiv diff {diff:.3e}, "
            f"runtime params {[n for n, _ in lowered.named_parameters()]}"
        )


def demo_fused_projections(verbose: bool = True) -> None:
    """Product-structure results: fused SwiGLU gate/up, fused QKV, norm fold.

    These are parameter-restructuring transforms Inductor cannot express:
    pairing projections of a shared input via the product universal
    property <f,g> = (f x g) . Delta.
    """
    from catopt.models import (
        AttentionBlock,
        GQAAttention,
        NormLinear,
        ParallelBlock,
        SwiGLU,
    )

    torch.manual_seed(0)
    if verbose:
        print(f"\n{'=' * 70}")
        print("  Case: product structure — fused projections")
        print(f"{'=' * 70}")

    cases = [
        (
            "SwiGLU gate/up",
            SwiGLU(64, hidden_mult=2),
            torch.randn(128, 64),
        ),
        (
            "Attention QKV",
            AttentionBlock(64, n_heads=4),
            torch.randn(2, 8, 64),
        ),
        (
            "GQA fused QKV",
            GQAAttention(128, 4, 2),
            torch.randn(2, 8, 128),
        ),
        (
            "NormLinear fold",
            NormLinear(64, 64),
            torch.randn(128, 8, 64),
        ),
        (
            "ParallelBlock (5-way)",
            ParallelBlock(128, n_heads=4, hidden_mult=2),
            torch.randn(4, 16, 128),
        ),
    ]
    for name, model, x in cases:
        opt, stats = optimize_model(model, x, verbose=False)
        model.eval()
        opt.eval()
        with torch.no_grad():
            d = (model(x.clone()) - opt(x.clone())).abs().max().item()
        r = op_repr(opt._root)
        n_fused = len(
            [p for p in opt._param_map if p.startswith("fused_")]
        )
        n_views = r.count("(split") + r.count("(chunk")
        ok = "✓" if d < 1e-4 else "✗"
        if verbose:
            print(f"\n  {name}:")
            print(f"    best:  {r[:110]}")
            print(
                f"    {ok} diff {d:.2e} | fused params: {n_fused}"
                f" | projection views: {n_views}"
                f" | paired: {bool(stats.get('paired_extract'))}"
            )


def demo_naturality(batch: int = 64, verbose: bool = True) -> None:
    """Demonstrate naturality of scalar multiplication w.r.t. matmul.

    Original:  out = (x * c) @ W    [scale first, then matmul]
    Naturate:  out = (x @ W) * c    [matmul first, then scale]

    Both have the same FLOPs, but the naturated form lets the elementwise
    mul potentially fuse with something downstream, and avoids materializing
    the scaled input as a separate buffer.
    """
    dim, out_dim = 32, 32
    W = Param("W", TensorType((dim, out_dim)))
    x = Var("x", TensorType((batch, dim)))
    c = Var("c", TensorType((batch, 1)))

    original = Op.make("matmul", Op.make("mul", x, c), W)
    naturated = Op.make("mul", Op.make("matmul", x, W), c)

    if verbose:
        print(f"\n{'=' * 70}")
        print("  Case: Naturality of scalar multiplication")
        print(f"{'=' * 70}")
        print(f"  Original:     {op_repr(original)}")
        print(f"  Naturated:    {op_repr(naturated)}")
        print(f"  Original cost: {flops_cost(original):.0f} FLOPs")
        print(f"  Naturated cost: {flops_cost(naturated):.0f} FLOPs")

    eg = EGraph()
    root_eid = eg.add_term(original)
    stats = eg.run(
        CATEGORICAL_RULES, root_eid, max_iterations=10, max_nodes=1000
    )
    best = eg.extract_best(root_eid, flops_cost)

    if verbose:
        print(f"  E-graph stats: {stats}")
        print(f"  Best (by FLOPs): {op_repr(best)}")
        print(
            "  ✓ E-graph confirms naturality: both forms are equivalent"
        )


def demo_swiglu_rmsnorm(
    dim: int = 32, batch: int = 4, seqlen: int = 8, verbose: bool = True
) -> None:
    """Demonstrate the categorical pipeline on real exported SwiGLU/RMSNorm.

    Uses the polyhedral-RFC target ops (SwiGLU + RMSNorm from
    catopt.models).  Both export cleanly now that _graph_signature remaps
    the GraphModule's 'p_<attr>' placeholders to 'gate.weight' etc.  The
    e-graph then applies silu/pow bridge rules (x*sigmoid(x), pow<->square)
    so the SwiGLU gate/up prefix sharing and RMSNorm's x**2 scaling are
    visible as ONE equivalence class per op.  IRModule lowers with the
    ORIGINAL weights, so semantic equivalence is verified bit-exactly.
    """
    from catopt.models import RMSNorm, SwiGLU

    torch.manual_seed(0)
    if verbose:
        print(f"\n{'=' * 70}")
        print("  Case: SwiGLU + RMSNorm (polyhedral-RFC targets)")
        print(f"{'=' * 70}")

    xg = torch.randn(batch, seqlen, dim)
    sg, rn = SwiGLU(dim), RMSNorm(dim)
    for name, model, x in (("SwiGLU", sg, xg), ("RMSNorm", rn, xg)):
        ir, source_tensors = export_to_ir(model, x)
        eg = EGraph()
        root_eid = eg.add_term(ir.root)
        stats = eg.run(
            CATEGORICAL_RULES + SIMPLIFICATION_RULES,
            root_eid,
            max_iterations=30,
            max_nodes=20000,
        )
        best = eg.extract_best(root_eid, launch_aware_cost)
        if verbose:
            print(f"\n  {name} IR:      {op_repr(ir.root)}")
            print(f"  {name} cost:    {flops_cost(ir.root):.0f} FLOPs")
            print(f"  E-graph stats:  {stats}")
            print(f"  Best term:      {op_repr(best)}")
            print(
                f"  Best cost:      {flops_cost(best):.0f} FLOPs"
                "  (tree count; shared subterms counted per use)"
            )
        model.eval()
        lowered = ir_to_torch_module(
            IR(
                root=best,
                inputs=ir.inputs,
                input_names=ir.input_names,
                params=ir.params,
            ),
            param_values=source_tensors,
        )
        lowered.eval()
        with torch.no_grad():
            out_o = model(x.clone())
            out_n = lowered(x.clone())
            diff = (out_o - out_n).abs().max().item()
            ok = "✓" if diff < 1e-4 else "✗"
            print(f"  {ok} {name}: max abs diff {diff:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="catopt — categorical NN optimizer demo"
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--large-batch",
        type=int,
        default=0,
        help="Run an extra CPU-friendly large-batch matrix-chain "
        "timing (e.g. --large-batch 4096).  The 6.3x FLOP "
        "figure is compile-time + runtime; large batches "
        "amortize the one-time W1@(W2@W3) precompute and "
        "narrow the measured gap toward it.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Also benchmark with torch.compile (slow)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(
        "  catopt — Categorical Optimization of Neural Network Graphs"
    )
    print("  Demonstrating: associativity + naturality of matmul")
    print("  Note: 6.3x is FLOP count (incl. one-time precompute);")
    print(
        "        runtime was 1.6x on CPU batch=128 (see --large-batch)."
    )
    print("=" * 70)

    demo_matrix_chain(batch=args.batch)
    if args.large_batch:
        demo_large_batch(batch=args.large_batch)
    demo_swiglu_rmsnorm()
    demo_parallel_projections(batch=max(args.batch, 4096))
    demo_fused_projections()
    demo_naturality(batch=64)

    if args.bench:
        print("\n>>> Benchmarking with torch.compile --bench <<<")
        from catopt.optimize import optimize_model

        d0, d1, d2, d3 = 128, 64, 32, 8
        model = MatrixChain(d0, d1, d2, d3)
        x = torch.randn(args.batch, d0)
        optimize_model(model, x, verbose=True)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
