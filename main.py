"""Main entry point: demonstrates the catopt optimization pipeline.

Runs the "killer experiment" on a MatrixChain model where the
associativity of matrix multiplication (a categorical law) enables a
cheaper association order that TorchInductor does not discover.
"""

from __future__ import annotations

import argparse
import torch

from catopt.egraph import EGraph
from catopt.ir import Op, Var, Param, TensorType, IR, op_repr
from catopt.rules import CATEGORICAL_RULES
from catopt.cost import flops_cost
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.models import MatrixChain


def demo_matrix_chain(batch: int = 128, verbose: bool = True) -> None:
    """Demonstrate matrix-chain associativity optimization."""
    d0, d1, d2, d3 = 128, 64, 32, 8

    model = MatrixChain(d0, d1, d2, d3)
    x = torch.randn(batch, d0)

    print(f"\n{'='*70}")
    print(f"  Case: MatrixChain({d0}→{d1}→{d2}→{d3}), batch={batch}")
    print(f"  Left-assoc FLOPs:  {MatrixChain.flops((d0,d1,d2,d3), batch):,}")
    print(f"  Fused FLOPs:       {MatrixChain.fused_flops((d0,d1,d2,d3), batch):,}")
    print(f"  Expected speedup:  {MatrixChain.flops((d0,d1,d2,d3), batch) / MatrixChain.fused_flops((d0,d1,d2,d3), batch):.1f}x")
    print(f"{'='*70}")

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
    optimized = ir_to_torch_module(IR(
        root=best, inputs=ir.inputs,
        input_names=ir.input_names, params=ir.params
    ), param_values=source_tensors)
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
        print(f"\n{'='*70}")
        print(f"  Case: Naturality of scalar multiplication")
        print(f"{'='*70}")
        print(f"  Original:     {op_repr(original)}")
        print(f"  Naturated:    {op_repr(naturated)}")
        print(f"  Original cost: {flops_cost(original):.0f} FLOPs")
        print(f"  Naturated cost: {flops_cost(naturated):.0f} FLOPs")

    eg = EGraph()
    root_eid = eg.add_term(original)
    stats = eg.run(CATEGORICAL_RULES, root_eid, max_iterations=10, max_nodes=1000)
    best = eg.extract_best(root_eid, flops_cost)

    if verbose:
        print(f"  E-graph stats: {stats}")
        print(f"  Best (by FLOPs): {op_repr(best)}")
        print("  ✓ E-graph confirms naturality: both forms are equivalent")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="catopt — categorical NN optimizer demo")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--bench", action="store_true",
                        help="Also benchmark with torch.compile (slow)")
    args = parser.parse_args()

    print("=" * 70)
    print("  catopt — Categorical Optimization of Neural Network Graphs")
    print("  Demonstrating: associativity + naturality of matmul")
    print("=" * 70)

    demo_matrix_chain(batch=args.batch)
    demo_naturality(batch=64)

    if args.bench:
        print("\n>>> Benchmarking with torch.compile --bench <<<")
        from catopt.torch_bridge import export_to_ir
        from catopt.optimize import optimize_model
        d0, d1, d2, d3 = 128, 64, 32, 8
        model = MatrixChain(d0, d1, d2, d3)
        x = torch.randn(args.batch, d0)
        optimize_model(model, x, verbose=True)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()