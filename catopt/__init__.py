"""catopt — Categorical optimization of neural-network computation graphs.

This package implements a prototype pipeline that:

1. Converts a PyTorch computation graph (via torch.export / ATen) into a
   categorical / string-diagram IR.
2. Performs equality saturation over an e-graph, applying rewrite rules
   derived from categorical laws (associativity, symmetry/commutativity,
   distributivity/naturality, etc.).
3. Extracts the lowest-cost equivalent program according to a cost model.
4. Lowers the result back to ATen/FX for TorchInductor compilation and
   benchmarking against vanilla Inductor output.
"""

# locked: IRModule(ir, param_values={...}) lowers right-assoc
# matrix chains to ONE fused runtime matmul; catopt == Inductor both go
# through torch.compile.  Measured on CPU: 1.6x wall-clock.
from catopt.ir import IR, Op, Var, Const, Param, TensorType  # noqa: E401,F401
from catopt.egraph import EGraph, ENode, Rewrite  # noqa: E401,F401
from catopt.rules import all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES  # noqa: E401,F401
from catopt.cost import CostModel, count_cost, flops_cost  # noqa: E401,F401
from catopt.torch_bridge import export_to_ir, ir_to_torch_module, IRModule  # noqa: E401,F401
from catopt.optimize import optimize_model  # noqa: E401,F401
from catopt.omd_lower import (  # noqa: E401,F401
    BatchedOmdModule, to_batched_omd_module, is_omd_apply_term,
    build_omd_plan,
)

__version__ = "0.1.0dev"
__all__ = [
    "IR", "Var", "Const", "Param", "Op", "TensorType",
    "EGraph", "ENode", "Rewrite",
    "all_rules", "SIMPLIFICATION_RULES", "CATEGORICAL_RULES",
    "CostModel", "count_cost", "flops_cost",
    "export_to_ir", "ir_to_torch_module", "IRModule",
    "optimize_model",
    "BatchedOmdModule", "to_batched_omd_module", "is_omd_apply_term",
    "build_omd_plan",
]
