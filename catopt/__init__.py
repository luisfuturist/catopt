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
from catopt.cost import (
    CostModel,
    count_cost,
    flops_cost,
)
from catopt.egraph import EGraph, ENode, Rewrite
from catopt.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
)
from catopt.omd_lower import (
    BatchedOmdModule,
    build_omd_plan,
    is_omd_apply_term,
    to_batched_omd_module,
)
from catopt.optimize import optimize_model
from catopt.rules import (
    CATEGORICAL_RULES,
    SIMPLIFICATION_RULES,
    all_rules,
)
from catopt.torch_bridge import (
    IRModule,
    export_to_ir,
    ir_to_torch_module,
)

__version__ = "0.1.0dev"
__all__ = [
    "CATEGORICAL_RULES",
    "IR",
    "SIMPLIFICATION_RULES",
    "BatchedOmdModule",
    "Const",
    "CostModel",
    "EGraph",
    "ENode",
    "IRModule",
    "Op",
    "Param",
    "Rewrite",
    "TensorType",
    "Var",
    "all_rules",
    "build_omd_plan",
    "count_cost",
    "export_to_ir",
    "flops_cost",
    "ir_to_torch_module",
    "is_omd_apply_term",
    "optimize_model",
    "to_batched_omd_module",
]
