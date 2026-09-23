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

from catopt.ir import IR, Var, Const, Param, Op, TensorType
from catopt.egraph import EGraph, ENode, Rewrite
from catopt.rules import all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES
from catopt.cost import CostModel, count_cost, flops_cost
from catopt.torch_bridge import export_to_ir, ir_to_torch_module, IRModule
from catopt.optimize import optimize_model

__version__ = "0.1.0dev"
__all__ = [
    "IR", "Var", "Const", "Param", "Op", "TensorType",
    "EGraph", "ENode", "Rewrite",
    "all_rules", "SIMPLIFICATION_RULES", "CATEGORICAL_RULES",
    "CostModel", "count_cost", "flops_cost",
    "export_to_ir", "ir_to_torch_module", "IRModule",
    "optimize_model",
]
