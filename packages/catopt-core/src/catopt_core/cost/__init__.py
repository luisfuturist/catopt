"""Cost models for e-graph extraction.

The cost model assigns a scalar "cost" to a term, used by
EGraph.extract_best to find the minimum-cost representative.

Models provided include:
* count_cost - counts the number of operations (simplest).
* flops_cost - estimates FLOPs using shape information.
* param_bytes_cost - counts stored parameter values (the storage
  axis; what lets extraction prefer weight-sharing members).
* executor_overhead / executor_cost_for - price the LOWERING:
  the serial evaluator's per-node dispatch+kernel time, the level-
  batched executors' leaf-gather/level-compose schedule, the compiled
  executor's fusion regions.
* fusion_regions - partition a term's op-DAG into Inductor-style
  pointwise-fusion regions (one region = one compiled kernel).
* fusion_member_key - the per-member tie-break extraction uses to
  prefer fusion-friendly members among near-cost-ties.
* fused_cost_for - the compiled lowering's price: each fusion region
  costs one launch + the max of its summed member FLOPs vs its
  external/boundary memory traffic; one dispatch per compiled graph.
* lowering_aware_cost_for - min over lowerings: extraction picks the
  term whose best lowering is cheapest.

For the "killer experiment", the FLOPs-based model matters: it rewards
the associativity / distributivity / naturality rewrites that produce
fewer total floating-point operations.

The implementation is split by concern across submodules:
:mod:`~catopt_core.cost.basic`, :mod:`~catopt_core.cost.params`,
:mod:`~catopt_core.cost.roofline`, :mod:`~catopt_core.cost.fusion`,
:mod:`~catopt_core.cost.executor` and :mod:`~catopt_core.cost.backend`.
This module re-exports the full public (and cross-module private)
surface, so every ``from catopt_core.cost import X`` keeps working.
"""

from __future__ import annotations

from catopt_core.typing import (
    _INVALID as _INVALID,
)
from catopt_core.typing import (
    _broadcast as _broadcast,
)
from catopt_core.typing import (
    _infer_op_shape as _infer_op_shape,
)
from catopt_core.typing import (
    _numel as _numel,
)
from catopt_core.typing import (
    _shape_of as _shape_of,
)
from catopt_core.typing import (
    has_var_leaf as has_var_leaf,
)

from .backend import (
    CostModel as CostModel,
)
from .backend import (
    _ops_supported as _ops_supported,
)
from .backend import (
    backend_cost as backend_cost,
)
from .basic import (
    _INVALID_COST as _INVALID_COST,
)
from .basic import (
    _LAUNCH_PENALTY as _LAUNCH_PENALTY,
)
from .basic import (
    _OP_FLOPS as _OP_FLOPS,
)
from .basic import (
    _VIEW_OPS as _VIEW_OPS,
)
from .basic import (
    _CostMarkers as _CostMarkers,
)
from .basic import (
    _flops_of as _flops_of,
)
from .basic import (
    _local_cost as _local_cost,
)
from .basic import (
    _memo_dispatch as _memo_dispatch,
)
from .basic import (
    count_cost as count_cost,
)
from .basic import (
    flops_cost as flops_cost,
)
from .basic import (
    launch_aware_cost as launch_aware_cost,
)
from .executor import (
    _SCAN_COMPOSE_OPS as _SCAN_COMPOSE_OPS,
)
from .executor import (
    _SCAN_ROOT_OPS as _SCAN_ROOT_OPS,
)
from .executor import (
    LOWERINGS as LOWERINGS,
)
from .executor import (
    _batched_scan_latency as _batched_scan_latency,
)
from .executor import (
    _batched_scan_overhead as _batched_scan_overhead,
)
from .executor import (
    _generic_latency_ns as _generic_latency_ns,
)
from .executor import (
    _generic_overhead as _generic_overhead,
)
from .executor import (
    _leaf_gather_base as _leaf_gather_base,
)
from .executor import (
    _leaf_shared_a as _leaf_shared_a,
)
from .executor import (
    _outside_overhead as _outside_overhead,
)
from .executor import (
    executor_cost_for as executor_cost_for,
)
from .executor import (
    executor_overhead as executor_overhead,
)
from .executor import (
    lowering_aware_cost_for as lowering_aware_cost_for,
)
from .fusion import (
    _FUSIBLE_OPS as _FUSIBLE_OPS,
)
from .fusion import (
    _FUSION_TRANSPARENT_OPS as _FUSION_TRANSPARENT_OPS,
)
from .fusion import (
    _SOLVER_FACTOR as _SOLVER_FACTOR,
)
from .fusion import (
    _SOLVER_OPS as _SOLVER_OPS,
)
from .fusion import (
    _exposes_pointwise as _exposes_pointwise,
)
from .fusion import (
    _fused_cost as _fused_cost,
)
from .fusion import (
    _region_traffic as _region_traffic,
)
from .fusion import (
    fused_cost_for as fused_cost_for,
)
from .fusion import (
    fusion_member_key as fusion_member_key,
)
from .fusion import (
    fusion_regions as fusion_regions,
)
from .params import (
    _FOLDABLE_ELEMWISE as _FOLDABLE_ELEMWISE,
)
from .params import (
    _FOLDABLE_VIEWS as _FOLDABLE_VIEWS,
)
from .params import (
    _FUSION_POINTWISE_OPS as _FUSION_POINTWISE_OPS,
)
from .params import (
    _fold_ewidth as _fold_ewidth,
)
from .params import (
    _fold_numel as _fold_numel,
)
from .params import (
    _folds_to_param as _folds_to_param,
)
from .params import (
    _has_var_leaf as _has_var_leaf,
)
from .params import (
    _param_index as _param_index,
)
from .params import (
    _param_numel as _param_numel,
)
from .params import (
    _param_resolves as _param_resolves,
)
from .params import (
    dag_cost as dag_cost,
)
from .params import (
    param_bytes_cost as param_bytes_cost,
)
from .params import (
    param_bytes_cost_for as param_bytes_cost_for,
)
from .roofline import (
    _LAUNCH_S as _LAUNCH_S,
)
from .roofline import (
    _MEASURED_GATHER_OPS as _MEASURED_GATHER_OPS,
)
from .roofline import (
    _MEASURED_REDUCE_OPS as _MEASURED_REDUCE_OPS,
)
from .roofline import (
    _PEAK_BW as _PEAK_BW,
)
from .roofline import (
    _PEAK_FLOPS as _PEAK_FLOPS,
)
from .roofline import (
    _STRIDE_PENALTY as _STRIDE_PENALTY,
)
from .roofline import (
    _bytes_of as _bytes_of,
)
from .roofline import (
    _depth_cost as _depth_cost,
)
from .roofline import (
    _is_strided as _is_strided,
)
from .roofline import (
    _kernel_lookup as _kernel_lookup,
)
from .roofline import (
    _kernel_signature as _kernel_signature,
)
from .roofline import (
    _local_roofline as _local_roofline,
)
from .roofline import (
    _mm_signature as _mm_signature,
)
from .roofline import (
    _profile_constants as _profile_constants,
)
from .roofline import (
    _profile_dispatch_s as _profile_dispatch_s,
)
from .roofline import (
    _profile_graph_overhead_s as _profile_graph_overhead_s,
)
from .roofline import (
    _profile_kernel_table as _profile_kernel_table,
)
from .roofline import (
    _profile_leaf_eval_s as _profile_leaf_eval_s,
)
from .roofline import (
    _roofline_cost as _roofline_cost,
)
from .roofline import (
    depth_cost as depth_cost,
)
from .roofline import (
    depth_cost_for as depth_cost_for,
)
from .roofline import (
    roofline_cost as roofline_cost,
)
from .roofline import (
    roofline_cost_for as roofline_cost_for,
)

__all__ = [
    "LOWERINGS",
    "CostModel",
    "backend_cost",
    "count_cost",
    "dag_cost",
    "depth_cost",
    "depth_cost_for",
    "executor_cost_for",
    "executor_overhead",
    "flops_cost",
    "fused_cost_for",
    "fusion_member_key",
    "fusion_regions",
    "has_var_leaf",
    "launch_aware_cost",
    "lowering_aware_cost_for",
    "param_bytes_cost",
    "param_bytes_cost_for",
    "roofline_cost",
    "roofline_cost_for",
]
