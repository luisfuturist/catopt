# ruff: noqa: E402 — the alias loop must precede the public imports
"""catopt — Categorical optimization of neural-network computation graphs.

Façade + compatibility layer over the domain packages (plan 0003).
The engine is split across distributions — ``catopt-core`` (the
torch-free semantic engine: IR, attrs, typing, egraph, laws, cost,
meta, rulecache, ports, ops), ``catopt-torch`` (the PyTorch adapter:
bridge, executors, models, reports), ``catopt-carriers`` (om/xcarrier/
trace lifts + lowerers), ``catopt-eps`` (the opt-in approximation
toolkit), ``catopt-optimize`` (the pipeline orchestrators).

``sys.modules`` aliases below keep every historical ``catopt.X`` path
resolving to its new home — ``from catopt.cost import flops_cost``,
``import catopt.egraph.core``, ``from catopt.rules import all_rules``
etc. all keep working unchanged.
"""

# -- compat: alias every historical module path BEFORE anything else --
import sys as _sys
from importlib import import_module as _imp

_ALIAS = {
    "catopt.act_eps": "catopt_eps.act_eps",
    "catopt.attrs": "catopt_core.attrs",
    "catopt.calibrate": "catopt_optimize.calibrate",
    "catopt.cost": "catopt_core.cost",
    "catopt.egraph": "catopt_core.egraph",
    "catopt.egraph.certs": "catopt_core.egraph.certs",
    "catopt.egraph.core": "catopt_core.egraph.core",
    "catopt.egraph.extract": "catopt_core.egraph.extract",
    "catopt.egraph.proof": "catopt_core.egraph.proof",
    "catopt.egraph.terms": "catopt_core.egraph.terms",
    "catopt.egraph.types": "catopt_core.egraph.types",
    "catopt.eps": "catopt_eps.eps",
    "catopt.executors": "catopt_torch.executors",
    "catopt.executors.base": "catopt_torch.executors.base",
    "catopt.ibp": "catopt_eps.ibp",
    "catopt.ir": "catopt_core.ir",
    "catopt.laws": "catopt_core.laws",
    "catopt.laws.base": "catopt_core.laws.base",
    "catopt.laws.pairing": "catopt_core.laws.pairing",
    "catopt.laws.scan": "catopt_core.laws.scan",
    "catopt.laws.tensor": "catopt_core.laws.tensor",
    "catopt.meta": "catopt_core.meta",
    "catopt.models": "catopt_torch.models",
    "catopt.models.hybrid": "catopt_torch.models.hybrid",
    "catopt.models.ssm": "catopt_torch.models.ssm",
    "catopt.om": "catopt_carriers.om",
    "catopt.om_lower": "catopt_carriers.om_lower",
    "catopt.omd_lower": "catopt_carriers.omd_lower",
    "catopt.ops": "catopt_core.ops",
    "catopt.optimize": "catopt_optimize.optimize",
    "catopt.ports": "catopt_core.ports",
    "catopt.regime": "catopt_optimize.regime",
    "catopt.report": "catopt_torch.report",
    "catopt.rulecache": "catopt_core.rulecache",
    "catopt.rules": "catopt_core.rules",
    "catopt.scan_lower": "catopt_carriers.scan_lower",
    "catopt.torch_bridge": "catopt_torch.torch_bridge",
    "catopt.trace": "catopt_carriers.trace",
    "catopt.trace_lift": "catopt_carriers.trace_lift",
    "catopt.typing": "catopt_core.typing",
    "catopt.xcarrier": "catopt_carriers.xcarrier",
}

for _old, _new in _ALIAS.items():
    try:
        _mod = _imp(_new)
        _sys.modules.setdefault(_old, _mod)
        # attribute access too — `catopt.egraph`/`catopt.optimize` as
        # getattr targets (import machinery only sees sys.modules).
        # Only direct children: `catopt.egraph.core` resolves via the
        # real `catopt.egraph` package attr naturally.
        if _old.count(".") == 1:
            _sys.modules[__name__].__dict__.setdefault(
                _old.split(".", 1)[1], _mod
            )
    except ModuleNotFoundError:  # pragma: no cover — only fires when a
        pass  # domain package isn't installed (partial install)

del _sys, _imp

# -- public API ---------------------------------------------------------
from catopt_carriers.omd_lower import (
    BatchedOmdModule,
    build_omd_plan,
    is_omd_apply_term,
    to_batched_omd_module,
)
from catopt_core.cost import (
    CostModel,
    count_cost,
    flops_cost,
)
from catopt_core.egraph import EGraph, ENode, Rewrite
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
)
from catopt_core.rules import (
    CATEGORICAL_RULES,
    SIMPLIFICATION_RULES,
    all_rules,
)
from catopt_optimize.optimize import optimize_model
from catopt_torch.torch_bridge import (
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
