"""Fixed torch_bridge.py — uses exported._graph_signature.inputs_to_parameters
to correctly map graph placeholder targets (p_w1, p_w2, ...) to actual
model parameter names (W1, W2, ...) and retrieve their shapes.
"""
from __future__ import annotations

from typing import Any

import torch

from catopt.ir import IR, Op, Var, Const, Param, TensorType


_ATEN_TO_IR: dict[str, str] = {
    "add": "add",
    "mul": "mul",
    "sub": "sub",
    "neg": "neg",
    "silu": "silu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": "gelu",
    "exp": "exp",
    "square": "square",
    "sum": "sum",
    "mean": "mean",
    "matmul": "matmul",
    "transpose": "transpose",
    "reshape": "reshape",
    "view": "reshape",
    "amax": "max",
    "matmul.default": "matmul",
}


def _strip_backend_suffix(name: str) -> str:
    for suffix in (".default", ".deterministic", ".backend", ".assigned"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _aten_name(target: Any) -> str:
    if hasattr(target, "__name__"):
        return _strip_backend_suffix(target.__name__)
    if isinstance(target, str):
        return _strip_backend_suffix(target)
    s = str(target)
    if "aten::" in s:
        name = s.split("aten::")[1].split(".")[0]
        return _strip_backend_suffix(name)
    return _strip_backend_suffix(s)


def _infer_shape(node_or_value: Any) -> tuple:
    if hasattr(node_or_value, "shape"):
        return tuple(
            int(d) if d is not None else None for d in node_or_value.shape
        )
    return (None,)


def export_to_ir(model: torch.nn.Module, example_input: torch.Tensor) -> IR:
    """Export a PyTorch model to a catopt IR via torch.export.

    Uses exported._graph_signature.inputs_to_parameters to map graph
    placeholder node targets (e.g. 'p_w1') to actual model attribute
    names (e.g. 'W1'), so we can retrieve parameter shapes correctly.
    """
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,))

    graph = exported.graph
    mod = exported.module()  # the actual Module instance

    # Dict to collect original parameter tensors for IRModule
    source_tensors: dict[str, torch.Tensor] = {}

    # Map from graph placeholder target -> model attribute name
    # e.g. {'p_w1': 'W1', 'p_w2': 'W2', 'p_w3': 'W3'}
    inputs_to_params: dict[str, str] = {}
    try:
        inputs_to_params = dict(exported._graph_signature.inputs_to_parameters)
    except Exception:
        pass

    env: dict[str, Any] = {}
    inputs: list[Var] = []
    params: dict[str, Param] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            target = node.target  # e.g. 'p_w1' or 'x'
            model_attr_name = inputs_to_params.get(target)

            if model_attr_name is not None:
                # It's a parameter — get its real name, shape, and original tensor
                tensor = getattr(mod, model_attr_name)
                shape = tuple(int(d) for d in tensor.shape)
                param = Param(name=node.name, typ=TensorType(shape))
                env[node.name] = param
                params[node.name] = param
                source_tensors[node.name] = tensor.clone()
            else:
                # It's a model input (user-provided tensor)
                shape = _infer_shape(example_input)
                var = Var(name=node.name, typ=TensorType(shape))
                env[node.name] = var
                inputs.append(var)

        elif node.op == "get_attr":
            tensor = _resolve_attr(mod, node.target)
            shape = tuple(int(d) for d in tensor.shape)
            param = Param(name=node.name, typ=TensorType(shape))
            env[node.name] = param
            params[node.name] = param

        elif node.op == "call_function":
            op_name = _aten_name(node.target)
            ir_op = _ATEN_TO_IR.get(op_name, op_name)
            args = []
            attrs = {}
            for i, arg_node in enumerate(node.args):
                if isinstance(arg_node, (int, float)) and not isinstance(arg_node, bool):
                    attrs[f"arg{i}"] = arg_node
                elif isinstance(arg_node, str):
                    continue
                else:
                    key = arg_node.name if hasattr(arg_node, "name") else str(arg_node)
                    if key in env:
                        args.append(env[key])
            for k, v in node.kwargs.items():
                if isinstance(v, (int, float)):
                    attrs[k] = v
            if args:
                ir_node = Op.make(ir_op, *args, **attrs)
            else:
                ir_node = Op.make(ir_op, **attrs)
            env[node.name] = ir_node

    # Find root
    root = None
    for node in reversed(graph.nodes):
        if node.op == "output" and node.args:
            arg = node.args[0]
            if hasattr(arg, "name") and arg.name in env:
                root = env[arg.name]
            break
    if root is None and env:
        root = list(env.values())[-1]

    
    return IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    ), source_tensors


def _resolve_attr(module: torch.nn.Module, target: str) -> torch.Tensor:
    obj = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


_IR_TO_TORCH: dict[str, Any] = {
    "matmul": torch.matmul,
    "add": torch.add,
    "mul": torch.mul,
    "sub": torch.sub,
    "neg": torch.neg,
    "silu": torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "gelu": torch.nn.functional.gelu,
    "exp": torch.exp,
    "square": torch.square,
    "sum": lambda x, **kw: x.sum(dim=kw.get("dim", -1)),
    "mean": lambda x, **kw: x.mean(),
    "transpose": lambda x, **kw: x.t() if x.dim() == 2 else x.transpose(-2, -1),
    "reshape": lambda x, **kw: x.reshape(-1),
    "broadcast": lambda x, **kw: x,
    "linear": lambda x, **kw: torch.nn.functional.linear(
        x, kw.get("weight"), kw.get("bias")
    ),
}


class IRModule(torch.nn.Module):
    """A torch module reconstructed from a catopt IR term.

    The IR must contain `inputs` (Var list) and `params` (Param dict).

    Parameters
    ----------
    ir : IR
        The categorical IR to lower.
    param_values : dict[str, torch.Tensor] | None
        Optional mapping from IR param names (e.g. 'p_w1') to actual
        tensor values (e.g. a clone of the original model's W1).
        If not provided, random parameters are generated.
    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._inputs = ir.inputs
        self._param_map: dict[str, torch.nn.Parameter] = {}
        self._param_values = param_values or {}
        # Phase 3b: materialize weight-only subtrees (e.g. W1 @ (W2 @ W3))
        # at construction time, so runtime is a single matmul per fused chain.
        self._root = self._fold_weight_chains(ir.root)
        self._build_params()

    @staticmethod
    def _uses_input(term: Any) -> bool:
        """True if the term mentions any data-dependent leaf (Var/Const input)."""
        if isinstance(term, Var):
            return True
        if isinstance(term, Op):
            return any(IRModule._uses_input(a) for a in term.args)
        return False

    def _fold_weight_chains(self, term: Any) -> Any:
        """Bottom-up: replace weight-only subtrees with a single fused Param.

        A subtree with no Var leaves is parameter-only and can be evaluated
        once at construction time (using _param_values when available).
        Its result is stored as a new fused Param, so the runtime graph
        contains one matmul instead of a chain of them.  Matmul is handled
        with torch.matmul; other ops fall back to eager _eval.
        """
        from catopt.ir import Op as _Op
        from catopt.ir import Param as _Param

        if isinstance(term, _Op):
            folded_args = tuple(self._fold_weight_chains(a) for a in term.args)
            term = _Op.make(term.op, *folded_args, **dict(term.attrs))
            if term.op == "matmul" and not self._uses_input(term):
                left = self._param_values.get(term.args[0].name) \
                    if isinstance(term.args[0], _Param) else None
                right = self._param_values.get(term.args[1].name) \
                    if isinstance(term.args[1], _Param) else None
                if left is not None and right is not None:
                    with torch.no_grad():
                        fused = torch.matmul(left, right)
                    fused_name = f"fused_{len(self._param_map) + len(self._param_values)}"
                    self._param_values[fused_name] = fused.detach().clone()
                    shape = tuple(int(d) for d in fused.shape)
                    from catopt.ir import TensorType
                    return _Param(name=fused_name, typ=TensorType(shape))
        return term

    def _build_params(self) -> None:
        from catopt.ir import Op as _Op
        from catopt.ir import Param as _Param

        param_shapes: dict[str, tuple] = {}

        def collect(t: Any) -> None:
            if isinstance(t, _Param):
                if t.name not in param_shapes and t.typ.size is not None:
                    param_shapes[t.name] = tuple(
                        d if d is not None else 1 for d in t.typ.shape
                    )
            elif isinstance(t, _Op):
                for a in t.args:
                    collect(a)

        collect(self._root)
        for name, shape in param_shapes.items():
            if name in self._param_values:
                # Use the original model's parameter value
                p = torch.nn.Parameter(self._param_values[name].clone())
            else:
                p = torch.nn.Parameter(torch.randn(*shape) * 0.02)
            # Use the IR name (e.g. p_w1) so _eval can find it
            setattr(self, name, p)
            self._param_map[name] = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env: dict[str, torch.Tensor] = {"self": x}
        # Fill env with input placeholders
        for inp in self._inputs:
            if inp.name == "x" or not inp.name.startswith("p_"):
                env[inp.name] = x
        return self._eval(self._root, env, x)

    def _eval(
        self, term: Any, env: dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(term, Var):
            return env.get(term.name, env.get("self", x))
        if isinstance(term, Const):
            return torch.tensor(term.value)
        if isinstance(term, Param):
            pid = self._param_map.get(term.name)
            if pid is not None:
                return pid
            shape = tuple(d if d is not None else 1 for d in term.typ.shape)
            return torch.randn(*shape)
        if isinstance(term, Op):
            fn = _IR_TO_TORCH.get(term.op)
            if fn is None:
                raise ValueError(f"No torch binding for op '{term.op}'")
            args = [self._eval(a, env, x) for a in term.args]
            kwargs = dict(term.attrs) if term.attrs else {}
            return fn(*args, **kwargs)
        raise TypeError(f"Cannot evaluate term: {term}")


def ir_to_torch_module(ir: IR, param_values: dict[str, torch.Tensor] | None = None) -> IRModule:
    """Convert a catopt IR into a torch.nn.Module."""
    return IRModule(ir, param_values=param_values)
