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
    "pow": "pow",
    "sum": "sum",
    "mean": "mean",
    "matmul": "matmul",
    "linear": "linear",
    "transpose": "transpose",
    "reshape": "reshape",
    "view": "reshape",
    "amax": "max",
    "matmul.default": "matmul",
    "contiguous": "contiguous",
    "clone": "contiguous",
    "scaled_dot_product_attention": "sdpa",
}

#: ATen overload-specific names (e.g. 'mul.Tensor') that do not survive
#: naive suffix stripping.
_IR_TO_TORCH_EXTRA: dict[str, str] = {
    "mul.Tensor": "mul",
    "add.Tensor": "add",
    "sub.Tensor": "sub",
    "div.Tensor": "div",
    "pow.Tensor_Scalar": "pow",
    "pow.Tensor_Tensor": "pow",
    "mean.dim": "mean",
    "sum.dim_IntList": "sum",
    "linear.default": "linear",
    "silu.default": "silu",
    "rsqrt.default": "rsqrt",
    "sqrt.default": "sqrt",
    "neg.default": "neg",
    "transpose.int": "transpose",
    "reshape.default": "reshape",
    "view.default": "reshape",
    "clone.default": "contiguous",
    "amax.default": "max",
    "scaled_dot_product_attention.default": "sdpa",
}


#: Ops where a numeric argument is a scalar OPERAND (not an attribute).
_SCALAR_OPERAND_OPS: set[str] = {
    "pow", "mul", "div", "add", "sub", "rsub", "rmul", "rdiv",
    "clamp", "clamp_min", "clamp_max", "leaky_relu",
}


def _strip_backend_suffix(name: str) -> str:
    for suffix in (".default", ".deterministic", ".backend", ".assigned"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _canon_aten_name(name: str) -> str:
    """Map a raw ATen name to a catopt generator, handling overloads."""
    if name in _ATEN_TO_IR:
        return _ATEN_TO_IR[name]
    stripped = _strip_backend_suffix(name)
    if stripped in _ATEN_TO_IR:
        return _ATEN_TO_IR[stripped]
    if name in _IR_TO_TORCH_EXTRA:
        return _IR_TO_TORCH_EXTRA[name]
    return stripped


def _aten_name(target: Any) -> str:
    if hasattr(target, "__name__"):
        return _canon_aten_name(target.__name__)
    if isinstance(target, str):
        return _canon_aten_name(target)
    s = str(target)
    if "aten::" in s:
        name = s.split("aten::")[1].split(".")[0]
        return _canon_aten_name(name)
    return _canon_aten_name(s)


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
    state_dict = exported.state_dict if isinstance(
        exported.state_dict, dict) else dict(exported.state_dict)

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
                # via the exported state_dict (handles dotted names like
                # 'gate.weight'; GraphModule attrs may not itself hold them).
                tensor = state_dict.get(model_attr_name)
                if tensor is None:
                    tensor = _resolve_attr(mod, model_attr_name)
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
                    # Scalar operands of arithmetic ops are real operands
                    # (e.g. pow(x, 2)); for reductions the scalar is a
                    # dim/arg attribute (e.g. x.mean(-1)).
                    if ir_op in _SCALAR_OPERAND_OPS:
                        args.append(Const(float(arg_node)))
                    else:
                        attrs[f"arg{i}"] = arg_node
                elif isinstance(arg_node, str):
                    continue
                elif isinstance(arg_node, (list, tuple)):
                    # e.g. dim=[-1] lists for reductions; for view/reshape
                    # the list is the target SHAPE, not a dim.
                    if ir_op == "reshape":
                        attrs["shape"] = tuple(arg_node)
                    else:
                        attrs["dim"] = tuple(arg_node)
                elif isinstance(arg_node, bool):
                    attrs["keepdim"] = arg_node
                else:
                    key = arg_node.name if hasattr(arg_node, "name") else str(arg_node)
                    if key in env:
                        args.append(env[key])
            for k, v in node.kwargs.items():
                if k == "dim" and isinstance(v, (list, tuple)):
                    attrs[k] = tuple(v)
                elif isinstance(v, (int, float, bool)):
                    attrs[k] = v
                elif isinstance(v, list):
                    attrs[k] = tuple(v)
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
    "div": torch.div,
    "sub": torch.sub,
    "neg": torch.neg,
    "silu": torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "gelu": torch.nn.functional.gelu,
    "exp": torch.exp,
    "pow": torch.pow,
    "square": torch.square,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "sum": lambda x, *a, **kw: x.sum(*_dim_args(a, kw)),
    "mean": lambda x, *a, **kw: x.mean(*_dim_args(a, kw)),
    "transpose": lambda x, *a, **kw: (
        x.t() if x.dim() == 2 and "arg1" not in kw
        else x.transpose(kw.get("arg1", -2), kw.get("arg2", -1))
    ),
    "reshape": lambda x, *a, **kw: x.reshape(
        tuple(kw["shape"]) if "shape" in kw else (-1,)
    ),
    "contiguous": lambda x, *a, **kw: x.contiguous(),
    "sdpa": lambda q, k, v, **kw: torch.nn.functional
        .scaled_dot_product_attention(
            q, k, v,
            **{kk: vv for kk, vv in kw.items() if vv is not None},
        ),
    "broadcast": lambda x, *a, **kw: x,
    "linear": lambda x, w, *a, **kw: torch.nn.functional.linear(
        x, w, (a[0] if a else kw.get("bias"))
    ),
    "concat": lambda a, b, dim=0, **kw: torch.cat((a, b), dim=dim),
    "chunk": lambda t, chunks=2, dim=-1, index=0, **kw: torch.chunk(
        t, chunks, dim=dim
    )[index],
    "split": lambda t, sizes=(), dim=-1, index=0, **kw: torch.split(
        t, list(sizes), dim=dim
    )[index],
}


def _dim_args(args: tuple, kwargs: dict) -> tuple:
    """Extract a (dim, keepdim) argument tuple from IR attrs/args."""
    if args:
        return tuple(args)
    dim = kwargs.get("dim", kwargs.get("axis", -1))
    if isinstance(dim, (list, tuple)):
        dim = tuple(int(d) for d in dim)
    keep = kwargs.get("keepdim", False)
    if dim is None:
        return ()
    return (dim, bool(keep))


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
        self._fold_memo: dict[int, Any] = {}
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
        with torch.matmul; elementwise weight-only chains (add/mul/neg/
        silu/sigmoid/square) are folded eagerly via _eval on _param_values.
        """
        from catopt.ir import Op as _Op
        from catopt.ir import Param as _Param

        # Memoize by identity of the CALLER's term object: the extracted
        # term may share a subtree object between several parents (e.g.
        # one fused GEMM under two chunk projections).  Returning the
        # same folded object preserves that sharing for _eval.  Must
        # capture id() before `term` is rebound to the rebuilt node.
        orig_id = id(term)
        memo_hit = self._fold_memo.get(orig_id)
        if memo_hit is not None:
            return memo_hit

        if isinstance(term, _Op):
            folded_args = tuple(self._fold_weight_chains(a) for a in term.args)
            term = _Op.make(term.op, *folded_args, **dict(term.attrs))
            if not self._uses_input(term):
                if term.op == "matmul":
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
                        result = _Param(name=fused_name, typ=TensorType(shape))
                        self._fold_memo[orig_id] = result
                        return result
                elif term.op in (
                    "add", "mul", "sub", "div", "neg", "square", "sqrt",
                    "sigmoid", "silu", "tanh", "gelu", "exp", "pow",
                    "concat",
                ):
                    # Try eager compile-time fold via _IR_TO_TORCH bindings
                    vals: list[torch.Tensor] = []
                    ok = True
                    for a in term.args:
                        if isinstance(a, _Param) and a.name in self._param_values:
                            vals.append(self._param_values[a.name])
                        elif isinstance(a, Const):
                            vals.append(torch.tensor(a.value))
                        else:
                            ok = False
                            break
                    if ok:
                        try:
                            fn = _IR_TO_TORCH[term.op]
                            with torch.no_grad():
                                fused = fn(*vals, **dict(term.attrs))
                            if isinstance(fused, torch.Tensor):
                                fused_name = (
                                    f"fused_{len(self._param_map) + len(self._param_values)}"
                                )
                                self._param_values[fused_name] = fused.detach().clone()
                                shape = tuple(int(d) for d in fused.shape)
                                from catopt.ir import TensorType
                                result = _Param(name=fused_name, typ=TensorType(shape))
                                self._fold_memo[orig_id] = result
                                return result
                        except Exception:
                            pass
        self._fold_memo[orig_id] = term
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
        return self._eval(self._root, env, x, {})

    def _eval(
        self, term: Any, env: dict[str, torch.Tensor], x: torch.Tensor,
        memo: dict[int, torch.Tensor],
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
            # Shared subtrees (e.g. one fused GEMM read by two chunk
            # projections) are the SAME object — compute once.
            hit = memo.get(id(term))
            if hit is not None:
                return hit
            fn = _IR_TO_TORCH.get(term.op)
            if fn is None:
                raise ValueError(f"No torch binding for op '{term.op}'")
            args = [self._eval(a, env, x, memo) for a in term.args]
            kwargs = dict(term.attrs) if term.attrs else {}
            result = fn(*args, **kwargs)
            memo[id(term)] = result
            return result
        raise TypeError(f"Cannot evaluate term: {term}")


def ir_to_torch_module(ir: IR, param_values: dict[str, torch.Tensor] | None = None) -> IRModule:
    """Convert a catopt IR into a torch.nn.Module."""
    return IRModule(ir, param_values=param_values)
