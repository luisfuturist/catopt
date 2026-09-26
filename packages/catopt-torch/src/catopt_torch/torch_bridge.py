# ruff: noqa: RUF002, RUF003
"""Fixed torch_bridge.py — uses exported._graph_signature.inputs_to_parameters
to correctly map graph placeholder targets (p_w1, p_w2, ...) to actual
model parameter names (W1, W2, ...) and retrieve their shapes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch
from catopt_core.attrs import ATTR_SCHEMA, attr_of
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.ops import (
    OpTable,
    register_ambient_bindings,
    register_core_bindings,
)

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
    "amin": "min",
    "matmul.default": "matmul",
    "contiguous": "contiguous",
    "clone": "contiguous",
    "scaled_dot_product_attention": "sdpa",
    "conv2d": "conv2d",
    "cat": "concat",
    # List-form split exports as aten.split_with_sizes; unsafe_*
    # variants are the functionalization-internal spellings of the
    # same op.  All lower through the "split" binding.
    "split_with_sizes": "split",
    "unsafe_split": "split",
    "unsafe_split_with_sizes": "split",
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
    "amin.default": "min",
    "scaled_dot_product_attention.default": "sdpa",
    "conv2d.default": "conv2d",
}

#: Positional-argument → named-attribute mapping lives in
#: ``catopt_core.attrs.ATTR_SCHEMA`` (plan 0001 phase 1b): for every
#: schema'd op, non-node args at declared positions land under the
#: canonical attr name directly — ``cat(ts, -2)`` becomes
#: ``concat(dim=-2)``, ``F.sdpa(..., is_causal=True)`` becomes
#: ``is_causal=True``, never ``argN``.  Ops *not* in the schema keep
#: the legacy ``arg{i}`` spelling so undeclared positionals fail
#: loudly at ``Op.make`` validation instead of being silently
#: accepted under a guessed name.  (What used to be ``_ATTR_RENAMES``
#: for concat/chunk/split/rms_norm is fully covered by the schema.)


#: Ops where a numeric argument is a scalar OPERAND (not an attribute).
_SCALAR_OPERAND_OPS: set[str] = {
    "pow",
    "mul",
    "div",
    "add",
    "sub",
    "rsub",
    "rmul",
    "rdiv",
    "clamp",
    "clamp_min",
    "clamp_max",
    "leaky_relu",
    # masked_fill(x, mask, -inf) / eq(x, 0) — the scalar is a Const
    # operand so rewrites can bind and check it.
    "masked_fill",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}


def _strip_backend_suffix(name: str) -> str:
    for suffix in (
        ".default",
        ".deterministic",
        ".backend",
        ".assigned",
    ):
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
    # Generic overload stripping: 'flatten.using_ints' → 'flatten',
    # 'to.dtype' → 'to'.  Try the base packet name against both maps.
    base = name.split(".")[0]
    if base in _ATEN_TO_IR:
        return _ATEN_TO_IR[base]
    if base in _IR_TO_TORCH_EXTRA:  # pragma: no cover — every EXTRA key contains a dot
        return _IR_TO_TORCH_EXTRA[base]
    return base if base in _IR_TO_TORCH else stripped


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
            int(d) if d is not None else None
            for d in node_or_value.shape
        )
    return (None,)


def export_to_ir(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple,
) -> tuple[IR, dict[str, torch.Tensor]]:
    """Export a PyTorch model to a catopt IR via torch.export.

    ``example_input`` may be a single tensor or a tuple of positional
    args for multi-input modules.

    Returns ``(ir, source_tensors)`` — the IR plus the concrete
    parameter/buffer values keyed by IR param name (e.g. ``p_w1``),
    used to materialise the lowered module and to compare exact
    weights in the non-local passes.

    Uses exported._graph_signature.inputs_to_parameters to map graph
    placeholder node targets (e.g. 'p_w1') to actual model attribute
    names (e.g. 'W1'), so we can retrieve parameter shapes correctly.
    """
    model.eval()
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    with torch.no_grad():
        exported = torch.export.export(model, args)

    graph = exported.graph
    mod = exported.module()  # the actual Module instance
    state_dict = (
        exported.state_dict
        if isinstance(exported.state_dict, dict)
        else dict(exported.state_dict)
    )

    # Dict to collect original parameter tensors for IRModule
    source_tensors: dict[str, torch.Tensor] = {}

    # Map from graph placeholder target -> model attribute name
    # e.g. {'p_w1': 'W1', 'p_w2': 'W2', 'p_w3': 'W3'}
    inputs_to_params: dict[str, str] = {}
    with contextlib.suppress(Exception):
        inputs_to_params = dict(
            exported._graph_signature.inputs_to_parameters
        )
    # Buffers (e.g. nanoGPT's causal-mask `bias`) are placeholders too —
    # they are constants, not user inputs: lift them like parameters.
    with contextlib.suppress(Exception):
        inputs_to_params.update(
            exported._graph_signature.inputs_to_buffers
        )

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
                # It's a model input (user-provided tensor).  Prefer the
                # exported node's own metadata shape — correct for
                # multi-input graphs where placeholders differ.
                meta_shape = getattr(
                    getattr(node, "meta", {}), "get", lambda k: None
                )("val")
                if meta_shape is not None and hasattr(
                    meta_shape, "shape"
                ):
                    shape = tuple(int(d) for d in meta_shape.shape)
                elif len(inputs) < len(args):
                    shape = _infer_shape(args[len(inputs)])
                else:
                    shape = _infer_shape(args[0])
                var = Var(name=node.name, typ=TensorType(shape))
                env[node.name] = var
                inputs.append(var)

        elif node.op == "get_attr":  # pragma: no cover — torch 2.14 export lifts everything to placeholders
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
            positional_attrs = ATTR_SCHEMA.get(ir_op, {})
            for i, arg_node in enumerate(node.args):
                if (
                    i in positional_attrs
                    and arg_node is not None
                    and not hasattr(arg_node, "name")
                ):
                    # e.g. conv2d(x, w, b, [s,s], [p,p], [d,d], g) —
                    # non-node positionals at schema-declared positions
                    # are named op attributes.  ``None`` positionals
                    # (an absent sdpa attn_mask) are skipped — a
                    # present-but-None attr would poison e-matching's
                    # exact attr-set comparison.
                    v = arg_node
                    attrs[positional_attrs[i]] = (
                        tuple(v) if isinstance(v, (list, tuple)) else v
                    )
                    continue
                if isinstance(
                    arg_node, (int, float)
                ) and not isinstance(arg_node, bool):
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
                    if any(hasattr(a, "name") for a in arg_node):
                        # A list of FX nodes (stack/cat take tensor lists)
                        # — each element is an operand.
                        for a in arg_node:
                            key = (
                                a.name if hasattr(a, "name") else str(a)
                            )
                            if key in env:
                                args.append(env[key])
                    # e.g. dim=[-1] lists for reductions; for view/reshape
                    # the list is the target SHAPE, not a dim.
                    elif ir_op == "reshape" or ir_op in (
                        "expand",
                        "repeat",
                    ):
                        attrs["shape"] = tuple(arg_node)
                    elif ir_op in ("split", "chunk"):
                        attrs["sizes"] = tuple(arg_node)  # pragma: no cover — schema covers all real positions
                    else:
                        attrs["dim"] = tuple(arg_node)
                elif isinstance(arg_node, bool):
                    if ir_op in (
                        "mean",
                        "sum",
                        "max",
                        "min",
                        "prod",
                        "var",
                        "std",
                        "amax",
                        "amin",
                    ):
                        attrs["keepdim"] = arg_node
                    else:
                        attrs[f"arg{i}"] = arg_node
                else:
                    key = (
                        arg_node.name
                        if hasattr(arg_node, "name")
                        else str(arg_node)
                    )
                    if key in env:
                        args.append(env[key])
            for k, v in node.kwargs.items():  # pragma: no cover — export normalises kwargs
                if k == "dim" and isinstance(v, (list, tuple)):
                    attrs[k] = tuple(v)
                elif isinstance(v, (int, float, bool)):
                    attrs[k] = v
                elif isinstance(v, list):
                    attrs[k] = tuple(v)
            # Canonical attr spellings are guaranteed below by
            # ``Op.make`` (ATTR_SCHEMA): positional attrs were named at
            # emission above; any ``argN`` still present for a schema'd
            # op fails loudly at mint.
            if (
                ir_op == "getitem"
                and args
                and isinstance(args[0], Op)
                and args[0].op in ("split", "chunk", "unbind")
                and isinstance(node.args[1], int)
            ):
                # getitem(split(t), i) — fold the index into the
                # splitter so the op yields element i directly.
                base = args[0]
                ir_node = Op.make(
                    base.op,
                    *base.args,
                    **{**dict(base.attrs), "index": node.args[1]},
                )
            elif args:
                ir_node = Op.make(ir_op, *args, **attrs)
            else:
                ir_node = Op.make(ir_op, **attrs)
            env[node.name] = ir_node

    # Find root
    root = None
    for node in reversed(graph.nodes):  # pragma: no branch — output node is always last, first when reversed
        if node.op == "output" and node.args:  # pragma: no branch — same invariant
            arg = node.args[0]
            if hasattr(arg, "name") and arg.name in env:  # pragma: no branch — tuples don't carry .name
                root = env[arg.name]  # pragma: no cover — tuples don't carry .name
            break  # pragma: no cover — same
    if root is None and env:  # pragma: no branch — env is never empty when root missed
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


def _expand_torch(t: Any, *a: Any, **kw: Any) -> Any:
    """``t.expand`` binding: the target shape arrives as ``shape`` /
    ``dim`` attrs or positional args — a size sequence OR a bare int.
    Splat sequences; pass a scalar through (``t.expand(*4)`` would be a
    ``TypeError``, ``t.expand(4)``/``t.expand(-1)`` is legal)."""
    dim_v = kw.get("shape") or kw.get("dim") or a
    if isinstance(dim_v, (tuple, list)):
        return t.expand(*dim_v)
    return t.expand(dim_v)


#: Core torch lowering bindings — the base ``OpTable``'s table (plan
#: 0001 phase 2c).  Carrier ops (``trace``/``omd_*``/``cmask``/
#: ``aquant``...) are deliberately ABSENT: they live in each carrier
#: module's ``TORCH_BINDINGS`` export and compose via
#: :class:`catopt_core.ops.OpTable`, not by mutating this dict at import.
#: ``_IR_TO_TORCH`` below is the ambient view over this table.
_CORE_TORCH_BINDINGS: dict[str, Any] = {
    "matmul": torch.matmul,
    "add": torch.add,
    "mul": torch.mul,
    "div": torch.div,
    "sub": torch.sub,
    "neg": torch.neg,
    "silu": torch.nn.functional.silu,
    "relu": torch.nn.functional.relu,
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
    "max": lambda x, *a, **kw: x.amax(*_dim_args(a, kw)),
    "min": lambda x, *a, **kw: x.amin(*_dim_args(a, kw)),
    # Canonical names per ATTR_SCHEMA are listed first in each get;
    # the argN spellings remain live fallbacks — rule variants and
    # hand-minted terms legitimately carry them (Op.make preserves
    # declared-position argN; only the export boundary is canonical).
    "transpose": lambda x, *a, **kw: (
        x.t()
        if x.dim() == 2 and "arg1" not in kw and "dim0" not in kw
        else x.transpose(
            attr_of(kw, "dim0", "arg1", default=-2),
            attr_of(kw, "dim1", "arg2", default=-1),
        )
    ),
    "reshape": lambda x, *a, **kw: x.reshape(
        tuple(kw["shape"]) if "shape" in kw else (-1,)
    ),
    "contiguous": lambda x, *a, **kw: x.contiguous(),
    "sdpa": lambda q, k, v, *a, **kw: (
        torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            **({"attn_mask": a[0]} if a else {}),
            **{
                (
                    "attn_mask"
                    if kk == "arg3"
                    else "dropout_p"
                    if kk == "arg4"
                    else "is_causal"
                    if kk == "arg5"
                    else "scale"
                    if kk == "arg6"
                    else "enable_gqa"
                    if kk == "arg7"
                    else kk
                ): vv
                for kk, vv in kw.items()
                if vv is not None
            },
        )
    ),
    "broadcast": lambda x, *a, **kw: x,
    "linear": lambda x, w, *a, **kw: torch.nn.functional.linear(
        x, w, (a[0] if a else kw.get("bias"))
    ),
    "conv2d": lambda x, w, *a, **kw: torch.nn.functional.conv2d(
        x,
        w,
        a[0] if a else kw.get("bias"),
        stride=kw.get("stride", 1),
        padding=kw.get("padding", 0),
        dilation=kw.get("dilation", 1),
        groups=kw.get("groups", 1),
    ),
    "concat": lambda *ts, **kw: torch.cat(
        list(ts), dim=int(attr_of(kw, "dim", "arg1", default=0))
    ),
    "chunk": lambda t, chunks=2, dim=-1, index=0, **kw: torch.chunk(
        t,
        int(attr_of(kw, "chunks", "arg1", default=chunks)),
        dim=int(attr_of(kw, "dim", "arg2", default=dim)),
    )[int(attr_of(kw, "index", "arg3", default=index))],
    "split": lambda t, sizes=(), dim=-1, index=0, **kw: torch.split(
        t,
        _split_sizes(sizes, kw),
        dim=int(attr_of(kw, "dim", "arg2", default=dim)),
    )[int(attr_of(kw, "index", "arg3", default=index))],
    # torch.export emits aten.dropout with train=False in eval mode —
    # the op is a semantic identity there.  This binding is only valid
    # because export_to_ir always exports eval()-mode graphs.
    "dropout": lambda x, *a, **kw: x,
    # dtype casts are identity at the precision we verify (float32)
    "to": lambda x, *a, **kw: x,
    "clone": lambda x, *a, **kw: x.clone(),
    "getitem": lambda t, **kw: t[attr_of(kw, "index", "arg1", default=0)],
    "unbind": lambda t, *a, **kw: torch.unbind(
        t, dim=int(attr_of(kw, "dim", "arg1", default=0))
    )[int(attr_of(kw, "index", "arg2", "arg3", default=0))],
    "stack": lambda *ts, **kw: torch.stack(
        list(ts), dim=int(attr_of(kw, "dim", "arg1", default=0))
    ),
    "expand": _expand_torch,
    "flatten": lambda x, *a, **kw: x.flatten(
        int(attr_of(kw, "start_dim", "arg1", default=0)),
        int(attr_of(kw, "end_dim", "arg2", default=-1)),
    ),
    "slice": lambda t, *a, **kw: t[
        (slice(None),) * int(attr_of(kw, "dim", "arg1", default=0))
        + (slice(kw.get("arg2"), kw.get("arg3"), kw.get("arg4")),)
    ],
    "unsqueeze": lambda t, *a, **kw: t.unsqueeze(
        int(attr_of(kw, "dim", "arg1", default=-1))
    ),
    "squeeze": lambda t, *a, **kw: t.squeeze(
        int(attr_of(kw, "dim", "arg1", default=-1))
    ),
    "select": lambda t, *a, **kw: t.select(
        int(attr_of(kw, "dim", "arg1", default=0)),
        int(attr_of(kw, "index", "arg2", default=0)),
    ),
    # embedding(W, idx) — row gather; factorised form gathers the small
    # factor then projects (eps.low_rank_gather).
    "embedding": lambda w, idx, *a, **kw: torch.nn.functional.embedding(
        idx, w
    ),
    # gather along one axis.  The index is normally an attr (a tuple of
    # ints produced by share_duplicate_param_slices); a second tensor
    # operand (the raw aten spelling) is accepted too.
    "index_select": lambda t, *a, **kw: torch.index_select(
        t,
        int(attr_of(kw, "dim", "arg1", default=0)),
        (
            a[0]
            if a and torch.is_tensor(a[0])
            else torch.as_tensor(
                [
                    int(v)
                    for v in attr_of(
                        kw, "index", "arg2", default=a[0] if a else ()
                    )
                ],
                dtype=torch.long,
                device=t.device,
            )
        ),
    ),
    "type_as": lambda x, t, *a, **kw: x.type_as(t),
    "cos": torch.cos,
    "sin": torch.sin,
    "float": lambda x, *a, **kw: x.to(
        getattr(torch, str(kw.get("dtype", "float32")))
        if isinstance(kw.get("dtype"), str)
        else torch.float32
    ),
    "alias": lambda x, *a, **kw: x,
    "softmax": lambda x, *a, **kw: torch.nn.functional.softmax(
        x, dim=int(attr_of(kw, "dim", "arg1", default=-1))
    ),
    # aten.rms_norm(x, normalized_shape, weight, eps) — canonical
    # attrs dim=normalized_shape, eps=eps (Llama-family normalization);
    # normalized_shape/arg3 are the legacy minted spellings.
    "rms_norm": lambda x, w=None, *a, **kw: (
        torch.nn.functional.rms_norm(
            x,
            tuple(attr_of(kw, "dim", "normalized_shape")),
            weight=w,
            eps=float(attr_of(kw, "eps", "arg3", default=1e-6)),
        )
    ),
    "layer_norm": lambda x, w=None, b=None, *a, **kw: (
        torch.nn.functional.layer_norm(
            x,
            tuple(attr_of(kw, "dim", "normalized_shape")),
            weight=w,
            bias=b,
            # eps is arg4 canonically — NOT arg5 (the cudnn flag):
            # reading arg5 silently produced eps=0.0 before.
            eps=float(attr_of(kw, "eps", "arg4", default=1e-5)),
        )
    ),
    "masked_fill": lambda x, m, v, *a, **kw: x.masked_fill(m, v),
    "eq": lambda x, y, *a, **kw: x == y,
    "ne": lambda x, y, *a, **kw: x != y,
    "lt": lambda x, y, *a, **kw: x < y,
    "le": lambda x, y, *a, **kw: x <= y,
    "gt": lambda x, y, *a, **kw: x > y,
    "ge": lambda x, y, *a, **kw: x >= y,
    "logical_not": lambda x, *a, **kw: torch.logical_not(x),
    "where": lambda c, x, y, *a, **kw: torch.where(c, x, y),
    # Affine-map monoid (scan domain).  An aff is a pair (A, b)
    # denoting h ↦ A@h + b; compose/apply pass pairs around — only
    # ``apply`` returns a tensor.
    "aff": lambda A, b, *a, **kw: (A, b),
    "aff_compose": lambda f, g, *a, **kw: (
        f[0] @ g[0],
        f[0] @ g[1] + f[1],
    ),
    "apply": lambda f, h, *a, **kw: f[0] @ h + f[1],
    # Online-softmax monoid (FlashAttention's combine as a monoid law —
    # see catopt/om.py).  An ``om`` value is a triple (m, l, a):
    # running row-max, running exp-sum denominator, running weighted
    # numerator.  Only ``om_apply`` returns a tensor — a / l, UNCLAMPED,
    # so a fully-masked row yields NaN exactly like dense softmax.
    "om": lambda m, lv, a, *x, **kw: (m, lv, a),
    "om_elem": lambda s, v, *a, **kw: _om_elem(s, v),
    "om_compose": lambda f, g, *a, **kw: _om_compose(f, g),
    "om_apply": lambda f, *a, **kw: f[2] / f[1],
    # Diagonal-affine monoid (scan domain for elementwise SSMs — the
    # Mamba-faithful ``h ↦ a⊙h + x`` step).  An aff_diag is a pair
    # (a, b) of same-shaped tensors; compose/apply are elementwise —
    # O(d) work per node instead of the dense carrier's d×d products.
    "aff_diag": lambda a, b, *x, **kw: (a, b),
    "affd_compose": lambda f, g, *a, **kw: (
        f[0] * g[0],
        f[0] * g[1] + f[1],
    ),
    "applyd": lambda f, h, *a, **kw: f[0] * h + f[1],
}


class _AmbientTorchBindings(dict):
    """The ambient lowering table — legacy ``_IR_TO_TORCH`` semantics.

    Seeded eagerly with ``_CORE_TORCH_BINDINGS``; carrier bindings
    (each module's ``TORCH_BINDINGS`` export) resolve ON DEMAND through
    :func:`catopt_core.ops.carrier_torch_bindings` and are cached back into
    the dict, so ``_IR_TO_TORCH["omd_elem"]`` works whether or not a
    carrier module was imported — and imports register nothing.

    Still deliberately mutable: ``_IR_TO_TORCH[op] = fn`` overrides the
    binding for every consumer reading the ambient table (including
    ``IRModule`` instances built earlier — the pre-2c dispatch
    semantics).  An explicit :class:`catopt_core.ops.OpTable` owns a private
    dict and is NOT routed through here.
    """

    def _resolve(self, key: str) -> Any:
        """Pull ``key``'s binding from the carrier modules' exports,
        caching the hit back into the dict.  ``None`` when unknown."""
        from catopt_core.ops import carrier_torch_bindings

        fn = carrier_torch_bindings().get(key)
        if fn is not None:
            dict.__setitem__(self, key, fn)
        return fn

    def __missing__(self, key: str) -> Any:
        fn = self._resolve(key)
        if fn is None:
            raise KeyError(key)
        return fn

    def get(self, key: str, default: Any = None) -> Any:
        # ``dict.get`` never consults ``__missing__`` — route misses
        # through the carrier resolver so ``_IR_TO_TORCH.get`` sees the
        # same table ``[]`` does.
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True


#: Ambient lowering table — see :class:`_AmbientTorchBindings`.  New
#: code should take an explicit :class:`catopt_core.ops.OpTable` (the
#: ``ops`` parameter on ``IRModule`` / ``optimize_model``); this dict
#: remains for back-compat readers and post-hoc binding overrides.
_IR_TO_TORCH: _AmbientTorchBindings = _AmbientTorchBindings(
    _CORE_TORCH_BINDINGS
)

# -- push the torch tables into core (core imports no adapter) --------
# catopt-core owns ``OpTable.core()`` / ``OpTable.full()`` but never
# names this module; the wiring is registered here, at adapter import
# time.  ``_CORE_TORCH_BINDINGS`` becomes the base table ``core()``
# folds in; ``_IR_TO_TORCH`` is the ambient dict ``full()`` seats its
# bindings on (so post-hoc ``_IR_TO_TORCH[op] = fn`` overrides and lazy
# carrier resolution keep reaching already-built evaluators).
register_core_bindings(_CORE_TORCH_BINDINGS)
register_ambient_bindings(_IR_TO_TORCH)


def _om_elem(s: torch.Tensor, v: torch.Tensor):
    """elem(s, v) = (rowmax s, Σ exp(s−m), exp(s−m) @ v) — the
    online-softmax monoid element for one key block."""
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    return (m, e.sum(dim=-1, keepdim=True), e @ v)


def _om_compose(f, g):
    """(m1,l1,a1) ⊕ (m2,l2,a2): the FlashAttention combine.

    The ``isfinite`` guard covers the −inf − −inf case: a fully-masked
    block has m = −inf (and NaN l/a from elem's exp(s−m)); the where
    contributes 0 for it instead of exp(NaN)·NaN, so a masked block
    contaminates nothing — while a fully-masked ROW still produces
    l = 0 and a = 0, and om_apply's 0/0 = NaN preserves dense softmax's
    NaN semantics.
    """
    m1, l1, a1 = f
    m2, l2, a2 = g
    mx = torch.maximum(m1, m2)
    fin1, fin2 = torch.isfinite(m1), torch.isfinite(m2)
    e1 = torch.where(fin1, torch.exp(m1 - mx), torch.zeros_like(mx))
    e2 = torch.where(fin2, torch.exp(m2 - mx), torch.zeros_like(mx))
    lv = torch.where(
        fin1, l1 * e1, torch.zeros_like(l1)
    ) + torch.where(fin2, l2 * e2, torch.zeros_like(l2))
    a = torch.where(fin1, a1 * e1, torch.zeros_like(a1)) + torch.where(
        fin2, a2 * e2, torch.zeros_like(a2)
    )
    return (mx, lv, a)


def _split_sizes(sizes: Any, kw: dict):
    """split(x, sizes_list) and split(x, int) both land under the
    canonical ``sizes`` attr at the boundary; minted terms may still
    carry the positional ``arg1`` spelling."""
    sz = kw.get("sizes", sizes)
    if isinstance(sz, (list, tuple)) and sz:
        return list(sz)
    return int(kw.get("arg1", sz if isinstance(sz, int) else 1))


def _dim_args(args: tuple, kwargs: dict) -> tuple:
    """Extract a (dim, keepdim) argument tuple from IR attrs/args."""
    if args:
        return tuple(args)
    dim = attr_of(kwargs, "dim", "axis", default=-1)
    if isinstance(dim, (list, tuple)):
        dim = tuple(int(d) for d in dim)
    keep = kwargs.get("keepdim", False)
    if dim is None:
        return ()
    return (dim, bool(keep))


#: Sentinel for "no default supplied" — distinct from ``None`` so a
#: legitimate ``None`` default (e.g. a zero-arg forward's ``x``) still
#: resolves instead of reading as failure.
_MISS: Any = object()


def _randn_param(term: Param) -> torch.Tensor:
    """``IRModule._eval``'s unregistered-Param fallback: a fresh randn
    of the declared shape (``None`` dims materialise as extent 1)."""
    shape = tuple(d if d is not None else 1 for d in term.typ.shape)
    return torch.randn(*shape)


def eval_term(
    term: Any,
    *,
    var_env: dict | None = None,
    param_env: dict | None = None,
    bindings: Any = None,
    memo: dict | None = None,
    strict: bool = False,
    var_default: Any = _MISS,
    param_default: Callable[[Param], Any] | None = None,
    tensor_only: bool = False,
) -> Any:
    """Evaluate an IR term against concrete environments.

    Single source for the eval-term family (plan 0002 phase D) —
    :meth:`IRModule._eval` (strict runtime eval), the permissive
    compile-time folds ``optimize._eval_const`` /
    ``ibp._eval_concrete``, and ``_fold_weight_chains``' shallow fold
    all delegate here.  Leaf semantics:

    * ``Var``   → ``var_env[name]``; on a miss, ``var_default`` when
      the caller supplied one (``IRModule`` passes the forward's first
      input — the single-input "self" fallback), else failure.
    * ``Param`` → ``param_env[name]``; on a miss,
      ``param_default(term)`` when given (``IRModule`` passes
      :func:`_randn_param` — the unshaped-Param fallback), else
      failure.
    * ``Const`` → ``torch.tensor(value)``.
    * ``Op``    → ``bindings[op](*args, **attrs)``; ``memo`` (when
      given) dedups DAG-shared subtrees — interned terms are content
      keys.

    ``strict=True`` is the runtime contract: a missing binding raises
    ``ValueError``, binding errors propagate with grad enabled, and an
    unknown term type raises ``TypeError``.  ``strict=False`` is the
    fold contract: any un-evaluatable piece — missing leaf, missing
    binding, raising binding — yields ``None`` instead, and op calls
    run under ``torch.no_grad()``.  ``tensor_only`` (compile-time
    folds) additionally treats a non-Tensor op result (e.g. an ``aff``
    pair) as failure AT EVERY level — mirroring the recursive
    isinstance check of the fold sites it replaced, not a top-level
    filter.  The two env-miss raises under ``strict`` are nominal —
    every strict caller passes both defaults.
    """

    def go(t: Any, rec: Callable[..., Any]) -> Any:
        # Recursion goes through the explicit ``rec`` parameter, not
        # the closure: a self-referencing ``go`` would sit in its own
        # ``__closure__``, leaving a reference cycle that keeps this
        # call's memo/env cells (every intermediate tensor) alive
        # until cyclic GC — an OOM source inside CUDA timing windows.

        if isinstance(t, Var):
            v = (var_env or {}).get(t.name, var_default)
            if v is _MISS:
                if strict:  # pragma: no cover — strict callers pass var_default
                    raise KeyError(t.name)
                return None
            return v
        if isinstance(t, Const):
            return torch.tensor(t.value)
        if isinstance(t, Param):
            v = (param_env or {}).get(t.name)
            if v is None:
                if param_default is not None:
                    return param_default(t)
                if strict:  # pragma: no cover — strict callers pass param_default
                    raise KeyError(t.name)
                return None
            return v
        if isinstance(t, Op):
            if memo is not None:
                hit = memo.get(t)
                if hit is not None:
                    return hit
            fn = (bindings or {}).get(t.op)
            if fn is None:
                if strict:
                    raise ValueError(f"No torch binding for op '{t.op}'")
                return None
            args = [rec(a, rec) for a in t.args]
            if not strict and any(a is None for a in args):
                return None
            if strict:
                out = fn(*args, **dict(t.attrs))
            else:
                try:
                    with torch.no_grad():
                        out = fn(*args, **dict(t.attrs))
                except Exception:
                    return None
            if tensor_only and not isinstance(out, torch.Tensor):
                if strict:  # pragma: no cover — no strict caller sets tensor_only
                    raise TypeError(
                        f"eval_term: op '{t.op}' returned "
                        f"{type(out).__name__}, not a tensor"
                    )
                return None
            if memo is not None:
                memo[t] = out
            return out
        if strict:
            raise TypeError(f"Cannot evaluate term: {t}")
        return None

    return go(term, go)


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
    ops : OpTable | None
        The op table to lower through (plan 0001 phase 2c).  ``None``
        (default) resolves to ``OpTable.full()`` — the ambient table
        sharing ``_IR_TO_TORCH``, so post-construction binding
        overrides keep reaching ``_eval``.  An explicit table owns its
        own dict: an op absent from it fails loudly at eval.

    Ports layer: an ``IRModule`` is a minimal
    :class:`catopt_core.ports.Executor` — ``forward(*xs) -> Tensor`` — and
    is itself the ``eval_mod`` the planned wrappers embed (see
    :class:`catopt_core.ports.PlannedExecutor`); ``ops`` conforms to
    :class:`catopt_core.ports.OpRegistry`.
    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
        ops: OpTable | None = None,
    ) -> None:
        super().__init__()
        self._ops = ops if ops is not None else OpTable.full()
        self._torch_bindings = self._ops.torch_bindings
        self._inputs = ir.inputs
        self._param_map: dict[str, torch.nn.Parameter] = {}
        self._param_values = param_values or {}
        # Phase 3b: materialize weight-only subtrees (e.g. W1 @ (W2 @ W3))
        # at construction time, so runtime is a single matmul per fused chain.
        # Keyed by interned term OBJECTS (content-hashed), not ints.
        self._fold_memo: dict[Any, Any] = {}
        self._uses_memo: dict = {}
        self._root = self._fold_weight_chains(ir.root)
        self._build_params()

    def _uses_input(self, term: Any) -> bool:
        """True if the term mentions any data-dependent leaf (Var input).

        Delegates to :func:`catopt_core.typing.has_var_leaf` with the
        instance's content-keyed memo: extracted terms are
        shared-subterm DAGs, and an unmemoised walk is exponential in
        DAG depth.
        """
        from catopt_core.typing import has_var_leaf

        return has_var_leaf(term, self._uses_memo)

    def _fold_weight_chains(self, term: Any) -> Any:
        """Bottom-up: replace weight-only subtrees with a single fused Param.

        A subtree with no Var leaves is parameter-only and can be evaluated
        once at construction time (using _param_values when available).
        Its result is stored as a new fused Param, so the runtime graph
        contains one matmul instead of a chain of them.  Matmul is handled
        with torch.matmul; elementwise weight-only chains (add/mul/neg/
        silu/sigmoid/square) are folded eagerly via _eval on _param_values.
        """
        from catopt_core.ir import Op as _Op
        from catopt_core.ir import Param as _Param

        # Memoize by identity of the CALLER's term object: the extracted
        # term may share a subtree object between several parents (e.g.
        # one fused GEMM under two chunk projections).  Returning the
        # same folded object preserves that sharing for _eval.  Must
        # capture the key before `term` is rebound to the rebuilt node.
        orig_key = term
        memo_hit = self._fold_memo.get(orig_key)
        if memo_hit is not None:
            return memo_hit

        if isinstance(term, _Op):
            folded_args = tuple(
                self._fold_weight_chains(a) for a in term.args
            )
            term = _Op.make(term.op, *folded_args, **dict(term.attrs))
            if not self._uses_input(term):
                if term.op == "matmul":
                    left = (
                        self._param_values.get(term.args[0].name)
                        if isinstance(term.args[0], _Param)
                        else None
                    )
                    right = (
                        self._param_values.get(term.args[1].name)
                        if isinstance(term.args[1], _Param)
                        else None
                    )
                    if left is not None and right is not None:
                        with torch.no_grad():
                            fused = torch.matmul(left, right)
                        fused_name = f"fused_{len(self._param_map) + len(self._param_values)}"
                        self._param_values[fused_name] = (
                            fused.detach().clone()
                        )
                        shape = tuple(int(d) for d in fused.shape)
                        from catopt_core.ir import TensorType

                        result = _Param(
                            name=fused_name, typ=TensorType(shape)
                        )
                        self._fold_memo[orig_key] = result
                        return result
                elif term.op in (
                    "add",
                    "mul",
                    "sub",
                    "div",
                    "neg",
                    "square",
                    "sqrt",
                    "sigmoid",
                    "silu",
                    "tanh",
                    "gelu",
                    "exp",
                    "pow",
                    "concat",
                ):
                    # Try eager compile-time fold via the op table's
                    # torch bindings (ambient _IR_TO_TORCH for the
                    # default full table; the custom table's own dict
                    # for an explicit ``ops``).  SHALLOW by design:
                    # eval_term only fires when every arg already
                    # resolved to a leaf — a surviving Op arg means the
                    # fold list declined it below, and descending now
                    # would fold ops outside that list.
                    fused = None
                    if all(
                        isinstance(a, (_Param, Const)) for a in term.args
                    ):
                        fused = eval_term(
                            term,
                            param_env=self._param_values,
                            bindings=self._torch_bindings,
                            tensor_only=True,
                        )
                    if fused is not None:
                        fused_name = f"fused_{len(self._param_map) + len(self._param_values)}"
                        self._param_values[fused_name] = (
                            fused.detach().clone()
                        )
                        shape = tuple(int(d) for d in fused.shape)
                        from catopt_core.ir import TensorType

                        result = _Param(
                            name=fused_name,
                            typ=TensorType(shape),
                        )
                        self._fold_memo[orig_key] = result
                        return result
        self._fold_memo[orig_key] = term
        return term

    def _build_params(self) -> None:
        from catopt_core.ir import Op as _Op
        from catopt_core.ir import Param as _Param

        param_shapes: dict[str, tuple] = {}

        # Interned term objects are content-hashed — safe set members
        # (no id()/GC hazards).
        seen: set[Any] = set()

        def collect(t: Any) -> None:
            if t in seen:
                return
            seen.add(t)
            if isinstance(t, _Param):
                if (
                    t.name not in param_shapes
                    and t.typ.size is not None
                ):
                    param_shapes[t.name] = tuple(
                        d if d is not None else 1 for d in t.typ.shape
                    )
            elif isinstance(t, _Op):
                for a in t.args:
                    collect(a)

        collect(self._root)
        for name, shape in param_shapes.items():
            if name in self._param_values:
                # Use the original model's parameter value.  Integer
                # tensors (quantized params offered by eps passes) are
                # registered without grad — Parameters require float.
                v = self._param_values[name].clone()
                p = torch.nn.Parameter(
                    v,
                    requires_grad=v.is_floating_point()
                    or v.is_complex(),
                )
            else:
                p = torch.nn.Parameter(torch.randn(*shape) * 0.02)
            # Use the IR name (e.g. p_w1) so _eval can find it
            setattr(self, name, p)
            self._param_map[name] = p

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0] if xs else None
        env: dict[str, torch.Tensor] = {"self": x}
        # Map input placeholders positionally to forward args
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        return self._eval(self._root, env, x, {})

    def _eval(
        self,
        term: Any,
        env: dict[str, torch.Tensor],
        x: torch.Tensor,
        memo: dict[Any, torch.Tensor],
    ) -> torch.Tensor:
        """Strict runtime evaluation — delegates to :func:`eval_term`.

        ``var_default=x`` is the single-input "self" fallback (``env``
        always carries ``"self"`` → ``x``, so a Var missing from ``env``
        resolves to the first input); ``param_default=_randn_param``
        materialises Params that never registered (unshaped
        ``typ.size is None``); shared subtrees dedup through ``memo``.
        """
        return eval_term(
            term,
            var_env=env,
            var_default=x,
            param_env=self._param_map,
            param_default=_randn_param,
            bindings=self._torch_bindings,
            memo=memo,
            strict=True,
        )


def ir_to_torch_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
    ops: OpTable | None = None,
) -> IRModule:
    """Convert a catopt IR into a torch.nn.Module.

    ``ops`` selects the lowering table; ``None`` (default) uses
    ``OpTable.full()`` — the ambient ``_IR_TO_TORCH`` table.  The
    return type stays the concrete :class:`IRModule` rather than the
    :class:`catopt_core.ports.Executor` port — callers use module attrs the
    port doesn't name (``state_dict``/``parameters``/``_eval``).
    """
    return IRModule(ir, param_values=param_values, ops=ops)
