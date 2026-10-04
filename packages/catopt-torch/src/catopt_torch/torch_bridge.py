# ruff: noqa: RUF002, RUF003
"""Fixed torch_bridge.py.

Uses exported._graph_signature.inputs_to_parameters to correctly map
graph placeholder targets (p_w1, p_w2, ...) to actual model parameter
names (W1, W2, ...) and retrieve their shapes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, cast

import torch
from catopt_core.attrs import (
    ATTR_SCHEMA,
    attr_of,
    is_positional_attr,
)
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
    # The specialised GEMM spellings all unify under "matmul" so the
    # matmul laws and shape rule see them.
    "mm": "matmul",
    "bmm": "matmul",
    "mv": "matmul",
    "dot": "matmul",
    "linear": "linear",
    "transpose": "transpose",
    # aten.t() is transpose(-2, -1) — the transpose binding's default.
    "t": "transpose",
    "reshape": "reshape",
    "view": "reshape",
    # aten.amax/.amin are values-only reductions; aten.max/.min are
    # the values+indices namedtuple picked apart by getitem — they
    # are different ops and must not collapse into one name.
    "amax": "amax",
    "amin": "amin",
    "matmul.default": "matmul",
    "contiguous": "contiguous",
    "clone": "contiguous",
    "lift_fresh_copy": "contiguous",
    # aten.trace is the diagonal SUM — NOT the carrier ``trace``
    # morphism; renaming at the boundary keeps the carrier binding
    # from claiming exported terms it cannot evaluate.
    "trace": "diag_sum",
    # In-place index writes functionalise to index_put under export.
    "index_put_": "index_put",
    "unsafe_index_put": "index_put",
    "_unsafe_index_put": "index_put",
    # torch.tile(t, dims) IS repeat — the same ``shape`` list attr.
    "tile": "repeat",
    # torch.linalg.inv / torch.inverse → the ``inv`` op the trace
    # carrier already binds (torch.linalg.inv) and types.
    "linalg_inv": "inv",
    "inverse": "inv",
    "scaled_dot_product_attention": "sdpa",
    "conv2d": "conv2d",
    "cat": "concat",
    # List-form split exports as aten.split_with_sizes; unsafe_*
    # variants are the functionalization-internal spellings of the
    # same op.  All lower through the "split" binding.
    "split_with_sizes": "split",
    "unsafe_split": "split",
    "unsafe_split_with_sizes": "split",
    # fill_.Tensor/fill_.Scalar/masked_fill_* are deliberately NOT in
    # this table: they are ``copy_``-family mutations, dispatched by
    # raw-name prefix below (like copy_/zero_ themselves) — a
    # canonical entry would mint an unbound IR op and break the
    # binding-coverage contract.
    # torch.special.gammaln exports as aten.special_gammaln — same op
    # as the ``gammaln`` the binding already lowers.
    "special_gammaln": "gammaln",
    # Infix ``a | b``/``a & b`` spell aten.__or__/__and__ — the bound
    # bitwise ops (bitwise_or is dtype-polymorphic: bool tensors get
    # the logical-or semantics the spelling means).
    "__or__": "bitwise_or",
    "__and__": "bitwise_and",
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
    "amax.default": "amax",
    "amin.default": "amin",
    "scaled_dot_product_attention.default": "sdpa",
    "conv2d.default": "conv2d",
    # aten.upsample_nearest2d.vec — the two-list overload torch.export
    # emits for F.interpolate/nn.Upsample in ``nearest`` mode; the
    # vec suffix is an overload tag, not a different op.
    "upsample_nearest2d.vec": "upsample_nearest2d",
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
    # scatter.value's fill, lerp's weight and the addcmul/addcdiv
    # ``value`` kwarg-position are real operands.  (arange's bounds
    # stay ATTRS: Const would erase the int/float distinction and a
    # float arange can't produce int64 indices.)
    "scatter",
    "lerp",
    "addcmul",
    "addcdiv",
    # Scalar second operands of these accept a Number — ``x % 2``,
    # ``x < 2``, ``maximum(x, 0.5)`` all pass the scalar as a real
    # operand.  As an ``argN`` attr they'd reach the torch fn as a
    # bogus keyword arg.
    "remainder",
    "fmod",
    "xlogy",
    "heaviside",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    # full_like/full's fill value is a Const operand like pow's
    # exponent — position and value both matter to a rewrite.
    "full",
    "full_like",
    "new_full",
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
    if (
        base in _IR_TO_TORCH_EXTRA
    ):  # pragma: no cover — every EXTRA key contains a dot
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


def _fill_src(node: Any, dst_fx: Any, env: dict[str, Any]) -> Any:
    """Materialise a ``fill_``/``zero_`` scalar write as a term.

    The destination's ``meta["val"]`` shape is the broadcast target —
    an absent meta means the write cannot be sized honestly, so the
    ``KeyError`` surfaces as an export rejection rather than minting a
    term that drops the mutation.  ``fill_.Tensor``'s source is a
    lifted 0-dim tensor (broadcast); ``fill_.Scalar``'s is an inline
    number (a ``full`` creator); ``zero_`` carries none (``zeros``).
    """
    val = dst_fx.meta["val"]
    shape = tuple(int(d) for d in val.shape)
    dtype = str(val.dtype).split(".")[-1]
    src_fx = node.args[1] if len(node.args) > 1 else None
    if _aten_name(node.target).startswith("zero_"):
        return Op.make("zeros", shape=shape, dtype=dtype)
    if hasattr(src_fx, "name"):
        # KeyError on an unminted source is an honest rejection.
        return Op.make("broadcast_to", env[src_fx.name], shape=shape)
    return Op.make(
        "full",
        Const(cast("int | float", src_fx)),
        shape=shape,
        dtype=dtype,
    )


def _handle_copy_(node: Any, env: dict[str, Any]) -> None:
    """Thread a functionalized ``copy_``/``fill_``/``zero_`` write.

    ``copy_(dst, src)`` where dst is a ``slice``/``select`` view rewrites
    the VIEWED BASE's env binding to the matching scatter — downstream
    readers of the base then see the post-write value.  ``fill_`` and
    ``zero_`` are the scalar spellings of the same write: the value is
    first broadcast to the destination's shape (``_fill_src``), so a
    ``y[:, :4] = 0`` slice fill lowers through ``slice_scatter`` like
    the ``copy_`` view-write it functionalises from.  A copy on a
    whole tensor rewrites the dst binding itself to a ``copy`` op.
    """
    if not node.args:
        return
    dst_fx = node.args[0]
    dst = env.get(dst_fx.name)
    if dst is None:
        return
    target = _aten_name(node.target)
    if target.startswith(("fill_", "zero_")):
        src = _fill_src(node, dst_fx, env)
    elif target.startswith("masked_fill_"):
        # masked_fill_(dst, mask, v) — the functional ``masked_fill``
        # term over dst is the written value; the view dispatch below
        # scatters it onto the base exactly like a copy_ source.  An
        # unminted mask/value raises (KeyError/IndexError) into the
        # caller's rejection record — never a silently-dropped write.
        mask = env[node.args[1].name]
        v_fx = node.args[2]
        v = env[v_fx.name] if hasattr(v_fx, "name") else Const(v_fx)
        src = Op.make("masked_fill", dst, mask, v)
    else:
        src = env.get(node.args[1].name)
        if src is None:
            return
    dst_target = _aten_name(dst_fx.target)
    view_args = getattr(dst_fx, "args", ())
    if dst_target == "slice" and len(view_args) >= 4:
        base_fx = view_args[0]
        base = env.get(base_fx.name)
        if base is None:
            return
        dim = view_args[1]
        start = view_args[2]
        end = view_args[3]
        step = view_args[4] if len(view_args) > 4 else 1
        new_base = Op.make(
            "slice_scatter",
            base,
            src,
            dim=dim,
            start=start,
            end=end,
            step=step,
        )
        env[base_fx.name] = new_base
        # The view node still exists — it reads the post-write base.
        env[dst_fx.name] = Op.make(
            "slice", new_base, dim=dim, start=start, end=end, step=step
        )
    elif dst_target == "select" and len(view_args) >= 3:
        base_fx = view_args[0]
        base = env.get(base_fx.name)
        if base is None:
            return
        dim, index = view_args[1], view_args[2]
        new_base = Op.make(
            "select_scatter", base, src, dim=dim, index=index
        )
        env[base_fx.name] = new_base
        env[dst_fx.name] = Op.make(
            "select", new_base, dim=dim, index=index
        )
    else:
        # Whole-tensor write (or an unrecognised view): dst's readers
        # see the broadcast-cloned source.
        env[dst_fx.name] = Op.make("copy", dst, src)
    env[node.name] = env[dst_fx.name]


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
    # Tensor constants created inside forward (``torch.tensor(0.5)``)
    # lift to placeholders keyed into ``exported.constants`` — also
    # constants, never user inputs.  Leaving them as Vars would let
    # eval's "self" fallback silently substitute the first input.
    with contextlib.suppress(Exception):
        inputs_to_params.update(
            exported._graph_signature.inputs_to_lifted_tensor_constants
        )
    constants = getattr(exported, "constants", None) or {}

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
                    # Lifted tensor constants live in
                    # ``exported.constants``, not the state dict.
                    tensor = constants.get(model_attr_name)
                if tensor is None or not hasattr(
                    tensor, "shape"
                ):  # pragma: no cover — every signature-mapped tensor
                    # lives in the state dict or exported.constants;
                    # this is the get_attr-era safety net.
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
                else:  # pragma: no cover — placeholders consume an
                    # example arg in order; an unmapped extra (e.g. a
                    # lifted custom object) is the only overrun.
                    shape = _infer_shape(args[0])
                var = Var(name=node.name, typ=TensorType(shape))
                env[node.name] = var
                inputs.append(var)

        elif (
            node.op == "get_attr"
        ):  # pragma: no cover — torch 2.14 export lifts everything to placeholders
            tensor = _resolve_attr(mod, node.target)
            shape = tuple(int(d) for d in tensor.shape)
            param = Param(name=node.name, typ=TensorType(shape))
            env[node.name] = param
            params[node.name] = param

        elif node.op == "call_function":
            op_name = _aten_name(node.target)
            ir_op = _ATEN_TO_IR.get(op_name, op_name)
            if op_name.startswith(
                ("copy_", "fill_", "zero_", "masked_fill_")
            ):
                # Functionalized in-place write ``dst = src`` — the
                # FX graph is not SSA here: ``copy_``/``fill_``/
                # ``zero_``/``masked_fill_`` MUTATE dst's tensor and
                # downstream nodes keep referencing the pre-write
                # node.  Thread the
                # mutation through env: a write through a
                # ``slice``/``select`` view becomes the matching
                # scatter on the viewed base; a whole-tensor write is
                # the ``copy`` op — either way consumers of the dst
                # node see the post-write value.
                _handle_copy_(node, env)
                continue
            args = []
            attrs: dict[str, Any] = {}
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
                        # Keep ints as ints — coercing to float
                        # makes e.g. ``x % 2`` promote to float at
                        # eval (aten ``*.Scalar`` is weak-typed).
                        args.append(
                            Const(
                                arg_node
                                if isinstance(arg_node, int)
                                else float(arg_node)
                            )
                        )
                    else:
                        attrs[f"arg{i}"] = arg_node
                elif isinstance(arg_node, str):  # pragma: no cover —
                    # schema'd str positionals (pad.mode, conv padding,
                    # einsum.equation) are named upstream; an unschemed
                    # str positional would land here.
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
                        if ir_op in ("index", "index_put"):
                            # Advanced indexing keeps real semantics in
                            # the key's None slots: x[:, i] is NOT x[i].
                            # ``layout`` marks which positions carry an
                            # index operand (True) vs a full slice
                            # (False) so the binding can rebuild the
                            # exact key tuple.
                            attrs["layout"] = tuple(
                                hasattr(a, "name") for a in arg_node
                            )
                    # e.g. dim=[-1] lists for reductions; for view/reshape
                    # the list is the target SHAPE, not a dim.
                    elif ir_op == "reshape" or ir_op in (
                        "expand",
                        "repeat",
                        "broadcast_to",
                        # Tensor creators take the shape tuple as the
                        # same-named attr — zeros((2, 3)) etc.
                        "zeros",
                        "ones",
                        "empty",
                        "randn",
                        "rand",
                        "full",
                        "new_zeros",
                        "new_ones",
                        "new_empty",
                        "new_full",
                    ):
                        attrs["shape"] = tuple(arg_node)
                    elif ir_op in ("split", "chunk"):
                        attrs["sizes"] = tuple(
                            arg_node
                        )  # pragma: no cover — schema covers all real positions
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
                elif isinstance(arg_node, torch.dtype):
                    # ``to``/creator overloads carry a positional
                    # ScalarType; record the short name just like the
                    # dtype kwarg channel below — a dropped cast mints
                    # a wrong-dtype term (one_hot's long->float).
                    attrs["dtype"] = str(arg_node).split(".")[-1]
                else:
                    key = (
                        arg_node.name
                        if hasattr(arg_node, "name")
                        else str(arg_node)
                    )
                    if key in env:
                        args.append(env[key])
            for (
                k,
                v,
            ) in (
                node.kwargs.items()
            ):  # pragma: no cover — export normalises kwargs
                if k == "dim" and isinstance(v, (list, tuple)):
                    attrs[k] = tuple(v)
                elif isinstance(v, (int, float, bool)):
                    attrs[k] = v
                elif isinstance(v, list):
                    attrs[k] = tuple(v)
                elif isinstance(v, str):
                    # String-valued flags are semantics: gelu's
                    # ``approximate``, pad's ``mode``, scatter_reduce's
                    # ``reduce``.  Attrs are the only channel that
                    # reaches the binding.
                    attrs[k] = v
                elif isinstance(v, torch.dtype):
                    # ``dtype`` is semantics, not a dispatch detail —
                    # ``zeros(shape, dtype=float64)`` must not lower
                    # to an fp32 tensor.  Record the short name
                    # ("float64"): torch.dtype objects do not survive
                    # term serialization.
                    attrs[k] = str(v).split(".")[-1]
            # Canonical attr spellings are guaranteed below by
            # ``Op.make`` (ATTR_SCHEMA): positional attrs were named at
            # emission above; any ``argN`` still present for a schema'd
            # op fails loudly at mint.
            #
            # aten elides trailing arguments at their defaults, so an
            # implicit ``dim=0`` on split*/chunk never reaches the FX
            # node.  Record it: an absent ``dim`` reads as the last
            # axis (the minted-term/law convention), which is the
            # WRONG axis for aten's positional default — the packed
            # QKV in-proj split is a dim-0 split of a (3E, E) weight.
            if ir_op in ("split", "chunk") and "dim" not in attrs:
                attrs["dim"] = 0
            if (
                ir_op == "getitem"
                and args
                and isinstance(args[0], Op)
                and args[0].op
                in ("split", "chunk", "unbind", "tensor_split")
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
    for node in reversed(
        graph.nodes
    ):  # pragma: no branch — output node is always last, first when reversed
        if (
            node.op == "output" and node.args
        ):  # pragma: no branch — same invariant
            arg = node.args[0]
            # The FX output arg is a tuple ``(out,)`` — unwrap to the
            # node so env rewrites (e.g. the copy_ mutation threader
            # rebinding ``clone`` to its scatter) are honoured.
            arg = (
                arg[0]
                if isinstance(arg, (tuple, list)) and arg
                else arg
            )
            if (
                hasattr(arg, "name") and arg.name in env
            ):  # pragma: no branch — tuples don't carry .name
                root = env[
                    arg.name
                ]  # pragma: no cover — tuples don't carry .name
            break  # pragma: no cover — same
    if (
        root is None and env
    ):  # pragma: no cover — output args resolve via the tuple unwrap
        # above; this only fires on a structurally-degenerate graph.
        root = list(env.values())[-1]

    return IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    ), source_tensors


def _resolve_attr(module: torch.nn.Module, target: str) -> torch.Tensor:
    obj: Any = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _expand_torch(t: Any, *a: Any, **kw: Any) -> Any:
    """``t.expand`` binding: the target shape arrives as attrs or args.

    ``shape`` / ``dim`` attrs or positional args — a size sequence OR a
    bare int.  Splat sequences; pass a scalar through (``t.expand(*4)``
    would be a ``TypeError``, ``t.expand(4)``/``t.expand(-1)`` is
    legal).
    """
    dim_v = kw.get("shape") or kw.get("dim") or a
    if isinstance(dim_v, (tuple, list)):
        return t.expand(*dim_v)
    return t.expand(dim_v)


def _scalar_value(v: Any) -> Any:
    """Unwrap a Const-evaluated scalar to a Python number.

    A Const-evaluated scalar arrives as a 0-dim tensor — unwrap it
    to the Python number the aten call expects.
    """
    if torch.is_tensor(v):
        return v.item() if v.dim() == 0 else v
    return v


def _scatter_torch(t: Any, idx: Any, v: Any = None, **kw: Any) -> Any:
    """``scatter`` covers both aten spellings.

    ``scatter.src`` (src operand) and ``scatter.value`` (Const
    operand that evals to a 0-dim tensor — unwrap it to the Number
    the value overload wants).
    """
    dim = int(attr_of(kw, "dim", default=0))
    if torch.is_tensor(v) and v.dim() == 0:
        v = v.item()
    if torch.is_tensor(v):
        return t.scatter(dim, idx, v)
    return t.scatter(dim, idx, value=v)


def _index_torch(t: Any, *idxs: Any, **kw: Any) -> Any:
    """``index`` — numpy-style advanced indexing.

    ``layout`` marks the key positions that carry an index operand
    (True) vs a full slice (False); minted terms without it index
    the leading axes.
    """
    layout = attr_of(kw, "layout", default=None) or (True,) * len(idxs)
    it = iter(idxs)
    key = tuple(
        next(it) if is_idx else slice(None) for is_idx in layout
    )
    return t[key]


def _index_put_torch(t: Any, *a: Any, **kw: Any) -> Any:
    """``index_put`` — the functional ``x[key] = v``.

    The index operands come first (counted by ``layout``'s True
    entries), the values operand last; ``accumulate`` is an attr.
    """
    layout = attr_of(kw, "layout", default=None) or ()
    n_idx = sum(1 for flag in layout if flag)
    if layout:
        it = iter(a[:n_idx])
        key = tuple(
            next(it) if is_idx else slice(None) for is_idx in layout
        )
        value = a[n_idx] if len(a) > n_idx else None
    else:
        # Minted term without layout: all-but-last operands index
        # leading axes, the last is the value.
        key = tuple(a[:-1])
        value = a[-1] if a else None
    acc = bool(attr_of(kw, "accumulate", default=False))
    return torch.index_put(
        t, cast(Any, key), cast(Any, value), accumulate=acc
    )


def _arange_torch(*a: Any, **kw: Any) -> Any:
    """``arange`` — the bounds arrive as Const operands.

    ``arange`` is in ``_SCALAR_OPERAND_OPS``; operands are
    evaluated to 0-dim tensors.
    """
    vals = [_scalar_value(v) for v in a]
    if not vals:
        vals = [
            _scalar_value(kw[k])
            for k in sorted(
                (k for k in kw if is_positional_attr(k)),
                key=lambda k: int(k[3:]),
            )
        ]
    return torch.arange(*vals, dtype=_creator_dtype(kw))


def _batch_norm_torch(x: Any, *a: Any, **kw: Any) -> Any:
    """``aten.batch_norm`` → F.batch_norm's operand order.

    ``aten.batch_norm(x, w, b, rm, rv, ...)`` →
    F.batch_norm's (x, rm, rv, w, b, ...) operand order.  None
    weight/bias are dropped at export, so the affine pair is
    whatever sits between x and the running stats (a lone mid
    operand reads as weight).
    """
    rm, rv = a[-2], a[-1]
    mid = a[:-2]
    w = mid[0] if len(mid) >= 1 else None
    b = mid[1] if len(mid) >= 2 else None
    return torch.nn.functional.batch_norm(
        x,
        rm,
        rv,
        weight=w,
        bias=b,
        training=bool(attr_of(kw, "training", default=False)),
        momentum=float(attr_of(kw, "momentum", default=0.1)),
        eps=float(attr_of(kw, "eps", default=1e-5)),
    )


def _instance_norm_torch(x: Any, *a: Any, **kw: Any) -> Any:
    """``aten.instance_norm`` → F.instance_norm's operand order.

    ``aten.instance_norm(x, w, b, rm, rv, use_input_stats, momentum,
    eps, cudnn)`` — None operands drop at export, so the operand list
    is whatever survived: the affine pair fills the leading slots,
    and the running-stats pair is the tail iff ``use_input_stats`` is
    False (the eval-mode export of a ``track_running_stats=True``
    module).  ``cudnn_enabled`` is a dispatch hint, not semantics —
    F.instance_norm has no such kwarg.
    """
    uis = bool(attr_of(kw, "use_input_stats", default=True))
    stats_ok = not uis and len(a) >= 2
    rm, rv = (a[-2], a[-1]) if stats_ok else (None, None)
    mid = a[:-2] if stats_ok else a
    w = mid[0] if len(mid) >= 1 else None
    b = mid[1] if len(mid) >= 2 else None
    return torch.nn.functional.instance_norm(
        x,
        running_mean=rm,
        running_var=rv,
        weight=w,
        bias=b,
        use_input_stats=uis,
        momentum=float(attr_of(kw, "momentum", default=0.1)),
        eps=float(attr_of(kw, "eps", default=1e-5)),
    )


def _upsample_nearest2d_torch(x: Any, *a: Any, **kw: Any) -> Any:
    """``aten.upsample_nearest2d[.vec]`` → F.interpolate nearest.

    The vec overload's two list arguments land under the canonical
    ``size`` / ``scale`` attrs (``ATTR_SCHEMA``); exactly one is
    present on any export.  Scalar spellings broadcast to a pair.
    """
    size = attr_of(kw, "size", default=None)
    scale = attr_of(kw, "scale", default=None)
    if isinstance(size, int) and not isinstance(size, bool):
        size = (size, size)
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = (float(scale), float(scale))
    return torch.nn.functional.interpolate(
        x,
        size=size,
        scale_factor=scale,
        mode="nearest",
    )


def _lstm_input_torch(
    x: Any, h0: Any, c0: Any, *params: Any, **kw: Any
):
    """``aten.lstm.input`` — the recurrent kernel passthrough.

    The two operand lists (``hx``, the flat ``params`` table) arrive
    flattened at export: ``h0``/``c0`` are always exactly the first
    two slots, the weights/biases fill the rest (4 tensors per layer
    per direction).  The scalar tail is schema'd (has_biases,
    num_layers, dropout, train, bidirectional, batch_first).  The op
    returns the ``(output, h_n, c_n)`` triple — a ``getitem``
    consumer picks one.
    """
    return torch.ops.aten.lstm.input(
        x,
        [h0, c0],
        list(params),
        bool(attr_of(kw, "has_biases", default=True)),
        int(attr_of(kw, "num_layers", default=1)),
        float(attr_of(kw, "dropout", default=0.0)),
        bool(attr_of(kw, "train", default=False)),
        bool(attr_of(kw, "bidirectional", default=False)),
        bool(attr_of(kw, "batch_first", default=False)),
    )


#: Core torch lowering bindings — the base ``OpTable``'s table (plan
#: 0001 phase 2c).  Carrier ops (``trace``/``omd_*``/``cmask``...)
#: are deliberately ABSENT: they live in each carrier
#: module's ``TORCH_BINDINGS`` export and compose via
#: :class:`catopt_core.ops.OpTable`, not by mutating this dict at import.
#: ``_IR_TO_TORCH`` below is the ambient view over this table.
_CORE_TORCH_BINDINGS: dict[str, Any] = {
    "matmul": torch.matmul,
    "add": torch.add,
    "mul": torch.mul,
    # aten.div.Tensor_mode carries rounding_mode as a kwarg — x/y
    # without it silently drops a "floor" div down to trunc.
    "div": lambda x, y, *a, **kw: torch.div(
        x, y, rounding_mode=kw.get("rounding_mode")
    ),
    "sub": torch.sub,
    # aten.rsub(self, other) = other - self — the ``1 - x`` spelling.
    "rsub": lambda x, s, *a, **kw: torch.sub(s, x),
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
    "amax": lambda x, *a, **kw: x.amax(*_dim_args(a, kw)),
    "amin": lambda x, *a, **kw: x.amin(*_dim_args(a, kw)),
    # aten.max/.min — the (values, indices) pair when a dim is given,
    # the scalar value for the whole-tensor spelling.  getitem picks.
    "max": lambda x, *a, **kw: (
        torch.max(x)
        if attr_of(kw, "dim", "axis", default=None) is None
        else torch.max(
            x,
            dim=_dim_args(a, kw)[0],
            keepdim=bool(attr_of(kw, "keepdim", default=False)),
        )
    ),
    "min": lambda x, *a, **kw: (
        torch.min(x)
        if attr_of(kw, "dim", "axis", default=None) is None
        else torch.min(
            x,
            dim=_dim_args(a, kw)[0],
            keepdim=bool(attr_of(kw, "keepdim", default=False)),
        )
    ),
    "var_mean": lambda x, *a, **kw: torch.var_mean(
        x,
        dim=attr_of(kw, "dim", default=None),
        correction=attr_of(kw, "correction", default=1),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "std_mean": lambda x, *a, **kw: torch.std_mean(
        x,
        dim=attr_of(kw, "dim", default=None),
        correction=attr_of(kw, "correction", default=1),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    # Every binding reads ONLY the canonical ATTR_SCHEMA name — the
    # dual-spelling argN fallbacks are gone (the boundary emits
    # canonical; nothing downstream mints a bare positional).
    "transpose": lambda x, *a, **kw: (
        x.t()
        if x.dim() == 2 and "dim0" not in kw and "dim1" not in kw
        else x.transpose(
            attr_of(kw, "dim0", default=-2),
            attr_of(kw, "dim1", default=-1),
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
        list(ts), dim=int(attr_of(kw, "dim", default=0))
    ),
    "chunk": lambda t, chunks=2, dim=-1, index=0, **kw: torch.chunk(
        t,
        int(attr_of(kw, "chunks", default=chunks)),
        dim=int(attr_of(kw, "dim", default=dim)),
    )[int(attr_of(kw, "index", default=index))],
    "split": lambda t, sizes=(), dim=-1, index=0, **kw: torch.split(
        t,
        _split_sizes(sizes, kw),
        dim=int(attr_of(kw, "dim", default=dim)),
    )[int(attr_of(kw, "index", default=index))],
    # torch.export emits aten.dropout with train=False in eval mode —
    # the op is a semantic identity there.  This binding is only valid
    # because export_to_ir always exports eval()-mode graphs.
    "dropout": lambda x, *a, **kw: x,
    # ``to`` — dtype/device casts.  A cast the boundary recorded must
    # apply: ``one_hot(...)``'s long→float ``to(dtype)`` feeds a matmul
    # that raises on dtype mismatch when the cast is dropped.  With no
    # recorded dtype the cast is still identity at the precision we
    # verify — the exported graph runs fp64 end to end.
    "to": lambda x, *a, **kw: (
        x.to(dtype=_creator_dtype(kw))
        if _creator_dtype(kw) is not None
        else x
    ),
    "clone": lambda x, *a, **kw: x.clone(),
    "getitem": lambda t, **kw: t[attr_of(kw, "index", default=0)],
    "unbind": lambda t, *a, **kw: torch.unbind(
        t, dim=int(attr_of(kw, "dim", default=0))
    )[int(attr_of(kw, "index", default=0))],
    "stack": lambda *ts, **kw: torch.stack(
        list(ts), dim=int(attr_of(kw, "dim", default=0))
    ),
    "expand": _expand_torch,
    "flatten": lambda x, *a, **kw: x.flatten(
        int(attr_of(kw, "start_dim", default=0)),
        int(attr_of(kw, "end_dim", default=-1)),
    ),
    "slice": lambda t, *a, **kw: t[
        (slice(None),) * int(attr_of(kw, "dim", default=0))
        + (slice(kw.get("start"), kw.get("end"), kw.get("step")),)
    ],
    "unsqueeze": lambda t, *a, **kw: t.unsqueeze(
        int(attr_of(kw, "dim", default=-1))
    ),
    # squeeze() with no dim removes EVERY size-1 axis (aten.squeeze);
    # squeeze(dim) removes just that axis (aten.squeeze.dim).
    "squeeze": lambda t, *a, **kw: (
        t.squeeze()
        if attr_of(kw, "dim", default=None) is None
        else t.squeeze(int(attr_of(kw, "dim")))
    ),
    "select": lambda t, *a, **kw: t.select(
        int(attr_of(kw, "dim", default=0)),
        int(attr_of(kw, "index", default=0)),
    ),
    # embedding(W, idx) — row gather.
    "embedding": lambda w, idx, *a, **kw: torch.nn.functional.embedding(
        idx, w
    ),
    # gather along one axis.  The index is normally an attr (a tuple of
    # ints produced by share_duplicate_param_slices); a second tensor
    # operand (the raw aten spelling) is accepted too.
    "index_select": lambda t, *a, **kw: torch.index_select(
        t,
        int(attr_of(kw, "dim", default=0)),
        (
            a[0]
            if a and torch.is_tensor(a[0])
            else torch.as_tensor(
                [
                    int(v)
                    for v in attr_of(
                        kw, "index", default=a[0] if a else ()
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
        x, dim=int(attr_of(kw, "dim", default=-1))
    ),
    # aten.rms_norm(x, normalized_shape, weight, eps) — canonical
    # attrs dim=normalized_shape, eps=eps (Llama-family normalization).
    "rms_norm": lambda x, w=None, *a, **kw: (
        torch.nn.functional.rms_norm(
            x,
            tuple(attr_of(kw, "dim", "normalized_shape")),
            weight=w,
            eps=float(attr_of(kw, "eps", default=1e-6)),
        )
    ),
    "layer_norm": lambda x, w=None, b=None, *a, **kw: (
        torch.nn.functional.layer_norm(
            x,
            tuple(attr_of(kw, "dim", "normalized_shape")),
            weight=w,
            bias=b,
            # eps is the canonical ``eps`` attr — the old binding read
            # the cudnn flag's position and silently produced eps=0.0.
            eps=float(attr_of(kw, "eps", default=1e-5)),
        )
    ),
    # --- shape ops ----------------------------------------------------
    # broadcast_to materialises like expand; the exported size list
    # lands under the ``shape`` attr.
    "broadcast_to": lambda t, *a, **kw: torch.broadcast_to(
        t, tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ()))
    ),
    # repeat(*sizes) — the size list lands under ``shape``.
    "repeat": lambda t, *a, **kw: t.repeat(
        *tuple(attr_of(kw, "shape", default=a if a else ()))
    ),
    "unflatten": lambda t, *a, **kw: t.unflatten(
        int(attr_of(kw, "dim", default=0)),
        tuple(attr_of(kw, "sizes", default=a[0] if a else (-1,))),
    ),
    "permute": lambda t, *a, **kw: (
        t.permute(*tuple(attr_of(kw, "dim", "dims", default=a)))
        if attr_of(kw, "dim", "dims", default=None) is not None or a
        else t.permute(*reversed(range(t.dim())))
    ),
    "movedim": lambda t, *a, **kw: torch.movedim(
        t,
        attr_of(kw, "source", default=a[0] if a else 0),
        attr_of(kw, "destination", default=a[1] if len(a) > 1 else -1),
    ),
    "flip": lambda t, *a, **kw: torch.flip(
        t,
        tuple(attr_of(kw, "dim", "dims", default=a[0] if a else ())),
    ),
    "roll": lambda t, *a, **kw: torch.roll(
        t,
        shifts=attr_of(kw, "shifts", default=a[0] if a else 1),
        dims=attr_of(kw, "dims", "dim", default=None),
    ),
    "expand_as": lambda t, o, *a, **kw: t.expand_as(o),
    "narrow": lambda t, *a, **kw: t.narrow(
        int(attr_of(kw, "dim", default=0)),
        int(attr_of(kw, "start", default=0)),
        int(attr_of(kw, "length", default=1)),
    ),
    # --- indexing -------------------------------------------------------
    "gather": lambda t, idx, *a, **kw: t.gather(
        int(attr_of(kw, "dim", default=0)), idx
    ),
    "take_along_dim": lambda t, idx, *a, **kw: torch.take_along_dim(
        t, idx, dim=int(attr_of(kw, "dim", default=0))
    ),
    "scatter": _scatter_torch,
    "scatter_add": lambda t, idx, src, *a, **kw: t.scatter_add(
        int(attr_of(kw, "dim", default=0)), idx, src
    ),
    "scatter_reduce": lambda t, idx, src, *a, **kw: t.scatter_reduce(
        int(attr_of(kw, "dim", default=0)),
        idx,
        src,
        reduce=attr_of(kw, "reduce", default="sum"),
    ),
    "index_add": lambda t, idx, src, *a, **kw: t.index_add(
        int(attr_of(kw, "dim", default=0)), idx, src
    ),
    # index(t, idx...): ``layout`` records which key positions carry
    # an index operand vs a full slice — x[:, i] ≠ x[i].
    "index": _index_torch,
    "index_put": _index_put_torch,
    "slice_scatter": lambda t, src, *a, **kw: t.slice_scatter(
        src,
        dim=int(attr_of(kw, "dim", default=0)),
        start=attr_of(kw, "start", default=None),
        end=attr_of(kw, "end", default=None),
        step=int(attr_of(kw, "step", default=1)),
    ),
    "select_scatter": lambda t, src, *a, **kw: t.select_scatter(
        src,
        dim=int(attr_of(kw, "dim", default=0)),
        index=int(attr_of(kw, "index", default=0)),
    ),
    # --- reductions -----------------------------------------------------
    "argmax": lambda t, *a, **kw: torch.argmax(
        t,
        dim=attr_of(kw, "dim", default=None),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "argmin": lambda t, *a, **kw: torch.argmin(
        t,
        dim=attr_of(kw, "dim", default=None),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "prod": lambda t, *a, **kw: t.prod(*_dim_args(a, kw)),
    "var": lambda t, *a, **kw: t.var(
        dim=attr_of(kw, "dim", default=None),
        correction=attr_of(kw, "correction", default=1),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "std": lambda t, *a, **kw: t.std(
        dim=attr_of(kw, "dim", default=None),
        correction=attr_of(kw, "correction", default=1),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "any": lambda t, *a, **kw: t.any(*_dim_args(a, kw)),
    "all": lambda t, *a, **kw: t.all(*_dim_args(a, kw)),
    "nansum": lambda t, *a, **kw: torch.nansum(t, *_dim_args(a, kw)),
    "nanmean": lambda t, *a, **kw: torch.nanmean(t, *_dim_args(a, kw)),
    "count_nonzero": lambda t, *a, **kw: torch.count_nonzero(
        t, dim=attr_of(kw, "dim", default=None)
    ),
    "cumsum": lambda t, *a, **kw: torch.cumsum(
        t, int(attr_of(kw, "dim", default=0))
    ),
    "cumprod": lambda t, *a, **kw: torch.cumprod(
        t, int(attr_of(kw, "dim", default=0))
    ),
    "logcumsumexp": lambda t, *a, **kw: torch.logcumsumexp(
        t, int(attr_of(kw, "dim", default=0))
    ),
    # cummax/cummin/median/topk/kthvalue/sort return value+index
    # pairs — the IR term is the pair; a getitem consumer picks one.
    "cummax": lambda t, *a, **kw: torch.cummax(
        t, int(attr_of(kw, "dim", default=0))
    ),
    "cummin": lambda t, *a, **kw: torch.cummin(
        t, int(attr_of(kw, "dim", default=0))
    ),
    "median": lambda t, *a, **kw: (
        torch.median(t)
        if attr_of(kw, "dim", default=None) is None
        else torch.median(
            t,
            dim=int(attr_of(kw, "dim")),
            keepdim=bool(attr_of(kw, "keepdim", default=False)),
        )
    ),
    "kthvalue": lambda t, *a, **kw: torch.kthvalue(
        t,
        int(attr_of(kw, "k", default=a[0] if a else 1)),
        dim=int(attr_of(kw, "dim", default=-1)),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "argsort": lambda t, *a, **kw: torch.argsort(
        t,
        dim=int(attr_of(kw, "dim", default=-1)),
        descending=bool(attr_of(kw, "descending", default=False)),
    ),
    "topk": lambda t, *a, **kw: torch.topk(
        t,
        int(attr_of(kw, "k", default=a[0] if a else 1)),
        dim=int(attr_of(kw, "dim", default=-1)),
        largest=bool(attr_of(kw, "largest", default=True)),
        sorted=bool(attr_of(kw, "sorted", default=True)),
    ),
    "sort": lambda t, *a, **kw: torch.sort(
        t,
        dim=int(attr_of(kw, "dim", default=-1)),
        descending=bool(attr_of(kw, "descending", default=False)),
    ),
    "mode": lambda t, *a, **kw: torch.mode(
        t,
        dim=int(attr_of(kw, "dim", default=-1)),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "linalg_vector_norm": lambda t, *a, **kw: torch.linalg.vector_norm(
        t,
        ord=attr_of(kw, "ord", default=2),
        dim=attr_of(kw, "dim", default=None),
        keepdim=bool(attr_of(kw, "keepdim", default=False)),
    ),
    "diag_sum": lambda t, *a, **kw: torch.trace(t),
    # --- elementwise ----------------------------------------------------
    "clamp": lambda t, *a, **kw: torch.clamp(
        t,
        min=attr_of(kw, "min", default=a[0] if a else None),
        max=attr_of(kw, "max", default=a[1] if len(a) > 1 else None),
    ),
    "clamp_min": lambda t, *a, **kw: torch.clamp_min(
        t, attr_of(kw, "min", default=a[0] if a else 0)
    ),
    "clamp_max": lambda t, *a, **kw: torch.clamp_max(
        t, attr_of(kw, "max", default=a[0] if a else 0)
    ),
    "hardtanh": lambda t, *a, **kw: torch.nn.functional.hardtanh(
        t,
        min_val=attr_of(kw, "min", default=-1.0),
        max_val=attr_of(kw, "max", default=1.0),
    ),
    "leaky_relu": lambda t, *a, **kw: torch.nn.functional.leaky_relu(
        t,
        float(a[0])
        if a
        else float(attr_of(kw, "negative_slope", default=0.01)),
    ),
    "elu": lambda t, *a, **kw: (
        attr_of(kw, "scale", default=1.0)
        * torch.nn.functional.elu(
            t * attr_of(kw, "input_scale", default=1.0),
            alpha=attr_of(kw, "alpha", default=1.0),
        )
    ),
    "celu": lambda t, *a, **kw: torch.nn.functional.celu(t),
    "softplus": lambda t, *a, **kw: torch.nn.functional.softplus(
        t,
        beta=attr_of(kw, "beta", default=1),
        threshold=attr_of(kw, "threshold", default=20),
    ),
    "softsign": lambda t, *a, **kw: torch.nn.functional.softsign(t),
    "hardswish": lambda t, *a, **kw: torch.nn.functional.hardswish(t),
    "hardsigmoid": lambda t, *a, **kw: torch.nn.functional.hardsigmoid(
        t
    ),
    "mish": lambda t, *a, **kw: torch.nn.functional.mish(t),
    "relu6": lambda t, *a, **kw: torch.nn.functional.relu6(t),
    "glu": lambda t, *a, **kw: torch.nn.functional.glu(
        t, dim=int(attr_of(kw, "dim", default=-1))
    ),
    "prelu": lambda t, w, *a, **kw: torch.nn.functional.prelu(t, w),
    # Scalar-operand spellings eval to 0-dim tensors; torch's
    # tensor-tensor promotion reproduces the aten weak-type result
    # (ints stay ints thanks to the int-preserving Const).
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "fmax": torch.fmax,
    "fmin": torch.fmin,
    "fmod": torch.fmod,
    "remainder": torch.remainder,
    "xlogy": torch.xlogy,
    "atan2": torch.atan2,
    "heaviside": torch.heaviside,
    "isclose": lambda a, b, *args, **kw: torch.isclose(
        a,
        b,
        rtol=kw.get("rtol", 1e-5),
        atol=kw.get("atol", 1e-8),
        equal_nan=kw.get("equal_nan", False),
    ),
    "lerp": lambda t, e, w, *a, **kw: torch.lerp(t, e, w),
    "addcmul": lambda t, t1, t2, *a, **kw: torch.addcmul(
        t,
        t1,
        t2,
        value=attr_of(kw, "value", default=a[0] if a else 1),
    ),
    "addcdiv": lambda t, t1, t2, *a, **kw: torch.addcdiv(
        t,
        t1,
        t2,
        value=attr_of(kw, "value", default=a[0] if a else 1),
    ),
    "logical_and": lambda x, y, *a, **kw: torch.logical_and(x, y),
    "logical_or": lambda x, y, *a, **kw: torch.logical_or(x, y),
    "logical_xor": lambda x, y, *a, **kw: torch.logical_xor(x, y),
    "bitwise_and": lambda x, y, *a, **kw: torch.bitwise_and(x, y),
    "bitwise_or": lambda x, y, *a, **kw: torch.bitwise_or(x, y),
    "log_softmax": lambda x, *a, **kw: torch.nn.functional.log_softmax(
        x, dim=int(attr_of(kw, "dim", default=-1))
    ),
    "abs": torch.abs,
    "sign": torch.sign,
    "floor": torch.floor,
    "ceil": torch.ceil,
    # aten.round.decimals carries decimals as a kwarg.
    "round": lambda t, *a, **kw: torch.round(
        t, decimals=kw.get("decimals", 0)
    ),
    "frac": torch.frac,
    "trunc": torch.trunc,
    "log": torch.log,
    "log2": torch.log2,
    "log10": torch.log10,
    "log1p": torch.log1p,
    "exp2": torch.exp2,
    "expm1": torch.expm1,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "erfinv": torch.erfinv,
    "gammaln": torch.special.gammaln,
    "digamma": torch.digamma,
    "i0": torch.i0,
    "sinc": torch.sinc,
    "reciprocal": torch.reciprocal,
    "nan_to_num": lambda t, *a, **kw: torch.nan_to_num(
        t,
        nan=float(kw.get("nan", 0.0)),
        posinf=kw.get("posinf"),
        neginf=kw.get("neginf"),
    ),
    "isnan": torch.isnan,
    "isinf": torch.isinf,
    "isfinite": torch.isfinite,
    "isposinf": torch.isposinf,
    "isneginf": torch.isneginf,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "asinh": torch.asinh,
    "acosh": torch.acosh,
    "atanh": torch.atanh,
    "triu": lambda t, *a, **kw: torch.triu(
        t, diagonal=int(attr_of(kw, "diagonal", default=0))
    ),
    "tril": lambda t, *a, **kw: torch.tril(
        t, diagonal=int(attr_of(kw, "diagonal", default=0))
    ),
    "pad": lambda t, *a, **kw: torch.nn.functional.pad(
        t,
        list(attr_of(kw, "pad", default=a[0] if a else ())),
        mode=attr_of(kw, "mode", default="constant"),
        value=attr_of(kw, "value", default=None),
    ),
    "one_hot": lambda t, *a, **kw: torch.nn.functional.one_hot(
        t,
        num_classes=int(attr_of(kw, "num_classes", default=-1)),
    ),
    "batch_norm": _batch_norm_torch,
    "group_norm": lambda t, *a, **kw: torch.nn.functional.group_norm(
        t,
        int(attr_of(kw, "num_groups", default=1)),
        a[0] if a else None,
        a[1] if len(a) >= 2 else None,
        eps=float(attr_of(kw, "eps", default=1e-5)),
    ),
    "instance_norm": _instance_norm_torch,
    "upsample_nearest2d": _upsample_nearest2d_torch,
    # The recurrent kernel — an aten passthrough (the term IS the
    # overload; ``aten.lstm.data`` — the PackedSequence variant — stays
    # unbound deliberately: same base name, different input type).
    "lstm.input": _lstm_input_torch,
    "conv1d": lambda x, w, *a, **kw: torch.nn.functional.conv1d(
        x,
        w,
        a[0] if a else kw.get("bias"),
        stride=kw.get("stride", 1),
        padding=kw.get("padding", 0),
        dilation=kw.get("dilation", 1),
        groups=kw.get("groups", 1),
    ),
    # --- multi-output / misc ---------------------------------------------
    "searchsorted": lambda srt, v, *a, **kw: torch.searchsorted(
        srt,
        v,
        out_int32=bool(attr_of(kw, "out_int32", default=False)),
        right=bool(attr_of(kw, "right", default=False)),
    ),
    "nonzero": lambda t, *a, **kw: torch.nonzero(t),
    "outer": torch.outer,
    "einsum": lambda *ts, **kw: torch.einsum(
        attr_of(kw, "equation", default=""), *ts
    ),
    "tensor_split": lambda t, *a, **kw: torch.tensor_split(
        t,
        attr_of(kw, "sections", default=2),
        dim=int(attr_of(kw, "dim", default=0)),
    )[int(attr_of(kw, "index", default=0))],
    "unfold": lambda t, *a, **kw: t.unfold(
        int(attr_of(kw, "dim", default=0)),
        int(attr_of(kw, "size", default=1)),
        int(attr_of(kw, "step", default=1)),
    ),
    "pixel_shuffle": lambda t, *a, **kw: (
        torch.nn.functional.pixel_shuffle(
            t, int(attr_of(kw, "upscale_factor", default=1))
        )
    ),
    "pixel_unshuffle": lambda t, *a, **kw: (
        torch.nn.functional.pixel_unshuffle(
            t, int(attr_of(kw, "downscale_factor", default=1))
        )
    ),
    "hstack": lambda *ts, **kw: torch.hstack(list(ts)),
    "vstack": lambda *ts, **kw: torch.vstack(list(ts)),
    # --- creators / casts / export artifacts ------------------------------
    "arange": _arange_torch,
    "zeros": lambda *a, **kw: torch.zeros(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "ones": lambda *a, **kw: torch.ones(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "empty": lambda *a, **kw: torch.empty(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "randn": lambda *a, **kw: torch.randn(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "rand": lambda *a, **kw: torch.rand(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "full": lambda *a, **kw: torch.full(
        tuple(attr_of(kw, "shape", "dim", default=())),
        _scalar_value(a[0]) if a else kw.get("fill_value", 0),
        dtype=_creator_dtype(kw),
    ),
    "zeros_like": lambda t, *a, **kw: torch.zeros_like(
        t, dtype=_creator_dtype(kw)
    ),
    "ones_like": lambda t, *a, **kw: torch.ones_like(
        t, dtype=_creator_dtype(kw)
    ),
    "full_like": lambda t, *a, **kw: torch.full_like(
        t,
        _scalar_value(a[0]) if a else kw.get("fill_value", 0),
        dtype=_creator_dtype(kw),
    ),
    "new_zeros": lambda t, *a, **kw: t.new_zeros(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "new_ones": lambda t, *a, **kw: t.new_ones(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "new_empty": lambda t, *a, **kw: t.new_empty(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        dtype=_creator_dtype(kw),
    ),
    "new_full": lambda t, *a, **kw: t.new_full(
        tuple(attr_of(kw, "shape", "dim", default=a[0] if a else ())),
        _scalar_value(a[-1]) if a else kw.get("fill_value", 0),
        dtype=_creator_dtype(kw),
    ),
    "int": lambda t, *a, **kw: t.int(),
    "long": lambda t, *a, **kw: t.long(),
    "double": lambda t, *a, **kw: t.double(),
    "half": lambda t, *a, **kw: t.half(),
    "bfloat16": lambda t, *a, **kw: t.bfloat16(),
    "bool": lambda t, *a, **kw: t.bool(),
    "byte": lambda t, *a, **kw: t.byte(),
    "char": lambda t, *a, **kw: t.char(),
    "short": lambda t, *a, **kw: t.short(),
    "detach": lambda t, *a, **kw: t.detach(),
    "detach_": lambda t, *a, **kw: t.detach(),
    "_assert_tensor_metadata": lambda t, *a, **kw: t,
    "copy": lambda t, s, *a, **kw: torch.broadcast_to(
        s, t.shape
    ).clone(),
    "item": lambda t, *a, **kw: t.item(),
    "numel": lambda t, *a, **kw: t.numel(),
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
        """Pull ``key``'s binding from the carrier modules' exports.

        Caches the hit back into the dict.  ``None`` when unknown.
        """
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

    def get(self, key: object, default: Any = None) -> Any:
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
    """elem(s, v) = (rowmax s, Σ exp(s−m), exp(s−m) @ v).

    The online-softmax monoid element for one key block.
    """
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
    lv = torch.where(fin1, l1 * e1, torch.zeros_like(l1)) + torch.where(
        fin2, l2 * e2, torch.zeros_like(l2)
    )
    a = torch.where(fin1, a1 * e1, torch.zeros_like(a1)) + torch.where(
        fin2, a2 * e2, torch.zeros_like(a2)
    )
    return (mx, lv, a)


def _split_sizes(sizes: Any, kw: dict):
    """``split(x, sizes_list)`` and ``split(x, int)``.

    Both land under the canonical ``sizes`` attr.
    """
    sz = kw.get("sizes", sizes)
    if isinstance(sz, (list, tuple)) and sz:
        return list(sz)
    return int(sz if isinstance(sz, int) else 1)


def _creator_dtype(kw: dict) -> Any:
    """Resolve a ``dtype`` attr to a ``torch.dtype`` — or ``None``.

    Exported terms record the dtype's short name ("float64" — see
    ``export_to_ir``); a minted term may carry the ``torch.dtype``
    object itself.  ``None`` leaves the torch default for the
    no-operand creators and the operand's dtype for ``*_like`` /
    ``new_*``.
    """
    d = attr_of(kw, "dtype")
    if d is None:
        return None
    if isinstance(d, torch.dtype):
        return d
    return getattr(torch, str(d).split(".")[-1])


def _dim_args(args: tuple, kwargs: dict) -> tuple:
    """Extract a (dim, keepdim) argument tuple from IR attrs/args.

    No dim attr → ``()``: aten's dim-less spelling (``x.sum()``,
    ``x.mean()``) is a FULL reduce, not a last-axis one — the
    typing layer agrees (``sum`` with no dim reports ``()``).
    """
    if args:
        return tuple(args)
    dim = attr_of(kwargs, "dim", "axis")
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
    """``IRModule._eval``'s unregistered-Param fallback.

    A fresh randn of the declared shape (``None`` dims materialise as
    extent 1).
    """
    shape = tuple(d if d is not None else 1 for d in term.typ.shape)
    return torch.randn(*shape)


#: ``_plan_cache`` sentinel for a term that cannot be plan-flattened
#: (a non-term root such as ``42`` — the strict-eval ``TypeError``
#: stays delegated to :func:`eval_term`).
_UNPLANNED: Any = object()


class _EvalPlan:
    """A flattened, slot-indexed eval tape for one term DAG.

    Plan 0003: the recursive :func:`eval_term` walk is the generic
    executor's dispatch floor — per node it pays an isinstance chain,
    a content-hash memo get/set, a ``bindings`` lookup, an args list
    build and a ``dict(term.attrs)`` copy.  The plan replaces all of
    it with a linear tape evaluated in one loop:

    * **Slots** — every distinct subterm (DAG-shared nodes once, keyed
      on the interned term object) owns one slot in a per-call
      ``vals`` list; operand references are slot indices, so no
      recursion and no memo dict ever runs.
    * **Leaves** — ``var_fills`` resolve ``Var`` nodes by name through
      the eval env (``env.get(name, x)``) or positionally from the
      forward args (``xs[pos] if pos < len(xs) else x`` — the same
      value, since ``forward``'s env only ever maps input names plus
      ``"self"``); ``param_fills`` re-read ``param_map.get(name)``
      EVERY call (``load_state_dict`` visibility) with the
      ``_randn_param`` fallback for unregistered Params;
      ``const_fills`` re-mint ``torch.tensor(value)`` per call so the
      ambient default dtype at CALL time applies — caching the tensor
      would freeze the build-time dtype.
    * **Steps** — ``(out_slot, op_idx, arg_slots, attrs)``; ``op_idx``
      indexes ``op_names``, whose bindings are resolved through the
      module's op table per call — post-construction
      ``_IR_TO_TORCH[op] = fn`` overrides keep reaching already-built
      modules, and a deleted binding still raises the strict
      ``ValueError`` at eval.

    Plans are immutable once built and shared via the module's
    ``_plan_cache`` — safe under concurrent forwards (two racers
    build equal plans).
    """

    __slots__ = (
        "const_fills",
        "n_slots",
        "op_names",
        "param_fills",
        "root_slot",
        "steps",
        "var_fills",
    )

    def __init__(
        self,
        n_slots: int,
        var_fills: tuple,
        param_fills: tuple,
        const_fills: tuple,
        steps: tuple,
        op_names: tuple,
        root_slot: int,
    ) -> None:
        self.n_slots = n_slots
        self.var_fills = var_fills
        self.param_fills = param_fills
        self.const_fills = const_fills
        self.steps = steps
        self.op_names = op_names
        self.root_slot = root_slot

    @staticmethod
    def build(term: Any, input_pos: dict[str, int]) -> _EvalPlan | None:
        """Flatten ``term`` into a slot tape; ``None`` when unplannable.

        Iterative post-order (scan-domain prefix chains nest O(T)
        deep — past the recursion limit on long sequences).  A node
        that is not a ``Var``/``Param``/``Const``/``Op`` (a stray
        metavariable, a bare ``42`` root) makes the whole term
        unplannable — the caller falls back to ``eval_term``'s strict
        ``TypeError`` path.
        """
        slot_of: dict[Any, int] = {}
        var_fills: list[tuple[int, str, int]] = []
        param_fills: list[tuple[int, str, Any]] = []
        const_fills: list[tuple[int, Any]] = []
        steps: list[tuple[int, int, tuple, Any]] = []
        op_names: list[str] = []
        op_index: dict[str, int] = {}
        n = 0
        stack: list[tuple[Any, bool]] = [(term, False)]
        while stack:
            t, done = stack.pop()
            if done:
                s = n
                n += 1
                oi = op_index.get(t.op)
                if oi is None:
                    oi = len(op_names)
                    op_index[t.op] = oi
                    op_names.append(t.op)
                steps.append(
                    (
                        s,
                        oi,
                        tuple(slot_of[a] for a in t.args),
                        t.attrs or None,
                    )
                )
                slot_of[t] = s
                continue
            if t in slot_of:
                continue
            if isinstance(t, Op):
                stack.append((t, True))
                stack.extend((a, False) for a in t.args)
            elif isinstance(t, Var):
                slot_of[t] = n
                var_fills.append((n, t.name, input_pos.get(t.name, -1)))
                n += 1
            elif isinstance(t, Param):
                slot_of[t] = n
                param_fills.append((n, t.name, t))
                n += 1
            elif isinstance(t, Const):
                slot_of[t] = n
                const_fills.append((n, t.value))
                n += 1
            else:
                return None
        return _EvalPlan(
            n_slots=n,
            var_fills=tuple(var_fills),
            param_fills=tuple(param_fills),
            const_fills=tuple(const_fills),
            steps=tuple(steps),
            op_names=tuple(op_names),
            root_slot=slot_of[term],
        )

    def run_xs(self, xs: tuple, pmap_get: Any, bget: Any) -> Any:
        """``forward`` entry: Vars resolve positionally from ``xs``.

        ``pmap_get``/``bget`` are the bound ``.get``s of the module's
        ``_param_map`` and ``_torch_bindings``.
        """
        v: list = [None] * self.n_slots
        n = len(xs)
        x = xs[0] if n else None
        for s, _name, pos in self.var_fills:
            v[s] = xs[pos] if 0 <= pos < n else x
        return self._run_tail(v, pmap_get, bget)

    def run_env(
        self, env: Any, x: Any, pmap_get: Any, bget: Any
    ) -> Any:
        """``_eval`` entry: Vars resolve through the caller's env dict."""
        v: list = [None] * self.n_slots
        eget = (env or {}).get
        for s, name, _pos in self.var_fills:
            v[s] = eget(name, x)
        return self._run_tail(v, pmap_get, bget)

    def _run_tail(self, v: list, pmap_get: Any, bget: Any) -> Any:
        """Fill param/const slots, resolve bindings, run the tape."""
        for s, name, t in self.param_fills:
            p = pmap_get(name)
            v[s] = p if p is not None else _randn_param(t)
        for s, c in self.const_fills:
            v[s] = torch.tensor(c)
        fns: list = []
        for nm in self.op_names:
            f = bget(nm)
            if f is None:
                raise ValueError(f"No torch binding for op '{nm}'")
            fns.append(f)
        for o, fi, a, kw in self.steps:
            f = fns[fi]
            if kw is None:
                na = len(a)
                if na == 1:
                    v[o] = f(v[a[0]])
                elif na == 2:
                    v[o] = f(v[a[0]], v[a[1]])
                elif na == 3:
                    v[o] = f(v[a[0]], v[a[1]], v[a[2]])
                else:
                    v[o] = f(*[v[i] for i in a])
            else:
                na = len(a)
                if na == 0:
                    v[o] = f(**kw)
                elif na == 1:
                    v[o] = f(v[a[0]], **kw)
                elif na == 2:
                    v[o] = f(v[a[0]], v[a[1]], **kw)
                elif na == 3:
                    v[o] = f(v[a[0]], v[a[1]], v[a[2]], **kw)
                else:
                    v[o] = f(*[v[i] for i in a], **kw)
        return v[self.root_slot]


def eval_term(
    term: Any,
    *,
    var_env: dict | None = None,
    param_env: dict | None = None,
    bindings: Any = None,
    memo_env: dict | None = None,
    strict: bool = False,
    var_default: Any = _MISS,
    param_default: Callable[[Param], Any] | None = None,
    tensor_only: bool = False,
) -> Any:
    """Evaluate an IR term against concrete environments.

    Single source for the eval-term family (plan 0002 phase D) —
    :meth:`IRModule._eval` (strict runtime eval; the hot path itself
    runs the :class:`_EvalPlan` slot tape and delegates here only for
    seeded-memo calls and unplannable terms), the permissive
    compile-time fold ``optimize._eval_const``, and
    ``_fold_weight_chains``' shallow fold
    all delegate here.  Leaf semantics:

    * ``Var``   → ``var_env[name]``; on a miss, ``var_default`` when
      the caller supplied one (``IRModule`` passes the forward's first
      input — the single-input "self" fallback), else failure.
    * ``Param`` → ``param_env[name]``; on a miss,
      ``param_default(term)`` when given (``IRModule`` passes
      :func:`_randn_param` — the unshaped-Param fallback), else
      failure.
    * ``Const`` → ``torch.tensor(value)``.
    * ``Op``    → ``bindings[op](*args, **attrs)``; ``memo_env`` (when
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
                if (
                    strict
                ):  # pragma: no cover — strict callers pass var_default
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
            if memo_env is not None:
                hit = memo_env.get(t)
                if hit is not None:
                    return hit
            fn = (bindings or {}).get(t.op)
            if fn is None:
                if strict:
                    raise ValueError(
                        f"No torch binding for op '{t.op}'"
                    )
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
            if memo_env is not None:
                memo_env[t] = out
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
        """Initialise the module, folding weight-only subtrees."""
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
        # Plan-tape eval (plan 0003): last-wins name→position table so
        # the forward runner resolves Vars straight from ``xs`` without
        # building the env dict.  ``_plan_cache`` memoises per-term
        # plans for ``_eval``/``_eval_fast`` leaf evaluation.
        self._input_pos = {
            inp.name: i for i, inp in enumerate(self._inputs)
        }
        self._plan_cache: dict[Any, Any] = {}
        self._root_plan = self._plan_for(self._root)

    def _uses_input(self, term: Any) -> bool:
        """Return True if the term mentions a data-dependent leaf.

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
                    # The widening semantic set — every single-tensor
                    # op folds identically (views, reductions, gathers
                    # and creators all evaluate to a fused Param).
                    "relu",
                    "leaky_relu",
                    "elu",
                    "celu",
                    "softplus",
                    "softsign",
                    "hardswish",
                    "hardsigmoid",
                    "mish",
                    "relu6",
                    "glu",
                    "prelu",
                    "hardtanh",
                    "softmax",
                    "log_softmax",
                    "abs",
                    "sign",
                    "floor",
                    "ceil",
                    "round",
                    "frac",
                    "trunc",
                    "log",
                    "log2",
                    "log10",
                    "log1p",
                    "exp2",
                    "expm1",
                    "erf",
                    "erfc",
                    "erfinv",
                    "gammaln",
                    "digamma",
                    "i0",
                    "sinc",
                    "reciprocal",
                    "nan_to_num",
                    "isnan",
                    "isinf",
                    "isfinite",
                    "isposinf",
                    "isneginf",
                    "sin",
                    "cos",
                    "asin",
                    "acos",
                    "atan",
                    "sinh",
                    "cosh",
                    "asinh",
                    "acosh",
                    "atanh",
                    "rsqrt",
                    "clamp",
                    "clamp_min",
                    "clamp_max",
                    "maximum",
                    "minimum",
                    "fmax",
                    "fmin",
                    "fmod",
                    "remainder",
                    "xlogy",
                    "atan2",
                    "heaviside",
                    "isclose",
                    "lerp",
                    "addcmul",
                    "addcdiv",
                    "logical_and",
                    "logical_or",
                    "logical_xor",
                    "logical_not",
                    "bitwise_and",
                    "bitwise_or",
                    "sum",
                    "mean",
                    "max",
                    "min",
                    "prod",
                    "argmax",
                    "argmin",
                    "var",
                    "std",
                    "any",
                    "all",
                    "nansum",
                    "nanmean",
                    "count_nonzero",
                    "linalg_vector_norm",
                    "cumsum",
                    "cumprod",
                    "logcumsumexp",
                    "diag_sum",
                    "transpose",
                    "permute",
                    "movedim",
                    "reshape",
                    "flatten",
                    "unflatten",
                    "unsqueeze",
                    "squeeze",
                    # NOT folded: the slice/extraction/view family
                    # (index_select/gather/index/select/slice/narrow/
                    # unbind/getitem/tensor_split/expand/broadcast_to/
                    # repeat/expand_as/nonzero/unfold/stack-excluded
                    # too).  Those are exactly how
                    # share_duplicate_param_slices re-materialises a
                    # deduplicated weight — folding them would bake
                    # the reconstruction back into a full-size param
                    # and erase every dedup saving, for near-zero
                    # runtime gain (views and gathers are cheap).
                    "scatter",
                    "scatter_add",
                    "scatter_reduce",
                    "index_add",
                    "index_put",
                    "slice_scatter",
                    "select_scatter",
                    "flip",
                    "roll",
                    "triu",
                    "tril",
                    "pad",
                    "arange",
                    "zeros",
                    "ones",
                    "full",
                    "zeros_like",
                    "ones_like",
                    "full_like",
                    "new_zeros",
                    "new_ones",
                    "new_full",
                    "one_hot",
                    "nonzero",
                    "outer",
                    "argsort",
                    "contiguous",
                    "to",
                    "float",
                    "detach",
                    "detach_",
                    "alias",
                    "clone",
                    "int",
                    "long",
                    "double",
                    "half",
                    "bfloat16",
                    "bool",
                    "byte",
                    "char",
                    "short",
                    "type_as",
                    "dropout",
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
                        isinstance(a, (_Param, Const))
                        for a in term.args
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
                # tensors are registered without grad — Parameters
                # require a floating/complex dtype.
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

    def _plan_for(self, term: Any) -> _EvalPlan | None:
        """Fetch (or build and cache) the eval plan for ``term``.

        ``None`` when the term cannot be flattened — a non-term node
        anywhere in the DAG.  Cached under the (interned, content-keyed)
        term, so DAG-equal terms share one plan; builds are read-only
        afterwards.
        """
        plan = self._plan_cache.get(term, _MISS)
        if plan is _MISS:
            built = _EvalPlan.build(term, self._input_pos)
            plan = built if built is not None else _UNPLANNED
            self._plan_cache[term] = plan
        return None if plan is _UNPLANNED else plan

    def forward(self, *xs: Any) -> Any:
        """Run the module: bind inputs and evaluate the root."""
        plan = self._root_plan
        if plan is not None:
            # The tape's positional Var fills compute exactly what the
            # env dict would: ``xs[i] if i < len(xs) else x`` per
            # input name, ``x`` for everything else.
            return plan.run_xs(
                xs, self._param_map.get, self._torch_bindings.get
            )
        # Unplannable root (e.g. a bare non-term): the strict recursive
        # path preserves eval_term's exact failure semantics.
        x = cast(torch.Tensor, xs[0] if xs else None)
        env: dict[str, Any] = {"self": x}
        # Map input placeholders positionally to forward args
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        return self._eval(self._root, env, x, {})

    def _eval(
        self,
        term: Any,
        env: dict[str, Any],
        x: Any,
        memo: dict[Any, Any],
    ) -> Any:
        """Strict runtime evaluation — plan tape, or :func:`eval_term`.

        ``var_default=x`` is the single-input "self" fallback (``env``
        always carries ``"self"`` → ``x``, so a Var missing from ``env``
        resolves to the first input); ``param_default=_randn_param``
        materialises Params that never registered (unshaped
        ``typ.size is None``); shared subtrees dedup through ``memo``.

        A non-empty ``memo`` means caller-seeded short-circuits (the
        omd carrier seeds batched map/projection results INTO memo —
        they may hit at ANY depth), so the recursive evaluator runs
        verbatim.  An empty/absent memo gets the plan tape — internal
        dedup is structural (slots), and the top-level result is still
        written back under ``term`` when it is an ``Op``, matching
        ``eval_term``'s observable memo contract.
        """
        if memo:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                memo_env=memo,
                strict=True,
            )
        plan = self._plan_for(term)
        if plan is None:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                memo_env=memo,
                strict=True,
            )
        out = plan.run_env(
            env, x, self._param_map.get, self._torch_bindings.get
        )
        if memo is not None and isinstance(term, Op):
            memo[term] = out
        return out

    def _eval_fast(self, term: Any, env: dict[str, Any], x: Any) -> Any:
        """Memo-free strict eval for leaf terms — the tape, always.

        Same semantics as ``_eval`` without the caller memo: used by
        :meth:`BatchedExecutorBase.ev_factory` where the memo contract
        is discharged by the closure itself.
        """
        plan = self._plan_for(term)
        if plan is None:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                strict=True,
            )
        return plan.run_env(
            env, x, self._param_map.get, self._torch_bindings.get
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
