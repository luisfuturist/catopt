"""Coverage-gap tests for catopt.torch_bridge.

``tests/test_torch_integration.py`` exercises the flagship pipeline
(export → eqsat → lower → verify).  This file drives the boundary and
lowering internals those flows never touch:

* name canonicalisation — ``_canon_aten_name``'s generic ``base``
  fallback (``gt.Scalar`` → ``gt``), ``_aten_name``'s non-``__name__``
  targets (string targets, ``aten::``-spelled objects), and
  ``_infer_shape``;
* ``export_to_ir`` edges — a non-persistent buffer lifting through
  ``_resolve_attr`` (the ``state_dict`` miss), non-tensor arg
  placeholders (int/bool inputs) on both the ``args[i]`` and
  ``args[0]`` fallback arms, scalar positionals on un-schema'd ops
  (``tril``/``argmax``), the string-arg skip (``einsum``'s equation),
  list args with a non-node element (``x[:, idx]``'s
  ``[None, node]``), a zero-operand ``call_function`` (``arange``),
  and the tensor-method vs functional spellings landing on the same
  aten ops;
* ``IRModule`` fold edges — param-only ``concat``/elementwise folds,
  a fold memo hit on a shared subtree, a non-foldable param-only op,
  a matmul fold skipped for missing ``param_values``, a fold whose
  binding raises, and a fold returning a non-tensor (custom table);
* ``_build_params`` DAG dedup, unshaped ``Param`` (``typ.size is
  None`` → eval-time ``randn`` fallback), integer params (no grad),
  missing-binding ``ValueError``, and ``_eval``'s non-term
  ``TypeError``;
* ``_AmbientTorchBindings`` lazy carrier resolution, ``__contains__``
  /``.get``/``__missing__``;
* ``_dim_args``'s positional-args and ``dim=None`` arms.

Defensive branches believed unreachable by construction are listed in
``test_defensive_branch_inventory``.
"""

import pytest
import torch
import torch.nn.functional as F

from catopt.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from catopt.ops import OpTable
from catopt.torch_bridge import (
    _IR_TO_TORCH,
    _aten_name,
    _canon_aten_name,
    _dim_args,
    _infer_shape,
    export_to_ir,
    ir_to_torch_module,
)

torch.manual_seed(0)


def _find(term, op_name, out=None):
    out = [] if out is None else out
    if isinstance(term, Op):
        if term.op == op_name:
            out.append(term)
        for a in term.args:
            _find(a, op_name, out)
    return out


# ---------------------------------------------------------------------------
#  name canonicalisation / shape inference helpers
# ---------------------------------------------------------------------------


def test_canon_aten_name_base_fallback():
    """``base`` (pre-suffix name) falls through to the ambient table;
    unknown names come back stripped."""
    # Generic overload stripping: 'gt.Scalar' isn't in either map, but
    # its base 'gt' is a core binding.
    assert _canon_aten_name("gt.Scalar") == "gt"
    assert _canon_aten_name("to.dtype") == "to"
    assert _canon_aten_name("le.Tensor") == "le"
    # A base with no binding at all returns the stripped name.
    assert _canon_aten_name("frobnicate.qq") == "frobnicate.qq"
    assert _canon_aten_name("_assert_tensor_metadata") == (
        "_assert_tensor_metadata"
    )
    # .default stripping still works on an unmapped op.
    assert _canon_aten_name("frobnicate.default") == "frobnicate"


def test_aten_name_target_variants():
    """``_aten_name`` accepts callables, raw strings, and objects whose
    str() spells aten:: — every non-``__name__`` path."""
    # A string target goes straight to the canonicaliser.
    assert _aten_name("mul.Tensor") == "mul"
    assert _aten_name("linear.default") == "linear"
    # A real OpOverload (no usable __name__): str -> aten.op form.

    class FakeAtenTarget:
        def __str__(self):
            return "aten::conv2d.default"

    assert _aten_name(FakeAtenTarget()) == "conv2d"

    class PlainTarget:
        def __str__(self):
            return "sigmoid"

    assert _aten_name(PlainTarget()) == "sigmoid"
    # And the __name__ path: real torch.ops.aten OpOverloads.
    assert _aten_name(torch.ops.aten.add.Tensor) == "add"


def test_infer_shape_helper():
    assert _infer_shape(torch.randn(2, 3)) == (2, 3)
    # Non-tensor values report the unknown-shape tuple.
    assert _infer_shape(3) == (None,)
    assert _infer_shape("x") == (None,)
    # A 0-dim tensor reports a genuine scalar shape.
    assert _infer_shape(torch.tensor(1.0)) == ()


# ---------------------------------------------------------------------------
#  export_to_ir — placeholder edge cases
# ---------------------------------------------------------------------------


def test_export_nonpersistent_buffer_resolves_via_attr():
    """Non-persistent buffers are placeholder params missing from the
    exported state_dict — values resolve through the module itself."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "mask", torch.randn(4, 4), persistent=False
            )

        def forward(self, x):
            return x @ self.mask

    m = M().eval()
    ir, source = export_to_ir(m, torch.randn(2, 4))
    assert "b_mask" in ir.params
    assert ir.params["b_mask"].typ.shape == (4, 4)
    # The value came through _resolve_attr — it IS the module buffer.
    assert torch.equal(source["b_mask"], m.mask)
    lowered = ir_to_torch_module(ir, param_values=source)
    x = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.allclose(lowered(x), m(x))


def test_export_nontensor_arg_placeholder():
    """An int arg exports as a placeholder without a tensor meta —
    its shape is inferred from the arg itself (unknown)."""

    class M(torch.nn.Module):
        def forward(self, x, n):
            return x * n

    m = M().eval()
    ir, _ = export_to_ir(m, (torch.randn(4), 3))
    names = [v.name for v in ir.inputs]
    assert names == ["x", "n"]
    assert ir.inputs[0].typ.shape == (4,)
    # int arg: no meta shape -> (None,) unknown.
    assert ir.inputs[1].typ.shape == (None,)


def test_export_lifted_constants_overrun_args_fallback():
    """Lifted tensor constants count as inputs; a trailing non-tensor
    arg past ``len(args)`` falls back to ``args[0]``'s shape."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = torch.randn(4)
            self.c2 = torch.randn(4)

        def forward(self, n, x):
            return self.c1 + self.c2 + x * n

    m = M().eval()
    ir, source = export_to_ir(m, (3, torch.randn(4)))
    # Lifted constants are user inputs (Vars), not params.
    assert ir.params == {}
    shapes = {v.name: v.typ.shape for v in ir.inputs}
    assert shapes["c_c1"] == (4,)
    assert shapes["c_c2"] == (4,)
    # n's placeholder has no meta val; by then inputs already overrun
    # args, so it inherits args[0] = 3's (non-tensor) shape.
    assert shapes["n"] == (None,)
    assert shapes["x"] == (4,)
    # The constants remain ordinary inputs at eval time — the env maps
    # each forward arg positionally onto (c_c1, c_c2, n, x).
    lowered = ir_to_torch_module(ir, param_values=source)
    x = torch.randn(4)
    with torch.no_grad():
        got = lowered(x, 2 * x, 0, x)
    # root is add(add(c_c1, c_c2), mul(x, 3.0)) — the n arg was
    # specialised into the graph, so the Var is unused.
    assert torch.allclose(got, x + 2 * x + x * 3)


# ---------------------------------------------------------------------------
#  export_to_ir — argument handling edges
# ---------------------------------------------------------------------------


def test_export_scalar_positional_unschemed_op():
    """Scalar positionals on ops outside _SCALAR_OPERAND_OPS and
    ATTR_SCHEMA land as ``argN`` attrs; bools on non-reduction ops
    do too."""

    class MTri(torch.nn.Module):
        def forward(self, x):
            return torch.tril(x, 1)

    ir, _ = export_to_ir(MTri().eval(), (torch.randn(4, 4),))
    tril = _find(ir.root, "tril")[0]
    assert dict(tril.attrs) == {"arg1": 1}

    class MArg(torch.nn.Module):
        def forward(self, x):
            return x.argmax(dim=1, keepdim=True)

    ir, _ = export_to_ir(MArg().eval(), (torch.randn(4, 8),))
    argmax = _find(ir.root, "argmax")[0]
    assert argmax.attrs["arg1"] == 1
    assert argmax.attrs["arg2"] is True


def test_export_string_arg_skipped():
    """einsum's equation string is an arg that is not an operand —
    it is skipped, leaving only the tensor operands."""

    class M(torch.nn.Module):
        def forward(self, a, b):
            return torch.einsum("ij,jk->ik", a, b)

    a, b = torch.randn(2, 3), torch.randn(3, 4)
    ir, _ = export_to_ir(M().eval(), (a, b))
    es = _find(ir.root, "einsum")
    assert len(es) == 1
    assert len(es[0].args) == 2
    # A private table (copy) — adding the binding must not touch the
    # ambient registry.
    table = OpTable.full().copy()
    table.torch_bindings["einsum"] = (
        lambda x, y, *a, **kw: torch.einsum("ij,jk->ik", x, y)
    )
    lowered = ir_to_torch_module(ir, ops=table)
    with torch.no_grad():
        assert torch.allclose(
            lowered(a, b), torch.einsum("ij,jk->ik", a, b)
        )


def test_export_index_list_with_none_element():
    """Advanced indexing emits ``index(x, [None, idx])`` — list
    elements without an env entry are skipped, the node operand
    survives."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x[:, torch.tensor([0, 2])]

    ir, _ = export_to_ir(M().eval(), (torch.randn(4, 4),))
    idx_op = _find(ir.root, "index.Tensor")
    assert len(idx_op) == 1
    # The None element contributes nothing; the detached index tensor
    # is the second operand.
    assert len(idx_op[0].args) == 2


def test_export_zero_operand_factory_op():
    """A factory call_function with no tensor operands mints
    ``Op.make(op, **attrs)`` — the no-args branch."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x + torch.arange(4)

    ir, _ = export_to_ir(M().eval(), (torch.randn(4),))
    arange = _find(ir.root, "arange")
    assert len(arange) == 1
    assert arange[0].args == ()
    # Scalar positional + bool kwarg both landed as attrs.
    assert arange[0].attrs["arg0"] == 4
    assert arange[0].attrs["pin_memory"] is False


def test_export_unbind_getitem_fold_and_stack():
    """``getitem(unbind(t), i)`` folds the index into the splitter op;
    ``stack`` keeps its dim attr."""

    class M(torch.nn.Module):
        def forward(self, x):
            parts = torch.unbind(x, dim=0)
            return parts[0] * parts[1] + torch.stack(
                [parts[0], parts[1]], dim=0
            )

    ir, _ = export_to_ir(M().eval(), (torch.randn(2, 4),))
    unbinds = _find(ir.root, "unbind")
    assert unbinds
    # getitem folded the section index onto the unbind term.
    assert any(dict(u.attrs).get("index") == 0 for u in unbinds)
    assert any(dict(u.attrs).get("index") == 1 for u in unbinds)
    assert _find(ir.root, "stack")

    x = torch.randn(2, 4)
    lowered = ir_to_torch_module(ir)
    m = M().eval()
    with torch.no_grad():
        assert torch.allclose(lowered(x), m(x))


def test_export_getitem_of_non_splitter_stays_getitem():
    """``getitem`` on a non split/chunk/unbind result keeps the op —
    sort's tuple is the canonical example."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x.sort(descending=True).values

    ir, _ = export_to_ir(M().eval(), (torch.randn(8),))
    getitems = _find(ir.root, "getitem")
    assert getitems
    # The tuple element stays a getitem op with an index attr (the
    # fold only applies to split/chunk/unbind bases).
    assert all("index" in dict(g.attrs) for g in getitems)
    assert _find(ir.root, "sort")


def test_export_narrow_select_index_select_dropout():
    """Schema'd positional tails land under canonical names for the
    less-common axis ops."""
    torch.manual_seed(0)

    class M(torch.nn.Module):
        def forward(self, x):
            y = x.narrow(1, 1, 4)
            z = x.index_select(1, torch.tensor([0, 1, 2, 3]))
            return F.dropout(y * z, 0.25)

    ir, _ = export_to_ir(M().eval(), (torch.randn(3, 8),))
    narrow = _find(ir.root, "narrow")[0]
    assert narrow.attrs["dim"] == 1
    assert narrow.attrs["start"] == 1
    assert narrow.attrs["length"] == 4
    drop = _find(ir.root, "dropout")
    assert drop and dict(drop[0].attrs).get("p") == 0.25
    assert _find(ir.root, "index_select")


def test_export_method_vs_functional_spellings():
    """``x.mean(-1)`` and ``torch.mean(x, -1)`` export to the same
    aten op and therefore the same IR term."""

    class MMethod(torch.nn.Module):
        def forward(self, x):
            return x.mean(-1)

    class MFunc(torch.nn.Module):
        def forward(self, x):
            return torch.mean(x, -1)

    x = torch.randn(2, 4)
    ir1, _ = export_to_ir(MMethod().eval(), x)
    ir2, _ = export_to_ir(MFunc().eval(), x)
    assert op_repr(ir1.root) == op_repr(ir2.root)
    assert ir1.inputs[0].typ.shape == ir2.inputs[0].typ.shape


def test_export_multi_input_tuple_example():
    """``example_input`` as a tuple feeds multi-arg modules; each
    placeholder keeps its own meta shape."""

    class M(torch.nn.Module):
        def forward(self, a, b):
            return a + b.sum()

    a, b = torch.randn(2, 3), torch.randn(2, 3)
    ir, _ = export_to_ir(M().eval(), (a, b))
    assert [v.name for v in ir.inputs] == ["a", "b"]
    assert all(v.typ.shape == (2, 3) for v in ir.inputs)


def test_export_to_dtype_canonicalises():
    """``x.to(dtype)`` hits the generic ``base in _IR_TO_TORCH``
    canonicalisation arm — the op is simply ``to``."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float64) + x.to(torch.float32)

    ir, _ = export_to_ir(M().eval(), (torch.randn(4),))
    assert _find(ir.root, "to")


# ---------------------------------------------------------------------------
#  IRModule — _fold_weight_chains variants
# ---------------------------------------------------------------------------


def _lower_root(root, param_values=None, ops=None, inputs=None):
    x = Var("x", TensorType((4, 4)))
    ir = IR(
        root=root,
        inputs=inputs if inputs is not None else [x],
        input_names={"x"} if inputs is None else {v.name for v in inputs},
        params={},
    )
    return ir_to_torch_module(ir, param_values=param_values or {}, ops=ops)


def test_fold_param_only_concat_materialises():
    """concat over stored weights folds to one fused parameter —
    the runtime graph reads a single cat'ed weight."""
    W = Param("W", TensorType((4, 4)))
    Q = Param("Q", TensorType((4, 4)))
    cat = Op.make("concat", W, Q, dim=1)  # (4,8): x(4,4) @ cat -> (4,8)
    x = Var("x", TensorType((4, 4)))
    root = Op.make("matmul", x, cat)
    wt = torch.randn(4, 4)
    qt = torch.randn(4, 4)
    mod = _lower_root(root, {"W": wt, "Q": qt})
    fused = [n for n, _ in mod.named_parameters() if "fused" in n]
    assert len(fused) == 1
    assert torch.equal(mod._param_map[fused[0]], torch.cat([wt, qt], dim=1))
    # The originals were consumed by the fold.
    assert "W" not in mod._param_map
    assert "Q" not in mod._param_map
    xv = torch.randn(4, 4)
    with torch.no_grad():
        assert torch.allclose(mod(xv), xv @ torch.cat([wt, qt], dim=1))


def test_fold_elementwise_param_chain():
    """Elementwise weight chains (mul/add/neg...) fold eagerly through
    the op table's torch bindings."""
    W = Param("W", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    root = Op.make("matmul", x, Op.make("mul", W, W))
    wt = torch.randn(4, 4)
    mod = _lower_root(root, {"W": wt})
    fused = [n for n, _ in mod.named_parameters() if "fused" in n]
    assert len(fused) == 1
    assert torch.allclose(mod._param_map[fused[0]], wt * wt)


def test_fold_memo_shared_subtree_single_materialisation():
    """A shared param-only subtree folds once — both parents read the
    same fused Param object."""
    W = Param("W", TensorType((4, 4)))
    Q = Param("Q", TensorType((4, 4)))
    cat = Op.make("concat", W, Q, dim=0)  # interned: ONE object
    x = Var("x", TensorType((4, 4)))
    root = Op.make("add", Op.make("mul", x, cat), Op.make("mul", x, cat))
    wt, qt = torch.randn(4, 4), torch.randn(4, 4)
    mod = _lower_root(root, {"W": wt, "Q": qt})
    fused = [n for n, _ in mod.named_parameters() if "fused" in n]
    assert len(fused) == 1
    # And _eval memoisation shares the runtime value, not two copies.
    assert mod._root.args[0].args[1] is mod._root.args[1].args[1]


def test_fold_matmul_missing_values_stays():
    """matmul over params folds only when BOTH operand tensors are in
    ``param_values`` — otherwise it stays a runtime matmul."""
    W = Param("W", TensorType((4, 4)))
    Q = Param("Q", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    root = Op.make("matmul", x, Op.make("matmul", W, Q))
    mod = _lower_root(root, {})
    assert not [n for n, _ in mod.named_parameters() if "fused" in n]
    # Both weights registered as ordinary (random) parameters.
    assert set(mod._param_map) == {"W", "Q"}
    # Root term is unchanged — the fold was skipped.
    assert mod._root.args[1].op == "matmul"


def test_fold_param_only_nonfoldable_op_stays():
    """A param-only subtree on an op outside the foldable set is left
    alone even though it has no Var leaves."""
    W = Param("W", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    root = Op.make(
        "matmul", x, Op.make("reshape", W, shape=(4, 4))
    )
    mod = _lower_root(root, {"W": torch.randn(4, 4)})
    assert not [n for n, _ in mod.named_parameters() if "fused" in n]
    assert "W" in mod._param_map


def test_fold_elementwise_binding_raises_keeps_term():
    """An eager fold whose torch binding throws is abandoned — the
    subtree survives as a runtime op."""
    W = Param("W", TensorType((2, 3)))
    Q = Param("Q", TensorType((4, 5)))
    x = Var("x", TensorType((4, 4)))
    # add(W(2,3), Q(4,5)) broadcasts to nothing — torch.add raises.
    root = Op.make("matmul", x, Op.make("add", W, Q))
    mod = _lower_root(root, {"W": torch.randn(2, 3), "Q": torch.randn(4, 5)})
    assert not [n for n, _ in mod.named_parameters() if "fused" in n]
    assert mod._root.args[1].op == "add"


def test_fold_non_tensor_result_keeps_term():
    """A binding that returns a non-tensor can't be a fused param —
    the guard keeps the original term."""
    W = Param("W", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    root = Op.make("matmul", x, Op.make("mul", W, W))
    table = OpTable.core()
    table.torch_bindings["mul"] = lambda *a, **kw: "not-a-tensor"
    mod = _lower_root(root, {"W": torch.randn(4, 4)}, ops=table)
    assert not [n for n, _ in mod.named_parameters() if "fused" in n]
    assert mod._root.args[1].op == "mul"
    assert "W" in mod._param_map


def test_fold_nested_matmul_chains_fully():
    """matmul(P, matmul(Q, R)) folds bottom-up: the inner product is
    registered before the outer is considered — one final param."""
    P = Param("P", TensorType((4, 3)))
    Q = Param("Q", TensorType((3, 2)))
    R = Param("R", TensorType((2, 4)))
    root = Op.make("matmul", P, Op.make("matmul", Q, R))
    pt, qt, rt = (
        torch.randn(4, 3),
        torch.randn(3, 2),
        torch.randn(2, 4),
    )
    x = Var("x", TensorType((4, 4)))
    mod = _lower_root(
        Op.make("add", x, root), {"P": pt, "Q": qt, "R": rt}
    )
    fused = [n for n, _ in mod.named_parameters() if "fused" in n]
    # Only the OUTERMOST fold reaches the root — the inner product is
    # baked into it, so one fused parameter stores P@(Q@R).
    assert len(fused) == 1
    expected = pt @ (qt @ rt)
    assert torch.allclose(
        mod._param_map[fused[0]], expected, atol=1e-6
    )


def test_uses_input_leaf_classification():
    x = Var("x", TensorType((4, 4)))
    mod = _lower_root(Op.make("neg", x))
    assert mod._uses_input(x) is True
    assert mod._uses_input(Param("P", TensorType((4, 4)))) is False
    assert mod._uses_input(Const(1.0)) is False
    assert mod._uses_input("metavar") is False
    # Memoised: second read hits the cache.
    shared = Op.make("mul", x, x)
    assert mod._uses_input(shared) is True
    assert mod._uses_input(shared) is True


# ---------------------------------------------------------------------------
#  _build_params / _eval edges
# ---------------------------------------------------------------------------


def test_build_params_shared_dag_subterm():
    """``collect`` walks each distinct node once — a shared subterm is
    visited once via the ``seen`` early exit."""
    x = Var("x", TensorType((4, 4)))
    W = Param("W", TensorType((4, 4)))
    m = Op.make("mul", x, W)  # interned — same object both slots
    root = Op.make("add", m, m)
    mod = _lower_root(root, {"W": torch.randn(4, 4)})
    assert list(mod._param_map) == ["W"]
    xv = torch.randn(4, 4)
    with torch.no_grad():
        expected = xv * mod._param_map["W"] * 2
        assert torch.allclose(mod(xv), expected)


def test_eval_unshaped_param_falls_back_to_randn():
    """A Param whose TensorType carries an unknown dim never registers
    (``typ.size is None``) — _eval materialises a placeholder randn."""
    x = Var("x", TensorType((1, 3)))
    P = Param("W", TensorType((None, 3)))
    root = Op.make("add", x, P)
    mod = _lower_root(root, {})
    torch.manual_seed(0)
    out = mod(torch.randn(1, 3))
    # None dims materialise as extent-1 under the randn fallback.
    assert out.shape == (1, 3)


def test_eval_unknown_term_type_raises():
    mod = _lower_root(42)
    with pytest.raises(TypeError, match="Cannot evaluate term"):
        mod(torch.randn(4, 4))


def test_eval_missing_binding_raises():
    """An op absent from the chosen OpTable fails loudly at eval."""
    x = Var("x", TensorType((4, 4)))
    table = OpTable.core()
    del table.torch_bindings["neg"]
    mod = _lower_root(Op.make("neg", x), ops=table)
    with pytest.raises(ValueError, match="No torch binding"):
        mod(torch.randn(4, 4))


def test_eval_attrs_and_multi_input_env():
    """``dict(term.attrs)`` flows to the binding; missing forward args
    inherit the first input."""
    x = Var("x", TensorType((2, 4)))
    y = Var("y", TensorType((2, 4)))
    sel = Op.make("select", x, dim=0, index=1)
    mod = _lower_root(sel)
    xv = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.equal(mod(xv), xv.select(0, 1))
    # Two inputs, one forward arg -> the second reads x as well.
    mod2 = _lower_root(Op.make("add", x, y), inputs=[x, y])
    with torch.no_grad():
        assert torch.allclose(mod2(xv), xv + xv)


def test_param_requires_grad_by_dtype():
    """Integer-valued params register without grad; floats with it."""
    W = Param("W", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    mod = _lower_root(
        Op.make("mul", x, W), {"W": torch.zeros(4, 4, dtype=torch.int64)}
    )
    assert mod._param_map["W"].requires_grad is False
    mod2 = _lower_root(
        Op.make("mul", x, W), {"W": torch.randn(4, 4)}
    )
    assert mod2._param_map["W"].requires_grad is True


# ---------------------------------------------------------------------------
#  ambient bindings table / _dim_args
# ---------------------------------------------------------------------------


def test_ambient_table_lazy_carrier_resolution():
    """Carrier bindings resolve on demand through the ambient dict and
    cache back into it; unknown ops raise KeyError."""
    # Remove a cached entry if a prior test resolved it.
    _IR_TO_TORCH.pop("cswap", None)
    fn = _IR_TO_TORCH["cswap"]  # misses dict -> resolves carrier
    assert callable(fn)
    assert _IR_TO_TORCH["cswap"] is fn  # cached now
    assert "cswap" in _IR_TO_TORCH
    assert _IR_TO_TORCH.get("bdiag") is not None
    with pytest.raises(KeyError):
        _IR_TO_TORCH["definitely_not_an_op"]
    assert _IR_TO_TORCH.get("definitely_not_an_op") is None
    assert _IR_TO_TORCH.get("definitely_not_an_op", "fallback") == (
        "fallback"
    )
    assert "definitely_not_an_op" not in _IR_TO_TORCH
    # Real carrier binding semantics: cswap mints the [a;b] -> [b;a]
    # swap matrix.
    x = torch.randn(6)
    m = _IR_TO_TORCH["cswap"](d1=2, d2=4)
    assert m.shape == (6, 6)
    assert torch.equal(m @ x, torch.cat([x[2:], x[:2]]))


def test_dim_args_helper():
    """``_dim_args`` reads positional args first, then ``dim``/``axis``
    and ``keepdim`` attrs."""
    # Positional args win outright.
    assert _dim_args((2,), {"dim": 0}) == (2,)
    # dim=None means "reduce all" — an empty arg tuple.
    assert _dim_args((), {"dim": None}) == ()
    # axis is a synonym; list dims normalise to tuples; keepdim rides.
    assert _dim_args((), {"axis": [0, 2]}) == ((0, 2), False)
    assert _dim_args((), {"dim": -1, "keepdim": True}) == (-1, True)
    # No attrs at all -> reduce the last axis, no keepdim.
    assert _dim_args((), {}) == (-1, False)


# ---------------------------------------------------------------------------
#  defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------


def test_defensive_branch_inventory():
    """Documents branches no real ``torch.export`` graph can produce on
    torch 2.14 — candidates for ``# pragma: no cover``:

    * ``export_to_ir`` ``get_attr`` nodes (~257-261): modern export
      lifts every tensor constant/buffer/attribute as a placeholder
      (``c_*``/``b_*``/``p_*`` names) — ``get_attr`` nodes are never
      emitted.  ``_resolve_attr`` IS covered via the non-persistent
      buffer ``state_dict`` miss.
    * ``elif ir_op in ("split", "chunk"): attrs["sizes"]`` (~315-316):
      schema'd positions intercept every list arg these ops can
      carry (``ATTR_SCHEMA["split"]`` covers 1-3), so a list can only
      arrive at an undeclared position — which doesn't exist.
    * ``node.kwargs`` ``dim``-list and list-attr arms (~343-348):
      torch.export normalises all declared-schema kwargs into
      positional args; only factory kwargs (device/dtype/layout/
      pin_memory/memory_format) remain — never lists, never ``dim``.
    * ``_canon_aten_name`` line ~144 (``return
      _IR_TO_TORCH_EXTRA[base]``): ``base`` is the text before the
      first dot, but every ``_IR_TO_TORCH_EXTRA`` key *contains* a dot
      (``mul.Tensor``/``linear.default``/...) — a dotless ``base``
      can never be a key.  The ``base in _IR_TO_TORCH`` check below
      it handles every real case (``gt.Scalar`` → ``gt``,
      ``to.dtype`` → ``to``).
    * root-finding arcs (~376-382, 377->376, 380, 382->385): exported
      graphs always end with a single ``output`` node whose args[0]
      is a *tuple* of nodes, so ``hasattr(arg, "name")`` is never
      true (380 dead), the loop never iterates past it (377->376),
      never exhausts (376->382), and ``root is None`` is always true
      making 382->385 (the non-None arc) dead.  Verified empirically:
      ``args: ((node,),)`` on every export in the suite.
    """
    # Evidence for the output-tuple claim — the invariant the whole
    # inventory rests on.
    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    ep = torch.export.export(M().eval(), (torch.randn(2),))
    out_node = list(ep.graph.nodes)[-1]
    assert out_node.op == "output"
    assert isinstance(out_node.args[0], tuple)
    assert not hasattr(out_node.args[0], "name")
