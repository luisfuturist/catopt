"""OpTable composition — plan 0001 phase 2c.

Pins the explicit-registry contract that replaced import-side-effect
registration:

* ``OpTable.core()`` carries the base bindings only — no carrier ops;
* ``OpTable.full()`` composes core + every carrier ``TORCH_BINDINGS`` —
  the ambient behavior ``ir_to_torch_module`` defaults to;
* a minimal custom table (core + one carrier) lowers a carrier term
  through ``IRModule(ops=...)``;
* deleting a binding from a custom table surfaces the missing-binding
  error at eval — custom tables never silently fall back to the
  ambient registry.
"""

import pytest
import torch

from catopt.ir import IR, Op, TensorType, Var
from catopt.ops import OpTable
from catopt.torch_bridge import (
    _IR_TO_TORCH,
    IRModule,
    ir_to_torch_module,
)

#: Ops whose bindings live in carrier ``TORCH_BINDINGS`` exports —
#: absent from ``OpTable.core()``, present in ``OpTable.full()``.
CARRIER_OPS = (
    # catopt.trace
    "trace",
    "bdiag",
    "parl",
    "eye",
    "cswap",
    "inv",
    # catopt.xcarrier
    "affd_a",
    "affd_b",
    "aff_A",
    "aff_b",
    "om_elem_affd",
    "om_elem_aff",
    "omd",
    "omd_elem",
    "omd_compose",
    "omd_apply",
    "omd_applym",
    # catopt.om
    "cmask",
    "fill",
    "attnbias",
    # catopt.act_eps
    "aquant",
    "adequant",
)

#: Base ops — the ``_CORE_TORCH_BINDINGS`` literal, including the
#: aff/om/affd carrier primitives that ship in core.
CORE_OPS = (
    "matmul",
    "add",
    "mul",
    "linear",
    "sdpa",
    "concat",
    "aff",
    "apply",
    "om",
    "om_elem",
    "om_compose",
    "om_apply",
    "aff_diag",
    "affd_compose",
    "applyd",
)


def _var(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _omd_apply_ir() -> IR:
    """``omd_apply(omd_elem(s, a, b), h)`` — a real xcarrier carrier
    term: s (Tq,K) scores, a/b (K,d) the affine fiber, h (d,) state."""
    s = _var("s", (3, 4))
    a = _var("a", (4, 5))
    b = _var("b", (4, 5))
    h = _var("h", (5,))
    inputs = [s, a, b, h]
    term = Op.make(
        "omd_apply", Op.make("omd_elem", s, a, b), h
    )
    return IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )


def _omd_apply_ref(s, a, b, h):
    """The omd carrier's denotation: softmax(s) applied to a⊙h+b."""
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    return ((e @ a) * h + e @ b) / e.sum(dim=-1, keepdim=True)


def _rand(*shape):
    return torch.randn(*shape)


class TestOpTableComposition:
    def test_core_has_no_carrier_bindings(self):
        t = OpTable.core()
        for op in CARRIER_OPS:
            assert op not in t.torch_bindings, op
        for op in CORE_OPS:
            assert op in t.torch_bindings, op

    def test_core_carries_shape_rules_and_attr_schemas(self):
        t = OpTable.core()
        for op in ("aff", "apply", "om", "om_elem", "trace", "eye"):
            assert op in t.shape_rules, op
        assert "sdpa" in t.attr_schemas
        assert 5 in t.attr_schemas["sdpa"]  # is_causal

    def test_full_has_all_bindings(self):
        t = OpTable.full()
        for op in (*CARRIER_OPS, *CORE_OPS):
            assert op in t.torch_bindings, op
            assert callable(t.torch_bindings[op])

    def test_full_seats_on_ambient_dict(self):
        """full() shares the ambient ``_IR_TO_TORCH`` — post-hoc
        binding overrides keep reaching IRModule, exactly the pre-2c
        global-registry semantics."""
        assert OpTable.full().torch_bindings is _IR_TO_TORCH

    def test_register_module_name_dict_and_kwargs(self):
        t = OpTable.core()
        t.register("catopt.trace")  # by module path
        t.register("om")  # bare name resolves to catopt.om
        assert "trace" in t.torch_bindings
        assert "cmask" in t.torch_bindings
        sentinel = lambda *a, **kw: "bound"  # noqa: E731
        t.register({"my_op": sentinel})  # plain dict of bindings
        assert t.torch_bindings["my_op"] is sentinel
        t.register(torch_bindings={"other": sentinel})
        assert t.torch_bindings["other"] is sentinel

    def test_copy_detaches_from_ambient(self):
        full = OpTable.full()
        snap = full.copy()
        assert "omd_elem" in snap.torch_bindings
        del snap.torch_bindings["omd_elem"]
        # the ambient table keeps the binding — the copy was private
        assert "omd_elem" in full.torch_bindings


def test_register_carrier_extends_composition(monkeypatch):
    """The extension point (Phase 3): a carrier module registered at
    runtime joins ``full()``/``carrier_torch_bindings()`` without
    editing core's built-in ``_CARRIER_MODULES`` tuple.  Appended in
    registration order, idempotent, and the built-in order is kept."""
    import sys
    from types import SimpleNamespace

    import catopt_core.ops as ops_mod

    # isolate the registry — monkeypatch restores it afterwards
    monkeypatch.setattr(ops_mod, "_registered_carriers", [])
    fake = SimpleNamespace(
        TORCH_BINDINGS={"mycarrier_op": lambda *a, **k: None}
    )
    monkeypatch.setitem(sys.modules, "my_carrier_pkg", fake)

    ops_mod.register_carrier("my_carrier_pkg")
    ops_mod.register_carrier("my_carrier_pkg")  # idempotent
    names = ops_mod._carrier_names()
    assert names[: len(ops_mod._CARRIER_MODULES)] == ops_mod._CARRIER_MODULES
    assert names.count("my_carrier_pkg") == 1
    # the registered module's bindings join the merged carrier table
    assert "mycarrier_op" in ops_mod.carrier_torch_bindings()


class TestCustomTableLowering:
    def test_minimal_table_lowers_carrier_term(self):
        """core + xcarrier lowers an omd carrier term — composed
        explicitly, no import side effects."""
        torch.manual_seed(0)
        import catopt.xcarrier as xc

        tbl = OpTable.core().register(xc)
        mod = IRModule(_omd_apply_ir(), ops=tbl)
        s, a, b, h = (
            _rand(3, 4),
            _rand(4, 5),
            _rand(4, 5),
            _rand(5),
        )
        out = mod(s, a, b, h)
        assert torch.allclose(
            out, _omd_apply_ref(s, a, b, h), atol=1e-10
        )

    def test_custom_table_lacks_other_carriers(self):
        """core + xcarrier alone cannot lower a trace op — the table
        boundary is real."""
        torch.manual_seed(0)
        import catopt.xcarrier as xc

        tbl = OpTable.core().register(xc)
        assert "trace" not in tbl.torch_bindings
        x = _var("x", (4, 4))
        ir = IR(
            root=Op.make("inv", x),
            inputs=[x],
            input_names={"x"},
            params={},
        )
        mod = IRModule(ir, ops=tbl)
        with pytest.raises(ValueError, match="No torch binding"):
            mod(_rand(4, 4))

    def test_removed_binding_fails_cleanly(self):
        """Deleting a binding from a custom table surfaces the
        'no torch binding' error at eval — no ambient fallback."""
        torch.manual_seed(0)
        import catopt.xcarrier as xc

        tbl = OpTable.core().register(xc)
        del tbl.torch_bindings["omd_elem"]
        mod = IRModule(_omd_apply_ir(), ops=tbl)
        with pytest.raises(
            ValueError, match="No torch binding for op 'omd_elem'"
        ):
            mod(_rand(3, 4), _rand(4, 5), _rand(4, 5), _rand(5))

    def test_default_lowering_uses_full_table(self):
        """Back-compat: ``ir_to_torch_module`` without ``ops`` lowers
        carrier terms — the ambient table preserves old behavior."""
        torch.manual_seed(0)
        mod = ir_to_torch_module(_omd_apply_ir())
        s, a, b, h = (
            _rand(3, 4),
            _rand(4, 5),
            _rand(4, 5),
            _rand(5),
        )
        assert torch.allclose(
            mod(s, a, b, h), _omd_apply_ref(s, a, b, h), atol=1e-10
        )
