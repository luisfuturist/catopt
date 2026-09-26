"""Runtime conformance for the ports layer (:mod:`catopt.ports`).

The hexagonal boundary: real adapters must satisfy their port
protocols (``isinstance`` on ``@runtime_checkable`` protocols checks
member *presence*), and ``signature_conforms`` adds the signature-shape
check presence can't express — a callable with the wrong arity is not
a ``CostFn``.
"""

import torch

from catopt.cost import (
    CostModel,
    count_cost,
    dag_cost,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost,
    param_bytes_cost_for,
)
from catopt.ir import IR, Op, TensorType, Var
from catopt.om_lower import (
    StreamingOMModule,
    to_batched_om_module,
    to_streaming_om_module,
)
from catopt.omd_lower import to_batched_omd_module
from catopt.ops import OpTable
from catopt.ports import (
    BatchedExecutor,
    CostFn,
    Executor,
    LawSet,
    OpRegistry,
    PlannedExecutor,
    RuleLike,
    RuleProvider,
    ShapeRule,
    TorchBinding,
    Verifier,
    signature_conforms,
)
from catopt.report import VerifyReport, verify_equiv
from catopt.rules import CATEGORICAL_RULES, all_rules
from catopt.scan_lower import to_batched_scan_module
from catopt.torch_bridge import _IR_TO_TORCH, IRModule
from catopt.typing import _SHAPE_RULES

# ---------------------------------------------------------------------------
#  Fixtures — minimal real terms for each executor
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _ir(root, inputs):
    return IR(
        root=root,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params={},
    )


def _neg_ir() -> IR:
    x = _v("x", 2, 2)
    return _ir(Op.make("neg", x), [x])


def _scan_ir() -> IR:
    a, b, h = _v("A", 3, 3), _v("b", 3), _v("h", 3)
    return _ir(Op.make("apply", Op.make("aff", a, b), h), [a, b, h])


def _om_ir() -> IR:
    s, v = _v("s", 2, 4), _v("v", 4, 2)
    return _ir(Op.make("om_apply", Op.make("om_elem", s, v)), [s, v])


def _omd_ir() -> IR:
    s, a, b, h = _v("s", 3, 4), _v("a", 4, 2), _v("b", 4, 2), _v("h", 2)
    return _ir(
        Op.make("omd_apply", Op.make("omd_elem", s, a, b), h),
        [s, a, b, h],
    )


# ---------------------------------------------------------------------------
#  Executor — the lowered-module port
# ---------------------------------------------------------------------------


def test_irmodule_is_executor():
    mod = IRModule(_neg_ir())
    assert isinstance(mod, Executor)
    assert not isinstance(mod, PlannedExecutor)  # no eval_mod fallback
    x = torch.ones(2, 2)
    assert torch.equal(mod(x), -x)


def test_batched_scan_module_is_batched_executor():
    mod = to_batched_scan_module(_scan_ir())
    assert isinstance(mod, Executor)
    assert isinstance(mod, PlannedExecutor)
    assert isinstance(mod, BatchedExecutor)
    assert mod.is_batched
    eye, b, h = torch.eye(3), torch.ones(3), torch.zeros(3)
    assert torch.allclose(mod(eye, b, h), eye @ h + b)


def test_batched_om_module_is_batched_executor():
    mod = to_batched_om_module(_om_ir())
    assert isinstance(mod, BatchedExecutor)
    assert mod.is_batched
    s, v = torch.randn(2, 4), torch.randn(4, 2)
    assert torch.allclose(mod(s, v), torch.softmax(s, dim=-1) @ v)


def test_batched_omd_module_is_batched_executor():
    mod = to_batched_omd_module(_omd_ir())
    assert isinstance(mod, BatchedExecutor)
    assert mod.is_batched
    s = torch.randn(3, 4)
    a, b, h = torch.randn(4, 2), torch.randn(4, 2), torch.randn(2)
    want = torch.softmax(s, dim=-1) @ (a * h + b)
    assert torch.allclose(mod(s, a, b, h), want)


def test_streaming_om_module_is_planned_not_batched():
    """Streaming diverges on the discriminator name by design:
    ``is_streaming``, not ``is_batched``."""
    mod = to_streaming_om_module(_om_ir())
    assert isinstance(mod, Executor)
    assert isinstance(mod, PlannedExecutor)
    assert not isinstance(mod, BatchedExecutor)
    s, v = torch.randn(2, 4), torch.randn(4, 2)
    assert torch.allclose(mod(s, v), torch.softmax(s, dim=-1) @ v)


def test_any_nn_module_is_executor():
    """Presence-level check: ``forward`` is the whole port surface."""
    assert isinstance(torch.nn.Linear(2, 2), Executor)


def test_executor_negatives():
    assert not isinstance(42, Executor)
    assert not isinstance(lambda x: x, Executor)
    assert not isinstance(StreamingOMModule, BatchedExecutor)  # class


# ---------------------------------------------------------------------------
#  CostFn — extract-time pricing
# ---------------------------------------------------------------------------


def test_builtin_cost_models_are_cost_fns():
    for fn in (
        flops_cost,
        launch_aware_cost,
        count_cost,
        param_bytes_cost,
        param_bytes_cost_for(),
        CostModel(),
    ):
        assert isinstance(fn, CostFn), fn
        assert signature_conforms(fn, CostFn), fn


def test_cost_fn_wrong_signature_rejected():
    """A callable requiring args the extract sites never supply is not
    a CostFn — ``isinstance`` can't see this, ``signature_conforms``
    can."""
    assert not signature_conforms(lambda: 0.0, CostFn)
    assert not signature_conforms(lambda a, b, c: 0.0, CostFn)
    # dag_cost is a meta-wrapper: needs a cost_fn argument of its own.
    assert not signature_conforms(dag_cost, CostFn)
    assert not isinstance(object(), CostFn)


# ---------------------------------------------------------------------------
#  Verifier — the equivalence gate
# ---------------------------------------------------------------------------


def test_verify_equiv_is_verifier():
    assert isinstance(verify_equiv, Verifier)
    assert signature_conforms(verify_equiv, Verifier, strict=True)
    t = torch.ones(4)
    rep = verify_equiv(t, t.clone())
    assert isinstance(rep, VerifyReport) and rep.passed


def test_verifier_wrong_signature_rejected():
    # Call sites always pass rtol/atol by name — strict probe.
    assert not signature_conforms(lambda a, b: None, Verifier, strict=True)
    assert not signature_conforms(object(), Verifier)


# ---------------------------------------------------------------------------
#  OpRegistry — OpTable IS the adapter registry
# ---------------------------------------------------------------------------


def test_optable_conforms_to_registry():
    assert isinstance(OpTable.core(), OpRegistry)
    assert isinstance(OpTable.full(), OpRegistry)
    assert not isinstance({}, OpRegistry)


def test_registry_contents_are_port_typed():
    table = OpTable.full()
    assert isinstance(table.torch_bindings["matmul"], TorchBinding)
    assert isinstance(table.shape_rules["om_elem"], ShapeRule)
    assert isinstance(table.attr_schemas["conv2d"], dict)


# ---------------------------------------------------------------------------
#  TorchBinding / ShapeRule — the per-op adapter surfaces
# ---------------------------------------------------------------------------


def test_torch_bindings_conform():
    assert isinstance(_IR_TO_TORCH["matmul"], TorchBinding)
    assert isinstance(_IR_TO_TORCH["omd_elem"], TorchBinding)
    assert isinstance(lambda *a, **k: None, TorchBinding)
    assert not isinstance(123, TorchBinding)


def test_shape_rules_conform():
    for name in ("om_elem", "aff", "apply", "eye"):
        assert isinstance(_SHAPE_RULES[name], ShapeRule)
        assert signature_conforms(_SHAPE_RULES[name], ShapeRule)
    assert not signature_conforms(lambda a: None, ShapeRule)


# ---------------------------------------------------------------------------
#  LawSet / RuleProvider / RuleLike — laws as data
# ---------------------------------------------------------------------------


def test_rule_collections_conform():
    assert isinstance(CATEGORICAL_RULES, LawSet)
    assert isinstance(all_rules(), LawSet)
    assert all(isinstance(r, RuleLike) for r in CATEGORICAL_RULES)
    assert isinstance(all_rules, RuleProvider)
    # presence-level: a dict IS iterable (LawSet) — documented caveat.
    assert not isinstance(42, LawSet)
    assert not isinstance(42, RuleProvider)
    assert not isinstance(object(), RuleLike)
