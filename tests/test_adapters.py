"""TorchSource / TorchSink — the concrete graph source/sink adapters.

The default pair behind ``optimize_model``: ``TorchSource`` lifts a
``torch.export`` graph to IR, ``TorchSink`` lowers IR back to an
:class:`IRModule`, reports the torch op set as ``supported_ops``, and
delegates the equivalence gate to ``report.verify_module``.
"""

import torch
import torch.nn as nn
from catopt.adapters import TorchSink, TorchSource
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.ops import OpTable


def _neg_ir() -> IR:
    x = Var("x", TensorType((2, 2)))
    return IR(
        root=Op.make("neg", x),
        inputs=[x],
        input_names={"x"},
        params={},
    )


def _scale_ir() -> IR:
    x = Var("x", TensorType((2, 2)))
    w = Param("w", TensorType((2, 2)))
    return IR(
        root=Op.make("mul", x, w),
        inputs=[x],
        input_names={"x"},
        params={"w": w},
    )


class _Neg(nn.Module):
    def forward(self, x):
        return -x


def test_torch_source_to_ir():
    m = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    ir, tensors = TorchSource().to_ir(m, x)
    assert isinstance(ir, IR)
    assert ir.root is not None
    assert len(tensors) == 2
    assert all(k.startswith("p_") for k in tensors)
    assert all(isinstance(t, torch.Tensor) for t in tensors.values())


def test_torch_sink_default_table_covers_carriers():
    sink = TorchSink()
    assert isinstance(sink.ops, OpTable)
    # Core op and carrier op are both lowerable by the full table.
    assert "matmul" in sink.supported_ops
    assert "omd_elem" in sink.supported_ops


def test_torch_sink_custom_table_bounds_ops():
    sink = TorchSink(ops=OpTable.core())
    assert "matmul" in sink.supported_ops
    assert "omd_elem" not in sink.supported_ops


def test_torch_sink_lower_without_params():
    mod = TorchSink().lower(_neg_ir())
    x = torch.randn(2, 2)
    assert torch.allclose(mod(x), -x)


def test_torch_sink_lower_with_params():
    w = torch.randn(2, 2)
    mod = TorchSink().lower(_scale_ir(), {"w": w})
    x = torch.randn(2, 2)
    assert torch.allclose(mod(x), x * w)


def test_torch_sink_verify_pass_and_atol():
    mod = TorchSink().lower(_neg_ir())
    x = torch.randn(2, 2)
    rep = TorchSink().verify(_Neg(), mod, x)
    assert rep.passed and rep.max_rel == 0.0
    rep2 = TorchSink().verify(_Neg(), mod, x, rtol=1e-9, atol=1e-6)
    assert rep2.passed
