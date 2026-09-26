"""Torch adapters for the graph source/sink ports.

The concrete pair behind ``optimize_model``'s defaults — the only
place PyTorch is required end-to-end:

* :class:`TorchSource` — ``torch.export`` → ATen → catopt IR
  (:class:`catopt_core.ports.Source`).
* :class:`TorchSink` — IR → :class:`IRModule`, plus the torch op table
  and the torch equivalence gate (:class:`catopt_core.ports.Sink`).

Nothing in ``catopt-core`` imports this module; a caller that passes a
different ``Source`` / ``Sink`` pair never touches torch (beyond the
export the source itself performs).  The ``supported_ops`` bound a
:class:`TorchSink` reports is exactly the key set of its
:class:`~catopt_core.ops.OpTable` — for the default ``full()`` table
that is every core op plus every installed carrier's ops, so the
backend-relative cost never excludes a form torch can lower.
"""

from __future__ import annotations

from typing import Any

from catopt_core.ir import IR
from catopt_core.ops import OpTable
from catopt_core.ports import Executor, OpRegistry

from catopt_torch.report import VerifyReport, verify_module
from catopt_torch.torch_bridge import export_to_ir, ir_to_torch_module

__all__ = ["TorchSink", "TorchSource"]


class TorchSource:
    """``catopt_core.ports.Source`` over ``torch.export``.

    ``to_ir`` is :func:`catopt_torch.torch_bridge.export_to_ir` verbatim:
    the model is exported in eval mode and its graph lifted to IR, with
    the concrete parameter/buffer values returned alongside so the
    paired sink can materialise the lowered module and the non-local
    passes can compare exact weights.
    """

    def to_ir(
        self, model: Any, example_inputs: Any
    ) -> tuple[IR, dict[str, Any]]:
        return export_to_ir(model, example_inputs)


class TorchSink:
    """``catopt_core.ports.Sink`` over an :class:`OpTable`.

    Parameters
    ----------
    ops : OpTable | None
        The lowering table.  ``None`` (default) resolves to
        ``OpTable.full()`` — the ambient table every core and installed
        carrier op is seated in, preserving ``optimize_model``'s
        historical default.  An explicit table owns its own dict: an op
        absent from it both drops out of :attr:`supported_ops` (so
        extraction never selects a form using it) and fails loudly at
        eval.
    """

    def __init__(self, ops: OpTable | None = None) -> None:
        self._ops = ops if ops is not None else OpTable.full()

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry const folds dispatch through."""
        return self._ops

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every op name this sink can lower — the table's key set."""
        return frozenset(self._ops.torch_bindings)

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor:
        """Materialise ``ir`` as an :class:`IRModule` (the concrete
        :class:`catopt_core.ports.Executor`)."""
        return ir_to_torch_module(
            ir, param_values=params, ops=self._ops
        )

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> VerifyReport:
        """Run ``ref`` and ``opt`` on ``inputs`` under ``no_grad`` and
        compare — delegates to ``report.verify_module``."""
        return verify_module(ref, opt, inputs, rtol=rtol, atol=atol)
