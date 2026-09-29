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

from collections.abc import Mapping
from typing import Any

from catopt_core.ir import IR
from catopt_core.ops import OpTable
from catopt_core.ports import (
    Executor,
    ExecutorSpec,
    OpRegistry,
)

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
        """Export ``model`` to IR plus its leaf values."""
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
        """Initialise the sink's op table."""
        self._ops = ops if ops is not None else OpTable.full()
        self._executors: dict[str, ExecutorSpec] | None = None

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry const folds dispatch through."""
        return self._ops

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every op name this sink can lower — the table's key set."""
        return frozenset(self._ops.torch_bindings)

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """The torch executor table — named carrier lowerers.

        The table ``_lower_extracted`` / the regime planner route
        through: carrier families first (scan / om / omd / om
        streaming / trace), the generic lowering last — mapping order
        IS routing order.  Built lazily: ``catopt-carriers`` imports
        this package, so the carrier lowerers resolve at first
        access, never at import time.
        """
        if self._executors is None:
            self._executors = _executor_specs(self._ops)
        return self._executors

    def specialize_causal(
        self, term: Any, params: dict, memo: dict
    ) -> Any:
        """Apply the causal-mask const fold — optional capability hook.

        ``sdpa(q,k,v, mask)`` → ``sdpa(q,k,v, is_causal=True)`` when
        the parameter-only mask evaluates to the causal lower
        triangle; ``memo["_hit"]`` records whether any rewrite fired.
        """
        from catopt_torch.folds import _specialize_causal

        return _specialize_causal(term, params, memo, ops=self._ops)

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor:
        """Materialise ``ir`` as an :class:`IRModule`.

        The concrete :class:`catopt_core.ports.Executor`.
        """
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
        """Run ``ref`` and ``opt`` on ``inputs`` under ``no_grad``.

        Compares the results — delegates to ``report.verify_module``.
        """
        return verify_module(ref, opt, inputs, rtol=rtol, atol=atol)


def _always_true(_: Any) -> bool:
    return True


def _is_trace_rooted(term: Any) -> bool:
    """Probe for a trace root — deferred to the orchestrator probe."""
    from catopt_optimize.regime import is_trace_rooted_term

    return is_trace_rooted_term(term)


def _executor_specs(ops: OpTable) -> dict[str, ExecutorSpec]:
    """Build the torch executor table for *ops*.

    The named :class:`ExecutorSpec` entries the pipeline routes
    through — the same table the regime ``EXECUTORS`` registry
    carries (``catopt_torch.regime`` registers it as the ambient
    default).  Mapping order is routing order: carrier executors
    first — disjoint ``accepts`` probes except ``om_batched``
    preceding the streaming alternative — the catch-all ``generic``
    last.

    Carrier machinery resolves at call time (``catopt_carriers``
    imports ``catopt_torch``, so module-level carrier references are
    impossible here); the generic/trace lowerers bind *ops* so a
    custom table lowers through its own registry.  A partial install
    without ``catopt_carriers`` still yields ``trace`` + ``generic``
    (the missing carrier families simply never route).
    """

    def generic_lower(ir: IR, params: dict | None = None) -> Executor:
        return ir_to_torch_module(ir, param_values=params, ops=ops)

    table: dict[str, ExecutorSpec] = {}
    try:
        from catopt_carriers.om_lower import (
            is_om_apply_term,
            to_batched_om_module,
            to_streaming_om_module,
        )
        from catopt_carriers.omd_lower import (
            is_omd_apply_term,
            to_batched_omd_module,
        )
        from catopt_carriers.scan_lower import (
            is_scan_apply_term,
            to_batched_scan_module,
        )
    except ModuleNotFoundError:
        pass
    else:
        table.update(
            {
                "scan": ExecutorSpec(
                    "scan",
                    lambda ir, p: to_batched_scan_module(
                        ir, param_values=p
                    ),
                    is_scan_apply_term,
                    lambda m: bool(getattr(m, "is_batched", False)),
                    (
                        frozenset({"apply", "applyd"}),
                        frozenset({"aff_compose", "affd_compose"}),
                        frozenset({"aff", "aff_diag"}),
                    ),
                ),
                "om_batched": ExecutorSpec(
                    "om_batched",
                    lambda ir, p: to_batched_om_module(
                        ir, param_values=p
                    ),
                    is_om_apply_term,
                    lambda m: bool(getattr(m, "is_batched", False)),
                    (
                        frozenset({"om_apply"}),
                        frozenset({"om_compose"}),
                        frozenset({"om", "om_elem"}),
                    ),
                ),
                "omd_batched": ExecutorSpec(
                    "omd_batched",
                    lambda ir, p: to_batched_omd_module(
                        ir, param_values=p
                    ),
                    is_omd_apply_term,
                    lambda m: bool(getattr(m, "is_batched", False)),
                    (
                        frozenset({"omd_apply", "omd_applym"}),
                        frozenset({"omd_compose"}),
                        frozenset({"omd", "omd_elem"}),
                    ),
                ),
                "om_streaming": ExecutorSpec(
                    "om_streaming",
                    lambda ir, p: to_streaming_om_module(
                        ir, param_values=p
                    ),
                    is_om_apply_term,
                    lambda m: bool(getattr(m, "is_streaming", False)),
                    (
                        frozenset({"om_apply"}),
                        frozenset({"om_compose"}),
                        frozenset({"om", "om_elem"}),
                    ),
                ),
            }
        )
    table["trace"] = ExecutorSpec(
        "trace",
        generic_lower,
        _is_trace_rooted,
        _always_true,
        (
            frozenset({"trace"}),
            frozenset({"parl", "bdiag"}),
            frozenset({"eye", "cswap"}),
        ),
    )
    table["generic"] = ExecutorSpec(
        "generic",
        generic_lower,
        _always_true,
        _always_true,
        None,
    )
    return table
