"""Backend-relative pricing — bound extraction to a sink's op set.

:func:`backend_cost` wraps a cost fn so terms using ops outside a
sink's ``supported_ops`` price at ``+inf``; :class:`CostModel` is the
configurable weighted-FLOP model.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, cast

from catopt_core.ir import Op
from catopt_core.typing import _infer_op_shape, _numel, _shape_of

from .basic import _OP_FLOPS

if TYPE_CHECKING:
    from catopt_core.ports import CostFn


class CostModel:
    """A configurable cost model for EGraph.extract_best."""

    def __init__(
        self,
        op_weights: dict[str, float] | None = None,
        weight_coeff: float = 1.0,
        matmul_coeff: float = 2.0,
    ) -> None:
        """Initialise the op weights and coefficients."""
        self.op_weights = op_weights or _OP_FLOPS
        self.weight_coeff = weight_coeff
        self.matmul_coeff = matmul_coeff

    def __call__(self, term: Any) -> float:
        """Price *term* in weighted FLOPs."""
        if isinstance(term, Op):
            shape = _infer_op_shape(term)
            n = _numel(shape)
            coeff = self.op_weights.get(term.op, self.weight_coeff)
            if term.op == "matmul":
                shapes = [_shape_of(a) for a in term.args]
                if shapes and shapes[1] is not None:
                    k_dim = (
                        cast("int", shapes[1][-2])
                        if len(shapes[1]) >= 2
                        else 1
                    )
                    base = 2 * n * k_dim
                else:
                    base = coeff * n
            elif term.op == "linear":
                # F.linear(x[.., in], W[out, in]) -> 2 * M * out * in
                shapes = [_shape_of(a) for a in term.args]
                if (
                    shapes
                    and shapes[0] is not None
                    and len(shapes[0]) >= 1
                ):
                    base = 2 * n * cast("int", shapes[0][-1])
                else:
                    base = 2 * n
            else:
                base = coeff * n
            for arg in term.args:
                base += self(arg)
            return float(base)
        return 0.0


# ---------------------------------------------------------------------------
#  Backend-relative pricing — the sink's supported-op bound
# ---------------------------------------------------------------------------


def _ops_supported(
    term: Any, allowed: frozenset[str], cache: dict[Any, bool]
) -> bool:
    """Return True iff ``term``'s DAG uses only ``allowed`` ops.

    Leaves (``Param`` / ``Const`` / ``Var``) are always supported.
    ``cache`` memoizes the content-keyed verdict, so a shared-subterm
    DAG costs one linear walk across every extraction probe rather than
    one walk per probe.
    """
    hit = cache.get(term)
    if hit is not None:
        return hit
    ok = not isinstance(term, Op) or (
        term.op in allowed
        and all(_ops_supported(a, allowed, cache) for a in term.args)
    )
    cache[term] = ok
    return ok


def backend_cost(
    cost_fn: CostFn, supported_ops: Collection[str]
) -> CostFn:
    """Price members outside a backend's op set at ``+inf``.

    Wraps ``cost_fn`` so extraction is *backend-relative*.

    ``supported_ops`` is the op-name set a
    :class:`~catopt_core.ports.Sink` can lower (its ``supported_ops``).
    Any term using an op outside the set can never win extraction, so
    the optimizer commits only to forms the sink can execute: the
    reachable equivalence class is bounded by the backend's semantic
    language rather than discovered and then rejected at lowering.
    Leaves are always supported.

    The wrapper preserves the wrapped model's ``charges_param_only`` /
    ``dag_exact`` / ``profile`` markers, so :func:`dag_cost` and
    extraction bill a ``param_bytes_cost``-based model exactly as
    before.  It always declares ``memo`` (so the extraction memo is
    threaded in) but forwards it only to a ``cost_fn`` that accepts it,
    matching :func:`_memo_dispatch`'s adaptive call.
    """
    import inspect

    allowed = frozenset(supported_ops)
    cache: dict[Any, bool] = {}
    try:
        accepts_memo = "memo" in inspect.signature(cost_fn).parameters
    except (TypeError, ValueError):  # uninspectable callable
        accepts_memo = False

    def priced(term: Any, memo: dict | None = None) -> float:
        if not _ops_supported(term, allowed, cache):
            return float("inf")
        return cost_fn(term, memo) if accepts_memo else cost_fn(term)

    priced.__name__ = getattr(cost_fn, "__name__", "backend_cost")
    for marker in ("charges_param_only", "dag_exact", "profile"):
        if hasattr(cost_fn, marker):
            setattr(priced, marker, getattr(cost_fn, marker))
    return priced
