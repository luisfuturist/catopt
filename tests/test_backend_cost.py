"""backend_cost — backend-relative extraction pricing.

``backend_cost(cost_fn, supported_ops)`` prices any term using an op
outside a sink's op set at ``+inf``, so the reachable equivalence class
is bounded by the backend's semantic language rather than discovered
and then rejected at lowering.
"""

from catopt.cost import (
    backend_cost,
    flops_cost,
    param_bytes_cost_for,
    roofline_cost_for,
)
from catopt.ir import Op, Param, TensorType, Var
from catopt.ports import CostFn, signature_conforms

_X = Var("x", TensorType((2, 2)))
_SUPPORTED = Op.make("matmul", _X, Op.make("add", _X, _X))
_UNSUPPORTED = Op.make("matmul", _X, Op.make("neg", _X))


def test_backend_cost_unsupported_is_inf():
    priced = backend_cost(flops_cost, {"matmul", "add"})
    assert priced(_UNSUPPORTED) == float("inf")
    assert priced(_SUPPORTED) != float("inf")


def test_backend_cost_supported_matches_inner():
    priced = backend_cost(flops_cost, {"matmul", "add"})
    assert priced(_SUPPORTED) == flops_cost(_SUPPORTED)


def test_backend_cost_leaves_always_supported():
    def inner(term, memo=None):
        return 1.0

    priced = backend_cost(inner, set())
    assert priced(_X) == 1.0
    assert priced(Param("w", TensorType((2, 2)))) == 1.0


def test_backend_cost_caches_support_verdict():
    priced = backend_cost(flops_cost, {"matmul", "add"})
    # Second evaluation hits the memoized verdict for the same term.
    assert priced(_SUPPORTED) == priced(_SUPPORTED)


def test_backend_cost_marker_propagation():
    priced = backend_cost(
        param_bytes_cost_for({}, by_bytes=True), {"matmul"}
    )
    assert priced.charges_param_only
    assert priced.dag_exact
    profiled = backend_cost(roofline_cost_for(), {"matmul"})
    assert hasattr(profiled, "profile")


def test_backend_cost_preserves_name():
    assert backend_cost(flops_cost, {"matmul"}).__name__ == "flops_cost"


def test_backend_cost_forwards_memo_when_accepted():
    seen = {}

    def inner(term, memo=None):
        seen["memo"] = memo
        return 2.0

    priced = backend_cost(inner, {"matmul", "add"})
    sentinel = {"k": 1}
    assert priced(_SUPPORTED, memo=sentinel) == 2.0
    assert seen["memo"] is sentinel


def test_backend_cost_calls_bare_fn_without_memo():
    def inner(term):
        return 3.0

    priced = backend_cost(inner, {"matmul", "add"})
    assert priced(_SUPPORTED) == 3.0
    assert priced.__name__ == "inner"


def test_backend_cost_uninspectable_callable():
    # ``min`` has no inspectable signature — the fallback name is used.
    priced = backend_cost(min, {"matmul"})
    assert priced.__name__ == "min"


def test_backend_cost_is_cost_fn():
    priced = backend_cost(flops_cost, {"matmul", "add"})
    assert isinstance(priced, CostFn)
    assert signature_conforms(priced, CostFn)
