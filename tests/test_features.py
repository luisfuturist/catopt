"""Static program features (:mod:`catopt_core.features`)."""

from catopt_core.features import (
    DIMENSIONS,
    ProgramFeatures,
    StaticProfiler,
    _elems,
    compute_features,
)
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.ports import PerformanceModel, Policy, Profiler


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def test_leaf_has_no_compute():
    f = compute_features(_v("x", 4, 4))
    assert f == ProgramFeatures(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0)


def test_const_leaf_has_no_compute():
    assert compute_features(Const(3.0)).operations == 0


def test_as_dict_and_vector():
    f = compute_features(_v("x", 2, 2))
    assert set(f.as_dict()) == set(DIMENSIONS)
    assert f.to_vector() == tuple(0.0 for _ in DIMENSIONS)
    assert f.to_vector(("flops", "depth")) == (0.0, 0.0)


def test_elementwise_features():
    x = _v("x", 2, 3)
    y = _v("y", 2, 3)
    f = compute_features(Op.make("add", x, y))
    assert f.operations == 1
    assert f.flops == 6.0
    assert f.bytes_written == 24.0
    assert f.bytes_read == 48.0
    assert f.temporary_bytes == 0.0
    assert f.depth == 0
    assert f.parallelism == 1.0
    assert f.reuse == 6.0 / 72.0


def test_matmul_flops():
    a = _v("a", 4, 8)
    b = _v("b", 8, 4)
    assert compute_features(Op.make("matmul", a, b)).flops == 256.0


def test_chain_depth_and_temporaries():
    x = _v("x", 4, 4)
    w = _v("w", 4, 4)
    y = Op.make("add", Op.make("matmul", x, w), x)
    f = compute_features(y)
    assert f.operations == 2
    assert f.depth == 1
    assert f.temporary_bytes == 16 * 4
    assert f.parallelism == 2.0


def test_param_leaves_counted():
    p = Param("w", TensorType((4, 4)))
    f = compute_features(Op.make("matmul", _v("x", 4, 4), p))
    assert f.param_leaves == 1


def test_zero_size_program_has_no_reuse():
    x = _v("x", 0, 0)
    f = compute_features(Op.make("mul", x, x))
    assert f.reuse == 0.0
    assert f.bytes_written == 0.0


def test_elems_guards():
    assert _elems(None) == 0
    assert _elems((2, None)) == 0
    assert _elems((2, 3)) == 6


def test_static_profiler_conforms_and_scales_with_itemsize():
    p = StaticProfiler()
    assert isinstance(p, Profiler)
    assert p.name == "static"
    t = Op.make("add", _v("x", 2, 2), _v("y", 2, 2))
    assert p.profile(t).bytes_written == 16.0
    assert StaticProfiler(itemsize=8).profile(t).bytes_written == 32.0


def test_view_ops_bill_no_traffic():
    """A transpose owns no storage — it must not be billed as a write."""
    x = _v("x", 4, 8)
    f = compute_features(Op.make("transpose", x))
    assert f.bytes_written == 0.0
    assert f.bytes_read == 0.0
    assert compute_features(Op.make("reshape", x)).bytes_written == 0.0


def test_param_only_subtrees_bill_no_traffic():
    """A param-only chain folds at compile time — no runtime traffic."""
    a = Param("a", TensorType((4, 4)))
    b = Param("b", TensorType((4, 4)))
    f = compute_features(Op.make("matmul", a, b))
    assert f.bytes_written == 0.0
    assert f.bytes_read == 0.0
    assert f.flops > 0.0  # the arithmetic is still described


def test_mixed_program_bills_only_the_runtime_half():
    """add(transpose(x), y): the view itself is free, its reader is not."""
    x = _v("x", 4, 8)
    y = _v("y", 8, 4)
    f = compute_features(Op.make("add", Op.make("transpose", x), y))
    # the add's output is 32 elements, fp32
    assert f.bytes_written == 32 * 4
    # the add reads both operands (32 + 32 elements); the transpose
    # node itself reads and writes nothing
    assert f.bytes_read == 64 * 4
    assert compute_features(Op.make("transpose", x)).bytes_read == 0.0


def test_new_port_negatives():
    assert not isinstance(object(), Profiler)
    assert not isinstance(object(), Policy)
    assert not isinstance(object(), PerformanceModel)
