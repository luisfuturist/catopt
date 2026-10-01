"""Tests for the IR / term algebra."""

import pytest
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
    generator,
    op_def,
    op_repr,
    op_repr_dag,
)


def test_var():
    v = Var("x", TensorType((32, 64)))
    assert v.name == "x"
    assert v.typ.shape == (32, 64)
    assert str(v) == "x"


def test_const():
    c = Const(0.0)
    assert c.value == 0.0
    assert str(c) == "0.0"


def test_param():
    p = Param("W", TensorType((64, 32)))
    assert p.name == "W"
    assert str(p) == "W"


def test_op_construction():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("matmul", x, Const(2))
    assert op.op == "matmul"
    assert len(op.args) == 2


def test_op_nesting():
    x = Var("x", TensorType((1, 4)))
    nested = Op.make("add", Op.make("mul", x, Const(2)), Const(1))
    assert nested.op == "add"
    assert nested.args[0].op == "mul"
    assert nested.args[1] == Const(1)


def test_op_repr():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("add", x, Const(0))
    assert "add" in repr(op)  # Op __repr__ includes the op name


def test_op_repr_s_expression():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("mul", x, Const(2))
    s = op_repr(op)
    assert "(mul" in s and "x" in s and "2" in s


def test_ir_construction():
    x = Var("x", TensorType((1, 4)))
    root = Op.make("neg", x)
    ir = IR(root=root, inputs=[x])
    assert ir.root == root
    assert len(ir.inputs) == 1


def test_tensor_type():
    t = TensorType((None, 64))
    assert t.size is None  # unknown dim
    t2 = TensorType((32, 64))
    assert t2.size == 32 * 64


def test_tensor_type_repr():
    assert repr(TensorType((2, None))) == "TensorType(2, ?)"
    assert repr(TensorType(())) == "TensorType()"


def test_generator_registry():
    g = generator("add")
    assert g is not None
    assert g.commutative is True
    assert g.associative is True

    g2 = generator("matmul")
    assert g2 is not None
    assert g2.commutative is False  # matmul is NOT commutative

    g3 = generator("nonexistent")
    assert g3 is None


def test_op_attrs():
    x = Var("x", TensorType((32, 64)))
    op = Op.make("sum", x, axis=-1)
    assert op.attrs["axis"] == -1


def test_ir_repr():
    x = Var("x", TensorType((1, 4)))
    ir = IR(root=Op.make("neg", x), inputs=[x])
    assert "neg" in repr(ir)


def test_op_repr_includes_attrs():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("sum", x, axis=-1)
    assert op_repr(op) == "(sum x, axis=-1)"
    assert repr(op) == "sum(x, axis=-1)"


def test_op_repr_renders_args():
    x = Var("x", TensorType((1, 4)))
    assert repr(Op.make("neg", x)) == "neg(x)"


def test_op_make_validates_declared_positional_attrs():
    # ``arg5`` is undeclared for ``concat`` (its schema is {1: "dim"}).
    with pytest.raises(ValueError):
        Op.make("concat", "a", "b", arg5=1)
    # ``validate=False`` skips the schema contract entirely.
    relaxed = Op.make("concat", "a", "b", arg5=1, validate=False)
    assert relaxed.attrs["arg5"] == 1


def test_op_make_unhashable_arg_raises():
    with pytest.raises(TypeError):
        Op.make("noop", [1, 2])


def test_op_repr_two_attrs_use_comma_separator():
    a = Var("a", TensorType((1, 4)))
    b = Var("b", TensorType((1, 4)))
    op = Op.make("concat", a, b, dim=0, extra=1)
    assert repr(op) == "concat(a, b, dim=0, extra=1)"
    assert op_repr(op) == "(concat a, b, dim=0, extra=1)"


def test_op_def_defaults():
    g = op_def("mutation_probe_default", 2)
    assert g.n_out == 1
    assert g.commutative is False
    assert g.associative is False
    assert g.identity is None
    assert g.law == ""


def test_op_def_round_trips_all_fields():
    g = op_def(
        "mutation_probe_op",
        2,
        3,
        commutative=True,
        associative=True,
        identity=0,
        law="probe",
    )
    assert g.name == "mutation_probe_op"
    assert g.n_in == 2
    assert g.n_out == 3
    assert g.commutative is True
    assert g.associative is True
    assert g.identity == 0
    assert g.law == "probe"
    assert generator("mutation_probe_op") is g


def test_unhashable_attr_values_keep_ops_distinct():
    a = Op.make("noop", "x", meta=[])
    b = Op.make("noop", "x", meta=[1])
    assert a != b


# ---------------------------------------------------------------------------
#  op_repr_dag — sharing-aware rendering (morphism joint stats)
# ---------------------------------------------------------------------------


def test_op_repr_dag_leaves_and_trees_match_op_repr():
    """No sharing → identical output to ``op_repr`` (stats-safe)."""
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    terms = [
        x,
        Const(3),
        Op.make("mul", x, Const(2)),
        Op.make("add", Op.make("neg", x), Op.make("exp", y)),
        Op.make("concat", x, y, dim=0),
    ]
    for t in terms:
        assert op_repr_dag(t) == op_repr(t)


def test_op_repr_dag_binds_shared_subterm_once():
    """A twice-referenced op renders as one ``#n`` binding."""
    x = Var("x", TensorType((1, 4)))
    f = Op.make("neg", x)
    t = Op.make("add", f, f)
    assert op_repr(t) == "(add (neg x), (neg x))"
    assert op_repr_dag(t) == "(let ((#0 (neg x))) (add #0, #0))"


def test_op_repr_dag_defs_precede_uses_in_postorder():
    """Bindings are post-ordered: a shared def may cite earlier #n."""
    x = Var("x", TensorType((1, 4)))
    f = Op.make("neg", x)  # shared
    g = Op.make("add", f, f)  # shared, references f
    t = Op.make("mul", g, g)
    s = op_repr_dag(t)
    assert s == "(let ((#0 (neg x)) (#1 (add #0, #0))) (mul #1, #1))"


def test_op_repr_dag_diamond_renders_shared_once():
    x = Var("x", TensorType((1, 4)))
    f = Op.make("neg", x)
    t = Op.make("add", Op.make("exp", f), Op.make("sqrt", f))
    s = op_repr_dag(t)
    assert s.count("(neg x)") == 1
    assert s.count("#0") == 3  # one binding + two use sites


def test_op_repr_dag_shared_node_keeps_attrs():
    x = Var("x", TensorType((1, 4)))
    f = Op.make("sum", x, axis=-1)
    t = Op.make("add", f, f)
    s = op_repr_dag(t)
    assert "(#0 (sum x, axis=-1))" in s


def test_op_repr_dag_exponential_dag_stays_linear():
    """``t_{i+1} = add(t_i, t_i)`` — a tree repr is 2**n leaves.

    The DAG repr binds each level once, so depth-64 (unrenderable as
    a tree) stays a few-KB string and returns immediately.
    """
    x = Var("x", TensorType((1, 4)))
    t: Op | Var = x
    depth = 64
    for _ in range(depth):
        t = Op.make("add", t, t)
    s = op_repr_dag(t)
    assert s.startswith("(let ")
    assert len(s) < 8_000
    # every level bound exactly once, named exactly twice downstream
    assert s.count("(add ") == depth


def test_op_repr_dag_shared_leaf_stays_inline():
    """Only ``Op`` nodes get bindings — Var/Param/Const leaves do not."""
    x = Var("x", TensorType((1, 4)))
    p = Param("W", TensorType((4, 4)))
    t = Op.make("add", Op.make("matmul", x, p), Op.make("matmul", x, p))
    s = op_repr_dag(t)
    assert "(#0 (matmul x, W))" in s
    assert s == "(let ((#0 (matmul x, W))) (add #0, #0))"
