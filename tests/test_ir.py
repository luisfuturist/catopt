"""Tests for the IR / term algebra."""

import pytest
from catopt.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
    generator,
    op_def,
    op_repr,
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
