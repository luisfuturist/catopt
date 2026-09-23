"""Tests for the IR / term algebra."""

import pytest
from catopt.ir import (
    Var, Const, Param, Op, IR, TensorType,
    generator, op_repr, op_def, _OP_REGISTRY,
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
