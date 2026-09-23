"""Categorical / string-diagram IR for tensor computation graphs.

Terms form a simply-typed algebra where each generator (operation) has
named inputs and outputs, mirroring morphisms in a symmetric monoidal
category.  Sequential composition is ordinary function application
(f ∘ g), and the monoidal product is parallel/tensor composition.

In the e-graph representation each term is an *ENode*: an operation
label plus a list of child e-class IDs.  The e-graph itself (see
:mod:`catopt.egraph`) maintains equivalence(classes of ENodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Var",
    "Const",
    "Param",
    "Op",
    "IR",
    "op_def",
    "generator",
    "op_repr",
    "TensorType",
]


# ---------------------------------------------------------------------------
#  Shape / type system (minimal — enough for our cost model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorType:
    """A tensor type characterized by its shape."""

    shape: tuple[int | None, ...]

    @property
    def size(self) -> int | None:
        if any(d is None for d in self.shape):
            return None
        result: int | None = 1
        for d in self.shape:
            result = result * d
        return result

    def __repr__(self) -> str:
        inner = ", ".join("?" if d is None else str(d) for d in self.shape)
        return f"TensorType({inner})"


# ---------------------------------------------------------------------------
#  Terms  (these become ENodes in the e-graph)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    """A variable / placeholder (input to the computation)."""

    name: str
    typ: TensorType

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    """A literal constant scalar."""

    value: float

    def __repr__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Param:
    """A learnable parameter (weight matrix, bias, etc.)."""

    name: str
    typ: TensorType

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Op:
    """An operation (an ENode in the e-graph)."""

    op: str
    args: tuple[Any, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(op: str, *args, **attrs) -> "Op":
        return Op(op, args, attrs)

    def __repr__(self) -> str:
        parts = [op_repr(a) for a in self.args]
        if self.attrs:
            attr_str = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
            parts.append(attr_str)
        inside = ", ".join(parts)
        return f"{self.op}({inside})"


def op_repr(term: Any) -> str:
    """Compact S-expression-style rendering of a term."""
    if isinstance(term, Op):
        parts = [op_repr(a) for a in term.args]
        if term.attrs:
            parts.append(
                ", ".join(f"{k}={v}" for k, v in term.attrs.items())
            )
        return f"({term.op} {', '.join(parts)})"
    return repr(term)


# ---------------------------------------------------------------------------
#  Generator registry — declares which ops exist and their signatures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenDef:
    """Definition of a generator (operation) in the categorical semantics.

    Documents the categorical law(s) that make the operation interesting
    for rewriting.
    """

    name: str
    n_in: int          # number of input wires
    n_out: int         # number of output wires
    commutative: bool = False
    associative: bool = False
    identity: Any | None = None
    law: str = ""      # human-readable description of the categorical law


_OP_REGISTRY: dict[str, GenDef] = {}


def op_def(
    name: str,
    n_in: int,
    n_out: int = 1,
    *,
    commutative: bool = False,
    associative: bool = False,
    identity: Any | None = None,
    law: str = "",
) -> GenDef:
    """Register a generator and return its definition."""
    g = GenDef(name, n_in, n_out, commutative, associative, identity, law)
    _OP_REGISTRY[name] = g
    return g


def generator(name: str) -> GenDef | None:
    """Look up a registered generator by name."""
    return _OP_REGISTRY.get(name)


# ---------------------------------------------------------------------------
#  Predefined generators
# ---------------------------------------------------------------------------

# Linear algebra
op_def(
    "matmul", 2, 1,
    law="Bilinear; distributes over elementwise add when weight is fixed.",
)
op_def(
    "add", 2, 1, commutative=True, associative=True, identity=0,
    law="Addition forms a commutative monoid (SMC symmetry + associativity).",
)
op_def(
    "mul", 2, 1, commutative=True, associative=True, identity=1,
    law="Multiplication forms a commutative monoid (Hadamard product).",
)
op_def("sub", 2, 1, commutative=False,
       law="Subtraction: a - b = a + neg(b)  (distributivity of negation).")
op_def("div", 2, 1, commutative=False,
       law="Division: a / b = a * (1/b)  (multiplicative inverse).")
op_def("pow", 2, 1, commutative=False,
       law="Power: x**n, with square(x)=pow(x,2) and rsqrt as inverse sqrt.")

# Unary element-wise
op_def("square", 1, 1, law="Self-composition: square(x) = mul(x, x).")
op_def("sqrt", 1, 1, law="Square root: sqrt(x)**2 = x.")
op_def("neg", 1, 1, law="Additive inverse: neg(x) + x = 0  (group inverse).")
op_def("rsqrt", 1, 1, law="rsqrt(x)*sqrt(x)=1 (group inverse under mul).")
op_def("exp", 1, 1, law="Exponential.")
op_def("sigmoid", 1, 1, law="Logistic sigmoid; silu(x)=x*sigmoid(x).")
op_def("silu", 1, 1, law="SiLU/Swish: silu(x) = x * sigmoid(x).")
op_def("tanh", 1, 1, law="Hyperbolic tangent.")
op_def("gelu", 1, 1, law="Gaussian Error Linear Unit.")

# Reductions
op_def("sum", 1, 1,
       law="Sum over axes; distributes over add (additivity of trace/sum).")
op_def("mean", 1, 1,
       law="Mean = sum / size (linearity of expectation).")
op_def("max", 1, 1,
       law="Maximum; sublinear (convexity).")

# Data movement
op_def("transpose", 1, 1, law="Symmetry/isomorphism in the SMC.")
op_def("reshape", 1, 1,
       law="Relabeling of wires; composes associatively.")
op_def("broadcast", 1, 1,
       law="Monoidal coherence: copy then f = f in parallel.")


# ---------------------------------------------------------------------------
#  IR  — top-level program (a single term with free variables)
# ---------------------------------------------------------------------------

@dataclass
class IR:
    """A program: a term plus the set of free variables (inputs).

    The ``root`` term is the computation; ``inputs`` lists the
    :class:`Var` objects that are the program's inputs.  Parameters
    (:class:`Param`) are implicitly tracked as the learnable weights
    inside the term.
    """

    root: Any  # Var | Const | Param | Op
    inputs: list[Var] = field(default_factory=list)
    input_names: set[str] = field(default_factory=set)
    params: dict[str, Param] = field(default_factory=dict)

    def __repr__(self) -> str:
        return op_repr(self.root)
